#!/usr/bin/env python3
"""Flash K-Means benchmark (large workloads only): measures throughput and verifies correctness."""

import sys
import os
import math
import torch

# ─── Workload definitions (large only) ──────────────────────────────────
WORKLOADS = [
    {"label": "large-dense",  "B": 32, "N": 65536,  "D": 128, "K": 4096},
    {"label": "large-scale",  "B": 8,  "N": 131072, "D": 128, "K": 1000},
    {"label": "stress",       "B": 4,  "N": 262144, "D": 128, "K": 8192},
]

MAX_ITERS = 10
SEED = 42
NUM_WARMUP = 3
NUM_TIMED = 10
INERTIA_TOL = 0.01  # 1% relative error allowed

DTYPE = torch.float16
DEVICE = "cuda"


def compute_inertia(x, cluster_ids, centroids):
    """External inertia: sum ||x - centroids[cluster_ids]||^2, computed in float32."""
    assigned = centroids.gather(
        1, cluster_ids.unsqueeze(-1).expand(-1, -1, centroids.shape[-1])
    )
    diff = x.float() - assigned.float()
    return (diff ** 2).sum().item()


def run_reference(x, init_centroids, K):
    """Run torch_fallback reference implementation."""
    from flash_kmeans.torch_fallback import batch_kmeans_Euclid_torch_native
    cluster_ids, centroids, _ = batch_kmeans_Euclid_torch_native(
        x, K, max_iters=MAX_ITERS, tol=-1.0, init_centroids=init_centroids.clone()
    )
    return cluster_ids, centroids


def run_agent(x, init_centroids, K):
    """Run the agent's flash_kmeans implementation."""
    from flash_kmeans import batch_kmeans_Euclid
    cluster_ids, centroids, _ = batch_kmeans_Euclid(
        x, K, max_iters=MAX_ITERS, tol=-1.0, init_centroids=init_centroids.clone()
    )
    return cluster_ids, centroids


def check_outputs(cluster_ids, centroids, B, N, K, D):
    """Validate output shapes and ranges."""
    if cluster_ids.shape != (B, N):
        return False, f"cluster_ids shape {cluster_ids.shape} != expected ({B}, {N})"
    if centroids.shape != (B, K, D):
        return False, f"centroids shape {centroids.shape} != expected ({B}, {K}, {D})"
    cmin, cmax = cluster_ids.min().item(), cluster_ids.max().item()
    if cmin < 0 or cmax >= K:
        return False, f"cluster_ids range [{cmin}, {cmax}] not in [0, {K})"
    return True, "OK"


def check_no_cuda_files():
    """Check that no .cu or .so files exist in flash_kmeans/."""
    pkg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flash_kmeans")
    for root, _, files in os.walk(pkg_dir):
        for f in files:
            if f.endswith(".cu"):
                return False, f"Found CUDA source: {os.path.join(root, f)}"
            if f.endswith(".so"):
                return False, f"Found shared object: {os.path.join(root, f)}"
    return True, "OK"


def print_summary(throughput, valid_wl, total_wl, valid):
    """Print parseable summary to stdout."""
    print(f"\n{'='*60}", file=sys.stderr)
    print("BENCHMARK COMPLETE", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    # Machine-readable output on stdout
    print(f"throughput_mpps={throughput:.4f}")
    print(f"valid_workloads={valid_wl}")
    print(f"total_workloads={total_wl}")
    print(f"valid={'true' if valid else 'false'}")


def main():
    # Anti-cheat: no .cu/.so files
    ok, msg = check_no_cuda_files()
    if not ok:
        print(f"ANTI-CHEAT FAILURE: {msg}", file=sys.stderr)
        print_summary(0.0, 0, len(WORKLOADS), False)
        return

    results = []
    valid_count = 0
    all_valid = True

    for idx, wl in enumerate(WORKLOADS):
        label = wl["label"]
        B, N, D, K = wl["B"], wl["N"], wl["D"], wl["K"]
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Workload: {label}  (B={B}, N={N}, D={D}, K={K})", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        try:
            # 1. Generate data with per-workload seed
            gen = torch.Generator(device=DEVICE)
            gen.manual_seed(SEED + idx)
            x = torch.randn(B, N, D, device=DEVICE, dtype=DTYPE, generator=gen)

            # 2. Generate shared initial centroids (random indices from data)
            indices = torch.randint(0, N, (B, K), device=DEVICE, generator=gen)
            init_centroids = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, D)).clone()

            # 3. Run reference (torch_fallback)
            ref_ids, ref_centroids = run_reference(x, init_centroids, K)
            ref_inertia = compute_inertia(x, ref_ids, ref_centroids)
            print(f"  Reference inertia: {ref_inertia:.2f}", file=sys.stderr)

            # 4. Warmup agent implementation
            for _ in range(NUM_WARMUP):
                run_agent(x, init_centroids, K)
            torch.cuda.synchronize()

            # 5. Timed runs with CUDA events
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMED)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMED)]

            for i in range(NUM_TIMED):
                start_events[i].record()
                agent_ids, agent_centroids = run_agent(x, init_centroids, K)
                end_events[i].record()

            torch.cuda.synchronize()
            times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            avg_time_ms = sum(times_ms) / len(times_ms)
            avg_time_sec = avg_time_ms / 1000.0

            # 6. Correctness check: shapes and ranges
            ok, msg = check_outputs(agent_ids, agent_centroids, B, N, K, D)
            if not ok:
                print(f"  OUTPUT CHECK FAILED: {msg}", file=sys.stderr)
                results.append({"label": label, "throughput": 0.0, "valid": False})
                all_valid = False
                continue

            # 7. Correctness check: inertia
            agent_inertia = compute_inertia(x, agent_ids, agent_centroids)
            print(f"  Agent inertia:     {agent_inertia:.2f}", file=sys.stderr)

            if ref_inertia > 0:
                rel_error = (agent_inertia - ref_inertia) / ref_inertia
            else:
                rel_error = 0.0

            print(f"  Relative error:    {rel_error:.6f}", file=sys.stderr)

            if rel_error > INERTIA_TOL:
                print(f"  CORRECTNESS FAILED: rel_error {rel_error:.6f} > {INERTIA_TOL}", file=sys.stderr)
                results.append({"label": label, "throughput": 0.0, "valid": False})
                all_valid = False
                continue

            # 8. Compute throughput: mega-points-iters/sec
            total_point_iters = B * N * MAX_ITERS
            throughput_mpps = (total_point_iters / avg_time_sec) / 1e6
            print(f"  Avg time:          {avg_time_ms:.2f} ms", file=sys.stderr)
            print(f"  Throughput:        {throughput_mpps:.4f} Mpts-iters/s", file=sys.stderr)

            results.append({"label": label, "throughput": throughput_mpps, "valid": True})
            valid_count += 1

            # Free memory between workloads
            del x, init_centroids, ref_ids, ref_centroids, agent_ids, agent_centroids
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  EXCEPTION: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            results.append({"label": label, "throughput": 0.0, "valid": False})
            all_valid = False

    # Geometric mean of throughputs (only if ALL workloads valid)
    valid_throughputs = [r["throughput"] for r in results if r["valid"] and r["throughput"] > 0]
    if valid_throughputs and all_valid:
        geo_mean = math.exp(sum(math.log(t) for t in valid_throughputs) / len(valid_throughputs))
    else:
        geo_mean = 0.0

    # Per-workload summary
    print(f"\n{'─'*60}", file=sys.stderr)
    for r in results:
        status = "PASS" if r["valid"] else "FAIL"
        print(f"  {r['label']:15s}  {r['throughput']:10.4f} Mpts-iters/s  [{status}]", file=sys.stderr)
    print(f"  {'GEO MEAN':15s}  {geo_mean:10.4f} Mpts-iters/s", file=sys.stderr)
    print(f"{'─'*60}", file=sys.stderr)

    print_summary(geo_mean, valid_count, len(WORKLOADS), all_valid)


if __name__ == "__main__":
    main()
