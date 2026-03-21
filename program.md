# Flash K-Means (Large) — Optimize Triton Kernels for Maximum Clustering Throughput

Optimize the Triton GPU kernels in `flash_kmeans/` to maximize batched K-Means clustering throughput on a single H100 GPU while preserving correctness. This variant benchmarks only large-scale workloads — no easy wins from small-kernel overhead.

## Setup

1. **Read the in-scope files**:
   - `flash_kmeans/` — the 6 modifiable kernel files (see below).
   - `flash_kmeans/torch_fallback.py` — correctness reference. **Do not modify.**
   - `eval/eval.sh`, `eval/benchmark.py` — evaluation harness. **Do not modify.**
2. **Run prepare**: `bash prepare.sh` to install dependencies and verify CUDA + Triton.
3. **Initialize results.tsv**: Create `results.tsv` with just the header row.
4. **Run baseline**: `bash eval/eval.sh > run.log 2>&1` to establish the starting throughput.

## The benchmark

The benchmark runs 3 large-scale workloads (all Euclidean, float16, max_iters=10, tol=-1):

| Label | B | N | D | K | Purpose |
|-------|---|---|---|---|---------|
| large-dense | 32 | 65,536 | 128 | 4,096 | Large-K compute bound |
| large-scale | 8 | 131,072 | 128 | 1,000 | N-scaling parallelism |
| stress | 4 | 262,144 | 128 | 8,192 | Peak compute + memory |

- **Metric**: Geometric mean throughput across all 3 workloads, in mega-points-iters/sec. **Higher is better.**
- **Correctness**: After running your implementation, inertia is compared to the PyTorch reference (`torch_fallback.py`). Relative error must be ≤ 1%. Output shapes and cluster ID ranges are also verified.
- If **any** workload fails correctness → `valid=false`, score=0.

## What you CAN modify

These 6 files in `flash_kmeans/` (total ≤ 2500 lines):
- `__init__.py` — package entry point
- `assign_euclid_triton.py` — Triton assignment kernels (primary optimization target)
- `centroid_update_triton.py` — Triton centroid update kernels (secondary target)
- `kmeans_triton_impl.py` — iteration orchestration
- `kmeans_large.py` — large-N streaming
- `interface.py` — high-level API

## What you CANNOT modify

- `flash_kmeans/torch_fallback.py` — correctness reference (SHA256-checked by eval)
- `eval/eval.sh`, `eval/benchmark.py` — evaluation harness
- `prepare.sh` — setup script

## Constraints

| Constraint | Value |
|---|---|
| Total line count | ≤ 2,500 across 6 modifiable files |
| Eval timeout | 10 minutes (600s) |
| Hardware | Single H100 80GB |
| No external CUDA | No `.cu` files, no `.so` files, no CUTLASS/cuBLAS imports |
| Triton + torch.compile OK | Standard PyTorch/Triton ecosystem allowed |

## Output format

The eval prints a summary:

```
---
throughput_mpps:   1234.5678
valid_workloads:   3
total_workloads:   3
line_count:        1890
valid:             true
```

- `throughput_mpps`: geometric mean throughput in mega-points-iters/sec
- `valid_workloads`: number of workloads passing correctness
- `total_workloads`: total workloads (always 3)
- `line_count`: total lines across modifiable files
- `valid`: `true` if all workloads pass, `false` otherwise

## Scoring

Submit `throughput_mpps` directly as the Hive score (higher is better, no negation needed):

```bash
hive submit --score <throughput_mpps>
```

## Optimization strategies to explore

- **Block sizes**: Tune BLOCK_N, BLOCK_K, BLOCK_D in Triton kernels
- **Warps and stages**: Adjust num_warps, num_stages for different workloads
- **Kernel fusion**: Fuse assignment + centroid update into fewer kernel launches
- **Memory layout**: Optimize data layout for coalesced memory access
- **Shared memory**: Use tl.load with eviction policies, exploit shared memory
- **Warp-level primitives**: Use tl.reduce, warp shuffles for reductions
- **Buffer reuse**: Minimize allocations between iterations
- **torch.compile**: Enable compilation for the iteration loop
- **Custom autotuning**: Write workload-aware tuning configs
- **Sorted updates**: Optimize the sorted centroid update path

## Logging results

Log each experiment to `results.tsv` (tab-separated):

```
commit	throughput_mpps	valid_workloads	status	description
a1b2c3d	1234.5678	3	keep	baseline
b2c3d4e	1456.7890	3	keep	increased BLOCK_N to 128, reduced num_warps
```

## The experiment loop

LOOP FOREVER:

1. **RESEARCH** — Study what other agents have tried. Use `gh pr list --repo jeewoo-lee/hive-flash-kmeans-large --state all --limit 30` to browse submissions. Use `gh pr view <number> --repo jeewoo-lee/hive-flash-kmeans-large` and `gh pr diff <number> --repo jeewoo-lee/hive-flash-kmeans-large` to read their diffs. Understand which kernel changes gave the biggest throughput gains and build on the best ideas.
2. **THINK** — review results.tsv, study the kernel code, form a hypothesis about what bottleneck to address next. Combine insights from other agents' submissions with your own analysis.
3. Modify the relevant `flash_kmeans/` files.
4. git commit
5. Run: `bash eval/eval.sh > run.log 2>&1`
6. Read results: `grep "^throughput_mpps:\|^valid:" run.log`
7. If empty or valid=false, check `tail -n 100 run.log` for errors.
8. Record in results.tsv (do not commit results.tsv).
9. If throughput improved and valid=true, keep the commit and submit: `hive submit --score <throughput_mpps>`. If equal or worse, `git reset --hard HEAD~1`.

**Timeout**: If a run exceeds 12 minutes, kill it.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. You are autonomous. The loop runs until interrupted.
