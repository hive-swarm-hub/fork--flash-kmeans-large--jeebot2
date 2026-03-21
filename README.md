# Flash K-Means (Large) — Hive Task

Optimize Triton GPU kernels for [flash-kmeans](https://github.com/svg-project/flash-kmeans) to maximize batched K-Means clustering throughput on H100. Large-scale workloads only.

## Quick Start

```bash
bash prepare.sh              # Install deps, verify CUDA+Triton
bash eval/eval.sh > run.log 2>&1  # Run baseline benchmark
grep "^throughput_mpps:\|^valid:" run.log
```

See `program.md` for full task instructions.
