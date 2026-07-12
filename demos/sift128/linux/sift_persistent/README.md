# Linux persistent demo

This demo mirrors the existing persistent SIFT sample: on the first run it creates a persistent nn20db database from an ANN Benchmarks HDF5 dataset, and on later runs it reopens the existing DB and only executes searches.

Two vector configurations are available, selected at build time:

- **PQ** (default): product quantization, 32 segments × 8 bits (4D subvectors, 32-byte codes). Codebooks are trained on a 100K-vector sample of the train set before ingest. Smaller nodes → less I/O per search. (16×8D was tried first but caps recall@10 at ~0.56 on SIFT1M.)
- **fp32**: full-precision float32 vectors (`USE_PQ=0`).

Configs live in `config_pq.h` / `config_fp32.h`; both use the same HNSW settings.

## Prerequisites

- Linux SDK installed with `scripts/install-sdk.sh linux ...`
- HDF5 development package available through `pkg-config`
- ANN Benchmarks dataset `sift-128-euclidean.hdf5`
  (download: `http://ann-benchmarks.com/sift-128-euclidean.hdf5`)
  
## Build

```bash
make              # PQ config (default)
make USE_PQ=0     # fp32 config
```

## Run

Use a separate DB path per configuration — the vector layout is decided at creation time:

```bash
./sift_persistent_demo /path/to/sift-128-euclidean.hdf5 ./db/siftpq    # PQ build
./sift_persistent_demo /path/to/sift-128-euclidean.hdf5 ./db/sift128   # fp32 build
```

Optional arguments: number of test queries, `--ef-search N` (default 64) for
recall/latency sweeps without recompiling, and `--rerank N` to fetch the top N
candidates by DB distance and re-rank them with exact L2 against the in-RAM
train vectors (benchmark-only — the PQ DB stores codes, so a device cannot do
this; it separates ADC ranking error from graph reachability).

```bash
./sift_persistent_demo /path/to/sift-128-euclidean.hdf5 ./db/siftpq 250
./sift_persistent_demo /path/to/sift-128-euclidean.hdf5 ./db/siftpq 250 --ef-search 256
./sift_persistent_demo /path/to/sift-128-euclidean.hdf5 ./db/siftpq 100 --rerank 100
```

## Notes

- The demo stores the DB under the path you pass as the second argument.
- Metadata is the original SIFT row index (`int32_t`), which is used to compute recall against the ground-truth neighbor list.
- The DB names above match the 8.3 SD-card paths expected by the ESP32 demo (`nand0/siftpq`, `nand0/sift128`).
- Generation is tuned via `tuning.hnsw_node_cache_capacity` (sized to hold the full 1M-node graph, ~1.3–1.8 GB RAM); reduce it in the config header on smaller machines.
