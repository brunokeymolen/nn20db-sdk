# Linux persistent demo

This demo mirrors the existing persistent SIFT sample: on the first run it creates a persistent nn20db database from an ANN Benchmarks HDF5 dataset, and on later runs it reopens the existing DB and only executes searches.

## Prerequisites

- Linux SDK installed with `scripts/install-sdk.sh linux ...`
- HDF5 development package available through `pkg-config`
- ANN Benchmarks dataset `sift-128-euclidean.hdf5`
  (download: `http://ann-benchmarks.com/sift-128-euclidean.hdf5`)
  
## Build

```bash
make
```

## Run

```bash
./sift_persistent_demo /path/to/sift-128-euclidean.hdf5 ./db/sift128
```

Optional third argument: number of test queries to evaluate.

```bash
./sift_persistent_demo /path/to/sift-128-euclidean.hdf5 ./db/sift128 250
```

## Notes

- The demo stores the DB under the path you pass as the second argument.
- Metadata is the original SIFT row index (`int32_t`), which is used to compute recall against the ground-truth neighbor list.
- The sample uses `METRIC_EUCLIDEAN_CONFIG` for portability. If your Linux SDK and deployment environment are consistently AVX2-capable, you can switch the metric in `main.c`.
