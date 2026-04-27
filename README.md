# nn20db SDK demos

![Linux SDK](https://img.shields.io/badge/Linux-SDK%201.0.0-2ea44f)
![ESP32--P4 SDK](https://img.shields.io/badge/ESP32--P4-SDK%201.0.0-e5532d)
![Python API](https://img.shields.io/badge/Python-API-3776ab)
![HNSW](https://img.shields.io/badge/Index-HNSW-7c3aed)
![Persistent Storage](https://img.shields.io/badge/Storage-LFS%20%2F%20NAND-f59e0b)

`nn20db` enables million-scale vector search on small devices. It runs on
Linux and embedded hardware such as the ESP32-P4, searches directly from
low-cost storage far beyond available RAM, and works fully offline, supporting
local intelligence without relying on cloud services or expensive hardware.

This repository is the public SDK/demo shell for `nn20db`. It contains install tooling,
demo applications, Python bindings, and release notes. The proprietary static
libraries are not committed; they are installed into `sdk/<target>/current`
from SDK tarballs.

The current local SDK layout uses:

- `sdk/linux/current` -> Linux SDK, currently `nn20db-sdk-linux-0.0.11`
- `sdk/esp32/current` -> ESP32 SDK, currently `nn20db-sdk-esp32-0.0.11`

## Supported devices

| Target | Tested environment |
| --- | --- |
| Linux | Ubuntu 24.04 |
| ESP32-P4 | Espressif ESP-IDF v5.x |


## Repository layout

| Path | Purpose |
| --- | --- |
| `scripts/install-sdk.sh` | Installs a Linux or ESP32 SDK tarball from a local path or release URL. |
| `api/python/` | Minimal Python API wrapper around the Linux SDK. |
| `demos/geo/linux/python/` | GeoNames nearest-city demo; builds a persistent 10k-city index on Linux. |
| `demos/geo/esp32/` | ESP32-P4 search-only version of the GeoNames demo. |
| `demos/sift128/linux/sift_persistent/` | Linux SIFT-128 persistent HNSW recall demo. |
| `demos/sift128/esp32/sift_persistent_search/` | ESP32-P4 search-only SIFT-128 recall demo. |

Generated SDK payloads, build outputs, downloaded datasets, and generated
databases are intentionally left out of git.

## Install the SDK

Install from local tarballs:

```bash
./scripts/install-sdk.sh linux /path/to/nn20db-sdk-linux-<version>.tar.gz
./scripts/install-sdk.sh esp32 /path/to/nn20db-sdk-esp32-<version>.tar.gz
```

Install from release asset URLs:

```bash
./scripts/install-sdk.sh linux \
  https://github.com/<owner>/<repo>/releases/download/<tag>/nn20db-sdk-linux-<version>.tar.gz

./scripts/install-sdk.sh esp32 \
  https://github.com/<owner>/<repo>/releases/download/<tag>/nn20db-sdk-esp32-<version>.tar.gz
```

The installer unpacks the tarball under `sdk/<target>/` and updates
`sdk/<target>/current` to point at that version.

Expected SDK contents:

- Linux: `include/`, `cmake/`,
  `linux/x86_64-linux-gnu/lib/libnn20db.a`, and
  `linux/x86_64-linux-gnu/pkgconfig/nn20db.pc`
- ESP32-P4: `esp32/component/nn20db/include` and
  `esp32/component/nn20db/lib/esp32p4/libnn20db.a`

## Build the Python API

Build the Python shared library once before running the Python demos:

```bash
make -C api/python
```

## GeoNames demo

The Linux/Python GeoNames demo downloads the GeoNames `cities15000` dataset,
indexes the first 10,000 cities as 3D unit-sphere vectors, and runs nearest
city queries for Brussels, Paris, New York, and Tokyo.

```bash
python demos/geo/linux/python/demo_geonames_10k.py
```

Rebuild the index from scratch:

```bash
python demos/geo/linux/python/demo_geonames_10k.py --rebuild
```

Useful options:

```text
--max-cities N   Index only the first N cities; default: 10000
--top-k K        Return K nearest neighbours; default: 10
--rebuild        Delete and recreate the index
```

The generated database is stored under:

```text
demos/geo/linux/python/data/geo10k
```

To run the same data on ESP32-P4, copy that directory to the SD card as
`/nand0/geo10k`, then build and flash:

```bash
cd demos/geo/esp32
make build
make flash-monitor
```

Use `PORT=/dev/ttyUSB0` or another serial device if needed.

## SIFT-128 demo

The Linux SIFT demo builds or reopens a persistent HNSW index from the ANN
Benchmarks `sift-128-euclidean.hdf5` dataset and prints recall against the
ground-truth neighbours.

```bash
cd demos/sift128/linux/sift_persistent
make
./sift_persistent_demo /path/to/sift-128-euclidean.hdf5 ./db/sift128
```

An optional third argument limits the number of test queries:

```bash
./sift_persistent_demo /path/to/sift-128-euclidean.hdf5 ./db/sift128 250
```

To run the same index on ESP32-P4, copy the generated database directory to
the SD card as `/nand0/sift128`, then build and flash:

```bash
cd demos/sift128/esp32/sift_persistent_search
make build
make flash-monitor
```

The ESP32 demos are intentionally search-only. The SDK exposes insertion APIs
on ESP32 too, but these demos build databases on Linux because inserts are
much slower on constrained flash/SD storage.

## nn20db architecture

`nn20db` is configured at database-open time with a single `nn20db_config`
structure. At a high level:

```mermaid
flowchart LR
  classDef app fill:#dbeafe,stroke:#2563eb,color:#0f172a
  classDef core fill:#ede9fe,stroke:#7c3aed,color:#0f172a
  classDef index fill:#dcfce7,stroke:#16a34a,color:#0f172a
  classDef storage fill:#ffedd5,stroke:#f97316,color:#0f172a
  classDef metric fill:#fce7f3,stroke:#db2777,color:#0f172a
  classDef cache fill:#fef9c3,stroke:#ca8a04,color:#0f172a

  App[Application]:::app --> API[Public C API]:::app
  API --> Core[DB core]:::core
  Core --> Config[Config + logger]:::core
  Core --> Index[Index layer]:::index
  Core --> Storage[Storage layer]:::storage
  Core --> Metric[Distance metric]:::metric

  Index --> Linear[Linear scan]:::index
  Index --> HNSW[HNSW graph]:::index
  HNSW --> HNSWCache[Entry-point read cache]:::cache

  Storage --> Cache[Optional storage cache]:::cache
  Cache --> Memory[Memory backend]:::storage
  Cache --> LFS[LFS backend]:::storage
  LFS --> LaneCache[Lane write cache]:::cache
  LFS --> ReadPath[Read-ahead + object caches]:::cache
```

The public API drives the DB core through operations such as create/open,
add vector, search, get vector, remove, sync, and compact. The DB core owns the
selected index, storage backend, metric implementation, vector shape, and
metadata size.

### Index layer

The index is selected with `cfg.index.type`:

| Type | Description |
| --- | --- |
| `NN20DB_INDEX_LINEAR_CONFIG` | Brute-force scan, useful for small data or testing. |
| `NN20DB_INDEX_HNSW_CONFIG` | Hierarchical Navigable Small World graph for approximate nearest-neighbour search. |

HNSW keeps graph nodes in storage and uses an in-memory read cache around the
entry-point neighbourhood to reduce repeated storage reads during traversal.
The main HNSW options are:

| Option | Meaning |
| --- | --- |
| `search_threads` | Parallel search threads; `0` uses the default. Not implemented yet, put it to 1|
| `max_levels` | Maximum number of HNSW graph levels. |
| `diversity_alpha` | Neighbour diversity heuristic relaxation; `0.0` disables it, `1.0` is strict, values above `1.0` are more relaxed. |
| `search_seen_set_capacity` | Initial capacity for the search visited set; `0` sizes it automatically. |
| `ef_search` | Candidate list size during search; higher improves recall at higher latency. |
| `level_config[n].M` | Maximum graph connections per node at level `n`. |
| `level_config[n].ef_construction` | Candidate list size during insertion; higher improves graph quality but slows builds. |

Mental model: `M` controls graph connectivity, `ef_construction` controls build
quality, `ef_search` controls search effort, and `k` controls how many results
the caller asks for.

### Storage layer

The storage layer is selected with `cfg.storage.type`:

| Type | Description |
| --- | --- |
| `NN20DB_STORAGE_MEMORY_CONFIG` | Segment-based RAM storage; optionally backed by a device path in the current header. |
| `NN20DB_STORAGE_LFS_CONFIG` | Log Structured and Lane-based filesystem backend. |

The optional storage-cache decorator wraps any backend:

| Option | Meaning |
| --- | --- |
| `cache.enabled` | Enables the in-RAM read cache. |
| `cache.max_entries` | Maximum cached object count; `0` uses the default of 256. |
| `cache.max_object_size_bytes` | Maximum cached object size; with LFS, `0` inherits `lfs.object_cache_size_bytes`. |

The LFS backend stores append-only lane files plus update logs. Writes are
coalesced through per-lane write buffers. Reads use read-ahead, object pointer
slots, merged-object caching, and log-index lookups so updates can be merged
with base objects without rewriting whole lanes immediately.

Important LFS options:

| Option | Meaning |
| --- | --- |
| `lfs.device_path` | Database path or device path. |
| `lfs.mount_point` | Platform mount point; empty means no platform mount is needed. |
| `lfs.lane_cache_size_kb` | Per-lane write buffer size. |
| `lfs.lane_size_mb` | Maximum lane file size. |
| `lfs.log_size_mb` | Update-log budget; `0` uses the backend default. |
| `lfs.log_index_buckets` | Per-lane log-index hash buckets. |
| `lfs.object_cache_size_bytes` | Per-slot object cache size; four rotating slots are allocated. |
| `lfs.read_ahead_size_bytes` | Opportunistic read-ahead per object read; `0` uses the default. |
| `lfs.block_size` | I/O alignment boundary; `0` uses the default. |
| `lfs.flags` | Storage flags, including `NN20DB_STORAGE_FLAGS_DISABLE_CRC`. |

### Vector and metric options

Each database has a fixed vector shape:

| Option | Meaning |
| --- | --- |
| `vector.type` | `NN20DB_DIMENSION_FLOAT32_CONFIG` or `NN20DB_DIMENSION_BIT_CONFIG`. |
| `vector.dimension` | Number of dimensions per vector. |
| `vector.metadata_size` | Application metadata bytes stored alongside each vector. |

Supported metric types in the current SDK header:

| Metric | Notes |
| --- | --- |
| `METRIC_COSINE_CONFIG` | Cosine distance. |
| `METRIC_EUCLIDEAN_CONFIG` | Euclidean/L2 distance. |
| `METRIC_EUCLIDEAN_AVX2_CONFIG` | AVX2-accelerated Euclidean distance on x86_64. |
| `METRIC_DOT_PRODUCT_CONFIG` | Dot product / inner product. |
| `METRIC_MANHATTAN_CONFIG` | Manhattan/L1 distance. |
| `METRIC_HAMMING_CONFIG` | Hamming distance for binary vectors. |
| `METRIC_JACCARD_CONFIG` | Jaccard distance. |
| `METRIC_COSINE_F32_CONFIG` | Single-precision cosine, preferred on ESP32 hardware FPU. |
| `METRIC_EUCLIDEAN_F32_CONFIG` | Single-precision Euclidean, preferred on ESP32 hardware FPU. |

### Logger

`cfg.logger` can override the global logger:

| Option | Meaning |
| --- | --- |
| `logger.enabled` | `0` leaves logger settings unchanged; `1` applies this config. |
| `logger.level` | Minimum emitted log level. |
| `logger.path` | Log file path; empty uses the SDK default. |


## ESP32 Demo Output

<p align="left">
  <img src="img/geo.png" width="60%" alt="GeoNames demo"/>


  <img src="img/sift128.png" width="60%" alt="SIFT-128 demo"/>
</p>
