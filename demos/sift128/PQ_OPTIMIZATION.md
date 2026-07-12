# Performance optimization with PQ — SIFT-128 findings

Findings from tuning product quantization on the SIFT1M benchmark
(`sift-128-euclidean.hdf5`, 1M × 128D train, 100 queries, recall@10),
nn20db SDK v1.2.0, Linux x86_64 (AVX2), July 2026.

## Settings

HNSW construction, identical for every DB in this document:

| Setting                    | L0  | L1  | L2 | L3 | L4 |
|----------------------------|-----|-----|----|----|----|
| M                          | 32  | 16  | 8  | 4  | 2  |
| ef_construction            | 250 | 120 | 60 | 30 | 15 |

| Setting                     | Value   |
|-----------------------------|---------|
| max_levels                  | 5       |
| diversity_alpha             | 1.2     |
| search_seen_set_capacity    | 20000   |
| search_threads              | 1       |
| metadata_size               | 4 B (train-set row index) |

Vector / metric settings per configuration:

| Setting          | fp32                    | PQ 16×8              | PQ 32×4              |
|------------------|-------------------------|----------------------|----------------------|
| vector type      | FLOAT32                 | PQ                   | PQ                   |
| metric           | EUCLIDEAN_AVX2 (Linux) / EUCLIDEAN_F32 (ESP32) | EUCLIDEAN_PQ | EUCLIDEAN_PQ |
| num_segments     | —                       | 16                   | 32                   |
| bits_per_segment | —                       | 8 (256 centroids)    | 8 (256 centroids)    |
| subvector_dim    | —                       | 8                    | 4                    |
| code size        | 512 B (raw vector)      | 16 B                 | 32 B                 |
| PQ train set     | —                       | 100K strided samples | 100K strided samples |

Storage / tuning (Linux generation): lane 128 MB, block_size 4096,
read_ahead 4096, storage cache 1 048 576 entries, DISABLE_CRC,
`tuning.hnsw_node_cache_capacity = 1048576` (whole 1M-node graph in RAM
during ingest → 99.9 % node-cache hits while building),
`hnsw_cache_warm_depth = 2`.

Storage / tuning (ESP32 runtime): block_size 512, read_ahead 1280 (PQ) /
2048 (fp32), storage cache 128 entries, DISABLE_CRC + READ_ONLY (enables
FATFS fast-seek), node cache 4096 (PQ) / 2048 (fp32), warm depth 2.

## Headline results

| Config          | Codes   | recall@10 (ef 64) | Build time | Node worst case |
|-----------------|---------|-------------------|-----------|-----------------|
| fp32            | —       | **0.987**         | 39m54s    | ~1698 B         |
| PQ 16 × 8D      | 16 B    | 0.559             | 37m45s    | ~1202 B         |
| PQ 32 × 4D      | 32 B    | **0.730**         | 42m42s    | ~1218 B         |

PQ codebook training (k-means, 100K evenly-strided samples): 31 s (16 seg) /
42 s (32 seg). Ingest rate ~400–450 vec/s in all configs — PQ does not
meaningfully change build cost.

## All measured recall points (SIFT1M, 100 queries, recall@10)

| Config   | ef_search | rerank | recall@10 | avg search (cold cache) |
|----------|-----------|--------|-----------|-------------------------|
| fp32     | 14        | —      | ~0.85     | (earlier measurement)   |
| fp32     | 24        | —      | ~0.90     | (earlier measurement)   |
| fp32     | 64        | —      | 0.987     | 7.32 ms                 |
| PQ 16×8  | 64        | —      | 0.559     | 7.17 ms                 |
| PQ 32×4  | 10        | —      | 0.665     | 2.78 ms                 |
| PQ 32×4  | 64        | —      | 0.730     | 6.98 ms                 |
| PQ 32×4  | 250       | —      | 0.734     | 15.00 ms                |

Synthetic 20K validation set (uniform random — PQ-hostile; not comparable
to SIFT1M rows above, but internally consistent):

| Config   | ef_search | rerank | recall@10 |
|----------|-----------|--------|-----------|
| fp32     | 64        | —      | 0.974     |
| PQ 32×4  | 64        | —      | 0.614     |
| PQ 32×4  | 64        | 50     | 0.952     |
| PQ 32×4  | 100       | 100    | 0.978     |

## ef_search sweep (PQ 32×4, existing DB, cold storage cache)

| ef  | recall@10 | avg search | preads / 100 q | bytes / query |
|-----|-----------|-----------|----------------|---------------|
| 10  | 0.665     | 2.78 ms   | 62 738         | ~3.9 MB       |
| 64  | 0.730     | 6.98 ms   | 179 098        | ~11 MB        |
| 250 | 0.734     | 15.00 ms  | 385 460        | ~24 MB        |

**Finding 1 — recall plateaus hard at ~0.73.** ef 64 → 250 buys +0.004 for
2× latency and I/O; ef 10 already reaches 0.665. The limit is the accuracy
of the PQ (ADC) distance estimates, not the graph traversal. Raising
`ef_construction` would not help: the fp32 twin built with identical
construction settings reaches 0.987, so the construction recipe is sound.

**Finding 2 — the sweet spot is below ef 64.** For quantization-limited PQ
search, ef 24–32 should retain ~0.72 recall at roughly half the I/O of
ef 64. Relevant for the ESP32, where every pread is an SD transaction.

## Exact re-ranking (`--rerank N`)

`--rerank N` fetches the top-N candidates by DB distance and re-ranks them
with exact L2 (in-RAM train vectors, looked up via the metadata row index).
Benchmark-only: the PQ database stores codes, not vectors.

Measured on the 20K-vector synthetic validation set (uniform random —
PQ-hostile, fp32 reference 0.974):

| rerank N | recall@10 |
|----------|-----------|
| off      | 0.614     |
| 50       | 0.952     |
| 100      | **0.978** |

**Finding 3 — the true neighbors ARE in the candidate list; ADC misorders
them.** Re-ranking recovers fp32-level recall, which means the PQ-built
graph reaches the right neighborhood and the recall loss is purely a
ranking error in the final top-k selection. It is not a graph-reachability
or construction-quality problem.

## I/O characteristics (from `io_stats`)

- Cold search, ef 64, 4096-node cache on a 1M-node graph: ~24.6 % cache
  hits, ~1790 preads/query. Linux reads ~6.1 KB/pread (block_size 4096 +
  read_ahead 4096); the ESP32 config (block 512, read_ahead 1280–2048)
  should transfer ~1.5–2 KB/pread → **roughly 2.7–3.5 MB per query at
  ef 64, or ~1–1.3 MB at ef 24** on the device.
- fp32 and PQ do nearly identical pread *counts* at the same ef (~1790/query)
  — PQ's I/O advantage comes from smaller node payloads, and grows on media
  with per-byte cost (SD), not per-op cost (NVMe).
- Warm-up at open touched only 66 nodes (`open+warm: gets=66`) on the 1M
  DB. A depth-2 BFS around the entry point at L0 with M=32 should touch
  up to ~4K nodes — worth checking whether `hnsw_cache_warm_depth` warms
  the level intended (core question).

## Recommendations, in value order

1. **Core: verify the PQ metric is asymmetric (ADC), not symmetric (SDC).**
   A 0.73 plateau at 256-bit codes on SIFT1M is low for ADC (literature
   suggests well above 0.8); if the query is being quantized too, distance
   error roughly doubles. Making search-time distance asymmetric (query
   kept full precision, per-segment lookup tables against the codebook)
   requires no format change and would lift the ceiling on existing DBs.
2. **Core: optional exact re-rank stage.** Needs raw vectors; the PQ format
   does not store them. One design: optional side file of raw (or fp16)
   vectors, fetched only for the final N candidates — N=100 × 512 B ≈
   50 KB/query, negligible next to the ~3 MB traversal, and it recovers
   recall from 0.73 to ~0.95+ (measured, Finding 3).
3. **Config: 64 × 2D codes (64 B)** as a no-core-change middle ground for
   on-device recall; expected ceiling ~0.8–0.85, +32 B/node. Only worth a
   rebuild after (1) is settled, since ADC could move the ceiling further
   for free.
4. **Runtime: drop ef to 24–32 on the ESP32** while recall is
   quantization-capped anyway; spend the saved I/O budget nowhere — it's
   pure latency win.

## Reproduction

```bash
cd demos/sift128/linux/sift_persistent
make                    # PQ 32×4 (default); make USE_PQ=0 for fp32
./sift_persistent_demo <sift-128-euclidean.hdf5> <db-path> [queries] \
    [--ef-search N] [--rerank N]
```

Configs: `config_pq.h` / `config_fp32.h` (Linux),
`../esp32/sift_persistent_search/main/config_{pq,fp32}.h` (device — PQ
segment layout must match the DB being opened).
