# SD I/O latency findings (ESP32-P4, waveshare_p4 board)

Investigation into why `ef_search=64` search timing was identical between
SDMMC clock speeds of 20 MHz (`SDMMC_FREQ_DEFAULT`) and 40 MHz
(`SDMMC_FREQ_HIGHSPEED`, `main/sd_storage_waveshare_p4.c`).

## Observation

Both clock speeds produced ~3.9s avg search time over 100 queries, and
identical recall/top-k results.

```
[sift-search] recall@10 = 0.9870  (987/1000 over 100 queries)  avg_search=3.894s
[sift-search] io_stats search: gets=237586 cache_hits=9001 (3.8%) backend_gets=228585 preads=228585 bytes_read=526848293
```

## Why results are identical regardless of clock

SDMMC bus clock only affects bit-transfer speed on CLK/CMD/D0-D3; the
protocol includes CRC checks on every command/data block, so bytes handed
up to the filesystem are bit-identical at 20 or 40 MHz. The HNSW/PQ search
is deterministic given identical input data, so top-k results and recall
can't differ between clock speeds — only latency/throughput can.

## Why timing is identical regardless of clock

Per-query numbers derived from the io_stats above:

- 100 queries x avg 3.894s = 389.4s total search time
- 228,585 preads -> **~1.70 ms average latency per pread**
- 526,848,293 bytes / 228,585 preads -> **~2,305 bytes per pread**

At 20 MHz/4-bit (~10 MB/s theoretical max), transferring 2,305 bytes takes
~230 us; at 40 MHz (~20 MB/s) it takes ~115 us. Doubling the clock therefore
only removes ~115 us from a 1.70 ms request — under 7% of the total, small
enough to disappear into normal run-to-run jitter. The remaining ~93% of
each pread is fixed SDMMC command/response round-trip latency and the SD
card's internal flash-translation-layer access time, both independent of
bus clock.

This is a **latency-bound (IOPS-bound)** workload, not a **bandwidth-bound**
one: `ef_search` traversal issues many small random preads (one per graph
node visited: vector + adjacency list), not one large sequential transfer.
Raising the clock only helps bandwidth-bound transfers.

## Would a raw block device (bypassing FATFS) help?

Unlikely to move the needle, for two reasons:

1. **The FAT-overhead problem this would target is already solved.**
   `main/config_pq.h` sets `NN20DB_STORAGE_FLAGS_READ_ONLY`, which makes IDF
   build FATFS's fast-seek cluster map (`CONFIG_FATFS_USE_FASTSEEK`) instead
   of walking the FAT chain from the start on every random read into the
   128 MB lane file. That's the exact cost a raw block device would remove
   — and it's already gone.
2. **No plug-in point exists anyway.** `nn20db_esp32_storage.h` only exposes
   `mount`/`unmount` hooks (pins, LDO, clock). The actual read path — lane
   files, object cache, `pread` calls — lives inside the precompiled
   `libnn20db` and hard-codes a POSIX file path (`device_path`) on a
   *mounted filesystem*. Swapping in raw sector/partition I/O would require
   an SDK-level raw-block backend from nn20db itself, not something
   addable from the demo.

Even if it existed, the remaining software path (newlib syscall -> esp_vfs
dispatch -> FATFS diskio w/ fast-seek -> SDMMC driver -> card command) is
already thin. Removing it saves tens of microseconds per call, not the
~1.5 ms/pread of fixed hardware/protocol latency that dominates.

## The actual lever: cache hit rate is only 3.8%

`main/config_pq.h` tunes:
- `hnsw_node_cache_capacity = 4096` with `hnsw_cache_warm_depth = 2`,
  explicitly to make warm-up BFS nodes "hit by every search descent"
- `cache.max_entries = 128` object cache

Yet 228,585 of 237,586 gets (96.2%) still fall through to physical SD
reads (`cache_hits=9001`, 3.8%). That means ~2,286 preads per query are
still cold, each costing ~1.7 ms of largely clock-independent latency.

Improving cache coverage (bigger cache, deeper/smarter warm-up, or
investigating why L0 traversal evicts warm entries faster than expected)
would cut the *number* of physical reads per query — this is where real
speedups live, not clock speed or storage-layer swaps.

## Diagnostic used

`print_io_stats()` in `main/main.c:56` prints `gets`, `cache_hits`,
`backend_gets`, `preads`, `bytes_read` after each phase — this is what
made the above per-pread latency/byte-size math possible. Useful to
re-run after any cache-tuning change to check hit-rate improvement
directly.
