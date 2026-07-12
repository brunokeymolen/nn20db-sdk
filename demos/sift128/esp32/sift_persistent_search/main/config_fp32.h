#ifndef CONFIG_FP32_H
#define CONFIG_FP32_H

#include "nn20db_config.h"

#include "sift_queries.h"

/* Path where the fp32 SIFT-128 database was copied on the SD card.
 * Build it on Linux with demos/sift128/linux/sift_persistent (USE_PQ=0),
 * then copy the database directory to the SD card at this path.
 * Make sure the full path is in 8.3 format. */
#define DB_PATH     "/sdcard/nand0/sift128"

/* Both targets take the cache arenas from PSRAM (SPIRAM_USE_MALLOC in
 * sdkconfig), so esp32s3 and esp32p4 share one config. */
static const nn20db_config s_config = {
    .vector = {
        .type          = NN20DB_DIMENSION_FLOAT32_CONFIG,
        .dimension     = SIFT_VECTOR_DIM,
        .metadata_size = (int)sizeof(int32_t),
    },
    .storage = {
        .type = NN20DB_STORAGE_LFS_CONFIG,
        .lfs = {
            .device_path             = DB_PATH,
            .mount_point             = "/sdcard",
            .lane_cache_size_kb      = 16,  /* values below 16 trigger invalid-arg in lane init */
            .lane_size_mb            = 128,
            .log_size_mb             = 1,
            .log_index_buckets       = 256,
            .object_cache_size_bytes = 4096,
            /* fp32 node worst case ≈ 1.7 KB (24B hdr + 5×6B levels
             * + (64+16+8+4+2)×12B edges + 512B vector + 4B metadata), so
             * 2048 covers every node in one pread. */
            .read_ahead_size_bytes   = 2048,
            /* 512-byte alignment (SD native sector) instead of 4096 cuts the
             * average alignment waste per get from ~2 KB to ~256 B. */
            .block_size              = 512,
            /* READ_ONLY opens lanes O_RDONLY, which is what lets IDF's FATFS
             * build the fast-seek cluster map (CONFIG_FATFS_USE_FASTSEEK).
             * Without it every random read walks the FAT chain of the 128 MB
             * lane files from the start. Search-only demo, no writes needed. */
            .flags                   = NN20DB_STORAGE_FLAGS_DISABLE_CRC
                                        | NN20DB_STORAGE_FLAGS_READ_ONLY,
        },
        .cache = {
            .enabled              = 1,
            .max_entries          = 128,   /* 128 × 2048 = 256 KB in PSRAM */
            .max_object_size_bytes = 2048, /* fp32 node size; without this the
                                              slot inherits the 4 KB
                                              object_cache_size_bytes */
        },
    },
    .metric = {
        .type = METRIC_EUCLIDEAN_F32_CONFIG, /* ESP optimized; AVX not supported on ESP32 */
    },
    .tuning = {
        /* Same ~4 MB PSRAM budget as the PQ config's node cache: fp32 nodes
         * are ~2x larger, so half the entries. Upper-level nodes (hit by
         * every search descent) displace L0-only entries once full. */
        .hnsw_node_cache_capacity = 2048,
        .hnsw_cache_warm_depth    = 2,
    },
};

#endif
