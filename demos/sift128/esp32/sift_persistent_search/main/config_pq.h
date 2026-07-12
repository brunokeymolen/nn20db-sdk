#ifndef CONFIG_PQ_H
#define CONFIG_PQ_H

#include "nn20db_config.h"

#include "sift_queries.h"

/* Path where the PQ SIFT-128 database was copied on the SD card.
 * Build it on Linux with demos/sift128/linux/sift_persistent (USE_PQ=1),
 * then copy the database directory to the SD card at this path.
 * Make sure the full path is in 8.3 format. */
#define DB_PATH     "/sdcard/nand0/siftpq"

/* Both targets take the cache arenas from PSRAM (SPIRAM_USE_MALLOC in
 * sdkconfig), so esp32s3 and esp32p4 share one config. */
static const nn20db_config s_config = {
    .vector = {
        .type          = NN20DB_DIMENSION_PQ_CONFIG,
        .dimension     = SIFT_VECTOR_DIM,
        .metadata_size = (int)sizeof(int32_t),
        .pq = {
            /* 32×4D (32-byte codes): 16-byte codes cap recall@10 on SIFT1M
             * at ~0.56-0.65; must match the layout the DB was built with. */
            .num_segments = 32,     /* M: number of subvectors */
            .bits_per_segment = 8,  /* K: 256 codes per segment */
            .subvector_dim = 4,     /* 128 / 32 = 4D per subvector */
        },
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
            /* PQ node worst case ≈ 1.2 KB (24B hdr + 5×6B levels
             * + (64+16+8+4+2)×12B edges + 32B codes + 4B metadata), so
             * 1280 covers every node in one pread. */
            .read_ahead_size_bytes   = 1280,
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
            .max_entries          = 128,   /* 128 × 1280 = 160 KB in PSRAM */
            .max_object_size_bytes = 1280, /* PQ node size; without this the
                                              slot inherits the 4 KB
                                              object_cache_size_bytes */
        },
    },
    .metric = {
        .type = METRIC_EUCLIDEAN_PQ_CONFIG,
    },
    .tuning = {
        /* Warm-up BFS around the EP at L0 with M=32 (64 edge slots) touches
         * up to ~4.2K nodes at depth 2; the default 512-node cache would drop
         * 7/8 of them. 4096 × ~1 KB PQ nodes ≈ 4 MB PSRAM. Upper-level nodes
         * (hit by every search descent) displace L0-only entries once full. */
        .hnsw_node_cache_capacity = 4096,
        .hnsw_cache_warm_depth    = 2,
    },
};

#endif
