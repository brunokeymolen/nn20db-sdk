#ifndef CONFIG_FP32_H
#define CONFIG_FP32_H

#include <stdio.h>

#include "nn20db_config.h"

#define SIFT_DIM        128
#define METADATA_SIZE   ((int)sizeof(int32_t))  /* train-set row index */

static nn20db_config make_config(int dim, const char *db_path)
{
    nn20db_config cfg = {
        .vector = {
            .type          = NN20DB_DIMENSION_FLOAT32_CONFIG,
            .dimension     = dim,
            .metadata_size = METADATA_SIZE,
        },
        .storage = {
            .type = NN20DB_STORAGE_LFS_CONFIG,
            .lfs = {
                .lane_cache_size_kb      = 8192,
                .lane_size_mb            = 128,
                .log_size_mb             = 500,
                .log_index_buckets       = 262144,
                .object_cache_size_bytes = 4096,
                .read_ahead_size_bytes   = 4096,
                .block_size              = 4096,
                .flags                   = NN20DB_STORAGE_FLAGS_DISABLE_CRC,
            },
            .cache = {
                .enabled     = 1,
                .max_entries = 1048576,
            },
        },
        .metric = {
            .type = METRIC_EUCLIDEAN_AVX2_CONFIG,
        },
        .index = {
            .type = NN20DB_INDEX_HNSW_CONFIG,
            .hnsw = {
                .search_threads          = 1,
                .max_levels              = 5,
                .diversity_alpha         = 1.2f,
                .search_seen_set_capacity = 20000,
                .ef_search               = 64,
                .level_config[0]         = { .M = 32, .ef_construction = 250 },
                .level_config[1]         = { .M = 16, .ef_construction = 120 },
                .level_config[2]         = { .M = 8,  .ef_construction = 60  },
                .level_config[3]         = { .M = 4,  .ef_construction = 30  },
                .level_config[4]         = { .M = 2,  .ef_construction = 15  },
            },
        },
        .tuning = {
            /* Every insert runs a graph search, so generation is dominated by
             * node fetches. Sized to hold the full 1M-node SIFT graph: fp32
             * nodes are ~1.7 KB worst case (24B hdr + 5×6B levels +
             * (64+16+8+4+2)×12B edges + 512B vector + 4B metadata) → ~1.8 GB
             * RAM. Reduce on smaller machines. */
            .hnsw_node_cache_capacity = 1048576,
            .hnsw_cache_warm_depth    = 2,
        },
    };

    snprintf(cfg.storage.lfs.device_path,
             sizeof(cfg.storage.lfs.device_path), "%s", db_path);

    return cfg;
}

#endif
