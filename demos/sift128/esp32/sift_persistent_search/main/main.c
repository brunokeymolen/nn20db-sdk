/*
 * nn20db-sdk
 *
 * Copyright (c) 2026 Bruno Keymolen
 * Contact: bruno.keymolen@gmail.com
 *
 * License:
 * This SDK, including all pre-compiled binaries and accompanying files,
 * is provided for private and educational use only.
 *
 * Commercial use is strictly prohibited without prior written agreement
 * from the author.
 *
 * Disclaimer:
 * This software is provided "as is", without any express or implied
 * warranties, including but not limited to the implied warranties of
 * merchantability and fitness for a particular purpose.
 *
 * In no event shall the author be held liable for any damages arising
 * from the use of this software.
 */

#include <stdio.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"

#include "nn20db.h"
#include "nn20db_config.h"

#include "sift_queries.h"

/* Path where the SIFT-128 database was copied on the SD card.
 * Build it on Linux with demos/sift128/linux/sift_persistent, then copy
 * the database directory to the SD card at this path. 
 * Make sure the full path in in 8.3 format.*/
#define DB_PATH     "/sdcard/nand0/sift128"
#define DB_K        10
#define DB_EF       24

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
            .lane_cache_size_kb      = 16,
            .lane_size_mb            = 128,
            .log_size_mb             = 1,
            .log_index_buckets       = 512,
            .object_cache_size_bytes = 4096,
            .read_ahead_size_bytes   = 2048,
            .block_size              = 4096,
            .flags                   = NN20DB_STORAGE_FLAGS_DISABLE_CRC,
        },
        .cache = {
            .enabled     = 1,
            .max_entries = 500
        },
    },
    .metric = {
        .type = METRIC_EUCLIDEAN_F32_CONFIG, //the original config has METRIC_EUCLIDEAN_AVX_CONFIG which is not supported on ESP32, ESP optimized config uses METRIC_EUCLIDEAN_F32_CONFIG which is preferred on ESP32 (hardware FPU) vs double (soft-emulated) 
    }
};

static void run_search(void *arg)
{
    NN20DB *db = NULL;
    int rc;
    size_t qi;
    size_t total_hits = 0;
    nn20db_vector_search_result results[DB_K];
    int32_t metadata[DB_K];

    (void)arg;

    printf("[sift-search] opening DB at %s\n", DB_PATH);
    rc = nn20db_open_with_config(&s_config, &db);
    if (rc != NN20DB_ERROR_OK || db == NULL) {
        printf("[sift-search] open failed rc=%d\n", rc);
        vTaskDelete(NULL);
        return;
    }
    printf("[sift-search] DB open OK\n");
    printf("[sift-search] DB ef_search=%d\n", DB_EF);

    for (qi = 0; qi < SIFT_QUERY_COUNT; ++qi) {
        const sift_query_t *q = &SIFT_QUERIES[qi];
        int ri, gi;

        int64_t t0 = esp_timer_get_time();
        rc = nn20db_vector_search_ef(db, q->values, DB_K, DB_EF, results);
        int64_t search_us = esp_timer_get_time() - t0;
        if (rc != NN20DB_ERROR_OK) {
            printf("[sift-search] %s: search failed rc=%d\n", q->name, rc);
            continue;
        }

        /* Fetch metadata (train-set index) for each result */
        for (ri = 0; ri < DB_K; ++ri) {
            if (nn20db_vector_get(db, results[ri].id, NULL, &metadata[ri]) != NN20DB_ERROR_OK) {
                metadata[ri] = -1;
            }
        }

        /* Count recall@K hits */
        size_t query_hits = 0;
        for (ri = 0; ri < DB_K; ++ri) {
            if (metadata[ri] < 0) {
                continue;
            }
            for (gi = 0; gi < SIFT_GT_K; ++gi) {
                if (metadata[ri] == q->gt[gi]) {
                    query_hits++;
                    break;
                }
            }
        }
        total_hits += query_hits;

        printf("[sift-search] %s: hits=%zu/%d  (top results: id=%llu d=%.1f)  time=%.3fs\n",
               q->name, query_hits, DB_K,
               (unsigned long long)results[0].id,
               (double)results[0].distance,
               (double)(search_us/1000000.0));
    }

    printf("[sift-search] recall@%d = %.4f  (%zu/%zu over %zu queries)\n",
           DB_K,
           (double)total_hits / (double)(SIFT_QUERY_COUNT * DB_K),
           total_hits,
           (size_t)(SIFT_QUERY_COUNT * DB_K),
           (size_t)SIFT_QUERY_COUNT);

    nn20db_dtor(db);
    printf("[sift-search] done\n");
    vTaskDelete(NULL);
}

void app_main(void)
{
    BaseType_t ok = xTaskCreatePinnedToCore(
        run_search, "sift_search", 32 * 1024, NULL, tskIDLE_PRIORITY + 1, NULL, 1);

    if (ok != pdPASS) {
        ok = xTaskCreatePinnedToCore(
            run_search, "sift_search", 32 * 1024, NULL, tskIDLE_PRIORITY + 1, NULL, 0);
    }

    if (ok != pdPASS) {
        printf("[sift-search] task creation failed\n");
    }
}
