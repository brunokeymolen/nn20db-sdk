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

#include "sd_storage.h"
#include "sift_queries.h"

/* Product quantization config toggle
 * Override: make build USE_PQ=0  (or: idf.py -DUSE_PQ=0 build) */
#ifndef USE_PQ
#define USE_PQ 1
#endif

#if USE_PQ
    #include "config_pq.h"
#else
    #include "config_fp32.h"
#endif

#define DB_K        10
#define DB_EF       64
// Recall 85, EF = 14
// Recall 90, EF = 24
// Recall 98, EF = 64

void nn20db_logger_set_level(int level);

static void print_io_stats(NN20DB *db, const char *phase)
{
    nn20db_io_stats io;

    if (nn20db_io_stats_get(db, &io) != NN20DB_ERROR_OK) {
        return; /* backend does not track statistics */
    }

    printf("[sift-search] io_stats %s: gets=%u cache_hits=%u (%.1f%%) backend_gets=%u preads=%u bytes_read=%llu\n",
           phase,
           (unsigned)io.gets,
           (unsigned)io.cache_hits,
           io.gets ? 100.0 * (double)io.cache_hits / (double)io.gets : 0.0,
           (unsigned)io.backend_gets,
           (unsigned)io.preads,
           (unsigned long long)io.bytes_read);
}

static void run_search(void *arg)
{
    NN20DB *db = NULL;
    int rc;
    size_t qi;
    size_t total_hits = 0;
    int64_t total_search_us = 0;
    nn20db_vector_search_result results[DB_K];
    int32_t metadata[DB_K];

    (void)arg;

    nn20db_logger_set_level(1); /* 4: verbose logging for demo purposes */

    printf("[sift-search] opening DB at %s\n", DB_PATH);
    rc = nn20db_open_with_config(&s_config, &db);
    if (rc != NN20DB_ERROR_OK || db == NULL) {
        printf("[sift-search] open failed rc=%d\n", rc);
        vTaskDelete(NULL);
        return;
    }
    printf("[sift-search] DB open OK\n");
    printf("[sift-search] DB ef_search=%d\n", DB_EF);
    print_io_stats(db, "open+warm");
    nn20db_io_stats_reset(db);

    for (qi = 0; qi < SIFT_QUERY_COUNT; ++qi) {
        const sift_query_t *q = &SIFT_QUERIES[qi];
        int ri, gi;

        int64_t t0 = esp_timer_get_time();
        rc = nn20db_vector_search_ef(db, q->values, DB_K, DB_EF, results);
        int64_t search_us = esp_timer_get_time() - t0;
        total_search_us += search_us;
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

    printf("[sift-search] recall@%d = %.4f  (%zu/%zu over %zu queries)  avg_search=%.3fs\n",
           DB_K,
           (double)total_hits / (double)(SIFT_QUERY_COUNT * DB_K),
           total_hits,
           (size_t)(SIFT_QUERY_COUNT * DB_K),
           (size_t)SIFT_QUERY_COUNT,
           (double)total_search_us / (double)SIFT_QUERY_COUNT / 1000000.0);
    print_io_stats(db, "search");

    nn20db_dtor(db);
    printf("[sift-search] done\n");
    vTaskDelete(NULL);
}

void app_main(void)
{
#ifndef SD_BOARD_NONE
    /* Board-specific SD driver (must be before any nn20db call); without it
     * nn20db mounts with its built-in target default. */
    sd_storage_register_driver();
#endif

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
