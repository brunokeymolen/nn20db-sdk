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
#include <string.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"

#include "nn20db.h"
#include "nn20db_config.h"

/* Path where the geo10k database lives on the SD card.
 * Build it on Linux with demos/geo/linux/python/demo_geonames_10k.py, then
 * copy the database directory to the SD card at this path.
 * Make sure the full path is in 8.3 format. */
#define DB_PATH         "/sdcard/nand0/geo10k"
#define GEO_DIM         3
#define GEO_TOP_K       2
#define GEO_EF          32
#define EARTH_RADIUS_KM 6371.0088f

/* city metadata layout — must match the Python packing format "<32s2sffi" (46 bytes) */
#pragma pack(push, 1)
typedef struct {
    char    name[32];
    char    country[2];
    float   lat;
    float   lon;
    int32_t pop;
} city_meta_t;
#pragma pack(pop)

_Static_assert(sizeof(city_meta_t) == 46, "city_meta_t must be 46 bytes");

typedef struct {
    const char *label;
    float lat;
    float lon;
} geo_query_t;

static const geo_query_t QUERIES[] = {
    { "Brussels, Belgium",  50.85045f,   4.34878f },
    { "Paris, France",      48.85660f,   2.35220f },
    { "New York, USA",      40.71280f, -74.00600f },
    { "Niagara Falls, Canada", 43.10012f, -79.06627f },
    { "Stoney Creek, Canada",  43.21681f, -79.76633f },
    { "Cold Spring, USA",     41.43290f, -73.95400f },
    { "Peekskill, USA",       41.28950f, -73.92000f },
    { "Tokyo, Japan",       35.67620f, 139.65030f },
    { "Gent, Belgium",      51.05000f,   3.71667f },
    { "Merelbeke, Belgium", 50.99447f,   3.74621f },
    { "Wetteren, Belgium",  51.00526f,   3.88341f },
    { "Trenton, USA",        40.21710f, -74.74290f },
    { "Long Island, USA",      40.78914f, -73.13496f },
};

static const nn20db_config s_config = {
    .vector = {
        .type          = NN20DB_DIMENSION_FLOAT32_CONFIG,
        .dimension     = GEO_DIM,
        .metadata_size = (int)sizeof(city_meta_t),
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
            .max_entries = 500,
        },
    },
    .metric = {
        /* Linux demo uses cosine similarity on 3-D unit vectors */
        .type = METRIC_COSINE_CONFIG,
    },
};

static void latlon_to_unit_vector(float lat_deg, float lon_deg, float out[3]) {
    const float lat = lat_deg * (float)M_PI / 180.0f;
    const float lon = lon_deg * (float)M_PI / 180.0f;
    out[0] = cosf(lat) * cosf(lon);
    out[1] = cosf(lat) * sinf(lon);
    out[2] = sinf(lat);
}

static float haversine_km(float lat1, float lon1, float lat2, float lon2) {
    const float dlat = (lat2 - lat1) * (float)M_PI / 180.0f;
    const float dlon = (lon2 - lon1) * (float)M_PI / 180.0f;
    const float lat1r = lat1 * (float)M_PI / 180.0f;
    const float lat2r = lat2 * (float)M_PI / 180.0f;
    const float a = sinf(dlat * 0.5f) * sinf(dlat * 0.5f)
                  + cosf(lat1r) * cosf(lat2r) * sinf(dlon * 0.5f) * sinf(dlon * 0.5f);
    return 2.0f * EARTH_RADIUS_KM * asinf(sqrtf(a));
}

static void copy_name_32(char *dst, size_t dst_size, const char src[32]) {
    size_t n = 0;
    while (n < 32 && src[n] != '\0') n++;
    if (n >= dst_size) n = dst_size - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

static void run_queries(void *arg) {
    (void)arg;
    NN20DB *db = NULL;
    int rc;

    printf("[geo] opening DB at %s\n", DB_PATH);
    rc = nn20db_open_with_config(&s_config, &db);
    if (rc != NN20DB_ERROR_OK || db == NULL) {
        printf("[geo] open failed rc=%d\n", rc);
        vTaskDelete(NULL);
        return;
    }
    printf("[geo] DB open OK\n");

    for (size_t qi = 0; qi < (sizeof(QUERIES) / sizeof(QUERIES[0])); ++qi) {
        const geo_query_t *q = &QUERIES[qi];
        float qvec[GEO_DIM];
        latlon_to_unit_vector(q->lat, q->lon, qvec);

        nn20db_vector_search_result results[GEO_TOP_K];

        int64_t t0 = esp_timer_get_time();
        rc = nn20db_vector_search_ef(db, qvec, GEO_TOP_K, GEO_EF, results);
        int64_t search_us = esp_timer_get_time() - t0;

        if (rc != NN20DB_ERROR_OK) {
            printf("[geo] search failed for '%s' rc=%d\n", q->label, rc);
            continue;
        }

        printf("\nQuery: %s (%.5f, %.5f)  [%.3f ms]\n",
               q->label, q->lat, q->lon, (double)(search_us / 1000.0));
        printf("Top %d nearest cities:\n", GEO_TOP_K);

        for (int i = 0; i < GEO_TOP_K; ++i) {
            city_meta_t meta;
            memset(&meta, 0, sizeof(meta));

            if (nn20db_vector_get(db, results[i].id, NULL, &meta) != NN20DB_ERROR_OK) {
                printf("  %2d. <metadata read failed>\n", i + 1);
                continue;
            }

            char name[33];
            char country[3] = { meta.country[0], meta.country[1], '\0' };
            copy_name_32(name, sizeof(name), meta.name);

            float dist = haversine_km(q->lat, q->lon, meta.lat, meta.lon);

            printf("  %2d. %-24s %s  pop=%-10ld  lat=%9.5f lon=%9.5f  distance=%8.1f km\n",
                   i + 1, name, country, (long)meta.pop,
                   (double)meta.lat, (double)meta.lon, (double)dist);
        }
    }

    nn20db_dtor(db);
    printf("\n[geo] done\n");
    vTaskDelete(NULL);
}

void app_main(void) {
    BaseType_t ok = xTaskCreatePinnedToCore(
        run_queries, "geo_queries", 32 * 1024, NULL, tskIDLE_PRIORITY + 1, NULL, 1);

    if (ok != pdPASS) {
        ok = xTaskCreatePinnedToCore(
            run_queries, "geo_queries", 32 * 1024, NULL, tskIDLE_PRIORITY + 1, NULL, 0);
    }

    if (ok != pdPASS) {
        printf("[geo] task creation failed\n");
    }
}