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

#define _POSIX_C_SOURCE 200809L

#include <hdf5.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "nn20db.h"

typedef struct {
    float *train;
    float *test;
    int32_t *neighbors;
    size_t n_train;
    size_t n_test;
    size_t dim;
    size_t k_gt;
} ann_data_t;

static void die(const char *message)
{
    perror(message);
    exit(1);
}

static double now_seconds(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void read_2d_float(hid_t file, const char *name, float **out, size_t *rows, size_t *cols)
{
    hid_t dataset = H5Dopen2(file, name, H5P_DEFAULT);
    hid_t space;
    hsize_t dims[2];

    if (dataset < 0) {
        die("H5Dopen2");
    }

    space = H5Dget_space(dataset);
    if (H5Sget_simple_extent_ndims(space) != 2) {
        die("expected 2D float dataset");
    }

    H5Sget_simple_extent_dims(space, dims, NULL);
    *rows = (size_t)dims[0];
    *cols = (size_t)dims[1];
    *out = malloc((*rows) * (*cols) * sizeof(**out));
    if (*out == NULL) {
        die("malloc");
    }

    if (H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, *out) < 0) {
        die("H5Dread");
    }

    H5Sclose(space);
    H5Dclose(dataset);
}

static void read_2d_i32(hid_t file, const char *name, int32_t **out, size_t *rows, size_t *cols)
{
    hid_t dataset = H5Dopen2(file, name, H5P_DEFAULT);
    hid_t space;
    hsize_t dims[2];

    if (dataset < 0) {
        die("H5Dopen2");
    }

    space = H5Dget_space(dataset);
    if (H5Sget_simple_extent_ndims(space) != 2) {
        die("expected 2D int32 dataset");
    }

    H5Sget_simple_extent_dims(space, dims, NULL);
    *rows = (size_t)dims[0];
    *cols = (size_t)dims[1];
    *out = malloc((*rows) * (*cols) * sizeof(**out));
    if (*out == NULL) {
        die("malloc");
    }

    if (H5Dread(dataset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, *out) < 0) {
        die("H5Dread");
    }

    H5Sclose(space);
    H5Dclose(dataset);
}

static ann_data_t load_ann_bench(const char *path)
{
    ann_data_t data = {0};
    hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    size_t test_dim = 0;
    size_t gt_rows = 0;

    if (file < 0) {
        die("H5Fopen");
    }

    read_2d_float(file, "train", &data.train, &data.n_train, &data.dim);
    read_2d_float(file, "test", &data.test, &data.n_test, &test_dim);
    read_2d_i32(file, "neighbors", &data.neighbors, &gt_rows, &data.k_gt);

    H5Fclose(file);

    if (test_dim != data.dim || gt_rows != data.n_test) {
        fprintf(stderr, "dataset shape mismatch\n");
        exit(1);
    }

    return data;
}

static void free_ann_data(ann_data_t *data)
{
    free(data->train);
    free(data->test);
    free(data->neighbors);
    memset(data, 0, sizeof(*data));
}

static nn20db_config make_config(const ann_data_t *data, const char *db_path)
{
    nn20db_config config = {
        .vector = {
            .type          = NN20DB_DIMENSION_FLOAT32_CONFIG,
            .dimension     = (int)data->dim,
            .metadata_size = (int)sizeof(int32_t),
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
    };

    snprintf(config.storage.lfs.device_path, sizeof(config.storage.lfs.device_path), "%s", db_path);

    return config;
}

static int ensure_parent_dir(const char *db_path)
{
    char *copy = strdup(db_path);
    char *slash;

    if (copy == NULL) {
        return -1;
    }

    slash = strrchr(copy, '/');
    if (slash == NULL) {
        free(copy);
        return 0;
    }

    *slash = '\0';
    if (copy[0] == '\0') {
        free(copy);
        return 0;
    }

    if (mkdir(copy, 0777) != 0 && errno != EEXIST) {
        free(copy);
        return -1;
    }

    free(copy);
    return 0;
}

static NN20DB *open_or_create_db(const nn20db_config *config, int *created_db)
{
    NN20DB *db = NULL;
    int rc;

    *created_db = 0;
    rc = nn20db_create(config, &db);
    if (rc == NN20DB_ERROR_DB_ALREADY_EXISTS) {
        rc = nn20db_open_with_config(config, &db);
    } else if (rc == NN20DB_ERROR_OK) {
        *created_db = 1;
    }

    if (rc != NN20DB_ERROR_OK || db == NULL) {
        fprintf(stderr, "Failed to open/create DB at %s (rc=%d)\n", config->storage.lfs.device_path, rc);
        return NULL;
    }

    return db;
}

static void print_progress(size_t done, size_t total, double elapsed)
{
    int bar_width = 40;
    double fraction = (total > 0) ? (double)done / (double)total : 0.0;
    int filled = (int)(fraction * bar_width);
    double rate = (elapsed > 0.0) ? (double)done / elapsed : 0.0;
    double eta = (rate > 0.0 && done < total) ? (double)(total - done) / rate : 0.0;
    int i;

    fprintf(stderr, "\r  [");
    for (i = 0; i < bar_width; ++i) {
        if (i < filled) {
            fputc('=', stderr);
        } else if (i == filled) {
            fputc('>', stderr);
        } else {
            fputc(' ', stderr);
        }
    }
    fprintf(stderr, "] %5.1f%%  %zu/%zu  %.0f vec/s  ETA %ds  ",
            fraction * 100.0, done, total, rate, (int)eta);
}

static int ingest_vectors(NN20DB *db, const ann_data_t *data)
{
    size_t i;
    double t_start = now_seconds();
    size_t print_interval = data->n_train / 1000;

    if (print_interval < 100) {
        print_interval = 100;
    }

    for (i = 0; i < data->n_train; ++i) {
        const float *vector = data->train + (i * data->dim);
        int32_t metadata = (int32_t)i;
        int rc = nn20db_vector_add(db, vector, &metadata);

        if (rc != NN20DB_ERROR_OK) {
            fprintf(stderr, "\nvector add failed at %zu (rc=%d)\n", i, rc);
            return 1;
        }

        if ((i + 1) % print_interval == 0 || (i + 1) == data->n_train) {
            print_progress(i + 1, data->n_train, now_seconds() - t_start);
        }

        if ((i + 1) % 50000 == 0) {
            if (nn20db_compact(db) != NN20DB_ERROR_OK || nn20db_sync(db) != NN20DB_ERROR_OK) {
                fprintf(stderr, "\nmaintenance failed after %zu vectors\n", i + 1);
                return 1;
            }
        }
    }

    fprintf(stderr, "\n");

    if (nn20db_compact(db) != NN20DB_ERROR_OK || nn20db_sync(db) != NN20DB_ERROR_OK) {
        fprintf(stderr, "final maintenance failed\n");
        return 1;
    }

    return 0;
}


static int run_queries(NN20DB *db, const ann_data_t *data, size_t query_limit)
{
    int k = data->k_gt < 10 ? (int)data->k_gt : 10;
    int ef_search = 64;

    size_t total_hits = 0;
    size_t qi;
    nn20db_vector_search_result *results;
    int32_t *metadata;
    double t_start;
    double elapsed;
    double qps;

    if (query_limit > data->n_test) {
        query_limit = data->n_test;
    }

    results = malloc((size_t)k * sizeof(*results));
    metadata = malloc((size_t)k * sizeof(*metadata));
    if (results == NULL || metadata == NULL) {
        fprintf(stderr, "result allocation failed\n");
        free(results);
        free(metadata);
        return 1;
    }

    t_start = now_seconds();
    for (qi = 0; qi < query_limit; ++qi) {
        const float *query = data->test + (qi * data->dim);
        const int32_t *gt_row = data->neighbors + (qi * data->k_gt);
        int rc = nn20db_vector_search_ef(db, query, k, ef_search, results);
        int ri;

        if (rc != NN20DB_ERROR_OK) {
            fprintf(stderr, "search failed at query %zu (rc=%d)\n", qi, rc);
            free(results);
            free(metadata);
            return 1;
        }

        for (ri = 0; ri < k; ++ri) {
            rc = nn20db_vector_get(db, results[ri].id, NULL, &metadata[ri]);
            if (rc != NN20DB_ERROR_OK) {
                metadata[ri] = -1;
            }
        }

        for (ri = 0; ri < k; ++ri) {
            size_t gi;

            if (metadata[ri] < 0) {
                continue;
            }
            for (gi = 0; gi < (size_t)k; ++gi) {
                if (metadata[ri] == gt_row[gi]) {
                    total_hits++;
                    break;
                }
            }
        }
    }
    elapsed = now_seconds() - t_start;
    qps = (elapsed > 0.0) ? (double)query_limit / elapsed : 0.0;

    printf("queries=%zu ef_search=%d qps=%.2f recall@%d=%.6f\n", query_limit, ef_search, qps, k,
           (double)total_hits / (double)((size_t)k * query_limit));

    free(results);
    free(metadata);
    return 0;
}

int main(int argc, char **argv)
{
    ann_data_t data;
    nn20db_config config;
    NN20DB *db;
    size_t query_limit = 100;
    int created_db;
    double t0;
    double t1;
    double t2;

    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s <dataset.hdf5> <db-path> [query-limit]\n", argv[0]);
        return 2;
    }

    if (argc == 4) {
        query_limit = (size_t)strtoull(argv[3], NULL, 10);
        if (query_limit == 0) {
            fprintf(stderr, "query-limit must be > 0\n");
            return 2;
        }
    }

    if (ensure_parent_dir(argv[2]) != 0) {
        perror("mkdir");
        return 1;
    }

    t0 = now_seconds();
    data = load_ann_bench(argv[1]);
    t1 = now_seconds();

    config = make_config(&data, argv[2]);
    db = open_or_create_db(&config, &created_db);
    if (db == NULL) {
        free_ann_data(&data);
        return 1;
    }

    if (created_db) {
        printf("created DB, ingesting %zu vectors\n", data.n_train);
        if (ingest_vectors(db, &data) != 0) {
            nn20db_dtor(db);
            free_ann_data(&data);
            return 1;
        }
    } else {
        printf("opened existing DB at %s\n", argv[2]);
    }

    t2 = now_seconds();
    if (run_queries(db, &data, query_limit) != 0) {
        nn20db_dtor(db);
        free_ann_data(&data);
        return 1;
    }

    printf("timing load=%.3fs ingest_or_open=%.3fs total=%.3fs\n",
           t1 - t0,
           t2 - t1,
           now_seconds() - t0);

    nn20db_dtor(db);
    free_ann_data(&data);
    return 0;
}
