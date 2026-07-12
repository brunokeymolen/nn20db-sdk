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

/*
 * sd_storage_waveshare_s3_touch.c - SDMMC driver for the
 * Waveshare ESP32-S3-Touch-LCD-1.47 board.
 *
 * Provides the board's SDMMC mount/unmount hooks for nn20db. Without this,
 * nn20db's built-in ESP32-S3 default would be used, which mounts 1-bit at
 * 20 MHz on the IDF default pins (CLK=14/CMD=15/D0=2) — the wrong wiring
 * for this board.
 */

#include "sd_storage.h"

#include <dirent.h>
#include <stdint.h>
#include <stdio.h>

#include "driver/sdmmc_host.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"

#include "platform/nn20db_esp32_storage.h"

static const char *TAG = "sd_storage";

/* SD card SDMMC pins - Waveshare ESP32-S3-Touch-LCD-1.47 (from official BSP) */
#define PIN_SD_CLK      16
#define PIN_SD_CMD      15
#define PIN_SD_D0       17
#define PIN_SD_D1       18
#define PIN_SD_D2       13
#define PIN_SD_D3       14

/* Number of SDMMC data lines (4 or 1). Override at build time with
 * make build SD_BUS_WIDTH=1 to benchmark 1-bit vs 4-bit on the same pins
 * and clock; 1-bit mode only drives CLK/CMD/D0. */
#ifndef SD_BUS_WIDTH
#define SD_BUS_WIDTH 4
#endif

static int sd_storage_mount(const char *mount_point, void **out_handle)
{
    ESP_LOGI(TAG, "Mounting SD card at %s (%d-bit) ...", mount_point, SD_BUS_WIDTH);

    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    /* SDMMC_HOST_DEFAULT() runs at SDMMC_FREQ_DEFAULT (20 MHz); high-speed
     * mode doubles the bus clock. Drop back to default if the card or
     * wiring proves unreliable at 40 MHz. */
    host.max_freq_khz = SDMMC_FREQ_HIGHSPEED;

    sdmmc_slot_config_t slot_cfg = {};
    slot_cfg.clk   = PIN_SD_CLK;
    slot_cfg.cmd   = PIN_SD_CMD;
    slot_cfg.d0    = PIN_SD_D0;
    slot_cfg.d1    = PIN_SD_D1;
    slot_cfg.d2    = PIN_SD_D2;
    slot_cfg.d3    = PIN_SD_D3;
    slot_cfg.width = SD_BUS_WIDTH;
    slot_cfg.cd    = SDMMC_SLOT_NO_CD;
    slot_cfg.wp    = SDMMC_SLOT_NO_WP;
    slot_cfg.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;

    esp_vfs_fat_sdmmc_mount_config_t mount_cfg = {
        .format_if_mount_failed = false,
        .max_files              = 24,
        .allocation_unit_size   = 16 * 1024,
    };

    sdmmc_card_t *card = NULL;
    esp_err_t err = esp_vfs_fat_sdmmc_mount(mount_point, &host, &slot_cfg,
                                             &mount_cfg, &card);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "SD mount failed: %s", esp_err_to_name(err));
        return -1;
    }

    ESP_LOGI(TAG, "SD mounted OK (%s, %llu MB, %d-bit @ %d kHz)",
             card->cid.name,
             (unsigned long long)((uint64_t)card->csd.capacity *
                                  card->csd.sector_size / (1024 * 1024)),
             SD_BUS_WIDTH, card->max_freq_khz);
    *out_handle = card;
    return 0;
}

static void sd_storage_unmount(const char *mount_point, void *handle)
{
    esp_vfs_fat_sdcard_unmount(mount_point, (sdmmc_card_t *)handle);
}

static const nn20db_esp32_storage_driver_t s_sd_driver = {
    .mount   = sd_storage_mount,
    .unmount = sd_storage_unmount,
};

void sd_storage_register_driver(void)
{
    nn20db_esp32_set_storage_driver(&s_sd_driver);
}

static void sd_storage_list_dir(const char *path, int depth)
{
    DIR *dir = opendir(path);
    if (!dir) {
        ESP_LOGW(TAG, "  [cannot open %s]", path);
        return;
    }

    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        char child[512];
        snprintf(child, sizeof(child), "%s/%s", path, ent->d_name);
        if (ent->d_type == DT_DIR) {
            ESP_LOGI(TAG, "%*s[%s/]", depth * 2, "", ent->d_name);
            sd_storage_list_dir(child, depth + 1);
        } else {
            ESP_LOGI(TAG, "%*s%s", depth * 2, "", ent->d_name);
        }
    }
    closedir(dir);
}

void sd_storage_log_contents(const char *mount_point)
{
    ESP_LOGI(TAG, "--- SD card contents (%s) ---", mount_point);
    sd_storage_list_dir(mount_point, 0);
    ESP_LOGI(TAG, "--- end of SD listing ---");
}
