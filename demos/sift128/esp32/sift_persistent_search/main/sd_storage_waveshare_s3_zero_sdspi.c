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
 * sd_storage_waveshare_s3_zero_sdspi.c - SD-over-SPI (SDSPI) driver for the
 * Waveshare ESP32-S3-Zero with an external microSD breakout.
 *
 * Unlike the Touch-LCD and P4 boards (which use the parallel SDMMC slot),
 * the ESP32-S3-Zero has no dedicated SD slot: the card is wired to a generic
 * SPI breakout (e.g. a DAOKAI microSD module) on the FSPI (SPI2) bus. SD in
 * SPI mode is inherently single-data-line, so SD_BUS_WIDTH does not apply
 * here. Without this driver nn20db's built-in ESP32-S3 default would be used,
 * which mounts 1-bit SDMMC at 20 MHz on the IDF default pins (CLK=14/CMD=15/
 * D0=2) — the wrong wiring and wrong transport for this board.
 *
 * Wiring (DAOKAI microSD breakout -> ESP32-S3-Zero, FSPI/SPI2):
 *   CS   -> GPIO10  (FSPICS0)
 *   MOSI -> GPIO11  (FSPID)
 *   SCK  -> GPIO12  (FSPICLK)
 *   MISO -> GPIO13  (FSPIQ)
 *   GND  -> GND
 *   3V3  -> 3V3
 */

#include "sd_storage.h"

#include <dirent.h>
#include <stdint.h>
#include <stdio.h>

#include "driver/sdspi_host.h"
#include "driver/spi_common.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"

#include "platform/nn20db_esp32_storage.h"

static const char *TAG = "sd_storage";

/* SD card SPI pins - Waveshare ESP32-S3-Zero + microSD breakout (FSPI/SPI2) */
#define PIN_SD_CS       10
#define PIN_SD_MOSI     11
#define PIN_SD_CLK      12
#define PIN_SD_MISO     13

/* SPI bus clock. SD SPI mode tops out well below the parallel slot; a breakout
 * on jumper wires is often unreliable above the 20 MHz default, so start there
 * and raise (e.g. SDMMC_FREQ_HIGHSPEED = 40 MHz) only if the wiring is clean.
 * Override at build time with make build SD_SPI_FREQ_KHZ=40000. */
#ifndef SD_SPI_FREQ_KHZ
#define SD_SPI_FREQ_KHZ SDMMC_FREQ_DEFAULT
#endif

typedef struct {
    sdmmc_card_t *card;
    spi_host_device_t spi_host;
} s3zero_sd_handle_t;

static s3zero_sd_handle_t s_handle;

static int sd_storage_mount(const char *mount_point, void **out_handle)
{
    ESP_LOGI(TAG, "Mounting SD card at %s (SPI, 1-bit) ...", mount_point);

    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    host.max_freq_khz = SD_SPI_FREQ_KHZ;

    spi_bus_config_t bus_cfg = {
        .mosi_io_num     = PIN_SD_MOSI,
        .miso_io_num     = PIN_SD_MISO,
        .sclk_io_num     = PIN_SD_CLK,
        .quadwp_io_num   = -1,
        .quadhd_io_num   = -1,
        .max_transfer_sz = 4000,
    };
    esp_err_t err = spi_bus_initialize(host.slot, &bus_cfg, SDSPI_DEFAULT_DMA);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "SPI bus init failed: %s", esp_err_to_name(err));
        return -1;
    }

    sdspi_device_config_t slot_cfg = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_cfg.gpio_cs = PIN_SD_CS;
    slot_cfg.host_id = host.slot;

    esp_vfs_fat_sdmmc_mount_config_t mount_cfg = {
        .format_if_mount_failed = false,
        .max_files              = 24,
        .allocation_unit_size   = 16 * 1024,
    };

    sdmmc_card_t *card = NULL;
    err = esp_vfs_fat_sdspi_mount(mount_point, &host, &slot_cfg,
                                   &mount_cfg, &card);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "SD mount failed: %s", esp_err_to_name(err));
        spi_bus_free(host.slot);
        return -1;
    }

    ESP_LOGI(TAG, "SD mounted OK (%s, %llu MB, SPI @ %u kHz)",
             card->cid.name,
             (unsigned long long)((uint64_t)card->csd.capacity *
                                  card->csd.sector_size / (1024 * 1024)),
             (unsigned int)card->max_freq_khz);

    s_handle.card = card;
    s_handle.spi_host = host.slot;
    *out_handle = &s_handle;
    return 0;
}

static void sd_storage_unmount(const char *mount_point, void *handle)
{
    s3zero_sd_handle_t *h = (s3zero_sd_handle_t *)handle;
    esp_vfs_fat_sdcard_unmount(mount_point, h->card);
    spi_bus_free(h->spi_host);
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
