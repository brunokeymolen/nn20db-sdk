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
 * sd_storage_waveshare_p4.c - SDMMC driver for the Waveshare ESP32-P4 board.
 *
 * Provides the board's SDMMC mount/unmount hooks for nn20db. The board uses
 * the P4's native IOMUX SDMMC slot (CLK=43/CMD=44/D0=39-D3=42) whose IO
 * domain is powered by the chip's on-chip LDO (LDO_VO4) rather than an
 * external supply — that LDO channel must be brought up before the slot is
 * mounted. Without this driver nn20db's built-in ESP32-P4 default is used,
 * which mounts the same native slot at SDMMC_FREQ_DEFAULT (20 MHz); this
 * driver mounts at SDMMC_FREQ_HIGHSPEED (40 MHz) instead.
 */

#include "sd_storage.h"

#include <dirent.h>
#include <stdint.h>
#include <stdio.h>

#include "driver/sdmmc_host.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "sd_pwr_ctrl_by_on_chip_ldo.h"
#include "sdmmc_cmd.h"

#include "platform/nn20db_esp32_storage.h"

static const char *TAG = "sd_storage";

/* SD card SDMMC pins - Waveshare ESP32-P4 (native IOMUX slot 0) */
#define PIN_SD_CLK      43
#define PIN_SD_CMD      44
#define PIN_SD_D0       39
#define PIN_SD_D1       40
#define PIN_SD_D2       41
#define PIN_SD_D3       42

/* On-chip LDO channel powering the SDMMC IO domain (LDO_VO4 on the P4
 * Function-EV-Board and boards following its reference schematic). */
#ifndef SD_LDO_CHAN_ID
#define SD_LDO_CHAN_ID  4
#endif

/* Number of SDMMC data lines (4 or 1). Override at build time with
 * make build SD_BUS_WIDTH=1 to benchmark 1-bit vs 4-bit on the same pins
 * and clock; 1-bit mode only drives CLK/CMD/D0. */
#ifndef SD_BUS_WIDTH
#define SD_BUS_WIDTH 4
#endif

typedef struct {
    sdmmc_card_t *card;
    sd_pwr_ctrl_handle_t pwr_ctrl_handle;
} p4_sd_handle_t;

static p4_sd_handle_t s_handle;

static int sd_storage_mount(const char *mount_point, void **out_handle)
{
    ESP_LOGI(TAG, "Mounting SD card at %s (%d-bit) ...", mount_point, SD_BUS_WIDTH);

    sd_pwr_ctrl_ldo_config_t ldo_cfg = {
        .ldo_chan_id = SD_LDO_CHAN_ID,
    };
    sd_pwr_ctrl_handle_t pwr_ctrl_handle = NULL;
    esp_err_t err = sd_pwr_ctrl_new_on_chip_ldo(&ldo_cfg, &pwr_ctrl_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to init SD IO LDO (chan %d): %s", SD_LDO_CHAN_ID, esp_err_to_name(err));
        return -1;
    }

    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    /* SDMMC_HOST_DEFAULT() runs at SDMMC_FREQ_DEFAULT (20 MHz); high-speed
     * mode doubles the bus clock. Drop back to default if the card or
     * wiring proves unreliable at 40 MHz. */
    host.max_freq_khz = SDMMC_FREQ_HIGHSPEED;
    host.pwr_ctrl_handle = pwr_ctrl_handle;

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
    err = esp_vfs_fat_sdmmc_mount(mount_point, &host, &slot_cfg,
                                   &mount_cfg, &card);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "SD mount failed: %s", esp_err_to_name(err));
        sd_pwr_ctrl_del_on_chip_ldo(pwr_ctrl_handle);
        return -1;
    }

    ESP_LOGI(TAG, "SD mounted OK (%s, %llu MB, %d-bit @ %u kHz)",
             card->cid.name,
             (unsigned long long)((uint64_t)card->csd.capacity *
                                  card->csd.sector_size / (1024 * 1024)),
             SD_BUS_WIDTH, (unsigned int)card->max_freq_khz);

    s_handle.card = card;
    s_handle.pwr_ctrl_handle = pwr_ctrl_handle;
    *out_handle = &s_handle;
    return 0;
}

static void sd_storage_unmount(const char *mount_point, void *handle)
{
    p4_sd_handle_t *h = (p4_sd_handle_t *)handle;
    esp_vfs_fat_sdcard_unmount(mount_point, h->card);
    sd_pwr_ctrl_del_on_chip_ldo(h->pwr_ctrl_handle);
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
