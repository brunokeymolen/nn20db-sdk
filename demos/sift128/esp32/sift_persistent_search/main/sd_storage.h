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
 * sd_storage.h - board-specific SD card driver for nn20db.
 *
 * Each supported board provides one sd_storage_<board>.c implementing this
 * interface; the board is selected at build time with the SD_BOARD CMake
 * variable (see main/CMakeLists.txt). SD_BOARD=none compiles no driver and
 * nn20db falls back to its built-in target default (e.g. the ESP32-P4
 * devkit's native 4-bit slot).
 *
 * Boards:
 *   waveshare_s3_touch — Waveshare ESP32-S3-Touch-LCD-1.47
 */

#pragma once

/* Register the board's SDMMC mount/unmount hooks with nn20db.
 * Must be called before any nn20db call. */
void sd_storage_register_driver(void);

/* Recursively log the SD card contents (diagnostics). */
void sd_storage_log_contents(const char *mount_point);
