# sift_persistent_search — SIFT-128 recall demo (ESP32-P4)

This demo opens a pre-built SIFT-128 database on an SD card, runs a set of
test queries and reports recall@10 — the same measurement as the Linux
counterpart in `demos/sift128/linux/sift_persistent/`.

> **Search-only demo.**  
> nn20db supports inserting vectors on the ESP32 (the SDK exposes the same
> insert API), but insertions are significantly slower than on a PC because
> of the constrained storage bandwidth.  The database is built once on Linux
> and then copied to the device.

---

## Prerequisites

- ESP-IDF v5.x installed and sourced (`idf.py` on `$PATH`)
- nn20db ESP32 SDK installed (`make build` will tell you if it is missing)
- The SIFT-128 database built with `demos/sift128/linux/sift_persistent/`

---


## PQ vs fp32

The demo builds in one of two vector configurations, selected with `USE_PQ`
(default `1`); the same toggle exists in the Linux generator:

Build              | Config file          | SD card DB path      | Node size
-------------------|----------------------|----------------------|----------
`USE_PQ=1` (deflt) | `main/config_pq.h`   | `/sdcard/nand0/siftpq`  | ~1.2 KB
`USE_PQ=0`         | `main/config_fp32.h` | `/sdcard/nand0/sift128` | ~1.7 KB

Both configs open the DB **read-only** (`NN20DB_STORAGE_FLAGS_READ_ONLY`),
which enables FATFS fast-seek (`CONFIG_FATFS_USE_FASTSEEK` in
`sdkconfig.defaults`) — without it every random read walks the FAT chain of
the 128 MB lane files. They also warm the HNSW node cache around the entry
point at open (`tuning.hnsw_cache_warm_depth`).

## SD card board driver

nn20db has a built-in per-target SD driver, but its ESP32-S3 default mounts
1-bit @ 20 MHz on the IDF default pins — the wrong wiring for most boards.
The demo therefore registers a board-specific driver, selected at build time
with `SD_BOARD` (one `main/sd_storage_<board>.c` per board; add a new file to
support new hardware):

Board                              | `SD_BOARD`           | Default on
-----------------------------------|----------------------|-----------
Waveshare ESP32-S3-Touch-LCD-1.47  | `waveshare_s3_touch` | esp32s3
nn20db built-in target default     | `none`               | esp32p4 (native 4-bit LDO slot)

The Waveshare driver mounts 4-bit @ 40 MHz. The number of data lines can be
overridden for I/O experiments:

```bash
make build                  # 4-bit (default)
make build SD_BUS_WIDTH=1   # 1-bit, same pins and clock
make build SD_BOARD=none    # no driver, nn20db built-in default
```

## Build the database (Linux)

```bash
cd demos/sift128/linux/sift_persistent
make                  # PQ (default); use make USE_PQ=0 for fp32
./sift_persistent_demo /path/to/sift-128-euclidean.hdf5 ./db/siftpq
```

The dataset file `sift-128-euclidean.hdf5` can be downloaded from:  
`http://ann-benchmarks.com/sift-128-euclidean.hdf5`

Then copy the database directory to the SD card (match the build variant):

```
demos/sift128/linux/sift_persistent/db/siftpq/   →  <SD card>/nand0/siftpq/    (PQ)
demos/sift128/linux/sift_persistent/db/sift128/  →  <SD card>/nand0/sift128/   (fp32)
```

Insert the SD card into the ESP32-P4 before flashing (or reset the ESP32
after inserting the card).

---

## sdkconfig 
File	                    | Key settings
----------------------------| ---
sdkconfig.defaults          | Shared: FreeRTOS stack sizes, FAT (4096 sectors, 2 volumes, no LFN), VFS 8 slots
sdkconfig.defaults.esp32p4  | HEX PSRAM, 200 MHz, malloc fallback, FATFS_ALLOC_PREFER_EXTRAM
sdkconfig.defaults.esp32s3  | Octal PSRAM, 40 MHz, malloc fallback, FATFS_ALLOC_PREFER_EXTRAM

### P4
rm -rf build sdkconfig && idf.py set-target esp32p4 && make build USE_PQ=0

### S3
rm -rf build sdkconfig && idf.py set-target esp32s3 && make build USE_PQ=0


test with 1 dataline mmc (1 bit): 
make build SD_BUS_WIDTH=1 USE_PQ=0


---

## Build, flash and run

```bash
cd demos/sift128/esp32/sift_persistent_search

make build          # compile (PQ config; use make build USE_PQ=0 for fp32)
make flash          # flash to device on /dev/ttyACM0, some s3 have /dev/ttyUSB0
make monitor        # open serial monitor (Ctrl-] to exit)

make flash-monitor  # flash and open monitor in one step
```

To use a different serial port:

```bash
make flash-monitor PORT=/dev/ttyUSB0

make PORT=/dev/ttyUSB0 flash monitor
```

To clean build artefacts:

```bash
make clean
```

---

## Expected output

```
[sift-search] opening DB at /sdcard/nand0/sift128
[sift-search] DB open OK
[sift-search] query_0000: hits=10/10  (top results: id=... d=...)  time=x.xxxs
[sift-search] query_0001: hits=10/10  ...
...
[sift-search] recall@10 = 0.9800  (980/1000 over 100 queries)
[sift-search] done
```

Recall figures match those reported by the Linux demo.

---

## Troubleshooting

### `chip revision in range [v3.1 - v3.99] (tin case of chip revision v1.0)`

```
A fatal error occurred: bootloader/bootloader.bin requires chip revision in range [v3.1 - v3.99] (this chip is revision v1.0). Use --force to flash anyway.
```

`sdkconfig` has `CONFIG_ESP32P4_REV_MIN_301=y` (minimum supported chip
revision v3.1), but the actual ESP32-P4 silicon on the (Waveshare) board is
an earlier revision (v1.0). ESP-IDF's ESP32-P4 default targets MP silicon
(rev >= v3.0); support for rev < v3.0 and rev >= v3.0 chips is mutually
exclusive (`CONFIG_ESP32P4_SELECTS_REV_LESS_V3`), so the bootloader built
against the v3.0+ family refuses to boot on the older chip and esptool
aborts the flash instead of writing a bootloader the chip can't run.

Fix: select the rev < v3.0 chip family and lower the minimum supported
revision to v1.0 to match the board, either via menuconfig:

```bash
idf.py menuconfig
# Component config -> Hardware Settings -> Chip revision ->
#   "Minimum Supported ESP32-P4 Revision" -> select "Rev v1.0" (or lower)
#   (this also flips CONFIG_ESP32P4_SELECTS_REV_LESS_V3 on)

rm -rf build && idf.py build && make flash
```

or by editing `sdkconfig` directly — this is the resulting diff:

```diff
@@ -1250,17 +1250,18 @@
 #
 # Read the help text of the option below for explanation
 #
-# CONFIG_ESP32P4_SELECTS_REV_LESS_V3 is not set
-# CONFIG_ESP32P4_REV_MIN_300 is not set
-CONFIG_ESP32P4_REV_MIN_301=y
-CONFIG_ESP32P4_REV_MIN_FULL=301
-CONFIG_ESP_REV_MIN_FULL=301
+CONFIG_ESP32P4_SELECTS_REV_LESS_V3=y
+# CONFIG_ESP32P4_REV_MIN_0 is not set
+# CONFIG_ESP32P4_REV_MIN_1 is not set
+CONFIG_ESP32P4_REV_MIN_100=y
+CONFIG_ESP32P4_REV_MIN_FULL=100
+CONFIG_ESP_REV_MIN_FULL=100
 
 #
-# Maximum Supported ESP32-P4 Revision (Rev v3.99)
+# Maximum Supported ESP32-P4 Revision (Rev v1.99)
 #
-CONFIG_ESP32P4_REV_MAX_FULL=399
-CONFIG_ESP_REV_MAX_FULL=399
+CONFIG_ESP32P4_REV_MAX_FULL=199
+CONFIG_ESP_REV_MAX_FULL=199
 CONFIG_ESP_EFUSE_BLOCK_REV_MIN_FULL=0
 CONFIG_ESP_EFUSE_BLOCK_REV_MAX_FULL=199
```

Since `sdkconfig` is regenerated from `sdkconfig.defaults` +
`sdkconfig.defaults.esp32p4` whenever you run `rm -rf build sdkconfig &&
idf.py set-target esp32p4` (see [sdkconfig](#sdkconfig) above), the two lines
```
CONFIG_ESP32P4_SELECTS_REV_LESS_V3=y
CONFIG_ESP32P4_REV_MIN_100=y
```
have also been added to `sdkconfig.defaults.esp32p4` so the fix survives a
clean re-target instead of reverting to IDF's v3.0+ default.

If you don't have menuconfig access, you can flash past the check with
`esptool.py ... --force`, but this is a dev-only workaround — the sdkconfig
should still be corrected so future `make flash` runs don't hit the same
error.
