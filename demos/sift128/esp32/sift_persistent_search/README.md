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

## Build the database (Linux)

```bash
cd demos/sift128/linux/sift_persistent
make
./sift_persistent_demo /path/to/sift-128-euclidean.hdf5 ./db/sift128
```

The dataset file `sift-128-euclidean.hdf5` can be downloaded from:  
`http://ann-benchmarks.com/sift-128-euclidean.hdf5`

Then copy the database directory to the SD card:

```
demos/sift128/linux/sift_persistent/db/sift128/  →  <SD card>/nand0/sift128/
```

Insert the SD card into the ESP32-P4 before flashing (or reset the ESP32
after inserting the card).

---

## Build, flash and run

```bash
cd demos/sift128/esp32/sift_persistent_search

make build          # compile
make flash          # flash to device on /dev/ttyACM0
make monitor        # open serial monitor (Ctrl-] to exit)

make flash-monitor  # flash and open monitor in one step
```

To use a different serial port:

```bash
make flash-monitor PORT=/dev/ttyUSB0
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
