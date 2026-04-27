# geo_esp32 — GeoNames nearest-city search demo (ESP32-P4)

This demo opens a pre-built GeoNames database on an SD card and runs four
nearest-city queries using nn20db's vector search.  It produces the same
results as the Linux/Python counterpart in `demos/geo/linux/python/`.

> **Search-only demo.**  
> nn20db supports inserting vectors on the ESP32 (the SDK exposes the same
> insert API), but insertions are significantly slower than on a PC because
> of the constrained storage bandwidth.  For this reason the demo is
> intentionally read-only: the database is built once on Linux and then
> copied to the device.

---

## Prerequisites

- ESP-IDF v5.x installed and sourced (`idf.py` on `$PATH`)
- nn20db ESP32 SDK installed (`make build` will tell you if it is missing)
- The geo10k database built with `demos/geo/linux/python/demo_geonames_10k.py`

---

## Build the database (Linux)

```bash
cd demos/geo/linux/python
python demo_geonames_10k.py   # creates geo10k/ in the current directory
```

Then copy the database directory to the SD card:

```
demos/geo/linux/python/data/geo10k/  →  <SD card>/nand0/geo10k/
```

Insert the SD card into the ESP32-P4 before flashing (or reset the esp32 after insert).

---

## Build, flash and run


* In case of Visual Studio Code, open this project and do F1: "Dev Containers : Reopen In Container"

```bash
cd demos/geo/esp32

get_idf             # once

make build          # compile
make flash          # flash to device on /dev/ttyACM0
make monitor        # open serial monitor (Ctrl-] to exit)

make flash-monitor  # flash and open monitor in one step
```

To use a different serial port:

```bash
make flash-monitor PORT=/dev/ttyUSB0
```

To clean the build artefacts:

```bash
make clean
```

---

## Expected output

```
[geo] opening DB at /sdcard/nand0/geo10k
[geo] DB open OK

Query: Brussels, Belgium (50.85045, 4.34878)  [x.xxx ms]
Top 10 nearest cities:
   1. Brussels                 BE  pop=1019022     lat= 50.85045 lon=  4.34878  distance=     0.0 km
   2. Koekelberg               BE  pop=21984       lat= 50.86117 lon=  4.33136  distance=     1.7 km
  ...

Query: Paris, France ...
Query: New York, USA ...
Query: Tokyo, Japan ...

[geo] done
```

Results match the Linux demo output.
