PORT ?= /dev/ttyACM0

# Product quantization config toggle
# Override: make build USE_PQ=0
USE_PQ ?= 1

# SDMMC data lines used by the board SD driver (4 or 1)
# Override: make build SD_BUS_WIDTH=1
SD_BUS_WIDTH ?= 4

# SD card board driver (see main/sd_storage.h); default picked per target
# in main/CMakeLists.txt. Override: make build SD_BOARD=none
IDF_ARGS := -DUSE_PQ=$(USE_PQ) -DSD_BUS_WIDTH=$(SD_BUS_WIDTH)
ifdef SD_BOARD
IDF_ARGS += -DSD_BOARD=$(SD_BOARD)
endif

.PHONY: all build flash monitor flash-monitor clean

all: build

build:
	idf.py $(IDF_ARGS) build

flash:
	idf.py -p $(PORT) flash

monitor:
	idf.py -p $(PORT) monitor --no-reset

flash-monitor:
	idf.py -p $(PORT) flash monitor --no-reset

set-target-p4:
	idf.py set-target esp32p4

set-target-s3:
	idf.py set-target esp32s3

clean:
	idf.py fullclean
