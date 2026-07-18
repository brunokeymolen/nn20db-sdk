PORT ?= /dev/ttyACM0

# Product quantization config toggle
# Override: make build USE_PQ=0
USE_PQ ?= 1

# SDMMC data lines used by the board SD driver (4 or 1)
# Override: make build SD_BUS_WIDTH=1
SD_BUS_WIDTH ?= 4

# SPI bus clock (kHz) for SD-over-SPI board drivers (e.g. waveshare_s3_zero_sdspi)
# Override: make build SD_SPI_FREQ_KHZ=40000
SD_SPI_FREQ_KHZ ?= 20000

# SD card board driver (see main/sd_storage.h); default picked per target
# in main/CMakeLists.txt. Override: make build SD_BOARD=none
IDF_ARGS := -DUSE_PQ=$(USE_PQ) -DSD_BUS_WIDTH=$(SD_BUS_WIDTH) -DSD_SPI_FREQ_KHZ=$(SD_SPI_FREQ_KHZ)
ifdef SD_BOARD
IDF_ARGS += -DSD_BOARD=$(SD_BOARD)
endif

# Board-specific sdkconfig override fragment (optional):
#   sdkconfig.board.<SD_BOARD>  — merged AFTER sdkconfig.defaults[.<target>], so
#   it overrides the shared defaults (e.g. PSRAM mode, flash size). Only applied
#   when sdkconfig is (re)generated, i.e. by `set-target` or a first build.
# idf.py reads this ';'-separated list from the environment.
SDKCONFIG_DEFAULTS := sdkconfig.defaults
ifdef SD_BOARD
BOARD_FRAGMENT := $(wildcard sdkconfig.board.$(SD_BOARD))
ifneq ($(BOARD_FRAGMENT),)
SDKCONFIG_DEFAULTS := sdkconfig.defaults;$(BOARD_FRAGMENT)
endif
endif
export SDKCONFIG_DEFAULTS

.PHONY: all build flash monitor monitor-reset flash-monitor clean \
        set-target-p4 set-target-s3 s3-touch s3-zero p4

all: build

build:
	idf.py $(IDF_ARGS) build

flash:
	idf.py -p $(PORT) flash

monitor:
	idf.py -p $(PORT) monitor --no-reset

# Like `monitor` but resets the board on attach, so you see the boot log and
# any open-time crash (use this to debug startup / OOM-at-open failures).
monitor-reset:
	idf.py -p $(PORT) monitor

flash-monitor:
	idf.py -p $(PORT) flash monitor --no-reset

set-target-p4:
	idf.py set-target esp32p4

set-target-s3:
	idf.py set-target esp32s3

# Convenience targets: regenerate sdkconfig for the board (so the correct
# sdkconfig.board.<board> override is merged) and build in one step.
# The SD_BOARD passed here selects both the SD driver and the board fragment.
s3-touch:
	$(MAKE) set-target-s3 SD_BOARD=waveshare_s3_touch
	$(MAKE) build SD_BOARD=waveshare_s3_touch

s3-zero:
	$(MAKE) set-target-s3 SD_BOARD=waveshare_s3_zero_sdspi
	$(MAKE) build SD_BOARD=waveshare_s3_zero_sdspi

p4:
	$(MAKE) set-target-p4 SD_BOARD=waveshare_p4
	$(MAKE) build SD_BOARD=waveshare_p4

clean:
	idf.py fullclean
