# SD card board driver: one sd_storage_<board>.c per supported board.
# "none" compiles no driver and nn20db falls back to its built-in target
# default (20 MHz on both the ESP32-P4 devkit's native slot and, on
# ESP32-S3, the IDF default 1-bit pins).
# Override: idf.py -DSD_BOARD=<board> build
idf_build_get_property(target IDF_TARGET)
if(target STREQUAL "esp32s3")
    set(SD_BOARD_DEFAULT "waveshare_s3_touch")
elseif(target STREQUAL "esp32p4")
    set(SD_BOARD_DEFAULT "waveshare_p4")
else()
    set(SD_BOARD_DEFAULT "none")
endif()
set(SD_BOARD "${SD_BOARD_DEFAULT}" CACHE STRING "SD card board driver (none = nn20db built-in default)")

set(srcs "main.c")
if(NOT SD_BOARD STREQUAL "none")
    list(APPEND srcs "sd_storage_${SD_BOARD}.c")
endif()

idf_component_register(
    SRCS ${srcs}
    INCLUDE_DIRS "."
    REQUIRES nn20db fatfs sdmmc esp_driver_sdmmc esp_driver_sdspi esp_driver_spi esp_psram
)

if(SD_BOARD STREQUAL "none")
    target_compile_definitions(${COMPONENT_LIB} PRIVATE SD_BOARD_NONE=1)
endif()

# Number of SDMMC data lines used by the board driver (4 or 1)
# Override: make build SD_BUS_WIDTH=1  (or: idf.py -DSD_BUS_WIDTH=1 build)
# (Ignored by SPI-mode drivers, e.g. waveshare_s3_zero_sdspi, which are 1-bit.)
set(SD_BUS_WIDTH "4" CACHE STRING "SDMMC data line count (4 or 1)")
target_compile_definitions(${COMPONENT_LIB} PRIVATE SD_BUS_WIDTH=${SD_BUS_WIDTH})

# SPI bus clock (kHz) for SD-over-SPI board drivers (e.g. waveshare_s3_zero_sdspi).
# Ignored by the parallel SDMMC drivers. Override: make build SD_SPI_FREQ_KHZ=40000
set(SD_SPI_FREQ_KHZ "20000" CACHE STRING "SDSPI bus clock in kHz")
target_compile_definitions(${COMPONENT_LIB} PRIVATE SD_SPI_FREQ_KHZ=${SD_SPI_FREQ_KHZ})

# Product quantization config toggle
# Override: idf.py -DUSE_PQ=0 build  (or: make build USE_PQ=0)
option(USE_PQ "Use product quantization config" ON)
target_compile_definitions(${COMPONENT_LIB} PRIVATE USE_PQ=$<BOOL:${USE_PQ}>)
