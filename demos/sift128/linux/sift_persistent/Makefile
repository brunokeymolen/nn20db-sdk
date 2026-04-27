CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra
TARGET = sift_persistent_demo
SRCS = main.c

REPO_ROOT := $(abspath ../../../..)
SDK_ROOT ?= $(REPO_ROOT)/sdk/linux/current
PKG_CONFIG_PATH := $(SDK_ROOT)/linux/x86_64-linux-gnu/pkgconfig:$(PKG_CONFIG_PATH)

NN20DB_CFLAGS := $(shell PKG_CONFIG_PATH="$(PKG_CONFIG_PATH)" pkg-config --cflags nn20db)
NN20DB_LIBS := $(shell PKG_CONFIG_PATH="$(PKG_CONFIG_PATH)" pkg-config --libs --static nn20db)
HDF5_CFLAGS := $(shell pkg-config --cflags hdf5)
HDF5_LIBS := $(shell pkg-config --libs hdf5)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(NN20DB_CFLAGS) $(HDF5_CFLAGS) -o $@ $(SRCS) $(HDF5_LIBS) $(NN20DB_LIBS)

clean:
	rm -f $(TARGET)


build-cmake:
	cmake -B build && cmake --build build
