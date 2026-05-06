PORT ?= /dev/ttyACM0

.PHONY: all build flash monitor flash-monitor clean

all: build

build:
	idf.py build

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
