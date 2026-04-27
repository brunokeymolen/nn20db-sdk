PORT ?= /dev/ttyACM0

.PHONY: all build flash monitor flash-monitor clean

all: build

build:
	idf.py build

flash:
	idf.py -p $(PORT) flash

monitor:
	idf.py -p $(PORT) monitor

flash-monitor:
	idf.py -p $(PORT) flash monitor

clean:
	idf.py fullclean
