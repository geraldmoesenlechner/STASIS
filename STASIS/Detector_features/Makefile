CC = gcc
CFLAGS += -fPIC -shared -lm -lfftw3 -lxml2 -fopenmp -O3 -lfftw3_omp -lSTASIS_utility -Wall
SRC_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
DEST_DIR = /usr/lib/
INCLUDE_DIR = /usr/include/


SOURCES = $(SRC_DIR)/STASIS_detector.c
TARGET = libSTASIS_detector.so


$(TARGET): $(SOURCES)
	$(CC) $(SOURCES) -o $(TARGET) $(CFLAGS)

.PHONY: clean
clean:
	-${RM} $(SOURCES:.c=.o)

.PHONY: install
install:
	cp -n $(TARGET) $(DEST_DIR)
	cp -n $(SOURCES:.c=.h) $(INCLUDE_DIR)

.PHONY: uninstall
uninstall:
	sudo rm $(DEST_DIR)$(TARGET)
	sudo rm $(INCLUDE_DIR)/STASIS_detector.h