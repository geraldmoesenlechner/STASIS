CC = gcc
CFLAGS += -fPIC -shared -lm -lfftw3 -lxml2 -fopenmp -O3 -lfftw3_omp -Wall
SRC_DIR = $(shell realpath Utilities)
DEST_DIR = /usr/lib/
INCLUDE_DIR = /usr/include/


SOURCES = $(SRC_DIR)/datasim_utility.c
TARGET = libdatasim_utility.so


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
	sudo rm $(INCLUDE_DIR)datasim_utility.h
