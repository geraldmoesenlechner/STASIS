#!/bin/bash

sudo make -f ./src/Makefile uninstall
sudo make -f ./src/Makefile all
sudo make -f ./src/Makefile install
python3 setup.py build_ext --inplace
