#!/bin/bash

sudo make -f ./src/Makefile uninstall
sudo make -f ./src/Makefile all
python3 setup.py build_ext --inplace
