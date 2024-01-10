#!/bin/bash

sudo make -f ./STASIS/Makefile uninstall
sudo make -f ./STASIS/Makefile all
python3 setup.py build_ext
