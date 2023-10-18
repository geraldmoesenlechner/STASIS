#!/bin/bash

sudo make uninstall

make all
echo "+----------------------------------------+"
echo "| building datasim_utility python module |"
echo "+----------------------------------------+"
cd ./Utilities/
python3 setup.py build_ext --inplace
echo "+-------------------------------------+"
echo "| built datasim_utility python module |"
echo "+-------------------------------------+"
echo "+------------------------------------------+"
echo "| building detector_features python module |"
echo "+------------------------------------------+"
cd ../Detector_features/
python3 setup.py build_ext --inplace
echo "+---------------------------------------+"
echo "| built detector_features python module |"
echo "+---------------------------------------+"

