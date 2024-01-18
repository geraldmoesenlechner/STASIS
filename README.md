[![codecov](https://codecov.io/gh/geraldmoesenlechner/STASIS/branch/main/graph/badge.svg?token=9LF4AT5F8S)](https://codecov.io/gh/geraldmoesenlechner/STASIS)

# STASIS

STASIS is a lightweight, cython based kinematic sky simulator for astronomical detectors. 
The documentaion can be found at [STASIS.readthedocs.io]()

## Installation

Install from source by

```
git clone https://github.com/geraldmoesenlechner/STASIS.git
cd STASIS/
sudo chmod u+x build.sh
./build.sh
python -m pip install -e .
```

## Usage

In order to use the simulator, you will need a star catalouge .xml following this scheme:

```
<star_field>
	<star pos_x="" pos_y="" ra="" dec="" signal="" is_target=""/>
    ...
</star_field>
```

