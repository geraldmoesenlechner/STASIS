import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append('../')
cimport _sc_kinematics_py as sc

def test():
    return sc.test()