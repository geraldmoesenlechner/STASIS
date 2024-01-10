"""
file    _STASIS_utility_py.pyx
author  Gerald Mösenlechner (gerald.moesenlechner@univie.ac.at)
date    November, 2021

Copyright
---------
This program is free software; you can redistribute it and/or modify it
under the terms and conditions of the GNU General Public License,
version 2, as published by the Free Software Foundation.

This program is distributed in the hope it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

Brief
-----
Library containing utility functions and the wrappers for the datasim_utilities C-library

Overview
--------
Python library containing the  utility functions and the wrappers for the datasim_utilities C-library
"""

import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append('../')
from lxml import etree as xmlTree
cimport numpy as np
cimport _STASIS_utility as utils
from cpython.pycapsule cimport *
from libc.stdlib cimport malloc, free
from libc.stdlib cimport bool as bool_c
import astropy.io.fits as fits
np.import_array()

cdef class ArrayWrapper:
    """ Internal class used for proper memory handling of arrays allocated in C
    and passed to Python. Mallocs a memory buffer of size (n*sizeof(int)) and
    sets up the numpy array.

    Attributes
    ----------
    size_x, size_y: int, Length of the array.
    data_ptr: void*, Pointer of the C-array
    """
    cdef void* data_ptr
    cdef int size_x
    cdef int size_y

    cdef set_data(self, int size_x, int size_y, void* data_ptr):
        self.data_ptr = data_ptr
        self.size_x = size_x
        self.size_y = size_y

    def __array__(self):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.size_x * self.size_y
        ndarray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, self.data_ptr)
        arr = np.asarray(ndarray)
        arr = np.reshape(ndarray, (-1, self.size_y))

        return arr

    def __dealloc__(self):
        """ Frees the array. """
        free(self.data_ptr)

def rand_poisson_array(double mean, unsigned int dim_x, unsigned int dim_y):
  """
  Function that returns a poisson distributed array of a defined dimension

  Parameters
  ----------
  mean: double, mean value of the distribution
  dim_x, dim_y:int, x-dim and y-dim in pixels

  Returns
  -------
  array, float array containing random values of the distribution
  """

  cdef double *rand_array

  res = ArrayWrapper()
  rand_array = utils.random_poisson_trm(mean, dim_x*dim_y)
  res.set_data(dim_x, dim_y, <void*>rand_array)

  return np.array(res)

def rand_normal_array(double mean, double sigma, unsigned int dim_x, unsigned int dim_y):
  """
  Function that returns a normal distributed array of a defined dimension

  Parameters
  ----------
  mean: double, mean value of the distribution
  sigma: double, standard deviation of the distribution
  dim_x, dim_y:int, x-dim and y-dim in pixels

  Returns
  -------
  array, float array containing random values of the distribution
  """

  cdef double *rand_array

  res = ArrayWrapper()
  rand_array = utils.random_normal_trm(mean, sigma, dim_x*dim_y)
  res.set_data(dim_x, dim_y, <void*>rand_array)

  return np.array(res)

def upsampling(image, unsigned int factor, bint copy):
  """
  Function that upsamples a given image by the defined factor

  Parameters
  ----------
  image: array, input array for upsampling
  factor: int, upsampling factor
  copy: bint, denotes if the intensity of original pixel shall be copied to each resulting upsampled pixels

  Returns
  -------
  array, upsampled image
  """

  cdef unsigned int dim_x = image.shape[0]
  cdef unsigned int dim_y = image.shape[1]
  cdef np.ndarray[double , ndim=1, mode="c"] image_cython = np.asarray(image.ravel(), dtype = np.double, order="C")
  cdef double *img_us

  res = ArrayWrapper()
  img_us = utils.upsample_image(&image_cython[0], dim_x, dim_y, factor, copy)
  res.set_data(dim_x*factor, dim_y*factor, <void*>img_us)

  return np.array(res)

def downsampling(image , unsigned int factor):
  """
  Function that downsamples a given image by the defined factor

  Parameters
  ----------
  image: array, input array for downsampling
  factor: int, downsampling factor

  Returns
  -------
  array, downsampled image
  """
  cdef unsigned int dim_x = image.shape[0]
  cdef unsigned int dim_y = image.shape[1]
  cdef np.ndarray[double , ndim=1, mode="c"] image_cython = np.asarray(image.ravel(), dtype = np.double, order="C")
  cdef double *img_ds

  res = ArrayWrapper()
  img_ds = utils.downsample_image(&image_cython[0], dim_x, dim_y, factor)
  res.set_data(np.floor(dim_x/factor), np.floor(dim_y/factor), <void*>img_ds)

  return np.array(res)


def convolve2D(data, kernel):
  """
  Function that convolves two 2-dimensional arrays using FFTW

  Parameters
  ----------
  data: array, array containing the original data (float)
  kernel: array, array containing the convolution kernel (float)

  Returns
  -------
  array, convolved image
  """
  cdef unsigned int dim_x = data.shape[0]
  cdef unsigned int dim_y = data.shape[1]
  cdef unsigned int dim_kernel_x = kernel.shape[0]
  cdef unsigned int dim_kernel_y = kernel.shape[1]
  cdef np.ndarray[double , ndim=1, mode="c"] data_cython = np.asarray(data.ravel(), dtype = np.double, order="C")
  cdef np.ndarray[double , ndim=1, mode="c"] kernel_cython = np.asarray(kernel.ravel(), dtype = np.double, order="C")
  cdef double *conv_image

  res = ArrayWrapper()
  conv_image = utils.convolve(&data_cython[0], &kernel_cython[0], dim_x, dim_y, dim_kernel_x, dim_kernel_y)
  res.set_data(dim_x, dim_y, <void*>conv_image)

  return np.array(res)


def create_directory_ifnotexists(path):
  """
  creates a directory if it does not exists already

  Parameters
  ----------
  path: string, name of the path to be created
  """
  if not os.path.exists(os.path.dirname(path)):
    os.makedirs(os.path.dirname(path))

def write_fits_file(filepath, images):
  """
  Writes a fits containing numpy arrays. If file exists the existing file is extended otherwise a new file is created.

  Parameters
  ----------
  path: string, path + filename
  images: list, image-data in form of a list of numpy array
  """
  create_directory_ifnotexists(filepath)
  if not os.path.exists(filepath):
    fits.writeto(filepath, np.array(images), overwrite=True)
    #print('Created file: ' + filepath)
  else:
    #print("Updating file with new results...")
    f = fits.open(filepath, mode='update')
    f[0].data = np.concatenate((f[0].data, images), axis=0)
    f.flush()
    #f.info()
    f.close()

def get_xml_entry(tree, key, valuetype):
  """
  Gets the value for an xml entry as string.
  """
  if tree is None or tree.find(key) is None:
    if valuetype == "noneifempty":
      return None
      print("Key: " + key + " not found in xml. Setting key to None.")
    elif valuetype == "string":
      value = "Not defined"
    elif valuetype == "bool":
      value = "False"
    elif valuetype == "int":
      value = "-1"
    elif valuetype == "float":
      value = "-1.0"
    print("Key: " + key + " not found in xml. Setting key to default value: " + value)
  else:
    value = tree.find(key).text
  return value;

def str2bool(sourcestr):
  """
  Converts a string to boolean.
  Parameters
  ----------
  sourcestr: string, source string that will be converted.
  """
  result = False
  if sourcestr.lower() in ("true", "1"):
    result = True
  elif sourcestr.lower() in ("false", "0"):
    result = False
  else:
    raise Exception("Cannot convert given value to boolean: ", sourcestr)
  return result

def string_param_to_array(source, newCol, newRow):
  """
  Converts a string parameter to an 2 dimensional float array. Mainly used to input an array as commandline param
  eg. source is 1/2//3/4 and newCol is '/' and newRow is '//' --> output 2x2 array
  Parameters
  ----------
  source: string, contains the array data
  newCol: char, column separator
  newRow: char, row separator
  """
  indexTillNextRow = source.find(newRow)
  if indexTillNextRow == -1: # only one row exists
    firstRow = source
  else:
    firstRow = source[:indexTillNextRow]
  countCols = 1 + firstRow.count(newCol)
  countRows = 1 + source.count(newRow)

  array = np.zeros((countRows, countCols))

  startindex = 0

  for rowIndex in range(countRows):

    #retrieve row
    indexOfNextRow = source.find(newRow, startindex) # first index of newRow -1 if not found

    if indexOfNextRow == -1:
      if source.rfind(newRow) != -1:
        row = source[source.rfind(newRow)+len(newRow):] # gets the last row, rfind = last index of newCol
      else:
        row = source
    else:
        row = source[startindex:indexOfNextRow]

    #parse the row and write each col to array
    collist = row.split(newCol)

    j = 0
    #print collist
    for col in collist:
      array[rowIndex,j] = float(col)
      j += 1

    startindex = indexOfNextRow + len(newRow)

  return array

def floatOrEmpty(sourcestr):
  """
  Converts a given string to float and returns None if empty.
  Parameters
  ----------
  sourcestr: string, string that will be converted.
  """
  if sourcestr is None or sourcestr == '' or sourcestr == 'None':
    return None
  else:
    return float(sourcestr)


