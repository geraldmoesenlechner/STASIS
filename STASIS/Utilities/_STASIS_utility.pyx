"""
file    _STASIS_utility_py.pyx
author  Gerald Mösenlechner (gerald.moesenlechner@univie.ac.at)
date    November, 2021

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

    :param size_x: Length of the array in x.
    :type size_x: int
    :param size_y: Length of the array in y.
    :type size_y: int
    :param data_ptr: Pointer of the C-array.
    :type data_ptr: void*
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
  """Function that returns a poisson distributed array of a defined dimension

  :param mean: mean value of the distribution
  :type mean: double
  :param dim_x: x dimension of the array
  :type dim_x: unsigned int
  :param dim_y: y dimension of the array
  :type dim_y: unsigned int
  :return: array containing the random numbers
  :rtype: array
  """

  cdef double *rand_array

  res = ArrayWrapper()
  rand_array = utils.random_poisson_trm(mean, dim_x*dim_y)
  res.set_data(dim_x, dim_y, <void*>rand_array)

  return np.array(res)

def rand_normal_array(double mean, double sigma, unsigned int dim_x, unsigned int dim_y):
  """Function that returns a normal distributed array of a defined dimension
  
  :param mean: mean value of the distribution
  :type mean: double
  :param sigma: standard deviation of the distribution
  :type sigma: double
  :param dim_x: x dimension of the array
  :type dim_x: unsigned int
  :param dim_y: y dimension of the array
  :type dim_y: unsigned int
  :return: array containing the random numbers
  :rtype: array
  """

  cdef double *rand_array

  res = ArrayWrapper()
  rand_array = utils.random_normal_trm(mean, sigma, dim_x*dim_y)
  res.set_data(dim_x, dim_y, <void*>rand_array)

  return np.array(res)

def upsampling(image, unsigned int factor, bint copy):
  """Function that upsamples a given image by the defined factor

  :param image: input array for upsampling
  :type image: array
  :param factor: upsampling factor
  :type factor: unsigned int
  :param copy: denotes if the intensity of original pixel shall be copied to each resulting upsampled pixels
  :type copy: bint
  :return: array containing the upsampled image
  :rtype: array
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
  """Function that downsamples a given image by the defined factor

  :param image: input array for upsampling
  :type image: array
  :param factor: downsampling factor
  :type factor: unsigned int
  :return: array containing the downsampled image
  :rtype: array
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
  """Function that convolves two 2-dimensional arrays using FFTW

  :param data: array containing the original data (float)
  :type data: array
  :param kernel: array containing the convolution kernel (float)
  :type kernel: array
  :return: array containing the convolved image
  :rtype: array
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
