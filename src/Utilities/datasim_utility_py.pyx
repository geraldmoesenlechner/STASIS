"""
file    datasim_utility_py.pyx
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
cimport datasim_utility_py as utils
from cpython.pycapsule cimport *
from libc.stdlib cimport malloc, free
from libc.stdlib cimport bool as bool_c
import Constants as const
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

def get_datasim_config_from_xml(filename, folder = None):
  """
  Reads information from the datasim_config.xml file that contains information about the data simulation such as star signal, psf position and so on.
  starsim_config.xml is defined in const.data_sim_config_filename. Location can be speficied by 'folder' otherwise current folder (of running script) will be used.

  Parameters
  ----------
  filename: string, the filename of the xml config
  folder: (optional, string) Specifies the folder that contains the config file.

  Returns
  -------
  The parameters used to run the simulation as list.
  """
  if folder is None:
    xmlConfigPath = filename
  else:
    if (os.path.isfile(folder)):
      folder = os.path.dirname(folder) + '/' # directory of file
    xmlConfigPath = folder + filename

  if os.path.isfile(xmlConfigPath):
    try:
      print("Reading configuration from " + xmlConfigPath)
      tree = xmlTree.parse(xmlConfigPath)
      root = tree.getroot()

      config = const.datasim_config

      config[const.detector_x]         = int(get_xml_entry(tree, const.detector_x, "int"))  # directly access the element star_pos_x0 (list is returned cause there could be more than one element with this name) and retrieve value by .text
      config[const.detector_y]         = int(get_xml_entry(tree, const.detector_y, "int"))
      config[const.subpixelflat]       = get_xml_entry(tree, const.subpixelflat, "string")
      config[const.flat_mode]           = get_xml_entry(tree, const.flat_mode, "string")
      config[const.flat_name]          = get_xml_entry(tree, const.flat_name, "string")
      config[const.flat_mean]          = float(get_xml_entry(tree, const.flat_mean, "float"))
      config[const.flat_sigma]         = float(get_xml_entry(tree, const.flat_sigma, "float"))
      config[const.flat_grad_lower_limit] = float(get_xml_entry(tree, const.flat_grad_lower_limit, "float"))
      config[const.flat_grad_upper_limit] = float(get_xml_entry(tree, const.flat_grad_upper_limit, "float"))
      config[const.flat_grad_changes]     = float(get_xml_entry(tree, const.flat_grad_changes, "float"))
      config[const.flat_grad_angle]       = float(get_xml_entry(tree, const.flat_grad_angle, "float"))

      psfTree = root.find(const.xml_psf)
      config[const.psf_mode]           = get_xml_entry(psfTree, const.psf_mode, "string")
      config[const.plot_psf]           = str2bool(get_xml_entry(psfTree, const.plot_psf, "bool"))
      psfReadTree = psfTree.find(const.xml_psf_mode_read)
      config[const.psf_filename]       = get_xml_entry(psfReadTree, const.psf_filename, "string")
      config[const.psf_is_oversampled] = str2bool(get_xml_entry(psfReadTree, const.psf_is_oversampled, "bool"))
      psfGenerateTree = psfTree.find(const.xml_psf_mode_generate)
      config[const.psf_fwhm_x]         = float(get_xml_entry(psfGenerateTree, const.psf_fwhm_x, "float"))
      config[const.psf_fwhm_y]         = float(get_xml_entry(psfGenerateTree, const.psf_fwhm_y, "float"))
      config[const.psf_angle]          = float(get_xml_entry(psfGenerateTree, const.psf_angle, "float"))

      config[const.place_target_on_random_position]          = str2bool(get_xml_entry(tree, const.place_target_on_random_position, "bool"))
      config[const.bias]               = float(get_xml_entry(tree, const.bias, "float"))
      config[const.dark_average]       = float(get_xml_entry(tree, const.dark_average, "float"))
      config[const.dark_count]         = int(get_xml_entry(tree, const.dark_count, "int"))
      config[const.readout_noise]      = float(get_xml_entry(tree, const.readout_noise, "float"))
      config[const.exposure_time]      = float(get_xml_entry(tree, const.exposure_time, "float"))
      config[const.background_signal]  = float(get_xml_entry(tree, const.background_signal, "float"))
      config[const.QE]                 = float(get_xml_entry(tree, const.QE, "float"))
      config[const.full_well_capacity] = int(get_xml_entry(tree, const.full_well_capacity, "int"))
      config[const.MCT_mode]           = str2bool(get_xml_entry(tree, const.MCT_mode, "bool"))
      config[const.reset_percentage]   = float(get_xml_entry(tree, const.reset_percentage, "float"))
      config[const.reset_noise]   = float(get_xml_entry(tree, const.reset_noise, "float"))
      config[const.hotpixel_from_map]  = str2bool(get_xml_entry(tree, const.hotpixel_from_map, "bool"))
      config[const.hp_map]       = get_xml_entry(tree, const.hp_map, "string")
      config[const.hotpixel_amount]    = float(get_xml_entry(tree, const.hotpixel_amount, "float"))
      config[const.hotpixel_lower_limit] = float(get_xml_entry(tree, const.hotpixel_lower_limit, "float"))
      config[const.hotpixel_upper_limit] = float(get_xml_entry(tree, const.hotpixel_upper_limit, "float"))
      config[const.calibrate_image]    = str2bool(get_xml_entry(tree, const.calibrate_image, "bool"))
      config[const.save_image_as_fits] = str2bool(get_xml_entry(tree, const.save_image_as_fits, "bool"))
      config[const.fits_file_name]     = get_xml_entry(tree, const.fits_file_name, "string")
      config[const.data_input_directory]  = get_xml_entry(tree, const.data_input_directory, "string")
      config[const.data_output_directory] = get_xml_entry(tree, const.data_output_directory, "string")
      config[const.flatfield_count]    = int(get_xml_entry(tree, const.flatfield_count, "int"))
      config[const.bias_count]         = int(get_xml_entry(tree, const.bias_count, "int"))
      # get closed loop simulation configuration
      #closedLoop = root.find(const.xml_closed_loop)
      #config[const.closed_loop_enabled] = str2bool(closedLoop.find(const.closed_loop_enabled).text)
      #gain tree
      gainTree = root.find(const.xml_split_gain)
      config[const.gain] = float(get_xml_entry(gainTree, const.gain, "float"))
      config[const.apply_split_gain] = str2bool(get_xml_entry(gainTree, const.apply_split_gain, "bool"))
      config[const.split_position] = int(get_xml_entry(gainTree, const.split_position, "float"))
      config[const.additional_gain] = float(get_xml_entry(gainTree, const.additional_gain, "float"))
      # jitter tree
      jitTree = root.find(const.xml_jitter)
      config[const.jit_apply_jitter]  = str2bool(get_xml_entry(jitTree, const.jit_apply_jitter, "bool"))
      config[const.jit_input_file]    = get_xml_entry(jitTree, const.jit_input_file, "string")
      config[const.jitter_mode]       = get_xml_entry(jitTree, const.jitter_mode, "string")
      config[const.jitter_starttime]  = float(get_xml_entry(jitTree, const.jitter_starttime, "float"))
      config[const.jitter_savePsf] = str2bool(get_xml_entry(jitTree, const.jitter_savePsf, "bool"))
      config[const.jitter_plotPsf] = str2bool(get_xml_entry(jitTree, const.jitter_plotPsf, "bool"))
      # rotation tree
      rotTree = root.find(const.xml_rotation)
      config[const.rot_angle_speed]  = float(get_xml_entry(rotTree, const.rot_angle_speed, "float"))
      config[const.rot_rotation_direction] = get_xml_entry(rotTree, const.rot_rotation_direction, "string")
      config[const.rot_axis_pos_x]  = float(get_xml_entry(rotTree, const.rot_axis_pos_x, "float"))
      config[const.rot_axis_pos_y]  = float(get_xml_entry(rotTree, const.rot_axis_pos_y, "float"))
      config[const.rot_ignore_target_star]  = str2bool(get_xml_entry(rotTree, const.rot_ignore_target_star, "bool"))

      # get multiple image config
      mulTree = root.find(const.xml_multiple_images)
      config[const.mul_image_count]  = int(get_xml_entry(mulTree, const.mul_image_count, "int"))
      config[const.mul_posx_step]    = float(get_xml_entry(mulTree, const.mul_posx_step, "float"))
      config[const.mul_posy_step]    = float(get_xml_entry(mulTree, const.mul_posy_step, "float"))
      config[const.static_target]    = str2bool(get_xml_entry(mulTree, const.static_target, "bool"))
      config[const.mul_signal_steps] = float(get_xml_entry(mulTree, const.mul_signal_steps, "float"))
      config[const.mul_imgs_per_iteration] = int(get_xml_entry(mulTree, const.mul_imgs_per_iteration, "int"))

      # smearing tree
      """
      smearingTree = root.find(const.xml_smearing)
      config[const.sm_apply_smearing] = str2bool(get_xml_entry(smearingTree, const.sm_apply_smearing, "bool"))
      config[const.smearing_mode] = get_xml_entry(smearingTree, const.smearing_mode, "string")
      config[const.smearing_position] = get_xml_entry(smearingTree, const.smearing_position, "string")
      config[const.max_smear_size]    = float(get_xml_entry(smearingTree, const.max_smear_size, "float"))
      config[const.smearing_decay]    = float(get_xml_entry(smearingTree, const.smearing_decay, "float"))
      #if config[const.xml_smearing_point] is not None:
      smearpointsTree = smearingTree.find(const.xml_smearing_points).findall(const.xml_smearing_point)
      point_count = len(smearpointsTree)
      pointArray = np.zeros((point_count, 2), dtype=np.int)
      pointnr = 0
      for xmlPoint in smearpointsTree:
          pointArray[pointnr] = [int(xmlPoint.get(g.xml_glitch_point_x)), int(xmlPoint.get(g.xml_glitch_point_y))]
          pointnr = pointnr + 1
      config[const.smearing_coordinates] = pointArray
      config[const.sm_apply_smearing_step] = str2bool(get_xml_entry(smearingTree, const.sm_apply_smearing_step, "bool"))
      config[const.smearing_step_x] = float(get_xml_entry(smearingTree, const.smearing_step_x, "float"))
      config[const.smearing_step_y] = float(get_xml_entry(smearingTree, const.smearing_step_y, "float"))
      config[const.plot_smearing_kernel] = str2bool(get_xml_entry(smearingTree, const.plot_smearing_kernel, "bool"))
      """
      """
      Glitches not yet implemented
      # get glitches
      glitchesRoot = root.find(const.xml_glitches_name)
      config[const.glitch_rate_average] = float(get_xml_entry(glitchesRoot, const.glitch_rate_average, "float"))
      config[const.glitch_rate_sigma]   = float(get_xml_entry(glitchesRoot, const.glitch_rate_sigma, "float"))
      config[const.plot_glitches]       = str2bool(get_xml_entry(glitchesRoot, const.plot_glitches, "bool"))
      glitchTree = glitchesRoot.findall(g.xml_glitch_name)

      glitches = []
      for xmlGlitch in glitchTree:
          gtype       = get_xml_entry(xmlGlitch, g.xml_glitch_type, "string")
          spread_type = get_xml_entry(xmlGlitch, g.xml_glitch_spread_type, "string")
          pointsTree  = xmlGlitch.find(g.xml_glitch_points).findall(g.xml_glitch_point)
          point_count = len(pointsTree)
          pointArray  = np.zeros((point_count, 2), dtype=np.int)
          pointnr = 0
          for xmlPoint in pointsTree:
              pointArray[pointnr] = [int(xmlPoint.get(g.xml_glitch_point_x)), int(xmlPoint.get(g.xml_glitch_point_y))]
              pointnr = pointnr + 1
          width  = int(get_xml_entry(xmlGlitch, g.xml_glitch_width, "int"))
          signal = float(get_xml_entry(xmlGlitch, g.xml_glitch_signal, "float"))
          decay  = float(get_xml_entry(xmlGlitch, g.xml_glitch_decay, "float"))

          glitch = g.Glitch(gtype, spread_type, pointArray, width, signal, decay)
          glitches.append(glitch)

      config[const.glitches] = glitches
      """
      #get simulation infos
      simulationinfos = []
      simulationsTree = root.find(const.xml_simulations)
      if (simulationsTree is not None):
          simulationsTree = simulationsTree.findall(const.xml_simulation)
          for xmlSimulation in simulationsTree:
              nr    = int(xmlSimulation.get(const.xml_simulation_number))
              pos_x = float(xmlSimulation.get(const.pos_x))
              pos_y = float(xmlSimulation.get(const.pos_y))
              time  = floatOrEmpty(xmlSimulation.get(const.global_time))
              star_signal  = floatOrEmpty(xmlSimulation.get(const.star_signal))
              simulationinfos.append(const.SimulationInfo(nr, pos_x, pos_y, time, star_signal))

      afterSimParamsTree = root.find(const.xml_after_simulation_parameters)
      if (afterSimParamsTree is not None):
          config[const.psf_brightest_pixel_fraction] = float(afterSimParamsTree.find(const.psf_brightest_pixel_fraction).text)

    except:
      print("Invalid " + const.data_sim_config_filename + " file! Error: ", sys.exc_info()[0])
      raise
  else:
      raise Exception("ERROR: No " + const.data_sim_config_filename + " could be found. Path: " + xmlConfigPath)

  return config, simulationinfos
