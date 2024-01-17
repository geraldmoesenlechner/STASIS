"""
file    _detector_features.pyx
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
Library containing functions for simulating scientific CCD detectors

Overview
--------
Python library containing the functions used for the simulation of CDD images
Wrapper for the detector_features C-library
"""

import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append('../')
cimport numpy as np
cimport _STASIS_detector as df
from cpython.pycapsule cimport *
from libc.stdlib cimport malloc, free
from libc.stdlib cimport bool as bool_c
np.import_array()
from lxml import etree as xmlTree



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

def gen_bias(double bias_value, double readout_noise, unsigned int dim_x, unsigned int dim_y, oversampling = 1):
    """
    Function that creates a simulated bias frame.

    :param bias_value: Constant offset of the image
    :type bias_value: double
    :param readout_noise: mean readout noise of the detector [ADU/s]
    :type readout_noise: double
    :param dim_x: x dimension of the array
    :type dim_x: unsigned int
    :param dim_y: y dimension of the array
    :type dim_y: unsigned int
    :param oversampling: oversampling of the image based on the overall sub-px flat, defaults to 1
    :type oversampling: unsigned int
    :return: array containing the simulated bias
    :rtype: array
    """

    cdef double *bias

    res = ArrayWrapper()
    bias = df.generate_bias(bias_value, readout_noise, dim_x, dim_y, oversampling)
    res.set_data(dim_x*oversampling, dim_y*oversampling, <void*>bias)
    return np.array(res)

def gen_dark(double dark_mean, double exp_time, hot_pixels, unsigned int dim_x, unsigned int dim_y, unsigned int oversampling = 1):
    """Function that creates a simulated dark frame.

    :param dark_mean: mean value of the dark in [ADU/s]
    :type dark_mean: double
    :param exp_time: exposure time in [s]
    :type exp_time: double
    :param hot_pixels: array containing a map of the hot pixels
    :type hot_pixels: array
    :param dim_x: x-dim of the array
    :type dim_x: unsigned int
    :param dim_y: y-dim of the array
    :type dim_y: unsigned int
    :param oversampling: oversampling of the image based on the overall sub-px flat, defaults to 1
    :type oversampling: unsigned int
    :return: array containing the dark frame
    :rtype: array
    """

    cdef double *dark
    cdef np.ndarray[double , ndim=1, mode="c"] hp_cython = np.asarray(hot_pixels.ravel(), dtype = np.double, order="C")
    cdef unsigned int hp_dim_x = hot_pixels.shape[0]
    cdef unsigned int hp_dim_y = hot_pixels.shape[1]

    if dim_x*oversampling != hp_dim_x or dim_y*oversampling != hp_dim_y:
        raise Exception("Hp map dimension doesn't match dark frame dimensions!")

    res = ArrayWrapper()
    dark = df.generate_dark(dark_mean, exp_time, &hp_cython[0], dim_x, dim_y, oversampling)
    res.set_data(dim_x*oversampling, dim_y*oversampling, <void*>dark)

    return np.array(res)


def gen_hotpixels(double percentage, double lower_limit, double upper_limit, unsigned int dim_x, unsigned int dim_y, oversampling = 1):
    """Function that creates a map containing a given number of randomly distributed hot pixels inside of the image.

    :param percentage: max percentage of pixels in the image that should be hp [0-1]
    :type percentage: double
    :param lower_limit: lowest possible signal value for hp
    :type lower_limit: double
    :param upper_limit: highest possible signal value for hp
    :type upper_limit: double
    :param dim_x: x-dim of the array
    :type dim_x: unsigned int
    :param dim_y: y-dim of the array
    :type dim_y: unsigned int
    :param oversampling: oversampling of the image based on the overall sub-px flat, defaults to 1
    :type oversampling: unsigned int
    :return: array containing the hot pixel image
    :rtype: array
    """

    cdef double *hp

    res = ArrayWrapper()
    hp = df.generate_hotpixels(percentage, lower_limit, upper_limit, dim_x, dim_y, oversampling)
    res.set_data(dim_x*oversampling, dim_y*oversampling, <void*>hp)

    return np.array(res)

def gen_flat(double flat_mean, double flat_sigma, subpxflat, double grad_lower, double grad_upper, double grad_var, double angle, double px_to_px_response, unsigned int dim_x, unsigned int dim_y):
    """Function that creates a simulated flatfield.

    :param flat_mean: mean values of the normally distributed pixel sensitivity [0-1]
    :type flat_mean: double
    :param flat_sigma: variation of the pixel sensitivity
    :type flat_sigma: double
    :param subpxflat: array containing the inter-pixel sensitivity
    :type subpxflat: array
    :param grad_lower: lower limit for a flat gradient [TBD]
    :type grad_lower: double
    :param grad_upper: upper limit for a flat gradient [TBD]
    :type grad_upper: double
    :param grad_var: variation of the flat gradient [TBD]
    :type grad_var: double
    :param angle: direction of the gradient [TBD]
    :type angle: double
    :param px_to_px_response: [TBD]
    :type px_to_px_response: double
    :param dim_x: x-dim of the array
    :type dim_x: unsigned int
    :param dim_y: y-dim of the array
    :type dim_y: unsigned int
    :return: array containing the flat field
    :rtype: array
    """

    cdef double *flat
    cdef unsigned int subpx_x
    cdef unsigned int subpx_y

    if subpxflat.all() != None:
        subpx_x = subpxflat.shape[0]
        subpx_y = subpxflat.shape[1]
        os = subpx_x
    else:
        subpxflat = np.zeros((2,2))
        subpx_x = 1
        subpx_y = 1
        os = 1

    if subpx_x != subpx_y:
        raise Exception("Sub-pixel flat must be square!")

    cdef np.ndarray[double , ndim=1, mode="c"] subpx_flat_cython = np.asarray(subpxflat.ravel(), dtype = np.double, order="C")

    res = ArrayWrapper()
    flat = df.generate_flat(flat_mean, flat_sigma, &subpx_flat_cython[0], grad_lower, grad_upper, grad_var, angle, px_to_px_response, os, dim_x, dim_y)
    res.set_data(dim_x*os, dim_y*os, <void*>flat)

    return res

def smear_stars(starmask, double x, double y, unsigned int oversampling = 1):
    """Function that creates a linear smearing kernel that is convolved with the given starmask. Used to emulate jitter during the image inegration.

    :param starmask: python array containing the original starmask
    :type starmask: array
    :param x: the x-lenght of the smear
    :type x: double
    :param y: the y-lenght of the smear
    :type y: double
    :param oversampling: oversampling of the image based on the overall sub-px flat, defaults to 1
    :type oversampling: unsigned int
    :return: array containing the smeared starmask
    :rtype: array
    """

    cdef unsigned int width = starmask.shape[0]
    cdef unsigned int height = starmask.shape[1]
    cdef np.ndarray[double, ndim = 1, mode = "c"] starmask_cython = np.asarray(starmask.ravel(), dtype = np.double, order="C")
    cdef double *starmask_smear

    res = ArrayWrapper()

    starmask_smear = df.smear_star_image(&starmask_cython[0], width, height, x, y, oversampling)
    res.set_data(width, height, <void*>starmask_smear)

    return res

cdef class Stars:

    """ Class that handles the data of the stars used for the simulation and
    provides star-image generation and updating of the star-data

    :param stars_config: xml file containing the star catalouge
    :type stars_config: string
    """

    cdef df.stars cstars
    _x = []
    _y = []
    _ra = []
    _dec = []
    _signal = []
    _is_target = []
    _number = 0

    def __init__(self, stars_config):
        """Constructor method
        """

        if os.path.isfile(stars_config):
            try:
                
                tree = xmlTree.parse(stars_config)
                root = tree.getroot()

                #get stars on detector
                starsTree = root.findall("star")
                stars_number = len(starsTree)
            
                i = 0
                for xmlStar in starsTree:

                    self._x.append(float(xmlStar.get("pos_x")))
                    self._y.append(float(xmlStar.get("pos_y")))
                    self._ra.append(float(xmlStar.get("ra")))
                    self._dec.append(float(xmlStar.get("dec")))
                    self._signal.append(float(xmlStar.get("signal")))
                    self._is_target.append(int(xmlStar.get("is_target")))


                self.cstars.number = np.uintc(len(self._x))

                self.cstars.x = <double *> malloc(self.cstars.number * sizeof(double))
                self.cstars.y = <double *> malloc(self.cstars.number * sizeof(double))
                self.cstars.ra = <double *> malloc(self.cstars.number * sizeof(double))
                self.cstars.dec = <double *> malloc(self.cstars.number * sizeof(double))
                self.cstars.signal = <double *> malloc(self.cstars.number * sizeof(double))
                self.cstars.is_target = <short *> malloc(self.cstars.number * sizeof(short))

                for i in range(self.cstars.number):
                    self.cstars.x[i] = np.double(self._x[i])
                    self.cstars.y[i] = np.double(self._y[i])
                    self.cstars.ra[i] = np.double(self._ra[i])
                    self.cstars.dec[i] = np.double(self._dec[i])
                    self.cstars.signal[i] = np.double(self._signal[i])
                    self.cstars.is_target[i] = np.short(self._is_target[i])

            except:
                print("Invalid " + stars_config + " file! Error: ", sys.exc_info()[0])
                raise
        else:
            raise Exception("ERROR: No " + stars_config + " could be found.")

    def __dealloc__(self):
        """Deallocator
        """
        free(self.cstars.x)
        free(self.cstars.y)
        free(self.cstars.ra)
        free(self.cstars.dec)
        free(self.cstars.signal)
        free(self.cstars.is_target)


    def gen_star_image(self, psf, double qe, double exposure_time, unsigned int dim_x, unsigned int dim_y, unsigned int oversampling, double smear_x = 0, double smear_y = 0):
        """Method for generating a star image based on the configured stars

        :param psf: point spread funciton to be used in the image generation
        :type psf: array
        :param qe:  quantum efficiency of the simulated detector
        :type qe: double
        :param exposure_time: exposure time of the simulated image
        :type exposure_time: double
        :param dim_x: x size of the image
        :type  dim_x: unsigned int
        :param dim_y: y size of the image
        :type  dim_y: unsigned int
        :param oversampling: the oversampling factor of the provided psf
        :type oversampling: unisgned int
        :param smear_x: size of the smearing in x direction [px], defaults to 0
        :type smear_x: double
        :param smear_y: size of the smearing in y direction [px], defaults to 0
        :type smear_y: double
        :return: image depicting the current FoV of the telescope
        :rtype: array
        """
        cdef unsigned int psf_dim_x = psf.shape[0]
        cdef unsigned int psf_dim_y = psf.shape[1]
        cdef np.ndarray[double , ndim=1, mode="c"] psf_cython = np.asarray(psf.ravel(), dtype = np.double, order="C")
        cdef double *star_mask
        cdef double *star_mask_smeared
        cdef double *star_image


        res = ArrayWrapper()
        star_mask = df.generate_starmask(self.cstars, qe, exposure_time, oversampling, dim_x, dim_y)
                
        
        if(smear_x != 0 or smear_y != 0):
            star_mask_smeared = df.smear_star_image(star_mask, dim_x, dim_y, smear_x, smear_y, oversampling)
            free(star_mask)
            star_image = df.generate_star_image(&psf_cython[0],star_mask_smeared, oversampling, dim_x, dim_y, psf_dim_x, psf_dim_y)
            free(star_mask_smeared)
        else:
            star_image = df.generate_star_image(&psf_cython[0],star_mask, oversampling, dim_x, dim_y, psf_dim_x, psf_dim_y)
            free(star_mask)
        
        res.set_data(dim_x*oversampling, dim_y*oversampling, <void*>star_image)
        return np.array(res)

    def rotate_star_position(self, double alpha, double x_origin, double y_origin):
        """Rotate the star coordinates around a given origin

        :param alpha: angle of the rotation [deg]
        :type alpha: double
        :param x_origin: x_origin of the rotation
        :type x_origin: double
        :param y_origin: y_origin of the rotation
        :type y_origin: double
        """
        cdef double alpha_rad

        alpha_rad = np.radians(alpha)

        df.rotate_stars(&self.cstars, alpha_rad, x_origin, y_origin)

        return

    def shift_stars(self, double x_step, double y_step, only_target=False):
        """Shift stars in a linear fashion

        :param x_step: shift in x [px]
        :type x_step: double
        :param y_step: shift in y [px]
        :type y_step: double
        :param only_target: flag to only shift target stars, defaults to False
        :type only_target: bool
        """
        if only_target:
            for i in range(self.cstars.number):
                if self.cstars.is_target[i] == 1:
                    self.cstars.x[i] = self.cstars.x[i] + x_step
                    self.cstars.y[i] = self.cstars.y[i] + y_step
        else:
            for i in range(self.cstars.number):
                self.cstars.x[i] = self.cstars.x[i] + x_step
                self.cstars.y[i] = self.cstars.y[i] + y_step

        return

    def set_target_pos(self, x, y):
        """Set the position of all targets to a specific coordinate

        :param x: x coordinate
        :type x: double
        :param y: y coordinate
        :type y: double
        """
        for i in range(self.cstars.number):
            if self.cstars.is_target[i] == 1:
                self.cstars.x[i] = x
                self.cstars.y[i] = y
        return


    def update_star_signal(self, double signal_step, only_target=False):
        """Increases the signal of stars by a defined value

        :param signal_step: Signal value to be added
        :type signal_step: double
        :param only_target: flag to only increase target stars, defaults to False
        :type only_target: bool
        """
        if only_target:
            for i in range(self.cstars.number):
                if self.cstars.is_target[i] == 1:
                    self.cstars.signal[i] = self.cstars.signal[i] + signal_step
        else:
            for i in range(self.cstars.number):
                self.cstars.signal[i] = self.cstars.signal[i] + signal_step

        return

    def return_target_data(self):
        """Method to return the current position for the target stars

        :return: coordinates and signal of the target
        :rtype: double, double, double
        """
        for i in range(self.cstars.number):
            if self.cstars.is_target[i] == 1:
                return self.cstars.x[i], self.cstars.y[i], self.cstars.signal[i]

    def convert_to_detector(self, quaternion, fov, platescale):
        """Method to determine the x/y coordinates of stars on the detector based on an input quaternion

        :param quaternion: Input quaternion, scalar first notation
        :type quaternion: array
        :param fov: the field of view size [deg]
        :type fov: double
        :param platescale: platescale of the detector
        :type platescale: double
        """
        star_quat = np.zeros(4)
        quaternion_inv = np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])
        for i in range(self.cstars.number):
            star_quat[0] = 0
            star_quat[1] = np.cos(self.cstars.dec[i] * (np.pi/180)) * np.cos(self.cstars.ra[i] * (np.pi/180))
            star_quat[2] = np.cos(self.cstars.dec[i] * (np.pi/180)) * np.sin(self.cstars.ra[i] * (np.pi/180))
            star_quat[3] = np.sin(self.cstars.dec[i] * (np.pi/180))

            tmp = multiply_quat(quaternion, star_quat)
            res_quat = multiply_quat(tmp, quaternion_inv)

            x_pos = res_quat[1]*(180/np.pi)
            y_pos = res_quat[2]*(180/np.pi)

            if np.abs(x_pos) < fov/2 and np.abs(y_pos) < fov/2:
                self.cstars.x[i] = (x_pos + fov/2) * 3600000 / platescale
                self.cstars.y[i] = (y_pos + fov/2) * 3600000 / platescale
            else:
                self.cstars.x[i] = -10
                self.cstars.y[i] = -10

    def get_stars(self):
        """Getter for stars"""
        for i in range(self.cstars.number):
            self._x[i] = float(self.cstars.x[i])
            self._y[i] = float(self.cstars.y[i])
            self._ra[i] = float(self.cstars.ra[i])
            self._dec[i] = float(self.cstars.dec[i])
            self._signal[i] = float(self.cstars.signal[i])
            self._is_target[i] = float(self.cstars.is_target[i])

        return self._x, self._y, self._ra, self._dec, self._signal, self._is_target


def gen_bkgr(psf, double background_signal, double qe, double exposure_time, unsigned int detector_dim_x, unsigned int detector_dim_y, unsigned int oversampling = 1):
    """Function that creates a simulated background image based on a random poisson distributed image that is convolved with the given Pointspread function.

    :param psf: python array containing the Pointspread function
    :type psf: array
    :param background_signal: mean signal of the background [photons/s]
    :type backlground_signal: double
    :param qe: quantum efficiency of the detector
    :type qe: double
    :param exposure_time: exposure time of the image in seconds
    :type exposure_time: double
    :param detector_dim_x: x-dim of the array
    :type detector_dim_x: unsigned int
    :param detector_dim_y: y-dim of the array
    :type detector:dim_y: unsigned int
    :param oversampling: oversampling of the image based on the overall sub-px flat, defaults to 1
    :type oversampling: unsigned int
    :return: array containing the background image
    :rtype: array
    """

    cdef unsigned int psf_dim_x = psf.shape[0]
    cdef unsigned int psf_dim_y = psf.shape[1]
    cdef np.ndarray[double , ndim=1, mode="c"] psf_cython = np.asarray(psf.ravel(), dtype = np.double, order="C")
    cdef double *bkgr

    res = ArrayWrapper()
    bkgr = df.generate_background(&psf_cython[0], background_signal, qe, exposure_time, oversampling, detector_dim_x, detector_dim_y, psf_dim_x, psf_dim_y)
    res.set_data(detector_dim_x*oversampling, detector_dim_y*oversampling, <void*>bkgr)

    return np.array(res)

def gen_shotnoise(image):
    """Function that creates a simulated background image based on a random poisson distributed image that is convolved with the given Pointspread function.

    :param image: python array containing the incomming flux
    :type image: array
    :return: array containing the shot noise of the input image
    :rtype: array
    """
    cdef unsigned int dim_x = image.shape[0]
    cdef unsigned int dim_y = image.shape[1]
    cdef np.ndarray[double , ndim=1, mode="c"] image_cython = np.asarray(image.ravel(), dtype = np.double, order="C")
    cdef double *shot
    
    res = ArrayWrapper()
    shot = df.generate_shot(&image_cython[0], dim_x, dim_y)
    res.set_data(dim_x, dim_y, <void*>shot)

    return np.array(res)

def multiply_quat(quat1, quat2):
    output_quat = np.zeros(4)
    vec1 = quat1[1:]
    vec2 = quat2[1:]
    s1 = quat1[0]
    s2 = quat2[0]

    cross_res = np.cross(vec1, vec2)
    output_quat[0] = s1 * s2 - np.dot(vec1, vec2)
    output_quat[1] = s1 * vec2[0] + s2 * vec1[0] + cross_res[0]
    output_quat[2] = s1 * vec2[1] + s2 * vec1[1] + cross_res[1]
    output_quat[3] = s1 * vec2[2] + s2 * vec1[2] + cross_res[2]

    return output_quat



