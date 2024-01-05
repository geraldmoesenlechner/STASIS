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
import doctest
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

def gen_bias(double bias_value, double readout_noise, unsigned int dim_x, unsigned int dim_y, oversampling = 1):
    """
    Function that creates a simulated bias frame.

    Parameters
    ----------
    bias_value: double, Constant offset of the image
    readout_noise: double, mean readout noise of the detector [ADU/s]
    dim_x, dim_y: int, x-dim and y-dim in px
    oversampling: int, oversampling of the image based on the overall sub-px flat

    Returns
    -------
    array, array containing the simulated bias
    """

    cdef double *bias

    res = ArrayWrapper()
    bias = df.generate_bias(bias_value, readout_noise, dim_x, dim_y, oversampling)
    res.set_data(dim_x*oversampling, dim_y*oversampling, <void*>bias)
    return np.array(res)

def gen_masterbias(biasframe, double bias_value, double readout_noise, unsigned int readouts):
    """
    Function that creates a simulated masterbias.

    Parameters
    ----------
    biasframe: array, reference frame for the master-bias generation
    bias_value: double, Constant offset of the image
    readout_noise: double, mean readout noise of the detector [ADU/s]
    readouts: int, number of frames simulated for the masterbias generation

    Returns
    -------
    array, array containing the masterbias
    """

    cdef double *masterbias
    cdef unsigned int bias_dim_x = biasframe.shape[0]
    cdef unsigned int bias_dim_y = biasframe.shape[1]
    cdef np.ndarray[double , ndim=1, mode="c"] bias_cython = np.asarray(biasframe.ravel(), dtype = np.double, order="C")

    res = ArrayWrapper()
    masterbias = df.generate_masterbias(&bias_cython[0], bias_value, readout_noise, readouts, bias_dim_x, bias_dim_y)
    res.set_data(bias_dim_x, bias_dim_y, <void*>masterbias)

    return np.array(res)

def gen_dark(double dark_mean, double exp_time, hot_pixels, unsigned int dim_x, unsigned int dim_y, unsigned int oversampling = 1, double sb_rate = 0., double sb_amp = 1000000., double sb_size_mean = 5., double sb_size_sig = 1.):
    """
    Function that creates a simulated dark frame.

    Parameters
    ----------
    dark_mean: double, mean value of the dark in [ADU/s]
    exp_time: double, exposure time in [s]
    hot_pixels: array, array containing a map of the hot pixels
    dim_x, dim_y: unsigned int, x-dim and y-dim in px
    oversampling: unsigned int, oversampling of sub-px flat

    Returns
    -------
    array, array containing the simulated darkframe
    """

    cdef double *dark
    cdef np.ndarray[double , ndim=1, mode="c"] hp_cython = np.asarray(hot_pixels.ravel(), dtype = np.double, order="C")
    cdef unsigned int hp_dim_x = hot_pixels.shape[0]
    cdef unsigned int hp_dim_y = hot_pixels.shape[1]

    if dim_x*oversampling != hp_dim_x or dim_y*oversampling != hp_dim_y:
        raise Exception("Hp map dimension doesn't match dark frame dimensions!")

    res = ArrayWrapper()
    dark = df.generate_dark(dark_mean, exp_time, &hp_cython[0], dim_x, dim_y, oversampling, sb_rate, sb_amp, sb_size_mean, sb_size_sig)
    res.set_data(dim_x*oversampling, dim_y*oversampling, <void*>dark)

    return np.array(res)

def gen_masterdark(darkframe, double dark_mean, double exp_time, hot_pixels, unsigned int readouts):
    """
    Function that creates a simulated masterdark

    Parameters
    ----------
    darkframe: array, reference frame for the masterdark generation
    dark_mean: double, mean value of the dark in [ADU/s]
    exp_time: double, exposure time in [s]
    hot_pixels: array, array containing a map of the hot pixels
    readouts: int, number of frames simulated for the masterbias generation

    Returns
    -------
    array, array containing the simulated masterdark
    """

    cdef double *masterdark
    cdef unsigned int dark_dim_x = darkframe.shape[0]
    cdef unsigned int dark_dim_y = darkframe.shape[1]
    cdef unsigned int hp_dim_x = hot_pixels.shape[0]
    cdef unsigned int hp_dim_y = hot_pixels.shape[1]

    cdef np.ndarray[double , ndim=1, mode="c"] hp_cython = np.asarray(hot_pixels.ravel(), dtype = np.double, order="C")
    cdef np.ndarray[double , ndim=1, mode="c"] darkframe_cython = np.asarray(darkframe.ravel(), dtype = np.double, order="C")

    if dark_dim_x != hp_dim_x or dark_dim_y != hp_dim_y:
        raise Exception("Hp map dimension doesn't match dark frame dimensions!")

    res = ArrayWrapper()
    masterdark = df.generate_masterdark(&darkframe_cython[0], dark_mean, exp_time, &hp_cython[0], readouts, dark_dim_x, dark_dim_y)
    res.set_data(dark_dim_x, dark_dim_y, <void*> masterdark)

    return np.array(res)

def gen_hotpixels(double percentage, double lower_limit, double upper_limit, unsigned int dim_x, unsigned int dim_y, oversampling = 1):
    """
    Function that creates a map containing a given number of randomly distributed hot pixels inside of the image.

    Parameters
    ----------
    percentage: double, max percentage of pixels in the image that should be hp [0-1]
    lower_limit: double, lowest possible signal value for hp
    upper_limit: double, highest possible signal value for hp
    dim_x, dim_y: int, x-dim and y-dim in px
    oversampling: int, oversampling of the Psf

    Returns
    -------
    array, array depicting the hot pixels in the image
    """

    cdef double *hp

    res = ArrayWrapper()
    hp = df.generate_hotpixels(percentage, lower_limit, upper_limit, dim_x, dim_y, oversampling)
    res.set_data(dim_x*oversampling, dim_y*oversampling, <void*>hp)

    return np.array(res)

def gen_flat(double flat_mean, double flat_sigma, subpxflat, double grad_lower, double grad_upper, double grad_var, double angle, double px_to_px_response, unsigned int dim_x, unsigned int dim_y):
    """
    Function that creates a simulated flatfield.

    Parameters
    ----------
    flat_mean: double, mean values of the normally distributed pixel sensitivity [0-1]
    flat_sigma: double, variation of the pixel sensitivity
    subpxflat: array, array containing the inter-pixel sensitivity
    grad_lower: double, lower limit for a flat gradient [TBD]
    grad_upper: double, upper limit for a flat gradient [TBD]
    grad_var: double, variation of the flat gradient [TBD]
    angle: double, direction of the gradient [TBD]
    px_to_px_response: double, [TBD]
    os: int, oversampling of the Psf
    dim_x, dim_y: int, x-dim and y-dim in px

    Returns
    -------
    array, array depicting the flat
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
    """
    Function that creates a linear smearing kernel that is convolved with the given starmask. Used to emulate jitter during the image inegration.

    Parameters
    ----------
    starmask: array, python array containing the original starmask
    x, y: double, x and y length of the smear
    oversampling:int, oversampling of the Psf

    Returns
    -------
    array, float array depicting the smeared starmask
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

    Attributes
    ----------
    cstars: struct stars, C-Struct containing the star-data. Is automatically
    generated from the inputfile

    Methods
    -------
    gen_star_image(psf, double qe, double exposure_time, unsigned int dim_x, unsigned int dim_y, unsigned int oversampling)
        returns a image of the stars stored in the class with the defined size and given Pointspread function
    rotate_star_position(double alpha, double x_origin, double y_origin)
        rotates the coordinates of the stored stars around a given point by alpha (deg)
    shift_stars(double x_step, double y_step, only_target=False)
        shifts the star coordinates by a given amount [px] in the linear x and y direction
    update_star_signal(double signal_step, only_target=False)
        increases the signal of the stars by a given step
    """

    cdef df.stars cstars

    def __init__(self, stars_config):

        if os.path.isfile(stars_config):
            try:
                
                tree = xmlTree.parse(stars_config)
                root = tree.getroot()

                #get stars on detector
                starsTree = root.findall("star")
                stars_number = len(starsTree)
                x = []
                y = []
                signal = []
                is_target = []
                number = []
                i = 0
                for xmlStar in starsTree:

                    x.append(float(xmlStar.get("pos_x")))
                    y.append(float(xmlStar.get("pos_y")))
                    signal.append(float(xmlStar.get("signal")))
                    is_target.append(int(xmlStar.get("is_target")))


                self.cstars.number = np.uintc(len(x))

                self.cstars.x = <double *> malloc(self.cstars.number * sizeof(double))
                self.cstars.y = <double *> malloc(self.cstars.number * sizeof(double))
                self.cstars.signal = <double *> malloc(self.cstars.number * sizeof(double))
                self.cstars.is_target = <short *> malloc(self.cstars.number * sizeof(short))

                for i in range(self.cstars.number):
                    self.cstars.x[i] = np.double(x[i])
                    self.cstars.y[i] = np.double(y[i])
                    self.cstars.signal[i] = np.double(signal[i])
                    self.cstars.is_target[i] = np.short(is_target[i])

            except:
                print("Invalid " + stars_config + " file! Error: ", sys.exc_info()[0])
                raise
        else:
            raise Exception("ERROR: No " + stars_config + " could be found.")

    def __dealloc__(self):
        free(self.cstars.x)
        free(self.cstars.y)
        free(self.cstars.signal)
        free(self.cstars.is_target)


    def gen_star_image(self, psf, double qe, double exposure_time, unsigned int dim_x, unsigned int dim_y, unsigned int oversampling, add_jitter, jitter_files, double smear_x = 0, double smear_y = 0):
        cdef unsigned int psf_dim_x = psf.shape[0]
        cdef unsigned int psf_dim_y = psf.shape[1]
        cdef np.ndarray[double , ndim=1, mode="c"] psf_cython = np.asarray(psf.ravel(), dtype = np.double, order="C")
        cdef np.ndarray[double , ndim=1, mode="c"] jitter_cython
        cdef unsigned int jitter_dim_x
        cdef unsigned int jitter_dim_y
        cdef double *star_mask
        cdef double *star_mask_smeared
        cdef double *star_image
        cdef double *psf_jitter

        res = ArrayWrapper()
        star_mask = df.generate_starmask(self.cstars, qe, exposure_time, oversampling, dim_x, dim_y)
                
        if add_jitter:
            kernel_name = jitter_files + "Kernel_" + str(np.random.randint(0,400)) + ".txt"
            jitter_kernel = np.genfromtxt(kernel_name) 
            jitter_dim_x = jitter_kernel.shape[0]
            jitter_dim_y = jitter_kernel.shape[1]
            jitter_cython = np.asarray(jitter_kernel.ravel(), dtype = np.double, order="C")

            psf_jitter = df.convolve_starmask_fast(&psf_cython[0], &jitter_cython[0], psf_dim_x, psf_dim_y, jitter_dim_x, jitter_dim_y)

            if(smear_x != 0 or smear_y != 0):
                star_mask_smeared = df.smear_star_image(star_mask, dim_x, dim_y, smear_x, smear_y, oversampling)
                free(star_mask)
                star_image = df.generate_star_image(psf_jitter, star_mask_smeared, oversampling, dim_x, dim_y, psf_dim_x, psf_dim_y)
                free(star_mask_smeared)
                free(psf_jitter)
            else:
                star_image = df.generate_star_image(psf_jitter, star_mask, oversampling, dim_x, dim_y, psf_dim_x, psf_dim_y)
                free(star_mask)
                free(psf_jitter)
        else:
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
        cdef double alpha_rad

        alpha_rad = np.radians(alpha)

        df.rotate_stars(&self.cstars, alpha_rad, x_origin, y_origin)

        return

    def shift_stars(self, double x_step, double y_step, only_target=False):
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
        for i in range(self.cstars.number):
            if self.cstars.is_target[i] == 1:
                self.cstars.x[i] = x
                self.cstars.y[i] = y
        return


    def update_star_signal(self, double signal_step, only_target=False):
        if only_target:
            for i in range(self.cstars.number):
                if self.cstars.is_target[i] == 1:
                    self.cstars.signal[i] = self.cstars.signal[i] + signal_step
        else:
            for i in range(self.cstars.number):
                self.cstars.signal[i] = self.cstars.signal[i] + signal_step

        return

    def return_target_data(self):
        for i in range(self.cstars.number):
            if self.cstars.is_target[i] == 1:
                return self.cstars.x[i], self.cstars.y[i], self.cstars.signal[i]
 


def background_generation(psf, double background_signal, double qe, double exposure_time, unsigned int detector_dim_x, unsigned int detector_dim_y, unsigned int oversampling = 1):
    """
    Function that creates a simulated background image based on a random poisson distributed image that is convolved with the given Pointspread function.

    Parameters
    ----------
    psf: array, python array containing the Pointspread function
    background_signal: double, mean signal of the background [photons/s]
    qe: double, quantu, efficiency of the detector
    exposure_time: double, exposure time of the image in seconds
    detector_dim_x, detector_dim_y: int, x-dim and y-dim in px
    oversampling:int, oversampling of the Psf

    Returns
    -------
    array, float array depicting the oversampled background image
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
    """
    Function that creates a simulated background image based on a random poisson distributed image that is convolved with the given Pointspread function.

    Parameters
    ----------
    image: array, python array containing the incomming flux
    
    Returns
    -------
    array, float array depicting the shot noise model of the image
    """
    cdef unsigned int dim_x = image.shape[0]
    cdef unsigned int dim_y = image.shape[1]
    cdef np.ndarray[double , ndim=1, mode="c"] image_cython = np.asarray(image.ravel(), dtype = np.double, order="C")
    cdef double *shot
    
    res = ArrayWrapper()
    shot = df.generate_shot(&image_cython[0], dim_x, dim_y)
    res.set_data(dim_x, dim_y, <void*>shot)

    return np.array(res)


if __name__ == "__main__":
    doctest.testmod()
