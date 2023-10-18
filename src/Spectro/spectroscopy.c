/**
 * @file    spectroscopy.c
 * @author  Gerald Mösenlechner (gerald.moesenlechner@univie.ac.at)
 * @date    October, 2022	
 *
 * @copyright
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 * @brief Functions for the generation of spectras.
 *
 * ## Overviee
 * C library containing functions to simulate sythetic spectral images for compression tests.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <float.h>
#include <complex.h>
#include <stdbool.h>
#include "./spectroscopy.h"

#define PLANCK 6.6261e-34
#define BOLTZMANN 1.3806e-23
#define C 299792458
#define TWOPI 6.28318531f


/**
 * @brief Calculates the black body ratiation for a given wavelenght range
 *
 * @param wavelentgth	array containing all wavelengths in microns
 * @param length	length of in- and output array
 * @param temp		temperature of the black body
 * @param [out] output	output array
 * 
 **/

int generate_black_body(double* wavelength, unsigned int length, double temp, double* output)
{
	unsigned int i;
	double tmp_wl;

	for(i = 0; i < length; i++)
	{
		tmp_wl = wavelength[i]/1e6;
		output[i] = 2/(tmp_wl*tmp_wl)*(PLANCK*(C/tmp_wl))/(exp(PLANCK*C/(tmp_wl*BOLTZMANN*temp))-1);
	}

	return 0;
}


double gaussian(double wavelength, struct line l)
{
	double res;
	
	res = l.depth/(l.width*sqrt(TWOPI))*exp(-0.5*pow((wavelength-l.wl_cent)/l.width, 2)); 
	

	return res;
}

int place_spectra(double* image, double* wl, double* signal, double slitwidth, double wl_scale, double x_start, double y_start, unsigned int os)
{
	
	



	return 0;
}


int main()
{
	double *wv, *out;
	unsigned int length = 5000;
	double t = 5500;
	struct line l;
	unsigned int i;
	double res;
	l.wl_cent = 1.2;
	l.width = 0.005;
	l.depth = 2e-10;

	wv = malloc(length * sizeof(double));
	out = malloc(length * sizeof(double));

	for(i = 0; i<length; i++)
	{
		wv[i] = 0.5 + i*0.001;

	}

	generate_black_body(wv, length, t, out);

	for(i = 0; i<length; i++)
	{
	    	out[i] += gaussian(wv[i], l);
		printf("%f %e\n", wv[i], out[i]);
	
	}

	free(out);
	free(wv);


	return 0;
}
