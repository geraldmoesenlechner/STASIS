/**
* @file    detector_features.c
* @author  Gerald Mösenlechner (gerald.moesenlechner@univie.ac.at)
* @date    October, 2021
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
* @brief Functions for the generation of detector features, such as bias, dark and flats.
*
*
* ## Overview
* C library containing functions for the creation of simulated images of detector features like bias, dark and flat.
*
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <complex.h>
#include <fftw3.h>
#include <stdbool.h>
#include "./detector_features.h"

#define SET_PIXEL(img,cols,x,y,val) ( img[x + cols * y] = val )
#define GET_PIXEL(img,cols,x,y) ( img[x + cols * y] )
#define TWO_PI 6.28318531f
#define SIGDIVFWHM 0.42466452f

#if 1
#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline")
#pragma GCC option("arch=native","tune=native","no-zero-upper")
#pragma GCC target("avx")
#endif

/**
 * @brief generation of the uniform randomly distributed bias.
 *
 * @param bias_value    mean value of the bias
 * @param readout_noise readout noise of the detector
 * @param width/height  size of the detector
 * @param os            oversampling of the final image
 *
 * @returns matrix containing the bias
 */

double *generate_bias(double bias_value, double readout_noise, unsigned int width, unsigned int height, unsigned int os)
{
  double *ro_noise_matrix, *bias, *bias_os;
  double time;
  clock_t start, end;
  unsigned int i;

  bias = malloc(width * height * sizeof(double));
#pragma omp parallel for
  for(i = 0; i<width*height; i++){
    bias[i] = bias_value;
  }

  if(readout_noise > 0.){
    ro_noise_matrix = random_normal_trm(0.0, readout_noise, width*height);
#pragma omp parallel for
    for(i = 0; i<width*height; i++){

      if(ro_noise_matrix[i] < - 2.*readout_noise){
        ro_noise_matrix[i] = - 2.*readout_noise;
      }
      else if(ro_noise_matrix[i] > 2.*readout_noise){
        ro_noise_matrix[i] = 2.*readout_noise;
      }
      bias[i] = bias[i] + ro_noise_matrix[i];
    }
    free(ro_noise_matrix);
  }

  if (os != 1){
    bias_os = upsample_image(bias, width, height, os, 0);
    free(bias);
    return bias_os;
  }
  else{
    return bias;
  }
}

/**
 * @brief generation of the masterbias by averaging multiple bias frames.
 *
 * @param biasframe     bias frame used for image generation
 * @param bias_value    mean value of the bias
 * @param readout_noise readout noise of the detector
 * @param readouts      amount of readouts generated for the average
 * @param width/height  size of the detector
 *
 * @returns matrix containing the masterbias
 */

double *generate_masterbias(double *biasframe, double bias_value, double readout_noise, unsigned int readouts, unsigned int width, unsigned int height)
{
  double *master_bias, *tmp_bias;
  unsigned int i, j;

  memcpy(master_bias, biasframe, width*height*sizeof(double));

  for(i = 0; i < (readouts-1); i++){
    tmp_bias = generate_bias(bias_value, readout_noise, width, height, 1);
    for(j = 0; j < width*height; j++){
      master_bias[j] = master_bias[j] + tmp_bias[j];
    }
  }

  for(i = 0; i < width*height; i++){
    master_bias[i] = master_bias[i]/readouts;
  }
  free(tmp_bias);

  return master_bias;
}

/**
 * @brief function for the generation of a darkframes.
 *
 * @param dark_mean     mean value of the dark (e/s)
 * @param exp_time      exposure time in s
 * @param hot_pixels    array containing the hot pixels
 * @param width/height  size of the detector
 * @param os            oversampling of the final image
 *
 * @returns matrix containing the darkframe
 */

double *generate_dark(double dark_mean, double exp_time, double *hot_pixels, unsigned int width, unsigned int height, unsigned int os, double sb_rate, double sb_amp, double sb_size_mean, double sb_size_sig)
{
  double *darkframe, *darkframe_os;
  unsigned int i;

  if(dark_mean == 0.){
    darkframe = malloc(width*height*sizeof(double));
    for(i = 0; i < width*height; i++){
      darkframe[i] = 0.0;
    }
  }
  else{
    darkframe = random_poisson_trm(dark_mean*exp_time, width*height);
  }

  if(os != 1){
    darkframe_os = upsample_image(darkframe, width, height, os, 0);
    if(rand()/RAND_MAX < sb_rate)
    {
      generate_space_ball(darkframe_os, width*os, height*os, (rand()/(double)RAND_MAX) * width * os, (rand()/(double) RAND_MAX) * height * os, random_normal_number(sb_size_mean*os, sb_size_sig*os), random_normal_number(sb_size_mean*os, sb_size_sig*os), sb_amp);
    }
    for(i = 0; i < width*os*height*os; i++){
      darkframe_os[i] = darkframe_os[i] * hot_pixels[i];
      free(darkframe);
      return darkframe_os;
    }
  }
  else{
    generate_space_ball(darkframe, width, height, rand()/(double)RAND_MAX * width, rand()/(double)RAND_MAX * height, random_normal_number(sb_size_mean, sb_size_sig), random_normal_number(sb_size_mean, sb_size_sig), sb_amp);

    for(i = 0; i < width*height; i++){
      darkframe[i] = darkframe[i] * hot_pixels[i];
    }
    return darkframe;
  }
}

/**
 * @brief function for the generation of a masterdark by averaging.
 *
 * @param darkframe     original dark used for image generation
 * @param dark_mean     mean value of the dark (e/s)
 * @param exp_time      exposure time in s
 * @param hot_pixels    array containing the hot pixels
 * @param readouts      number of frames used for averaging
 * @param width/height  size of the detector
 *
 * @returns matrix containing the masterdark
 */

double *generate_masterdark(double *darkframe, double dark_mean, double exp_time, double *hot_pixels, unsigned int readouts, unsigned int width, unsigned int height)
{
  double *master_dark, *tmp_dark;
  unsigned int i, j;

  memcpy(master_dark, darkframe, width*height*sizeof(double));

  for(i = 0; i < (readouts-1); i++){
    tmp_dark = generate_dark(dark_mean, exp_time, hot_pixels, width, height, 1, 0, 0, 0, 0);
    for(j = 0; j < width*height; j++){
      master_dark[j] = master_dark[j] + tmp_dark[j];
    }
  }

  for(i = 0; i < width*height; i++){
    master_dark[i] = master_dark[i]/readouts;
  }
  free(tmp_dark);

  return master_dark;
}

/**
 * @brief function for the generation of a hot pixel mask.
 *
 * @param amount        0-1, perentage of the detector with hot pixels
 * @param lower         lower boundary for the signal increase
 * @param upper         upper boundary for the signal increase
 * @param width/height  size of the detector
 *
 * @returns matrix containing the hot pixels
 */

double *generate_hotpixels(double amount, double lower, double upper, unsigned int width, unsigned int height, unsigned int os)
{
  double *hot_pixels, *hot_pixels_os;
  double value;
  unsigned int count, x, y;
  unsigned int i;

  srand(time(0));

  hot_pixels = malloc(width*height*sizeof(double));
  count = (unsigned int) ceil(width*height*amount);

  for(i = 0; i < width*height; i++){
    hot_pixels[i] = 1.;
  }

  for(i = 0; i < count; i++){
    x = (unsigned int) (rand() * (1. / RAND_MAX) * width);
    y = (unsigned int) (rand() * (1. / RAND_MAX) * height);
    value = lower + (rand() * (1. / RAND_MAX) + (upper - lower));
    SET_PIXEL(hot_pixels,width,x,y,value);
  }

  if (os != 1){
    hot_pixels_os = upsample_image(hot_pixels, width, height, os, 0);
    free(hot_pixels);
    return hot_pixels_os;
  }

  return hot_pixels;
}


double *generate_flat(double flat_mean, double flat_sigma, double *subpxflat, double grad_lower, double grad_upper, double grad_var, double angle, double px_to_px_response, unsigned int os, unsigned int width, unsigned int height)
{
  double *sflat, *flatfield, *flat_gradient, *upsampled_flat;
  unsigned int i, j;

  flatfield = random_normal_trm(flat_mean, flat_sigma, width*height);

  for(i = 0; i < width*height; i++){
    if(flatfield[i] > 1.0){
      flatfield[i] = 1.0;
    }
    else if(flatfield[i] < 0.0){
      flatfield[i] = 0.0;
    }
  }

  if(os > 1){
    upsampled_flat = upsample_image(flatfield, width, height, os, true);
    free(flatfield);
    sflat = tile_array(subpxflat, os, os, width*os, height*os);
    for (i=0; i < width*os*height*os; i++){
      upsampled_flat[i] = upsampled_flat[i] * sflat[i];
    }
    free(sflat);
    return upsampled_flat;
  }
  else{
    return flatfield;
  }

}


double *generate_flat_gradient(double grad_lower, double grad_upper, double grad_var, double angle, unsigned int width, unsigned int height)
{
  double *flat_grad, *grad_row, grad_step;//, *tmp;
  double grad_value = grad_lower;
  unsigned int i;
  srand(time(0));
  //no rotation availible
  /*
  if(angle > 0.){
    extended_dim = (unsigned int) (sqrt(width*width+height*height) + 1);

    tmp = malloc(extended_dim*extended_dim*sizeof(double));
    grad_row = malloc(extended_dim*sizeof(double));

    grad_step = (grad_upper-grad_lower)/extended_dim;
    for(i = 0; i < extended_dim; i++){
      grad_row[i] = grad_value + random_normal_number(0,grad_var);
      grad_value = grad_value + grad_step;
    }
    tmp = tile_array(grad_row, extended_dim, 1, extended_dim, extended_dim);
    free(grad_row);

  }
  else{
    */
    grad_row = malloc(width*sizeof(double));

    grad_step = (grad_upper-grad_lower)/width;
    for(i = 0; i < width; i++){
      grad_row[i] = grad_value + random_normal_number(0,grad_var);
      grad_value = grad_value + grad_step;
    }
    flat_grad = tile_array(grad_row, width, 1, width, height);
    free(grad_row);

    return flat_grad;
  //}
}

double *generate_tiled_flat(double mean, double std, double percent, unsigned int dim, unsigned int width_out, unsigned int height_out)
{
	double *output, *random_array;
	unsigned int i,j,k,l, x_tmp, y_tmp;
	double tmp, mean_intern, std_intern;

	output = malloc(width_out*height_out*sizeof(double));

	for(i=0; i < width_out; i = i+dim){
		for(j=0; j < height_out; j = j+dim){
      			mean_intern = random_normal_number(mean, std);
      			std_intern = mean_intern*(percent / 100);
     			 random_array = random_normal_trm(mean_intern, std_intern, dim*dim);
			for(k=0; k<dim; k++){
				for(l=0; l<dim; l++){
					x_tmp = i+k;
					y_tmp = j+l;
          				if((x_tmp < width_out) && (y_tmp < height_out)){
				     		tmp = GET_PIXEL(random_array, dim, k, l);
				     		SET_PIXEL(output, width_out, x_tmp, y_tmp, tmp);
          				}
				}
			}
		}
	}
	return output;
}

/*
 * @brief Function that psotitions a star at a given position with subpx
 * precision
 *
 * @param image image of the star mask
 * @param width/height dimension of the starmask
 * @param x/y position of the star
 * @param signal signal of the star
 *
*/

void set_star_in_image(double *image, unsigned int width, unsigned int height, double x, double y, double signal)
{
	unsigned int px_x_low, px_y_low, px_x_high, px_y_high;
	double x_orig, y_orig;

	/*Starmask must be fliped due to origin change from convolution*/
	x_orig = x;
	y_orig = y;
	x = x_orig + 0.5;
	y = y_orig + 0.5;

	px_x_low  =  (int) (x - 0.5);
  	px_y_low  =  (int) (y - 0.5);
  	px_x_high =  (int) (x + 0.5);
  	px_y_high =  (int) (y + 0.5);

	SET_PIXEL(image, height, px_x_low, px_y_low, (1.0-fabs((x_orig-px_x_low))) * (1.0-fabs((y_orig-px_y_low))) * signal);
	SET_PIXEL(image, height, px_x_high, px_y_high, (1.0-fabs((x_orig-px_x_high))) * (1.0-fabs((y_orig-px_y_high))) * signal);
	SET_PIXEL(image, height, px_x_low, px_y_high, (1.0-fabs((x_orig-px_x_low))) * (1.0-fabs((y_orig-px_y_high))) * signal);
	SET_PIXEL(image, height, px_x_high, px_y_low, (1.0-fabs((x_orig-px_x_high))) * (1.0-fabs((y_orig-px_y_low))) * signal);

}

/*
 * @brief Function that creates a mask with pointsources containing the sources
 * signal for the image generation.
 *
 * @param starfield strcut containing the positions and signals of all stars
 * @param exposure_time exposure time of the simulated image, needed for signal calculations
 * @param oversampling oversampling factor of the image (size of subpx flat)
 * @param detector_width/detector_height Dimension of the detector [px]
 *
 * @returns array containing the generated starmask
*/

double *generate_starmask(struct stars star_field, double qe, double exposure_time, unsigned int oversampling, unsigned int detector_width, unsigned int detector_height)
{
	double *mask, *mask_os;
	unsigned int i;

	mask = malloc(detector_width * detector_height * sizeof(double));

	for(i = 0; i < detector_width  * detector_height; i++){
		mask[i] = 0.;
	}

	for(i = 0; i<star_field.number; i++){
		if(star_field.x[i] > 0 && star_field.y[i] > 0 && star_field.x[i] < detector_width && star_field.y[i] < detector_height){
		    set_star_in_image(mask, detector_width, detector_height, star_field.x[i], star_field.y[i], star_field.signal[i]*exposure_time*qe);
    		}
  	}

	if(oversampling != 1)
	{
	    mask_os = upsample_image(mask, detector_width, detector_height, oversampling, 0);
	    free(mask);
	    return mask_os;

	}
	else{
		return mask;
	}
}

/*
 * @brief modified sliding window convolution used for the creation of stellar images
 *
 * @param starmask array containg the original starmask
 * @param psf array containing the pointspread function or convolution kernel
 * @param width_star/height_star dimensions of the starmask
 * @param width_psf/height_psf dimension of the Psf
 *
 * @returns starmask convolved with given kernel
*/

double *convolve_starmask_fast(double *starmask, double *psf, int width_star, int height_star, int width_psf, int height_psf)
{
	unsigned int i, j, k, l, i_start, j_start;
	double *result;

	result = (double *) malloc(width_star * height_star * sizeof(double));

	bzero(result, width_star * height_star * sizeof(double));
#pragma omp parallel for collapse(2)
	for(i=0; i<height_star; i++)
	{
		for(j=0; j<width_star; j++)
		{
				if (starmask[i+height_star*j] > 0)
				{
					i_start = i - width_psf/2 + 1;
					j_start = j - height_psf/2 + 1;
					for(k=0; k<width_psf; k++)
					{
						for(l=0; l<height_psf; l++)
						{
							if(((i_start + k) > 0) && ((j_start + l) > 0) && ((i_start + k) < height_star) && ((j_start + l) < width_star))
							{
								result[i_start + k + height_star * (j_start + l)] = result[i_start + k + height_star * (j_start + l)] + psf[k + width_psf*l] * starmask[i + height_star*j];
							}
						}
					}
				}
		}
	}

	return result;
}

/*
 * @brief function, that takes a starmask and convolves it with the instruments
 * Psf
 *
 * @param psf array containing the oversampled Psf
 * @param starmask array containing the star mask
 * @param oversampling oversampling factor of the Psf
 * @param width/height dimension of the star mask
 * @param width_psf/height_psf dimension of the Psf
 *
 * @returns image containing the observed stars
*/

double *generate_star_image(double *psf, double *starmask, int oversampling, int width, int height, int width_psf, int height_psf)
{
	unsigned int i;
	double *star_image;
	//star_image = convolve_sliding_window(starmask, psf, width*oversampling, height*oversampling, width_psf, height_psf);
	star_image = convolve_starmask_fast(starmask, psf, width*oversampling, height*oversampling, width_psf, height_psf);
	//star_image = convolve(starmask, psf, width*oversampling, height*oversampling, width_psf, height_psf);


	#if 0
	for (i = 0; i < width*oversampling*height*oversampling; i++) {
			printf("%e ", star_image[i]);
			if (i && !((i + 1) % (width*oversampling)))
					printf("\n");
	}
	#endif

	return star_image;
}


void rotate_stars(struct stars *star_field, double alpha, double x_origin, double y_origin)
{
  	double sin_a, cos_a, x_tmp;
  	unsigned int i;

  	sin_a = sin(alpha);
  	cos_a = cos(alpha);

  	for(i=0; i < star_field->number; i++) {
    		x_tmp = ((star_field->x[i]-x_origin)*cos_a - (star_field->y[i]-y_origin)*sin_a) + x_origin;
    		star_field->y[i] = ((star_field->x[i]-x_origin)*sin_a + (star_field->y[i]-y_origin)*cos_a) + y_origin;
    		star_field->x[i] = x_tmp;
  }

  return;
}

/**
 * @brief generation of the background noise
 *
 * @param psf Psf of the instrument
 * @param background_signal Mean signal of the noise distribution
 * @param exposure_time Exposure time of the simulated image
 * @param oversampling Oversampling of the image in accordance to the subpx flat
 * @param width_psf size of the psf in x direction
 * @param height_psf size of the psf in y direction
 *
 * @returns the convolved image
 *
 */

double *generate_background(double *psf, double background_signal, double qe, double exposure_time, int oversampling, int width, int height, int width_psf, int height_psf)
{
	double *noise_data, *bkgr, *tmp, *conv, *psf_ds, *bkgr_os;
    	double signal;
    	unsigned int i;
    	int width_psf_ds = width_psf / oversampling;
    	int height_psf_ds = height_psf / oversampling;
    	int im_size_x = width + 2*width_psf_ds;
    	int im_size_y = height + 2*height_psf_ds;

	bkgr = malloc(width * height * sizeof(double));
    	psf_ds = downsample_image(psf, width_psf, height_psf, oversampling);
    	signal = background_signal*exposure_time*qe;
    	noise_data = random_poisson_trm(signal, im_size_x*im_size_y);

    	conv = convolve(noise_data, psf_ds, im_size_x, im_size_y, width_psf_ds, height_psf_ds);

    	extract_image(bkgr, width, height, conv, im_size_x, im_size_y, width_psf_ds, height_psf_ds, width, height);
    	free(psf_ds);
    	free(noise_data);
    	free(conv);

    	if (oversampling!=1){
		bkgr_os = upsample_image(bkgr, width, height, oversampling, 0);
      		free(bkgr);
		return bkgr_os;
	}
    	return bkgr;
}

/**
 * @brief function that generates a linear smearing kernel based on the Bresenham line algorithm
 *
 * @param x length of the smear in x direction
 * @param y length of the smear in y direction
 * @param dim dimension of the smearing kernel
 * @param oversampling Oversampling of the image
 *
 * @returns a smearing kernel consiting of a line
 *
 */

double *generate_linear_smearing_kernel(double x, double y, unsigned int dim, unsigned int oversampling)
{
	double *smear_kernel;
	unsigned int x0, x1, y0, y1, i;
	double sum;
	int dx, dy, err, e2, sx, sy;

	smear_kernel = (double*) malloc(dim*oversampling*dim*oversampling*sizeof(double));
	bzero(smear_kernel, dim*oversampling*dim*oversampling*sizeof(double));
	x0 = (unsigned int) (dim*oversampling/2);
	y0 = (unsigned int) (dim*oversampling/2);
	x1 = x0 + floor(x*oversampling);
	y1 = y0 + floor(y*oversampling);

	dx =  abs(x1 - x0);
	sx = x0 < x1 ? 1 : -1;
 	dy = -abs(y1 - y0);
	sy = y0 < y1 ? 1 : -1;
  	err = dx + dy; /* error value e_xy */

  	for (;;){  /* loop */
    		SET_PIXEL(smear_kernel, dim*oversampling, x0, y0, 1);
    		if (x0 == x1 && y0 == y1)
	    		break;
    		e2 = 2 * err;
    		if (e2 >= dy){
	    		err += dy;
	    		x0 += sx;
		} /* e_xy+e_x > 0 */
    		if (e2 <= dx){
	    		err += dx;
	    		y0 += sy;
		} /* e_xy+e_y < 0 */
  	}

	sum = 0;

	for(i = 0; i < dim*oversampling*dim*oversampling; i++){
	    	sum = sum + smear_kernel[i];
	}



#pragma omp parallel for
	for(i = 0; i < dim*oversampling*dim*oversampling; i++){
	    	smear_kernel[i] = smear_kernel[i]/(sum);
	}

	return smear_kernel;
}

/**
 * @brief function that generates a linear smearing kernel and convolves it with a given star-map
 *
 * @param starmask input starmask that should be convolved with the smearing kernel
 * @param width x dimension of the starmask
 * @param height y dimension of the starmask
 * @param x length of the smear in x direction
 * @param y length of the smear in y direction
 * @param oversampling Oversampling of the image
 *
 * @returns a smeared starmask
 *
 */

double *smear_star_image(double *starmask, unsigned int width, unsigned int height, double x, double y, unsigned int oversampling)
{
	double *smear_kernel, *smeared_image;
	unsigned int dim, i;

	dim = (unsigned int) 2*(sqrt(x*x+y*y)) + 4;
	smear_kernel = generate_linear_smearing_kernel(x, y, dim, oversampling);
	smeared_image = convolve_starmask_fast(starmask, smear_kernel, width*oversampling, height*oversampling, dim*oversampling, dim*oversampling);

	free(smear_kernel);

	return smeared_image;
}


/**
 * @brief Function for generating a poisson-distributed shot noise model based on the provided image
 *
 * @param signal: combined stellar and bkgr flux of the image
 * @param width: width of the image buffers
 * @param height: height of the image buffers
 *
 * @returns array containing a poisson based shot noise model
 */


double *generate_shot(double *signal, unsigned int width, unsigned int height)
{
    unsigned int i;
    double mean;
	double *shot_noise;

    mean = 0;

	shot_noise = (double*) malloc(width*height*sizeof(double));

    for(i = 0; i < width*height; i++)
    {
        shot_noise[i] = random_poisson(sqrt(signal[i]));
        mean = mean + shot_noise[i];
    }

    mean = mean / (width * height);

    for(i = 0; i < width*height; i++)
    {
        shot_noise[i] = shot_noise[i] - mean;
    }

	return shot_noise;
}
