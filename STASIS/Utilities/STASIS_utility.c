/**
* @file    STASIS_utility_lib.c
* @author  Gerald Mösenlechner (gerald.moesenlechner@univie.ac.at)
* @date    October, 2019
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
* @brief Utility functions for the data simulator for the ARIEL space mission.
*
*
* ## Overview
* C library containing utility functions such as random number generation, image convolution and image transformations
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
/*
#include <libxml/parser.h>
#include <libxml/xmlversion.h>
#include <libxml/tree.h>
*/
#include "./STASIS_utility.h"

#define SET_PIXEL(img,cols,x,y,val) ( img[x + cols * y] = val )
#define GET_PIXEL(img,cols,x,y) ( img[x + cols * y] )
#define TWO_PI 6.28318531f
#define SIGDIVFWHM 0.42466452f

#if 1
#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline")
#pragma GCC target("avx")
#endif

size_t get_next_pow_2_bound(size_t n)
{
	size_t c = 0;

	n = n - 1;
	while (n >>= 1)
		c++;

	return (1 << (c + 1));
}

int random_poisson(double mean)
{
    double L;
    double p = 1;
    int result = 0;

    L = exp(-mean);
    do {
        result++;
        p *= ((double) rand()) / INT_MAX;
    } while (p > L);
    result--;
    return result;
};

/**
 * @brief Random poisson number generator based on the Transformed Rejection Method from Wolfgang Hoermann
 *
 * Algorithm for creating an array filled with random numbers following a poisson distribution
 *
 * @param    mean       mean value of the distribution
 * @param    size       size of the created array
 *
 * @return Array contaning random numbers
 */

double *random_poisson_trm(double mean, unsigned int size)
{
    unsigned int i;
    double k;
    double U, V, slam, loglam, a, b, invalpha, vr, us, enlam, prod, X;
    double *matrix;

    matrix = malloc(size * sizeof(double));

    slam = sqrt(mean);
    loglam = log(mean);
    b = 0.931 + 2.53*slam;
    a = -0.059 + 0.02483*b;
    invalpha = 1.1239 + 1.1328/(b-3.4);
    vr = 0.9277 - 3.6224/(b-2);
		if(mean >= 10)
		{
	    for (i=0; i<size; )
	    {
	        while (1) {
	            U = (((double) rand()) / INT_MAX) - 0.5;
	            V = (((double) rand()) / INT_MAX);
	            us = 0.5 - fabs(U);
	            k = (double) ((2 * a / us + b) * U + mean + 0.43);
	            if ((us >= 0.07) && (V <= vr)) {
	                matrix[i] = k;
	                i+=1;
	                break;
	            }
	            if ((k < 0) ||
	                ((us < 0.013) && (V > us))) {
	                continue;
	            }
	            if ((log(V) + log(invalpha) - log(a / (us * us) + b)) <=
	                (-mean + k * loglam - lgamma(k + 1))) {
	                matrix[i] = k;
	                i+=1;
	                break;
	            }
	        }
			 }
    }
		else
		{
				enlam = exp(-mean);
				for (i=0; i<size;)
				{
					X = 0;
					prod = 1.0;
					while (1)
					{
				    U = (((double) rand()) / INT_MAX);
				    prod *= U;
				    if (prod > enlam) {
				      X += 1;
				    }
						else
						{
							matrix[i] = X;
							i+=1;
							break;
				    }
					}
				}
		}

    return matrix;
}


/**
 * @brief Random normal distributed number generator based on the Box-Muller transform
 *
 * Algorithm for creating an array filled with random numbers following a normal distribution
 *
 * @param    mean       mean value of the distribution
 * @param		 sigma      sigma of the distribution
 * @param    size       size of the created array
 *
 * @return Array contaning random numbers
 */

double *random_normal_trm(double mean, double sigma, unsigned int size)
{
    unsigned int i;
    double z0, z1, u0, u1;
    double *matrix;
		bool gen = true;
		double eps = DBL_MIN;

    matrix = malloc(size * sizeof(double));
#pragma omp parallel for
		for(i = 0; i<size; i++) {
			if(!gen){
				matrix[i] = z1 * sigma + mean;
			}
			else{
				do {
					u0 = rand() * (1. / RAND_MAX);
					u1 = rand() * (1. / RAND_MAX);
				} while (u0 <= eps);

				z0 = sqrt(-2. * log(u0)) * cos(TWO_PI * u1);
				z1 = sqrt(-2. * log(u0)) * sin(TWO_PI * u1);
				matrix[i] = z0 * sigma + mean;
			}
			gen = !gen;
		}

    return matrix;
}


double random_normal_number(double mean, double sigma)
{
	double z0, u0, u1, val;
	static double z1;
	static bool gen = true;
	double eps = DBL_MIN;

	if(!gen){
		val = z1 * sigma + mean;
	}
	else{
		do {
			u0 = rand() * (1. / RAND_MAX);
			u1 = rand() * (1. / RAND_MAX);

		} while (u0 <= eps);

		z0 = sqrt(-2. * log(u0)) * cos(TWO_PI * u1);
		z1 = sqrt(-2. * log(u0)) * sin(TWO_PI * u1);
		val = z0 * sigma + mean;
	}
	gen = !gen;
	return val;
}


/**
 * @brief put one image into another image cyclically
 *
 * @param dst the destination matrix
 * @param dw the width of the destination
 * @param dh the height of the destination
 *
 * @param src the source matrix
 * @param sw the width of the source
 * @param sh the height of the source
 *
 * @param x the upper left corner destination x coordinate
 * @param y the upper left corner destination y coordinate
 *
 * @returns 0 on success, otherwise error
 *
 * @note src must be smaller or equal dst; src will wrap within dst,
 *	 x and y are taken as mod w and mod h respectively
 */

static int put_matrix(double complex *dst, int dw, int dh, const double *src, int sw, int sh, int x, int y)
{
    int i, j;
    int dx, dy;


    if (!dst || !src)
        return -1;

    if ((dw < sw) || (dh < sh))
        return -1;


#pragma omp parallel for private(dy)
    for (i = 0; i < sh; i++) {

        dy = y + i;

        /* fold into dest height */
        if (dy < 0 || dy > dh)
            dy = (dy + dh) % dh;

#pragma omp parallel for private(dx)
        for (j = 0; j < sw; j++) {

            dx = x + j;
            /* fold into dest width */
            if (dx < 0 || dx > dw)
                dx = (dx + dw) % dw;

            dst[dy * dw + dx] =  src[i * sw + j];
        }
    }

    return 0;
}


static int put_matrix_double(double *dst, int dw, int dh, const double *src, int sw, int sh, int x, int y)
{
    int i, j;
    int dx, dy;


    if (!dst || !src)
        return -1;

    if ((dw < sw) || (dh < sh))
        return -1;


#pragma omp parallel for private(dy)
    for (i = 0; i < sh; i++) {

        dy = y + i;

        /* fold into dest height */
        if (dy < 0 || dy > dh)
            dy = (dy + dh) % dh;

#pragma omp parallel for private(dx)
        for (j = 0; j < sw; j++) {

            dx = x + j;
            /* fold into dest width */
            if (dx < 0 || dx > dw)
                dx = (dx + dw) % dw;

            dst[dy * dw + dx] =  src[i * sw + j];
        }
    }

    return 0;
}

/**
 * @brief get one image from another image cyclically
 *
 * @param dst the destination matrix
 * @param dw the width of the destination
 * @param dh the height of the destination
 *
 * @param src the source matrix
 * @param sw the width of the source area
 * @param sh the height of the source area
 *
 * @param x the upper left corner source area x coordinate
 * @param y the upper left corner source area y coordinate
 * @param w the source area width
 * @param h the source area height
 *
 * @returns 0 on success, otherwise error
 *
 * @note dst must be larger or equal src
 *       the upper left corner of the sleected area area will be placed at 0,0
 *       within dst
 *       the src area will wrap within src, as x an y are taken as mod w and
 *       mod h respectively for both source and destination
 */

static int get_matrix(double *dst, int dw, int dh,
                      const double complex *src, int sw, int sh,
                      int x, int y, int w, int h)
{
    int i, j;
    int sx, sy;


    if (!dst || !src)
        return -1;

    if ((dw < w) || (dh < h))
        return -1;

    if ((w > sw) || (h > sh))
        return -1;


#pragma omp parallel for private(sy)
    for (i = 0; i < h; i++) {

        sy = y +i;

        /* fold into src height */
        if (sy < 0 || sy > sh)
            sy = (sy + sh) % sh;

#pragma omp parallel for private(sx)
        for (j = 0; j < w; j++) {

            sx  = x + j;

            /* fold into src width */
            if (sx < 0 || sx > sw)
                sx = (sx + sw) % sw;

            dst[i * dw + j] = creal(src[sy * sw + sx]);
        }
    }

    return 0;
}



/**
 * @brief extracts a roi from a given image
 *
 * @param dst the destination matrix
 * @param dw the width of the destination
 * @param dh the height of the destination
 *
 * @param src the source matrix
 * @param sw the width of the source area
 * @param sh the height of the source area
 *
 * @param x the upper left corner source area x coordinate
 * @param y the upper left corner source area y coordinate
 * @param w the source area width
 * @param h the source area height
 *
 * @returns 0 on success, otherwise error
 *
 * @note dst must be larger or equal src
 *       the upper left corner of the sleected area area will be placed at 0,0
 *       within dst
 *       the src area will wrap within src, as x an y are taken as mod w and
 *       mod h respectively for both source and destination
 */

int extract_image(double *dst, int dw, int dh, double *src, int sw, int sh, int x, int y, int w, int h)
{
    int i, j;
    int sx, sy;


    if (!dst || !src)
        return -1;

    if ((dw < w) || (dh < h))
        return -1;

    if ((w > sw) || (h > sh))
        return -1;


#pragma omp parallel for private(sy)
    for (i = 0; i < h; i++) {

        sy = y +i;

        /* fold into src height */
        if (sy < 0 || sy > sh)
            sy = (sy + sh) % sh;

#pragma omp parallel for private(sx)
        for (j = 0; j < w; j++) {

            sx  = x + j;

            /* fold into src width */
            if (sx < 0 || sx > sw)
                sx = (sx + sw) % sw;

            dst[i * dw + j] = src[sy * sw + sx];
        }
    }

    return 0;
}


/**
 * @param data the data matrix to transform
 *
 * @param inv flag for inverse tranformation
 *
 * @param n1 / n2 the fft size
 *
 * @returns 0 on succes, otherwise error
 */


static int fft2d_fftw(double complex *array, int inv, int n1, int n2)
{
    fftw_plan p;
    //double complex *tmp;

    fftw_plan_with_nthreads(4);
    p = fftw_plan_dft_2d(n1, n2, array, array, inv, FFTW_ESTIMATE);
    fftw_execute(p);

    //memcpy(array, array, n * n * sizeof(double complex));

    fftw_destroy_plan(p);
    //fftw_free(tmp);
    return 0;
}

/*
double *convolution_new(double *data, double *kernel, int width, int height, int width_kernel, int height_kernel)
{
	unsigned int width_pad, height_pad, i;
	double scale;

	double *data_pad;
	double *kernel_pad;
	double complex *data_conv;
	double complex *kernel_conv;
	double complex *conv;
	double complex *tmp;
  double *result;
	fftw_plan p_data, p_kernel, p_inv;


	width_pad = width + width_kernel - 1;
 	height_pad = height + height_kernel - 1;

	scale = 1.0/(width_pad*height_pad);

	data_pad = (double*) malloc(sizeof(double) * width_pad * height_pad);
	kernel_pad = (double*) malloc(sizeof(double) * width_pad * height_pad);
	data_conv = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * width_pad * height_pad);
	kernel_conv = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * width_pad * height_pad);
	conv = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * width_pad * height_pad);
	tmp = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * width_pad * height_pad);
	result = (double*) malloc(width * height * sizeof(double));

	bzero(data_pad, width_pad * height_pad * sizeof(double));
	bzero(kernel_pad, width_pad * height_pad * sizeof(double));


	put_matrix(data_pad, width_pad, height_pad, data, width, height, 0, 0);


	put_matrix(kernel_pad, width_pad, height_pad, kernel, width_kernel, height_kernel, -width_kernel/2, -height_kernel/2);

	p_data = fftw_plan_dft_r2c_2d(width_pad, height_pad, data_pad, data_conv, FFTW_ESTIMATE);
	p_kernel = fftw_plan_dft_r2c_2d(width_pad, height_pad, kernel_pad, kernel_conv, FFTW_ESTIMATE);
	p_inv = fftw_plan_dft_2d(width_pad, height_pad, conv, tmp, 1, FFTW_ESTIMATE);
	fftw_execute(p_data);
	fftw_execute(p_kernel);


	for (i = 0; i < width_pad * height_pad; i++)
			conv[i] = data_conv[i] * kernel_conv[i] * scale;

	fftw_execute(p_inv);

	get_matrix(result, width, height, tmp, width_pad, height_pad, width_pad-width, height_pad-height, width, height);

	free(tmp);
	free(data_conv);
	free(kernel_conv);
	free(data_pad);
	free(kernel_pad);
	free(conv);

	return result;
}
*/


double *convolve_sliding_window(double *data, double *kernel, unsigned int width, unsigned int height, unsigned int width_kernel, unsigned int height_kernel)
{
	unsigned int i,j, k, l, i_start, j_start, width_pad, height_pad;
	double tmp;
	double *data_pad, *result;

	width_pad = width + width_kernel;
	height_pad = height + height_kernel;

	data_pad = (double*) malloc(width_pad * height_pad * sizeof(double));
	result = (double*) malloc(width * height * sizeof(double));
	bzero(data_pad, width_pad * height_pad * sizeof(double));

	put_matrix_double(data_pad, width_pad, height_pad, data, width, height, width_kernel, height_kernel);

	#if 0
	for (i = 0; i < width_pad*height_pad; i++) {
			printf("%e ", data_pad[i]);
			if (i && !((i + 1) % (width_pad)))
					printf("\n");
	}
	#endif


	i_start = width_kernel/2;
	j_start = height_kernel/2;
#pragma omp parallel for private(tmp), collapse(2)
	for(i = 0; i < width; i++)
	{
		for(j = 0; j < height; j++)
		{
			tmp = 0;
			for(k = 0; k < width_kernel; k++)
			{
				for(l = 0; l < height_kernel; l++)
				{
					tmp = tmp + data_pad[(i_start + i + k) + width_pad * (j_start + j + l)] * kernel[k + width_kernel*l];
				}
			}
			result[i + width * j] = tmp;
		}
	}

	#if 0
	for (i = 0; i < width*height; i++) {
			printf("%e ", result[i]);
			if (i && !((i + 1) % (width)))
					printf("\n");
	}
	#endif
	free(data_pad);

	return result;
}

/**
 * @brief convolution of to imanges
 *
 * @param data first input image
 * @param kernel second image (in the general case the Psf)
 * @param width size of the first image in x direction
 * @param height size of the first image in y direction
 * @param width_kernel size of the second image in x direction
 * @param height_kernel size of the second image in y direction
 *
 * @returns the convolved image
 *
 */

double *convolve(double *data, double *kernel, int width, int height, int width_kernel, int height_kernel)
{
    int i, n;

    double *tmp;
    double *result;

    double complex *data_pad;
    double complex *kernel_pad;


    double complex *conv;

    double scale;

    /* determine fft size */
    if (width < height)
        n = get_next_pow_2_bound(height);
    else
        n = get_next_pow_2_bound(width);

		scale = 1.0/(n*n);
    /* allocate data, kernel and result */

    tmp = malloc(width * height * sizeof(double));
    result = malloc(width * height * sizeof(double));

    /* allocate padded data and convolution arrays */

    data_pad = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n * n);



    //kernel_pad = malloc(n * n * sizeof(double complex));
    kernel_pad = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n * n);

    //conv = malloc(n * n * sizeof(double complex));
    conv = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n * n);

    /* set zero */
    bzero(data_pad, n * n * sizeof(double complex));
    bzero(kernel_pad, n * n * sizeof(double complex));

    /* put kernel and shift center to upper left corner */
    put_matrix(data_pad, n, n, data, width, height, 0, 0);


    /* put kernel and shift center to upper left corner */
    put_matrix(kernel_pad, n, n, kernel, width_kernel, height_kernel, -width_kernel/2, -height_kernel/2);

    bzero(data, width * height * sizeof(double));
    bzero(kernel, width_kernel * height_kernel * sizeof(double));



    /* transform both forward */
    fft2d_fftw(data_pad, 0, n, n);

    fft2d_fftw(kernel_pad, 0, n, n);


		//fft2d_fftw_real(data_pad, kernel_pad, n, n);

    /* multiplication step */
    for (i = 0; i < n * n; i++)
        conv[i] = data_pad[i] * kernel_pad[i] * scale;

    /* transform result backwards */
    fft2d_fftw(conv, 1, n, n);


    get_matrix(tmp, width, height,
               conv, n, n,
               n-width, n-height, width, height);

    memcpy(result, tmp, width * height * sizeof(double));

    free(tmp);
    //free(data);
    //free(kernel);
    free(data_pad);
    free(kernel_pad);

    free(conv);

    return result;
}

/*
void read_stars_config(char *inputfile)
{
	xmlDoc         *document;
 	xmlNode        *root, *first_child, *node;
	//struct stars

	document = xmlReadFile(inputfile, NULL, 0);
	root = xmlDocGetRootElement(document);
	printf("Root is <%s> (%i)\n", root->name, root->type);
  first_child = root->children;
  for (node = first_child; node; node = node->next) {
      printf("\t Child is <%s> (%i)\n", node->name, node->type);
			printf("\t %s\n", (char *)node->content);
  }
	xmlFreeDoc(document);
	return;
}


void getReference (xmlDocPtr doc, xmlNodePtr cur) {

	xmlChar *uri;
	cur = cur->xmlChildrenNode;
	while (cur != NULL) {
	    if ((!xmlStrcmp(cur->name, (const xmlChar *)"reference"))) {
		    uri = xmlGetProp(cur, "star");
		    printf("uri: %s\n", uri);
		    xmlFree(uri);
	    }
	    cur = cur->next;
	}
	return;
}


void read_stars_config(char *docname) {

	xmlDocPtr doc;
	xmlNodePtr cur;

	doc = xmlParseFile(docname);

	if (doc == NULL ) {
		fprintf(stderr,"Document not parsed successfully. \n");
		return;
	}

	cur = xmlDocGetRootElement(doc);

	if (cur == NULL) {
		fprintf(stderr,"empty document\n");
		xmlFreeDoc(doc);
		return;
	}

	if (xmlStrcmp(cur->name, (const xmlChar *) "star_field")) {
		fprintf(stderr,"document of the wrong type, root node != story");
		xmlFreeDoc(doc);
		return;
	}

	getReference (doc, cur);
	xmlFreeDoc(doc);
	return;
}
*/

/**
 * @brief simple upsampling of an image
 *
 * @param image Original image
 * @param height/width dimension of the original image
 * @param os Upsampling factor
 * @param copy Flag if the values should be split over the upsampled pixel
 *
 * @returns Upsampled image
 *
 */

double *upsample_image(double *image, unsigned int width, unsigned int height, unsigned int os, int copy)
{
	unsigned int x, y, i, j, k, l;
	unsigned int x_tmp, y_tmp;
	double *upsampled_image;
	double value;
	unsigned int splitter;

	upsampled_image = malloc(width * os * height * os * sizeof(double));

	if(copy == 0){
		splitter = os*os;
	}
	else{
		splitter = 1;
	}

	for(i=0; i < width; i++){
		for(j=0; j < height; j++){
			value = image[i+width*j]/splitter;
			x = i*os;
			y = j*os;
			for(k=0; k<os; k++){
				for(l=0; l<os; l++){
					x_tmp = x+k;
					y_tmp = y+l;
					SET_PIXEL(upsampled_image, width * os, x_tmp, y_tmp, value);
				}
			}
		}
	}

	return upsampled_image;
}

/**
 * @brief simple downsampling of an image
 *
 * @param image Original image
 * @param height/width dimension of the original image
 * @param os downsampling factor
 *
 * @returns downsampled image
 *
 */

double *downsample_image(double *image, unsigned int width, unsigned int height, unsigned int os)
{
	unsigned int x, y, i, j, k, l, x_tmp, y_tmp;
	unsigned int width_new, height_new;
	double *downsampled_image;
	double value, tmp;

	width_new = (unsigned int) floor(width/os);
	height_new = (unsigned int) floor(height/os);



	downsampled_image = malloc(width_new * height_new * sizeof(double));
#pragma omp parallel for private(value)
	for(i=0; i < width_new; i++){
		for(j=0; j < height_new; j++){
			value = 0;
			x = i*os;
			y = j*os;
			for(k=0; k<os; k++){
				for(l=0; l<os; l++){
					x_tmp = x+k;
					y_tmp = y+l;

					tmp = GET_PIXEL(image, width, x_tmp, y_tmp);
					value = value + tmp;
				}
			}
			downsampled_image[i + width_new*j] = value;
		}
	}

	return downsampled_image;
}

/**
 * @brief Function that creates an image by tiling a smaller arry multiple times
 *
 * @param input_array						input which shal be tiled
 * @param height_in/width_in		dimension of the tile array
 * @param height_out/width_out	dimension of the output array
 *
 * @returns tiled image
 *
 */

double *tile_array(double *input_array, unsigned int width_in, unsigned int height_in, unsigned int width_out, unsigned int height_out)
{
	double *output;
	unsigned int i,j,k,l, x_tmp, y_tmp;
	double tmp;

	output = malloc(width_out*height_out*sizeof(double));

	for(i=0; i < width_out; i = i+width_in){
		for(j=0; j < height_out; j = j+height_in){
			for(k=0; k<width_in; k++){
				for(l=0; l<height_in; l++){
					x_tmp = i+k;
					y_tmp = j+l;
					if((x_tmp < width_out) && (y_tmp < height_out)){
				     tmp = GET_PIXEL(input_array, width_in, k, l);
				     SET_PIXEL(output, width_out, x_tmp, y_tmp, tmp);
          }
				}
			}
		}
	}
	return output;
}

/*WORK IN PROGRESS*/

void perform_shear_x(double *in, double *out, double shear, unsigned int width, unsigned int height)
{
	unsigned int i, j, skewi;
	double val, skew, skewf, oleft, left;

	for(j = 0; j < height; j++){
		if(j < height/2){
			skew = (j - height/2) * shear;
		}
		else{
			skew = -(j - height/2) * shear;
		}
		skewi = floor(skew);
		skewf = skew - skewi;
		oleft = 0;
		for(i = (width-1); i > 0; i--){

			val = GET_PIXEL(in, width, i, j);
			left = val * skewf;
			val = (val - left) * oleft;

			if(j < height/2){
				if(((i + skewi) >= 0) && ((i + skewi) < width)){
					SET_PIXEL(out, width, (j + skewi), j, val);
				}
			}
			else{
				if(((i - skewi) >= 0) && ((i - skewi) < width)){
					SET_PIXEL(out, width, (i - skewi), j, val);
				}
			}
			/*
			if(((i + skewi) >= 0) && ((i + skewi) < width)){
				SET_PIXEL(out, width, (i + skewi), j, val);
			}
			*/
			oleft = left;
		}
	}
}

void perform_shear_y(double *in, double *out, double shear, unsigned int width, unsigned int height)
{
	unsigned int i, j, skewi;
	double val, skew, skewf, oleft, left;

	for(i = 0; i < width; i++){
		if(i < width/2){
			skew = (i - width/2) * shear;
		}
		else{
			skew = -(i - width/2) * shear;
		}
		skewi = floor(skew);
		skewf = skew - skewi;
		oleft = 0;
		for(j = (height-1); j > 0; j--){

			val = GET_PIXEL(in, width, i, j);
			left = val * skewf;
			val = (val - left) * oleft;

			if(i < width/2){
				if(((j + skewi) >= 0) && ((j + skewi) < height)){
					SET_PIXEL(out, width, i, (j + skewi), val);
				}
			}
			else{
				if(((j - skewi) >= 0) && ((j - skewi) < height)){
					SET_PIXEL(out, width, i, (j - skewi), val);
				}
			}
			/*
			if(((i + skewi) >= 0) && ((i + skewi) < width)){
				SET_PIXEL(out, width, (i + skewi), j, val);
			}
			*/
			oleft = left;
		}
	}
}

/**
 * @brief Function used by the rotate image function. Applies anti aliasing
 * rotation by using shears (Paeth A. W.)
 * the image.
 *
 * @param input_image						original image
 * @param output_image					pointer to allocated memory for the output
 * @param angle									rotation angle
 * @param height/width					dimension of the image
 */

void apply_rotation(double *input_image, double *output_image, double angle, unsigned int width, unsigned int height)
{
	double shear_x, shear_y, rad;
	double *tmp;

	tmp = malloc(width * height * sizeof(double));
	rad = angle * M_PI/180;

	shear_x = tan(rad/2);
	shear_y = sin(rad);

	perform_shear_x(input_image, output_image, shear_x, width, height);
	perform_shear_y(output_image, tmp, shear_y, width, height);
	perform_shear_x(tmp, output_image, shear_x, width, height);
	free(tmp);

}

/**
 * @brief Function that takes an input image and returns a rotated version of
 * the input image.
 *
 * @param input_image						original image
 * @param angle									rotation angle
 * @param height/width					dimension of the image
 * @param apply_padding					flag, if the input already contains the
 *															necessary padding for the rotation
 *
 * @returns rotated image
 *
 */

double *rotate_image(double *input_image, double angle, unsigned int width, unsigned int height, int apply_padding)
{
	double *rotated_image, *padded_image, *tmp;
	double diag, val;
	unsigned int width_pad, height_pad, i, j, shift_x, shift_y;

	if(apply_padding != 0){
		diag = sqrt(width*width + height*height);
		width_pad = ceil(diag - width) + 2;
		height_pad = ceil(diag - height) + 2;

		padded_image = malloc((width + width_pad) * (height + height_pad) * sizeof(double));
		tmp = malloc((width + width_pad) * (height + height_pad) * sizeof(double));

		for(i = 0; i < (width + width_pad) * (height + height_pad); i++){
			padded_image[i] = 0;
		}

		shift_x = (uint) ceil(width_pad/2);
		shift_y = (uint) ceil(height_pad/2);

		for(i = 0; i < width; i++){
			for(j = 0; j < height; j++){
				val = GET_PIXEL(input_image, width, i, j);
				SET_PIXEL(padded_image, (width + width_pad), (i + shift_x), (j + shift_y), val);
			}
		}

		apply_rotation(padded_image, tmp, angle, width + width_pad, height + height_pad);
		free(padded_image);

		rotated_image = malloc(width * height * sizeof(double));

		for(i = 0; i < width; i++){
			for(j = 0; j < height; j++){
				val = GET_PIXEL(tmp, width + width_pad, (i + shift_x), (j + shift_y));
				SET_PIXEL(rotated_image, width, i, j, val);
			}
		}
		free(tmp);
	}
	else{
		rotated_image = malloc(width * height * sizeof(double));
		apply_rotation(input_image, rotated_image, angle, width, height);
	}

	return rotated_image;
}


/**
 * @brief Function for generating 2D gaussian.
 *
 * @param img										input array
 * @param cols/rows							dimension of the image
 * @param x0/y0									center of the gaussian
 * @param fwhm_x/fwhm_y					FWHM of the gaussian
 *
 */

/*
void get_2D_gaussian (float *img, unsigned int cols, unsigned int rows, float x0, float y0, float fwhm_x, float fwhm_y)
{
    unsigned int x, y;
    float value;
    float xdist, ydist;
    float sigx, sigy;
    float fval;

    sigx = fwhm_x * SIGDIVFWHM;
    sigy = fwhm_y * SIGDIVFWHM;

    fval = 1 / (sigx * sigy * TWO_PI);

    //Sx = -0.5f / (sigx * sigx);
    //Sy = -0.5f / (sigy * sigy);

    for (y = 0; y < rows; y++)
        for (x = 0; x < cols; x++)
        {
            xdist = x - x0;
            ydist = y - y0;
            value = fval*exp(-((xdist*xdist)/(2*sigx*sigx)+(ydist*ydist)/(2*sigy*sigy)));
	          SET_PIXEL(img, cols, x, y, value);
        }
    return;
}
*/
