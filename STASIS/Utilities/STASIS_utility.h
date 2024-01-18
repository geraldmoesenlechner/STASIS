#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <string.h>
#include <complex.h>
#include <stdbool.h>

#ifndef STASIS_UTILITY_LIB_H
#define STASIS_UTILITY_LIB_H

double *random_poisson_trm(double, unsigned int);

double *random_normal_trm(double, double, unsigned int);

double random_normal_number(double, double);

int extract_image(double *, int, int, double *, int, int, int, int, int, int);

double *convolve(double *, double *, int, int, int, int);

double *upsample_image(double *, unsigned int, unsigned int, unsigned int, int);

double *downsample_image(double *, unsigned int, unsigned int, unsigned int );

double *tile_array(double *, unsigned int, unsigned int, unsigned int, unsigned int);

double *rotate_image(double *, double, unsigned int, unsigned int, int);

#endif //STASIS_UTILITY_LIB_H
