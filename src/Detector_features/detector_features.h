#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <string.h>
#include <complex.h>
#include <float.h>
#include <fftw3.h>
#include <stdbool.h>
#include "../Utilities/datasim_utility.h"

#ifndef DETECTOR_FEATURE_LIB_H
#define DETECTOR_FEATURE_LIB_H

struct stars {
  double *signal;
  double *x;
  double *y;
  unsigned int number;
  short *is_target;
};

double *generate_bias(double, double, unsigned int, unsigned int, unsigned int);

double *generate_masterbias(double*, double, double, unsigned int, unsigned int, unsigned int);

double *generate_dark(double, double, double*, unsigned int, unsigned int, unsigned int, double, double, double, double);

double *generate_masterdark(double* ,double, double, double*, unsigned int, unsigned int, unsigned int);

double *generate_hotpixels(double, double, double, unsigned int, unsigned int, unsigned int);

double *generate_flat(double, double, double *, double, double, double, double, double, unsigned int, unsigned int, unsigned int);

double *generate_flat_gradient(double, double, double, double, unsigned int, unsigned int);

double *generate_tiled_flat(double, double, double, unsigned int, unsigned int, unsigned int);

double *generate_starmask(struct stars, double, double, unsigned int, unsigned int, unsigned int);

double *generate_star_image(double *, double *, int, int, int, int, int);

void rotate_stars(struct stars *, double, double, double);

double *generate_background(double *, double, double, double, int, int, int, int, int);

double *smear_star_image(double*, unsigned int, unsigned int, double, double, unsigned int);

double *convolve_starmask_fast(double *, double *, int, int, int, int);

double *generate_shot(double *, unsigned int, unsigned int);

#endif //DATASIM_UTILITY_LIB_H
