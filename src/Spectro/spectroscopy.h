#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <complex.h>
#include <float.h>
#include <stdbool.h>
#include "../Utilities/datasim_utility.h"
#include "../Detector_features/detector_features.h"

#ifndef SPECTROSCOPY_LIB_H
#define SPECTROSCOPY_LIB_H

struct line
{
	double wl_cent;
	double width;
	double depth;
};

int generate_black_body(double*, unsigned int, double, double*);

double gaussian(double, struct line);

#endif /*SPECTROSCOPY_LIB_H*/
