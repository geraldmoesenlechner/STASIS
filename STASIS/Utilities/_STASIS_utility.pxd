
cdef extern from "STASIS_utility.h":

    double *random_poisson_trm(double mean, unsigned int size)

    double *random_normal_trm(double mean, double sigma, unsigned int size)

    double *upsample_image(double *image, unsigned int width, unsigned int height, unsigned int os, int copy)

    double *downsample_image(double *image, unsigned int width, unsigned int height, unsigned int os)

    double *tile_array(double *input_array, unsigned int width_in, unsigned int height_in, unsigned int width_out, unsigned int height_out)

    double *rotate_image(double *input_image, double angle, unsigned int width, unsigned int height, int apply_padding)

    double *convolve(double *data, double *kernel, int width, int height, int width_kernel, int height_kernel)
