cdef extern from 'detector_features.h':
  cdef struct stars:
      double *signal
      double *x
      double *y
      unsigned int number;
      short *is_target

  double *generate_bias(double bias_value, double readout_noise, unsigned int width, unsigned int height, unsigned int os)

  double *generate_masterbias(double *biasframe, double bias_value, double readout_noise, unsigned int readouts, unsigned int width, unsigned int height)

  double *generate_dark(double dark_mean, double exp_time, double *hot_pixels, unsigned int width, unsigned int height, unsigned int os, double sb_rate, double sb_amp, double sb_size_mean, double sb_size_sig)

  double *generate_masterdark(double *darkframe, double dark_mean, double exp_time, double *hot_pixels, unsigned int readouts, unsigned int width, unsigned int height)

  double *generate_hotpixels(double amount, double lower, double upper, unsigned int width, unsigned int height, unsigned int os)

  double *generate_flat(double flat_mean, double flat_sigma, double *subpxflat, double grad_lower, double grad_upper, double grad_var, double angle, double px_to_px_response, unsigned int os, unsigned int width, unsigned int height)

  double *generate_flat_gradient(double grad_lower, double grad_upper, double grad_var, double angle, unsigned int width, unsigned int height)

  double *generate_tiled_flat(double mean, double std, double percent, unsigned int dim, unsigned int width_out, unsigned int height_out)

  double *generate_starmask(stars starfield, double qe, double exposure_time, unsigned int oversampling, unsigned int detector_width, unsigned int detector_height)

  double *generate_star_image(double *psf, double *starmask, int oversampling, int width, int height, int width_psf, int height_psf)

  void rotate_stars(stars *starfield, double alpha, double x_origin, double y_origin)

  double *generate_background(double *psf, double background_signal, double qe, double exposure_time, int oversampling, int width, int height, int width_psf, int height_psf)

  double *smear_star_image(double *starmask, unsigned int width, unsigned int height, double x, double y, unsigned int oversampling)

  double *convolve_starmask_fast(double *starmask, double *psf, int width_star, int height_star, int width_psf, int height_psf)

  double *generate_shot(double *signal, unsigned int width, unisgned int height)

