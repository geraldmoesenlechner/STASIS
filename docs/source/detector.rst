.. _STASIS_detector:

Detector Features
=====================

This module provides general functions to simulate astronomical detectors and images.

.. automodule:: Detector
   :members:

Functions
=====================

.. autofunction:: gen_bias
.. autofunction:: gen_dark
.. autofunction:: gen_hotpixels
.. autofunction:: gen_flat
.. autofunction:: smear_stars
.. autoclass:: Stars
    :members: cstars
    :private-members: _x, _y, _ra, _dec, _signal, _is_target

    .. automethod:: __init__
    .. automethod:: gen_star_image
    .. automethod:: rotate_star_position
    .. automethod:: shift_stars
    .. automethod:: set_target_pos
    .. automethod:: update_star_signal
    .. automethod:: return_target_data
    .. automethod:: convert_to_detector
    .. automethod:: get_stars
.. autofunction:: gen_bkgr
.. autofunction:: gen_shotnoise
