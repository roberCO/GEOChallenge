import numpy as np

from photutils import Background2D
from photutils import detect_sources
from photutils import deblend_sources
from photutils import MedianBackground
from photutils import source_properties
from photutils import EllipticalAperture

import astropy.units as u
from astropy.convolution import Gaussian2DKernel

def image_segmentation(data, npixels, r, sigma, threshold_value):

    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
    
    threshold = bkg.background + (threshold_value * bkg.background_rms)
    kernel = Gaussian2DKernel(sigma, x_size=2, y_size=2)
    kernel.normalize()
    segm = detect_sources(data, threshold, npixels=npixels, filter_kernel=kernel, connectivity=4)
    segm_deblend = deblend_sources(data, segm, npixels=npixels, filter_kernel=kernel, nlevels=32, contrast=0.001)
    
    cat = source_properties(data, segm_deblend)
    apertures = []
    for obj in cat:
        position = np.transpose((obj.xcentroid.value, obj.ycentroid.value))
        a = obj.semimajor_axis_sigma.value * r
        b = obj.semiminor_axis_sigma.value * r
        theta = obj.orientation.to(u.rad).value
        apertures.append(EllipticalAperture(position, a, b, theta=theta))


    return apertures