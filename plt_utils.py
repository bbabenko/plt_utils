"""
Copyright (c) 2015, Boris Babenko
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions
   and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
   and the following disclaimer in the documentation and/or other materials provided with the
   distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
   or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from matplotlib import pyplot as plt

def grayshow(image, clip=None, ax=None, heat=False):
    """
    Display a grayscale image, i.e. 2D numpy array.

    Parameters
    ----------
    image : 2D numpy array
    clip : (min_val, max_val) tuple, input image will be clipped to these values (see np.clip)
    ax : pyplot axis object to use (by default uses plt.gca())
    heat : bool, display as a heat map if True

    """
    assert image.ndim == 2, 'Can only display 2D images using this function'
    image = image.astype(np.float32)
    if clip:
        assert len(clip) == 2 and clip[0] < clip[1], 'Invalid range argument: {}'.format(range)
        image = np.clip(image, clip[0], clip[1])
    if ax is None:
        ax = plt.gca()
    ax.matshow(image, cmap='jet' if heat else 'gray')
    plt.draw()
    plt.show()

# british spelling?
greyshow = grayshow

def colorshow(image, ax=None):
    """
    Display an NxMx3 numpy array as a color (RGB) image.

    Parameters
    ----------
    image : 3D (NxMx3) numpy array, if float type will get rescaled to (0,1)
    ax : pyplot axis object to use (by default uses plt.gca())

    """
    assert image.ndim == 3 and image.shape[2] == 3, 'Can only display NxMx3 images using this function'
    if ax is None:
        ax = plt.gca()
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        image = (image - image.min())/image.max()
    ax.imshow(image, interpolation='none')
    plt.draw()
    plt.show()
    
def mosaic(images, num_row=None, num_col=None, normalize=False, clip=None, padding=1):
    """
    Stitch multiple images (2D or 3-channel 3D numpy arrays) into one.  All images must be of
    the same shape.  By default, arranges images in a square mosaic.  If only one of num_row or
    num_col arguments are passed in, the other will be determined based on number of images.  If
    both are passed in, num_rol * num_col must be at least as large as len(images).

    Parameters
    ----------
    images : list of NxM or NxMx3 numpy arrays
    num_row : number of rows in the mosaic
    num_col : number of columns in the mosaic
    normalize : normalize each image to have max value = 1, min value = 0 (this is done *before* 
        clipping)
    clip : (min_val, max_val) tuple, each image will be clipped to these values (see np.clip)
    padding : number of pixels of black padding to insert in between images

    """
    if not isinstance(images, list):
        images = [images[...,i] for i in range(images.shape[-1])]

    num_images = len(images)
    if num_row and not num_col:
        num_col = int(np.ceil(float(num_images)/num_row))
    elif num_col and not num_row:
        num_row = int(np.ceil(float(num_images)/num_col))
    elif not num_row and not num_col:
        num_col = int(np.ceil(np.sqrt(num_images)))
        num_row = int(np.ceil(float(num_images)/num_col))
        
    im_shape = images[0].shape
    assert all([i.shape == im_shape for i in images]), 'Images must have the same shape'
    assert num_images <= num_row*num_col, 'More images than grid cells'
    assert not clip or len(clip) == 2 and clip[0] < clip[1], (
            'Invalid range argument: {}'.format(range))
    
    result_shape = (num_row*im_shape[0] + padding*(num_row-1),
                    num_col*im_shape[1] + padding*(num_col-1))
    if len(im_shape) == 3:
        result_shape = result_shape + (3,)
    result = np.zeros(result_shape, dtype=np.float32)
    im_iter = iter(images)
    for row in range(num_row):
        for col in range(num_col):
            row_start = row * (im_shape[0] + padding)
            col_start = col * (im_shape[1] + padding)
            try:
                image = (im_iter.next()).astype(np.float32)
            except StopIteration:
                break
            if normalize:
                image = (image - image.min())/image.max()
            if clip:
                image = np.clip(image, clip[0], clip[1])
            result[row_start:row_start+im_shape[0], col_start:col_start+im_shape[1],...] = image
    return result
    
def imshow(image, ax=None, split=False):
    """
    Convenience wrapper around grayshow and colorshow.  The main input is a numpy array.  If the 
    array is 2D, it is displayed as a grayscale image (via grayshow()). If the array is NxMx3, 
    it is displayed as a color image (via colorshow()).  If the image NxMxC where C!=3 or if split
    is True, the channels of the image will be stitched into a mosaic (with normalization turned 
    on), and displayed as a grayscale image.

    Parameters
    ----------
    image : 2D or 3D numpy array
    ax : pyplot axis object to use (by default uses plt.gca())

    """
    if image.ndim == 3 and image.shape[2] == 3 and not split:
        colorshow(image, ax=ax)
    elif image.ndim == 3:
        grayshow(mosaic(image, num_row=1, normalize=True), ax=ax)
    elif image.ndim == 2:
        grayshow(image, ax=ax)
    else:
        raise ValueError('Invalid image dimensions: {}'.format(image.shape))
