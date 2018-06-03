#!/usr/bin/env python

# nadaraya watson interpolation for sparse depth images

import numpy as np
import os
import sys
from PIL import Image
import glob
import imageio
import numba
import copy

# the nadaraya watson interpolation is a gaussian over all valids
@numba.jit(nopython=True)   
def apply_nadaraya_watson( _inputImage, _kernelSize = 7, _alpha=1 ):

    dst = np.zeros( shape=_inputImage.shape )
    uv_lut = np.zeros( shape=(_kernelSize, _kernelSize) )

    halfKernel = int(np.floor(_kernelSize/2))

    mask = _inputImage > 0

    kernelRange = range(-halfKernel,halfKernel+1, 1)

    for k_v in kernelRange: 
        for k_u in kernelRange:
            uv_lut[k_v+halfKernel, k_u+halfKernel] = np.sqrt(k_v**2 + k_u**2)

    imgHeight = _inputImage.shape[0]
    imgWidth = _inputImage.shape[1]

    #nbprint "Performing nadaraya watson on image with size ({},{})".format(imgHeight,imgWidth)

    for v in range(imgHeight):
        for u in range(imgWidth):
            nw_numerator = 0
            nw_denominator = 0
            for k_v in kernelRange: 
                for k_u in kernelRange:
                    if v+k_v < imgHeight and v+k_v > 0 and u+k_u < imgWidth and u+k_u > 0:
                        nw_numerator+= np.exp(-_alpha*uv_lut[k_v+halfKernel, k_u+halfKernel]) * _inputImage[v+k_v, u+k_u]
                        nw_denominator+= np.exp(-_alpha*uv_lut[k_v+halfKernel, k_u+halfKernel]) * mask[v+k_v, u+k_u]
            if nw_denominator != 0:
                dst[v,u] = nw_numerator / nw_denominator
            else:
                dst[v,u] = 0 # nw_numerator / nw_denominator

    return dst

# nearest neighbor interpolation
@numba.jit(nopython=True)   
def apply_nearest_neighbor( _inputImage, _maxRadius=10 ):

    #destination image
    dst = np.zeros( shape=_inputImage.shape )
    imgHeight = _inputImage.shape[0]
    imgWidth = _inputImage.shape[1]

    #nbprint "Performing nadaraya watson on image with size ({},{})".format(imgHeight,imgWidth)

    for v in range(imgHeight):
        for u in range(imgWidth):
            searchRadius=1
            not_found = True
            nn_values = []
            while not_found:
                for k_v in range(-searchRadius,searchRadius,1): 
                    for k_u in range(-searchRadius,searchRadius,1):
                        if v+k_v < imgHeight and v+k_v > 0 and u+k_u < imgWidth and u+k_u > 0:
                            if _inputImage[v+k_v,u+k_u] != 0:
                                nn_values.append(_inputImage[v+k_v,u+k_u])
                                not_found=False
                searchRadius=searchRadius+1
                if not_found and searchRadius==_maxRadius-1:
                    nn_values.append(0)
                    not_found=False

            if len(nn_values):
                dst[v,u] = np.random.choice(np.array(nn_values))
            else:
                dst[v,u] = 0#np.random.choice(np.array(nn_values))

    return dst

if __name__ == '__main__':
    searchPaths = [ os.path.join(os.environ['KITTIPATH'], 'test_depth_completion_anonymous', 'velodyne_raw') + '/*.png',
                    os.path.join(os.environ['KITTIPATH'], 'val_selection_cropped', 'velodyne_raw') + '/*.png',
                    os.path.join(os.environ['KITTIPATH'], 'train') + '/*/proj_depth/velodyne_raw/*/*.png']

    for path in searchPaths:
        imageFiles = glob.glob(path)
        for i, imageFile in enumerate(imageFiles):

            if imageFiles.startswith(os.path.join(os.environ['KITTIPATH'], 'train')):
                rightOrLeft = os.path.basename(os.path.dirname(imageFile))
                target_nn = os.path.abspath(os.path.join(os.path.dirname(imageFile), "..", "..", "interpolated_nn", rightOrLeft))
                targetImg_nn = os.path.join( target_nn, os.path.basename(imageFile) )

                target_nw = os.path.abspath(os.path.join(os.path.dirname(imageFile), "..", "..", "interpolated_nw", rightOrLeft))
                targetImg_nw = os.path.join( target_nw, os.path.basename(imageFile) )
            else:
                target_nn = os.path.abspath(os.path.join(os.path.dirname(imageFile), "..", "interpolated_nn"))
                targetImg_nn = os.path.join( target_nn, os.path.basename(imageFile) )

                target_nw = os.path.abspath(os.path.join(os.path.dirname(imageFile), "..", "interpolated_nw"))
                targetImg_nw = os.path.join( target_nw, os.path.basename(imageFile) )

            if os.path.isfile( targetImg_nn ) and os.path.isfile( targetImg_nw ):
                continue

            try:
                gt = np.array( Image.open( imageFile ) )
            except Exception as e:
                continue
                
            image = copy.deepcopy( gt )

            interpolated_nn = apply_nearest_neighbor( image, 15 )
            if not os.path.isdir( target_nn ):
                os.makedirs(target_nn)    

            i_nn_16bit = np.asarray( interpolated_nn, dtype=np.uint16)
            imageio.imwrite(targetImg_nn, i_nn_16bit)

            
            interpolated_nw = apply_nadaraya_watson( image, 7, 20 )
            if not os.path.isdir( target_nw ):
                os.makedirs(target_nw)
     
            i_nw_16bit = np.asarray( interpolated_nw, dtype=np.uint16)
            imageio.imwrite(targetImg_nw, i_nw_16bit)

            print "{}/{}".format(i, len(imageFiles))

