#
#  IsoFeatures.py
#  Final Project
#
#  Created by Nishita Kharche & Eisen Montalvo on 4/20/22.
#

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure  # skikit-image
import math
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
from scipy.misc import derivative

smoothenedSize = 0
voasmooth2 = np.array([0 for k in range(smoothenedSize)])

def f(x):
    return voasmooth2[int(x)]

def smooth(x, window_len=11, window='hanning'):

    if window_len < 3:
        return x
        
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

def getRepresentativeIsosurfaces(path,dimx,dimy,dimz,bits):

    engineFile = open(path, 'rb')
    
    if bits == 8:
        engine_array = np.fromfile(engineFile, dtype=np.uint8)
    elif bits == 16:
        engine_array = np.fromfile(engineFile, dtype=np.uint16)
    else:
        print("Invalid Bit size!")
        return 0

    # Get data in original form
    # Store dimensions after reading
    xx = dimx
    yy = dimy
    zz = dimz
    minIso = engine_array.min()
    maxIso = engine_array.max()

    # Convert to 3D array
    arr_3d = engine_array.reshape(xx, yy, zz)

    #  Calculate gradient
    gx, gy, gz = np.gradient(arr_3d, 50, 50, 50)

    # G has gradient magnitude
    g = np.array([[[0 for k in range(xx)] for j in range(yy)] for i in range(zz)]).transpose()

    # Inverse of gradient magnitude
    inv = np.array([[[0 for k in range(xx)] for j in range(yy)] for i in range(zz)]).transpose()

    # For each voxel
    for i in range(0, 255):
        for j in range(0, 255):
            for k in range(0, 110):
                #   gradient magnitude
                g[i][j][k] = math.sqrt(gx[i][j][k] ** 2 + gy[i][j][k] ** 2 + gz[i][j][k] ** 2)
                if g[i][j][k] != 0:
                    #    inverse
                    inv[i][j][k] = 1 / g[i][j][k]

    #  Surface area
    sSigma = np.array([0 for k in range(maxIso)])
    
#    Surface area for isosurfaces
    for i in range(1, maxIso):
     #   Mesh for particular isovalue
        verts, faces, normals, values = measure.marching_cubes(arr_3d, i)
        sSigma[i] = measure.mesh_surface_area(verts, faces)

    # Gradient summation over an iso value
    cg = np.array([0 for k in range(maxIso+1)])

    #  For each iso value calculate the C(sigma)
    for i in range(0, xx-1):
        for j in range(0, yy-1):
            for k in range(0, zz-1):
                iso = arr_3d[i][j][k]
                cg[iso] += inv[i][j][k]

    cgsmooth = smooth(cg)

    voa = cgsmooth[1:sSigma.size-1] / sSigma [1:sSigma.size-1]

    voasmooth1 = smooth(voa)
    voasmooth1 = smooth(voasmooth1)
    voasmooth1 = smooth(voasmooth1)
    voasmooth1 = smooth(voasmooth1)
    global smoothenedSize
    smoothenedSize = smooth(voasmooth1).size
    global voasmooth2
    voasmooth2 = smooth(voasmooth1)

    differential = np.array([999.999 for k in range(voasmooth2.size)])
    for i in range(voasmooth2.size):
         differential[i] = derivative(f, i, dx=1e-1)

    minimaValues = np.array([0 for k in range(18)])
    cnt = 0 # Number of minima
    minval = 1e-4 # should be different
    for i in range(1, voasmooth2.size-1):
        if differential[i] < 0 and differential[i + 1] > 0:
            if abs(differential[i]-differential[i + 1]) < minval:
                minimaValues[cnt] = abs((i / voasmooth2.size) * (maxIso-minIso)) # scaling to our data range of 255
                cnt += 1
    newMin = np.array([0 for k in range(cnt)])
    for i in range(cnt):
        newMin[i] = minimaValues[i]
    return newMin
