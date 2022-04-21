# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure  # skikit-image
import math
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
from scipy.misc import derivative

#voasmooth2 = np.array([0 for k in range(307)])
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
    
    
    # print("engine_array type: ", type(engine_array))
    # print("engine_array shape: ", engine_array.shape)

    # Get data in original form
    # Store dimensions after reading
    xx = dimx
    yy = dimy
    zz = dimz
    minIso = engine_array.min()
    maxIso = engine_array.max()
    # print("min iso: ", minIso)
    # print("max iso: ", maxIso)

    # Convert to 3D array
    arr_3d = engine_array.reshape(xx, yy, zz)
    # print("arr_3d type: ", type(arr_3d))
    # print("arr_3d shape: ", arr_3d.shape)

    #  Calculate gradient
    gx, gy, gz = np.gradient(arr_3d, 50, 50, 50)
    # print("arr_3d type: ", type(arr_3d))
    # print("gx shape: ", gx.shape)

    # G has gradient magnitude
    g = np.array([[[0 for k in range(xx)] for j in range(yy)] for i in range(zz)]).transpose()
    # print("g type: ", type(g))
    # print("g shape: ", g.shape)

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

    # print("max: ", arr_3d.max())
    # print("min: ", arr_3d.min())


    #  Surface area
    sSigma = np.array([0 for k in range(maxIso)])
    # print("sSigma.shape : ", sSigma.shape)

    # Surface area for isosurfaces
    # for i in range(0, 256):
    # #   Mesh for particular isovalue
    #     verts, faces, normals, values = measure.marching_cubes(arr_3d, i)
    #     sSigma[i] = measure.mesh_surface_area(verts, faces)
    # #   print(i, sSigma[i])
    
#    Surface area for isosurfaces
    for i in range(1, maxIso):
     #   Mesh for particular isovalue
        verts, faces, normals, values = measure.marching_cubes(arr_3d, i)
        sSigma[i] = measure.mesh_surface_area(verts, faces)
#        print(i, sSigma[i])
#        if sSigma[i] == 0:
#            print(i, sSigma[i])

#    savedSSigma = [2014508,
#                   5100915,
#                   6575345,
#                   6848835,
#                   6625856,
#                   6282097,
#                   5956770,
#                   5684845,
#                   5469420,
#                   5299803,
#                   5165687,
#                   5057844,
#                   4967984,
#                   4890563,
#                   4821644,
#                   4758661,
#                   4700134,
#                   4644983,
#                   4592610,
#                   4542417,
#                   4493984,
#                   4446950,
#                   4401211,
#                   4356554,
#                   4312876,
#                   4270057,
#                   4227987,
#                   4186706,
#                   4146113,
#                   4106065,
#                   4066639,
#                   4027749,
#                   3989294,
#                   3951341,
#                   3913767,
#                   3876570,
#                   3839762,
#                   3803338,
#                   3767265,
#                   3731508,
#                   3696076,
#                   3660930,
#                   3626096,
#                   3591558,
#                   3557257,
#                   3523228,
#                   3489442,
#                   3455849,
#                   3422533,
#                   3389376,
#                   3356440,
#                   3323680,
#                   3291089,
#                   3258672,
#                   3226406,
#                   3194314,
#                   3162366,
#                   3130562,
#                   3098870,
#                   3067319,
#                   3035930,
#                   3004654,
#                   2973520,
#                   2942480,
#                   2911604,
#                   2880801,
#                   2850095,
#                   2819432,
#                   2788868,
#                   2758371,
#                   2727998,
#                   2697729,
#                   2667588,
#                   2637531,
#                   2607472,
#                   2577426,
#                   2547443,
#                   2517581,
#                   2487758,
#                   2458047,
#                   2428436,
#                   2398890,
#                   2369482,
#                   2340082,
#                   2310714,
#                   2281420,
#                   2252194,
#                   2223042,
#                   2193999,
#                   2165005,
#                   2136051,
#                   2107112,
#                   2078273,
#                   2049451,
#                   2020652,
#                   1991933,
#                   1963266,
#                   1934633,
#                   1906012,
#                   1877399,
#                   1848800,
#                   1820102,
#                   1791303,
#                   1762344,
#                   1733190,
#                   1703804,
#                   1674419,
#                   1645356,
#                   1616574,
#                   1588162,
#                   1560002,
#                   1532035,
#                   1504138,
#                   1476288,
#                   1448433,
#                   1420587,
#                   1392758,
#                   1364862,
#                   1336907,
#                   1308902,
#                   1280905,
#                   1252779,
#                   1224597,
#                   1196311,
#                   1167893,
#                   1139338,
#                   1110589,
#                   1081619,
#                   1052379,
#                   1022695,
#                   992524,
#                   961715,
#                   930277,
#                   898088,
#                   864871,
#                   830397,
#                   794745,
#                   757683,
#                   719255,
#                   679249,
#                   637382,
#                   593352,
#                   546823,
#                   497478,
#                   446637,
#                   395837,
#                   347724,
#                   304646,
#                   268599,
#                   239666,
#                   217984,
#                   202199,
#                   191079,
#                   183073,
#                   177158,
#                   172700,
#                   169143,
#                   166084,
#                   163295,
#                   160696,
#                   158210,
#                   155807,
#                   153461,
#                   151158,
#                   148891,
#                   146650,
#                   144435,
#                   142239,
#                   140061,
#                   137906,
#                   135769,
#                   133649,
#                   131542,
#                   129454,
#                   127380,
#                   125321,
#                   123279,
#                   121253,
#                   119242,
#                   117246,
#                   115263,
#                   113292,
#                   111334,
#                   109391,
#                   107463,
#                   105547,
#                   103645,
#                   101754,
#                   99879,
#                   98006,
#                   96143,
#                   94288,
#                   92439,
#                   90587,
#                   88733,
#                   86873,
#                   84985,
#                   83070,
#                   80558,
#                   75492,
#                   73106,
#                   71128,
#                   69297,
#                   67549,
#                   65865,
#                   64263,
#                   62733,
#                   61253,
#                   59793,
#                   58321,
#                   56835,
#                   55319,
#                   53773,
#                   52166,
#                   50482,
#                   48636,
#                   46481,
#                   44222,
#                   42263,
#                   40699,
#                   39344,
#                   38126,
#                   36983,
#                   35878,
#                   34798,
#                   33740,
#                   32700,
#                   31670,
#                   30650,
#                   29645,
#                   28650,
#                   27665,
#                   26690,
#                   25724,
#                   24767,
#                   23815,
#                   22874,
#                   21939,
#                   21018,
#                   20102,
#                   19198,
#                   18303,
#                   17414,
#                   16533,
#                   15662,
#                   14797,
#                   13937,
#                   13087,
#                   12232,
#                   11385,
#                   10540,
#                   9699,
#                   8858,
#                   7978,
#                   6936,
#                   6936]
    # plt.plot(savedSSigma, label="surface area")

    # Gradient summation over an iso value
    cg = np.array([0 for k in range(maxIso+1)])

    #  For each iso value calculate the C(sigma)
    for i in range(0, xx-1):
        for j in range(0, yy-1):
            for k in range(0, zz-1):
                iso = arr_3d[i][j][k]
                cg[iso] += inv[i][j][k]

    cgsmooth = smooth(cg)
    # plt.plot(cgsmooth[0:256], label="C(sigma)")  # https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

#    print("sizes: ", cgsmooth.shape, savedSSigma.__sizeof__())
    
#    voa = cgsmooth[0:sSigma.size] / savedSSigma
    voa = cgsmooth[1:sSigma.size-1] / sSigma [1:sSigma.size-1]

    voasmooth1 = smooth(voa)
    voasmooth1 = smooth(voasmooth1)
    voasmooth1 = smooth(voasmooth1)
    voasmooth1 = smooth(voasmooth1)
    global smoothenedSize
    smoothenedSize = smooth(voasmooth1).size
    global voasmooth2
    voasmooth2 = smooth(voasmooth1)

#    print(voasmooth2.shape)

    differential = np.array([999.999 for k in range(voasmooth2.size)])
    for i in range(voasmooth2.size):
         differential[i] = derivative(f, i, dx=1e-1)
         # print(i, " : ", differential[i])

    minimaValues = np.array([0 for k in range(18)])
    cnt = 0 # Number of minima
    minval = 1e-4 # should be different
    # for i in range(1, voasmooth2.size-3):
    # for i in range(1, 255):
    for i in range(1, voasmooth2.size-1):
        # if (differential[i-1] < 0 and differential[i+1] > 0) or (differential[i-1] > 0 and differential[i+1] < 0):
        if differential[i] < 0 and differential[i + 1] > 0:
            if abs(differential[i]-differential[i + 1]) < minval:
                # print(i)
                # print(i, abs(differential[i]-differential[i + 1]))
                minimaValues[cnt] = abs((i / voasmooth2.size) * (maxIso-minIso)) # scaling to our data range of 255
                cnt += 1
    newMin = np.array([0 for k in range(cnt)])
    for i in range(cnt):
        newMin[i] = minimaValues[i]
#    print("final values: ", newMin)
    return newMin


#  make spline from the points then sample it for 255

 # plt.plot(voasmooth2[0:256], label="VOA" )
# plt.plot(voasmooth2, label="VOA" )
# plt.yscale("log")
# plt.title("VOA")
# # plt.grid()
# plt.xticks(np.arange(0, voasmooth2.size, 10.0))
# plt.legend()
# plt.show()


