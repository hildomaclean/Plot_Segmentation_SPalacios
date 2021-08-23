import os
from osgeo import gdal, ogr
import cv2
import rasterio
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
from scipy.signal import convolve2d
from scipy import ndimage



#inImg_path ='data_papa/TTC00035_geo_CUT2.tif'
inImg_path ='data/TTC_0474_georeferenced_CUT.tif'
#inImg_path ='data/TTC_0455_georeferenced_CUT.tif'
#inImg_path ='data/20191129_40m_BGRREN_RMSE0_506_CUT_CUTSP.tif'
#inImg_path ='data/Test_190615_Umbeluzi_2doSET_Index_BGNRRedEdge.tif'

k=4



colorOnly=False
_degRot = range(-75, 90+1, 15)


def getBandUint8(band, dtype):
    if "float" in dtype:
        band[band < 0] = 0
        band_int8 = (band - band.min()) * 255 / \
                    (np.quantile(band, .999) - band.min())
        band_int8[band_int8 > 255] = 255
        return band_int8
    else:
        return band


def loadImg(path):
    rasObj = rasterio.open(path) #access to geospatial raster data
    ls_dtype = rasObj.dtypes #Rasterio attributes
    #print(ls_dtype)
    nCh = rasObj.count
    if nCh < 3:
        npImg = np.zeros((rasObj.height, rasObj.width, 3), dtype="uint8")
        for i in range(3):
            npImg[:, :, i] = getBandUint8(rasObj.read(1), ls_dtype[1])
    else:
        npImg = np.zeros((rasObj.height, rasObj.width, nCh), dtype="uint8")
        for i in range(nCh):
            npImg[:, :, i]= getBandUint8(rasObj.read(i + 1), ls_dtype[i])
            #aux = getBandUint8(rasObj.read(i + 1), ls_dtype[i])
            #npImg[:, :, i] = ndimage.median_filter(aux, size=3)
        
    transform = rasObj.transform
    try:
        crs = rasObj.crs.wkt
    except Exception:
        crs = False
        
    rasObj.close()
    return npImg, transform, crs, nCh,rasObj.height,rasObj.width

#KMEAN
def doKMeans(img, k=3, features=[0]):
    """
    ----------
    Parameters
    ----------
    """

    # data type conversion for opencv
    ## select features
    img = img[:, :, features].copy()
    ## standardize
    img_max, img_min = img.max(axis=(0, 1)), img.min(axis=(0, 1))-(1e-8)
    img = (img-img_min)/(img_max-img_min)
    ## convert to float32
    img_z = img.reshape((-1, img.shape[2])).astype(np.float32)
    
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    param_k = dict(data=img_z,
                   K=k,
                   bestLabels=None,
                   criteria=criteria,
                   attempts=10,
                #    flags=cv2.KMEANS_RANDOM_CENTERS)
                   flags=cv2.KMEANS_PP_CENTERS)

    # KMEANS_RANDOM_CENTERS
    cv2.setRNGSeed(99163)
    _, img_k_temp, center = cv2.kmeans(**param_k)

    # Convert back
    img_k = img_k_temp.astype(np.uint8).reshape((img.shape[0], -1))

    # return
    return img_k, center

imgInput,tiff_transform, crs, nCh, height, width = loadImg(inImg_path) #gimage.py
features= np.arange(0, nCh)
imgK, center = doKMeans(img=imgInput,k=k,features=features)

#C = int(imgK.max()) #maximum number of clusters
#print ("Maximum",C)
#get_pixels = lambda x: [np.count_nonzero(imgK== i) for i in range(0,C+1,1)]
#print("Pixeles x cluster",get_pixels(1)) #Print cuantos cluster por pixel
#print("centers",center)

plt.imshow(imgK)
plt.show()
