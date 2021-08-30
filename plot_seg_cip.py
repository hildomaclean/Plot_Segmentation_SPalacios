import os
from osgeo import gdal, ogr, osr
import cv2
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import pandas as pd
from scipy.signal import convolve2d
from scipy import ndimage
import proccessing_algo

#inImg_path ='data_papa/TTC00035_geo_CUT2.tif'
#inImg_path ='data/TTC_0474_georeferenced_CUT.tif'
#inImg_path ='data/TTC_0455_georeferenced_CUT.tif'
#inImg_path ='data/20191129_40m_BGRREN_RMSE0_506_CUT_CUTSP.tif'
#inImg_path ='data/Test_190615_Umbeluzi_2doSET_Index_BGNRRedEdge.tif'

#features=[0,1,2]

#Input Image
inImg_path ='data/20191129_40m_BGRREN_RMSE0_506_CUT_CUTSP.tif'
#output folder
output_folder='data_output'
#Number of cluster
k=3
#Cluster of crop
crop_cluster=2

result_folder=os.path.join(output_folder,os.path.basename(inImg_path).split('.')[0] )
if (os.path.isdir(result_folder) == False): os.mkdir(result_folder)
#print(image_name)
#Outputfiles
outRaster=os.path.join(result_folder,os.path.basename(inImg_path).split('.')[0] +  '_seg.tif')
shapefile_path=os.path.join(result_folder,os.path.basename(inImg_path).split('.')[0] +  '_seg.shp')
outcsvfile=os.path.join(result_folder,os.path.basename(inImg_path).split('.')[0] +  '_seg.csv')

#print(outres)
#outRaster='data_output/TTC_0474_seg_255.tif' 
#shapefile_path='data_output/TTC_0474_seg/SHAPE/TTC_0474_seg.shp' #
#outcsvfile='data_output/TTC_0474_seg/TTC_0474_seg_255_mod.csv'



""" 
Some of the code was adapted from [1]


[1] Chunpeng James Chen and Zhiwu Zhang. “GRID: A Python Package for 
FieldPlot Phenotyping Using Aerial Images”. 
In:Remote Sensing12.11 (2020).issn:2072-4292.doi:10.3390/rs12111697.
url:https://www.mdpi.com/2072-4292/12/11/1697.

"""

colorOnly=False

#Rotation range to chech the plot inclination
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

def getFourierTransform(sig):
    sigf = abs(np.fft.fft(sig)/len(sig))
    return sigf[2:int(len(sigf)/2)]


def rankCenters( k, center, imgK, colorOnly=False):
        scores = []

        if colorOnly:
            ratioK = [(center[i, 0]-center[i, 1])/center[i, :].sum()
                    for i in range(center.shape[0])]
            rank = np.flip(np.argsort(ratioK), axis=0)
        else:
            for i in range(k):
                imgB = (np.isin(imgK, i)*1).astype(np.int64)
                sigs = imgB.mean(axis=0)
                sigsF = getFourierTransform(sigs)
                scMaxF = round((max(sigsF)/sigsF.mean())/100, 4)  # [0, 1]
                scMean = round(1-(sigs.mean()), 4)  # [0, 1]
                try:
                    scPeaks = round(len(find_peaks(sigs, height=(sigs.mean()))
                                        [0])/len(find_peaks(sigs)[0]), 4)
                except Exception:
                    scPeaks = 0

                score = scMaxF*.25 + scMean*.25 + scPeaks*.5
                scores.append(score)
            rank = np.flip(np.array(scores).argsort(), axis=0)

        return rank


def binarizeSmImg(image, cutoff=0.5):
    """
    ----------
    Parameters
    Binariza en el cluster especifico
    ----------
    """
    imgOut = image.copy()
    imgOut[image == cutoff] = 1
    imgOut[image > cutoff] = 0
    imgOut[image < cutoff] = 0
    #imgOut[image > cutoff] = 1
    #imgOut[image <= cutoff] = 0

    return imgOut.astype(np.int64)

#Read the input image and extract its information
imgInput,tiff_transform, crs, nCh, height, width = loadImg(inImg_path) #gimage.py

features= np.arange(0, nCh)

paramKMs = {'k': -1,'center': None,'features': [],'lsSelect': [],'lsKTobin': [],'valShad': -1,'valSmth': -1}
imgK, center = doKMeans(img=imgInput,k=k,features=features)
paramKMs['center'] = center
paramKMs['rank'] = rankCenters(k=k, center=center, imgK=imgK, colorOnly=colorOnly)
lsSelect=[ 0,1]
clusterSelected = paramKMs['rank'][lsSelect]
binOrg=((np.isin(imgK, clusterSelected))*1).astype(np.int64)

#Calling Binarization function, the crop_cluster should be indicated
image_binary=binarizeSmImg(imgK,crop_cluster) 

#Improving the binarization
image_binary = image_binary.astype('uint8')
kernel = np.ones((3,3),np.uint8)
kernel2 = np.ones((3,3),np.uint8)
aux= ndimage.median_filter(image_binary ,size=3)
opening = cv2.morphologyEx(aux,cv2.MORPH_OPEN,kernel, iterations = 1)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2,iterations = 1)

#C = int(imgK.max()) #maximum number of clusters
#print ("Maximum",C)
#get_pixels = lambda x: [np.count_nonzero(imgK== i) for i in range(0,C+1,1)]
#print("Pixeles x cluster",get_pixels(1)) #Print cuantos cluster por pixel
#print("centers",center)
#plt.imshow(imgInput[:,:,0:3])
#plt.imshow(closing )
#plt.show()


"""
Angle detection

"""

def rotateBinNdArray(img, angle):
    # create border for the image
    img[:, 0:2] = 1
    img[0:2, :] = 1
    img[:, -2:] = 1
    img[-2:, :] = 1

    # padding
    sizePad = max(img.shape)
    imgP = np.pad(img, [sizePad, sizePad], 'constant')

    # rotate
    #pivot = tuple((np.array(imgP.shape[:2])/2).astype(np.int64)) 4.1.1
    pivot = tuple(np.array(imgP.shape[:2])/2) #version actual de opencv 4.5.2
    matRot = cv2.getRotationMatrix2D(pivot,-angle,1.0)
    imgR = cv2.warpAffine(
        imgP.astype(np.float32), matRot, imgP.shape, flags=cv2.INTER_LINEAR).astype(np.uint8)

    # crop
    sigX = np.where(imgR.sum(axis=0) != 0)[0]
    sigY = np.where(imgR.sum(axis=1) != 0)[0]
    imgC = imgR[sigY[0]:sigY[-1], sigX[0]:sigX[-1]]

    # return
    return imgC

def getFourierTransform(sig):
    sigf = abs(np.fft.fft(sig)/len(sig))
    return sigf[2:int(len(sigf)/2)]

def detectAngles(img, rangeAngle):
        # evaluate each angle
        sc = []
        for angle in rangeAngle:
            imgR = rotateBinNdArray(img, angle)
            sig = imgR.mean(axis=0)
            sigFour = getFourierTransform(sig)
            sc.append(max(sigFour))

        # angle with maximum score win
        scSort = sc.copy()
        scSort.sort()
        idxMax = [i for i in range(len(sc)) if (sc[i] in scSort[-2:])]
        angles = np.array([rangeAngle[idx] for idx in idxMax])

        # sort angles (one closer to 0/90 is major angle)
        idx_major = np.argmin(angles % 90)
        angles = angles[[idx_major, abs(idx_major-1)]]

        # return
        return angles


def smoothImg(image, n):
    """
    ----------
    Parameters
    ----------
    """

    kernel = np.array((
            [1, 4, 1],
            [4, 9, 4],
            [1, 4, 1]), dtype='int') / 29

    for _ in range(n):
        image = convolve2d(image, kernel, mode='same')

    return image

# GRID Binarized function
def binarizeSmImg_ori(image, cutoff=0.5):
    """
    ----------
    Parameters
    ----------
    """
    imgOut = image.copy()
    imgOut[image > cutoff] = 1
    imgOut[image <= cutoff] = 0

    return imgOut.astype(np.int64)




def blurImg(image, n, cutoff=0.5):
    image = smoothImg(image=image, n=n)
    return binarizeSmImg_ori(image, cutoff=cutoff)

nSmt = int(min(image_binary.shape[0], image_binary.shape[1])/300)
binSeg=blurImg(closing,n=nSmt)
imgBin=binSeg
#To find the angles, is it better to use image_binary ori
angles = detectAngles(img=imgBin, rangeAngle=_degRot)

#print(angles)

#plt.imshow(closing )
#plt.show()


"""
Save the binary image into a raster"""
imagdal= gdal.Open(inImg_path, gdal.GA_ReadOnly)

nCh = imagdal.RasterCount
nc = imagdal.RasterXSize
nl = imagdal.RasterYSize
GeoTransform =imagdal.GetGeoTransform()
Projection = imagdal.GetProjection()
driver = gdal.GetDriverByName('GTiff')
dst_ds = driver.Create(outRaster, nc, nl, 1, gdal.GDT_Byte)
dst_ds.SetGeoTransform(GeoTransform )
dst_ds.SetProjection(Projection)
dst_ds.GetRasterBand(1).WriteArray(closing[:,:],0,0)
del dst_ds 

"""
Opens the binary raster and creates the shapes and saves it
"""
#shapefile_path='data_output/TTC_0474_seg/SHAPE/TTC_0474_seg.shp'


source_raster = gdal.Open(outRaster) 
band= source_raster.GetRasterBand(1)   

drv  = ogr.GetDriverByName('ESRI Shapefile') 
out_data = drv.CreateDataSource(shapefile_path)

srs = source_raster.GetSpatialRef()
dst_layer = out_data.CreateLayer(shapefile_path, geom_type=ogr.wkbPolygon, srs=srs)
dst_fieldname = 'id'
fd = ogr.FieldDefn(dst_fieldname, ogr.OFTInteger)
dst_layer.CreateField(fd)
dst_field = 0
gdal.Polygonize(band, None, dst_layer, dst_field, [], callback=None)


out_data=None
source_raster = None

#Open el shapefile y elimina el feature 0, q es el shape de toda la imagen
ds = ogr.Open(shapefile_path, True)  # True allows to edit the shapefile
lyr = ds.GetLayer()

lyr.SetAttributeFilter("id = '0' ")

#lyr.DeleteFeature(lyr.GetFID()) 
#lyr.DeleteFeature(0)  # Deletes the first feature in the Layer

for feat in lyr:
    lyr.DeleteFeature(feat.GetFID())
    ds.ExecuteSQL('REPACK ' + lyr.GetName())
    ds.ExecuteSQL('RECOMPUTE EXTENT ON ' + lyr.GetName())
    # Repack and recompute extent
    # This is not mandatory but it organizes the FID's (so they start at 0 again and not 1)
    # and recalculates t

del ds

#Coloca numeros en orden en el feature id
shapeSource= ogr.Open(shapefile_path, True)
layer=shapeSource.GetLayer()
featureCount = layer.GetFeatureCount()
for feat in layer:
    feat.SetField("id",(feat.GetFID() + 1))
    layer.SetFeature(feat)
    #print(feat.GetFID())
    #print(feat.GetField("id"))
del shapeSource

#Rasteriza el shape y extrae los datos de las bandas 

ROI = proccessing_algo.rasterize(inImg_path, shapefile_path, 'id')
X, Y = proccessing_algo.get_samples_from_roi(inImg_path, ROI)
#YX=np.append(Y,X, axis=1) #Datos desordenados
#print( X.shape)
#print( Y.shape)
#outcsvfile='data_output/TTC_0474_seg/TTC_0474_seg_255_mod.csv'

                               
# Ordena los datos xq en la extraccion no sale ordenado
x = np.array([]).reshape(0, X.shape[1])#Trainig set
y = np.array([]).reshape(0, 1)
C = int(Y.max())#Number of classes
for i in range(C):
    t = np.where((i + 1) == Y)[0]
    x = np.concatenate((x, X[t[:], :]))
    y = np.concatenate((y, Y[t[:]]))
    #print(x.shape)
#Guarda en  csv
#print( x.shape)
#print( y.shape)
yx = np.append(y,x, axis=1)

cols_name = [] # Generate columns labels
for i in range(X.shape[1]):
    if i==0:
        cols_name.append('Shape')
        
    bname= 'band_' + f'{i+1}' 
    cols_name.append(bname)


data_csv=pd.DataFrame(yx,columns=cols_name)
data_csv.to_csv(outcsvfile)
#print(cols_name)
#plt.imshow(imgInput)
#plt.show()