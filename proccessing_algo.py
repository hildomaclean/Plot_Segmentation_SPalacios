import numpy as np
import os
import tempfile
from osgeo import (gdal, ogr)

def rasterize(inRaster, inShape, inField):
    filename = tempfile.mktemp('.tif')
    data = gdal.Open(inRaster, gdal.GA_ReadOnly)
    shp = ogr.Open(inShape)

    lyr = shp.GetLayer()

    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(
        filename,
        data.RasterXSize,
        data.RasterYSize,
        1,
        gdal.GDT_UInt16)
    dst_ds.SetGeoTransform(data.GetGeoTransform())
    dst_ds.SetProjection(data.GetProjection())
    OPTIONS = 'ATTRIBUTE=' + inField
    gdal.RasterizeLayer(dst_ds, [1], lyr, None, options=[OPTIONS])
    data, dst_ds, shp, lyr = None, None, None, None

    return filename

def get_samples_from_roi(raster_name, roi_name,
                         stand_name=False, getCoords=False):
    '''!@brief Get the set of pixels given the thematic map.
    Get the set of pixels given the thematic map. Both map should be of same size. Data is read per block.
        Input:
            raster_name: the name of the raster file, could be any file that GDAL can open
            roi_name: the name of the thematic image: each pixel whose values is greater than 0 is returned
        Output:
            X: the sample matrix. A nXd matrix, where n is the number of referenced pixels and d is the number of variables. Each
                line of the matrix is a pixel.
            Y: the label of the pixel
    Written by Mathieu Fauvel.
    '''
    # Open Raster
    raster = gdal.Open(raster_name, gdal.GA_ReadOnly)
    if raster is None:
        print('Impossible to open ' + raster_name)
        # exit()

    # Open ROI
    roi = gdal.Open(roi_name, gdal.GA_ReadOnly)
    if roi is None:
        print('Impossible to open ' + roi_name)
        # exit()

    if stand_name:
        # Open Stand
        stand = gdal.Open(stand_name, gdal.GA_ReadOnly)
        if stand is None:
            print('Impossible to open ' + stand_name)
            # exit()

    # Some tests
    if (raster.RasterXSize != roi.RasterXSize) or (
            raster.RasterYSize != roi.RasterYSize):
        print('Images should be of the same size')
        # exit()

    # Get block size
    band = raster.GetRasterBand(1)
    block_sizes = band.GetBlockSize()
    x_block_size = block_sizes[0]#9915
    y_block_size = block_sizes[1]#1
    del band

    # Get the number of variables and the size of the images
    d = raster.RasterCount
    nc = raster.RasterXSize
    nl = raster.RasterYSize

    ulx, xres, xskew, uly, yskew, yres = roi.GetGeoTransform()

    if getCoords:
        coords = np.array([], dtype=np.uint16).reshape(0, 2)
        """
    # Old function which computes metric distance...
    if getCoords :
        #list of coords
        coords = sp.array([]).reshape(0,2)

        # convert pixel position to coordinate pos
        def pixel2coord(coord):
            #Returns global coordinates from pixel x, y coords
            x,y=coord
            xp = xres * x + xskew * y + ulx
            yp = yskew * x + yres * y + uly
            return[xp, yp]
      """

    # Read block data
    X = np.array([]).reshape(0, d)
    Y = np.array([],dtype=np.uint16).reshape(0, 1)
    STD = np.array([],dtype=np.uint16).reshape(0, 1)

    for i in range(0, nl, y_block_size):
        if i + y_block_size < nl:  # Check for size consistency in Y
            lines = y_block_size
        else:
            lines = nl - i
        for j in range(0, nc, x_block_size):  # Check for size consistency in X
            if j + x_block_size < nc:
                cols = x_block_size
            else:
                cols = nc - j

            # Load the reference data

            ROI = roi.GetRasterBand(1).ReadAsArray(j, i, cols, lines)
            if stand_name:
                STAND = stand.GetRasterBand(1).ReadAsArray(j, i, cols, lines)

            t = np.nonzero(ROI)

            if t[0].size > 0:
                Y = np.concatenate(
                    (Y, ROI[t].reshape(
                        (t[0].shape[0], 1))))
                if stand_name:
                    STD = np.concatenate(
                        (STD, STAND[t].reshape(
                            (t[0].shape[0], 1))))
                if getCoords:
                    #coords = sp.append(coords,(i,j))
                    #coordsTp = sp.array(([[cols,lines]]))
                    #coords = sp.concatenate((coords,coordsTp))
                    # print(t[1])
                    # print(i)
                    # sp.array([[t[1],i]])
                    coordsTp = np.empty((t[0].shape[0], 2))
                    coordsTp[:, 0] = t[1]
                    coordsTp[:, 1] = [i] * t[1].shape[0]
                    """
                    for n,p in enumerate(coordsTp):
                        coordsTp[n] = pixel2coord(p)
                    """
                    coords = np.concatenate((coords, coordsTp))

                # Load the Variables
                Xtp = np.empty((t[0].shape[0], d))
                for k in range(d):
                    band = raster.GetRasterBand(
                        k +
                        1).ReadAsArray(
                        j,
                        i,
                        cols,
                        lines)
                    Xtp[:, k] = band[t]
                try:
                    X = np.concatenate((X, Xtp))
                except MemoryError:
                    print('Impossible to allocate memory: ROI too big')
                    exit()

    """
    # No conversion anymore as it computes pixel distance and not metrics
    if convertTo4326:
        import osr
        from pyproj import Proj,transform
        # convert points coords to 4326
        # if vector
        ## inShapeOp = ogr.Open(inVector)
        ## inShapeLyr = inShapeOp.GetLayer()
        ## initProj = Proj(inShapeLyr.GetSpatialRef().ExportToProj4()) # proj to Proj4

        sr = osr.SpatialReference()
        sr.ImportFromWkt(roi.GetProjection())
        initProj = Proj(sr.ExportToProj4())
        destProj = Proj("+proj=longlat +datum=WGS84 +no_defs") # http://epsg.io/4326

        coords[:,0],coords[:,1] = transform(initProj,destProj,coords[:,0],coords[:,1])
    """

    # Clean/Close variables
    del Xtp, band
    roi = None  # Close the roi file
    raster = None  # Close the raster file

    if stand_name:
        if not getCoords:
            return X, Y, STD
        else:
            return X, Y, STD, coords
    elif getCoords:
        return X, Y, coords
    else:
        return X, Y

def scale(x, M=None, m=None):  # TODO:  DO IN PLACE SCALING
    """!@brief Function that standardize the data
        Input:
            x: the data
            M: the Max vector
            m: the Min vector
        Output:
            x: the standardize data
            M: the Max vector
            m: the Min vector
    """
    [n, d] = x.shape
    if np.float64 != x.dtype.type:
        x = x.astype('float')

    # Initialization of the output
    xs = np.empty_like(x)

    # get the parameters of the scaling
    minMax = False
    if M is None:
        minMax = True
        M, m = np.amax(x, axis=0), np.amin(x, axis=0)

    den = M - m
    for i in range(d):
        if den[i] != 0:
            xs[:, i] = 2 * (x[:, i] - m[i]) / den[i] - 1
        else:
            xs[:, i] = x[:, i]

    if minMax:
        return xs, M, m
    else:
        return xs