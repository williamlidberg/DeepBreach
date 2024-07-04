import os
import numpy as np
from osgeo import gdal, osr
import imageio.v2 as imageio
import geopandas as gpd
from shapely.ops import unary_union
import whitebox
from unets import Attention_ResUNet
from scipy.ndimage import label, find_objects, sum_labels
import time
whitebox.download_wbt(linux_musl=True, reset=True)
wbt = whitebox.WhiteboxTools()
wbt.verbose = False

def remove_small_groups(binary_image, min_pixels):
    # Label all connected components in the binary image
    labeled_image, num_features = label(binary_image)
    
    # Create an output image, initially all zeros (all groups removed)
    output_image = np.zeros_like(binary_image)
    
    # Iterate through each connected component, checking its size
    for i in range(1, num_features + 1):  # Labels start at 1
        # Calculate the size of the connected component
        size = sum_labels(binary_image, labeled_image, index=i)
        if size >= min_pixels:
            # If the size is above the threshold, retain this component in the output
            output_image[labeled_image == i] = 1
    
    return output_image

def write_gtiff(array, gdal_obj, outputpath, dtype=gdal.GDT_Byte, options=0, color_table=0, nbands=1, nodata=False):
    gt = gdal_obj.GetGeoTransform()
    width = np.shape(array)[1]
    height = np.shape(array)[0]
    # Prepare destination file
    driver = gdal.GetDriverByName("GTiff")
    if options != 0:
        dest = driver.Create(outputpath, width, height, nbands, dtype, options)
    else:
        dest = driver.Create(outputpath, width, height, nbands, dtype)
    if dest is None:
        print(f"Failed to create destination file: {outputpath}")
        return
    # Write output raster
    if color_table != 0:
        dest.GetRasterBand(1).SetColorTable(color_table)
    if len(array.shape) == 3:  # 3D array (height, width, channels)
        if nbands == 1:
            # Take the first channel (or apply any logic suitable for your case)
            array = array[:, :, 0]
        band = dest.GetRasterBand(1)
        if band is not None:
            band.WriteArray(array)
        else:
            print("Failed to get band 1 for writing.")
    else:  # 2D array
        band = dest.GetRasterBand(1)
        if band is not None:
            band.WriteArray(array)
        else:
            print("Failed to get band 1 for writing.")
    if nodata is not False:
        dest.GetRasterBand(1).SetNoDataValue(nodata)
    # Set transform and projection
    dest.SetGeoTransform(gt)
    wkt = gdal_obj.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dest.SetProjection(srs.ExportToWkt())
    # Close output raster dataset
    dest = None
    print(f"File {outputpath} written successfully.")

def patchify_x(img, start_y, patches, tile_size, margin, width):
    start_x = 0
    while start_x + tile_size <= width:
        patches.append(img[start_y:start_y+tile_size,
                           start_x:start_x+tile_size].copy())
        start_x += tile_size - 2 * margin
        assert patches[-1].shape[:2] == (tile_size, tile_size),\
            'shape: {}'.format(patches[-1].shape)
    # handle right boarder
    if start_x < width:
        start_x = width - tile_size
        patches.append(img[start_y:start_y+tile_size,
                           start_x:start_x+tile_size].copy())
        assert patches[-1].shape[:2] == (tile_size, tile_size)

def patchify(img, tile_size, margin):
    patches = []

    height, width = img.shape[:2]
    start_y = 0
    while start_y + tile_size <= height:
        patchify_x(img, start_y, patches, tile_size, margin, width)
        start_y += tile_size - 2 * margin
    # handle bottom boarder
    if start_y < height:
        start_y = height - tile_size
        patchify_x(img, start_y, patches, tile_size, margin, width)

    return patches

def start_and_end(base, tile_size, margin, limit, remainder):
    if base == 0:
        src_start = 0
        src_end = tile_size - margin
    elif base + (tile_size - margin) > limit:
        src_start = tile_size - remainder
        src_end = tile_size
    else:
        src_start = margin
        src_end = tile_size - margin

    return src_start, src_end

def unpatchify(shape, patches, tile_size, margin):
    img = np.zeros(shape)
    height, width = shape
    remain_height = height % tile_size
    remain_width = width % tile_size

    dest_start_y = 0
    dest_start_x = 0

    for i, patch in enumerate(patches):
        remain_width = width - dest_start_x
        remain_height = height - dest_start_y
        src_start_y, src_end_y = start_and_end(dest_start_y, tile_size, margin,
                                               height, remain_height)
        src_start_x, src_end_x = start_and_end(dest_start_x, tile_size, margin,
                                               width, remain_width)
        y_length = src_end_y - src_start_y
        x_length = src_end_x - src_start_x
        img[dest_start_y:dest_start_y+y_length,
            dest_start_x:dest_start_x+x_length] = patch[src_start_y:src_end_y,
                                                        src_start_x:src_end_x]
        dest_start_x += x_length
        if dest_start_x >= width:
            dest_start_x = 0
            dest_start_y += y_length
    return img

def raster_to_polygon(img_name, vector_polygons):
    wbt.raster_to_vector_polygons(
    i = img_name, 
    output = vector_polygons)

def main(input_path, model_path, out_path_binary, out_path_vector, img_type, tile_size, margin,
         threshold):

    # setup paths
    if not os.path.exists(input_path):
        raise ValueError('Input path does not exist: {}'.format(input_path))
    if os.path.isdir(input_path):
        imgs = [os.path.join(input_path, f) for f in os.listdir(input_path)
                if f.endswith('.tif')]

    else:
        imgs = [input_path]

    # check tile size

    if tile_size != 256:
        print('WARNING: setting tile size to 256')
        tile_size = 256

    # load model
    input_shape = (tile_size, tile_size, 4)
    model = Attention_ResUNet(tile_size, tile_size, 4, 1)
    model.compile()
    model.load_weights(model_path)


    for img_path in imgs:
        try:
            start_time = time.time()
            predicted = []

            img = imageio.imread(img_path)
            #img = img[:, :, :3]
            img = img/255
            img = img.astype(np.float32)

            # we do not need to patchify image if image is too small to be split
            # into patches - assume that img width == img height
            do_patchify = True if tile_size < img.shape[0] else False

            if do_patchify:
                patches = patchify(img, tile_size, margin)
                
            else:
                patches = [img]

            for i in [784, 16, 8, 4, 2, 1]:
                if len(patches) % i == 0:
                    bs = i
                    print('batch size = ', i)
                    break

            # perform prediction
            for i in range(0, len(patches), bs):
                batch = np.array(patches[i:i+bs])
                batch = batch.reshape((bs, *input_shape))
                out = model.predict(batch, verbose=0)
                for o in out:
                    # Check the shape of o
                    if o.shape != (tile_size, tile_size, 1):
                        raise ValueError(f"Unexpected shape of o: {o.shape}. Expected {(tile_size, tile_size, 1)}")
                    o = o[:, :, 0]  # Assuming you want to keep the channel dimension
                    predicted.append(o)


            if do_patchify:
                out = unpatchify(img.shape[:2], predicted, tile_size, margin)
            else:
                out = predicted[0]
            out[out < threshold] = 0

            end_time = time.time()  # End time measurement
            print(f"Time taken to process {img_path}: {end_time - start_time:.3f} seconds")

            # write image
            img_name = os.path.basename(img_path).split('.')[0]
            InutFileWithKnownExtent = gdal.Open(img_path)
            raster_name = os.path.join(out_path_binary,'{}.{}'.format(img_name, img_type))
            binary_prediction = np.round(out)
            filtered_image = remove_small_groups(binary_prediction, 200) # 200 pixels on 0.5 m resolution is about 50 square meters.
            filtered_image = filtered_image * 10 # the culverts will later be burned 10 meter into the DEM.
            write_gtiff(filtered_image, InutFileWithKnownExtent, raster_name)
    
            # convert raster to vector polygon
            temp_vector_name = os.path.join(out_path_vector,'{}.{}'.format(img_name, 'shp'))
            try:
                raster_to_polygon(raster_name, temp_vector_name)
            except:
                print('conversion from raster to vector failed. Perhaps due to empty raster')
        except:
            print('failed to run prediction')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='Run inference on given '
                                   'image(s)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='Path to image or folder')
    parser.add_argument('model_path')
    parser.add_argument('out_path_binary', help='Path to output binary folder')
    parser.add_argument('out_path_vector', help='Path to output vector folder')
    parser.add_argument('--img_type', help='Output image file ending',
                        default='tif')
    parser.add_argument('--tile_size', help='Tile size', type=int,
                        default=256)
    parser.add_argument('--margin', help='Margin', type=int, default=40)
    parser.add_argument('--threshold', help='Decision threshold', type=float,
                        default=0.5)
 
    args = vars(parser.parse_args())
    main(**args)