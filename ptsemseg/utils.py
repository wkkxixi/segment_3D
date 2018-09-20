"""
Misc Utility functions
"""
import os
import time
import logging
import datetime
import numpy as np

from collections import OrderedDict
from skimage.draw import line_aa


def loadswc(filepath):
    '''
    Load swc file as a N X 7 numpy array
    '''
    swc = []
    with open(filepath) as f:
        lines = f.read().split("\n")
        for l in lines:
            if not l.startswith('#'):
                cells = l.split(' ')
                if len(cells) ==7:
                    cells = [float(c) for c in cells]
                    swc.append(cells)
    return np.array(swc)

def loadtiff3d(filepath):
    """Load a tiff file into 3D numpy array"""

    import tifffile as tiff
    a = tiff.imread(filepath)

    stack = []
    for sample in a:
        stack.append(np.rot90(np.fliplr(np.flipud(sample))))
    out = np.dstack(stack)

    return out

def writetiff2d(filepath, block):
    import tifffile as tiff
    try:
        os.remove(filepath)
    except OSError:
        pass
    with tiff.TiffWriter(filepath, bigtiff=False) as tif:
        tif.save(np.rot90(block))

def writetiff3d(filepath, block):
    import tifffile as tiff

    try:
        os.remove(filepath)
    except OSError:
        pass

    with tiff.TiffWriter(filepath, bigtiff=False) as tif:
        for z in range(block.shape[2]):
            saved_block = np.rot90(block[:, :, z])
            tif.save(saved_block.astype('uint8'), compress=0)
'''
Generate a file including all data's information (name; x; y; z)
'''
def info_generator(folder):
    datainfo_folder = folder + '/datainfo'
    if not os.path.isdir(os.path.join(os.getcwd(), datainfo_folder)):
        os.mkdir(folder + '/datainfo')
    else:
        print(datainfo_folder + ' already exists')
    img_folder = folder +'/images/'
    with open(datainfo_folder +'/datainfo.txt', 'w') as f:
        for filename in os.listdir(img_folder):
            img = loadtiff3d(img_folder+filename)
            x = img.shape[0]
            y = img.shape[1]
            z = img.shape[2]
            f.write(filename + ' ' + str(x) + ' ' + str(y) + ' ' + str(z) + '\n')

'''
Reconstruct 3d tiff image based on its swc annotation result
'''
def swc2tif(filepath, tif_filepath, output_path):
    img = loadtiff3d(tif_filepath)
    x_shape = img.shape[0]
    y_shape = img.shape[1]
    z_shape = img.shape[2]
    swc = loadswc(filepath)
    output = np.zeros_like(img)
    for row in range(swc.shape[0]):
        x = swc[row][2]
        y = swc[row][3]
        z = swc[row][4]
        r = swc[row][-2]
        p = swc[row][-1]
        output[int(max(0, x-r)):int(min(x_shape, x+r)), int(max(0, y-r)):int(min(y_shape, y+r)), int(max(0, z-r)):int(min(z_shape, z+r))] = 255
    writetiff3d(output_path, output)

'''
Reconstruction for all swc file in the specified folder
'''
def swc2tif_operation(folder):
    label_folder = folder + '/labels'
    if not os.path.isdir(os.path.join(os.getcwd(), label_folder)):
        os.mkdir(label_folder)
    else:
        print(label_folder + ' already exists')
    with open(folder + '/datainfo/datainfo.txt') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for c in content:
        filename = (c.split()[0]).split('.tif')[0]
        print(filename + '.swc is on processing')
        file_path = folder + '/ground_truth/' + filename + '.swc'
        converted_path = label_folder + '/' + filename + '.tif'
        tif_path = folder + '/images/' + filename + '.tif'
        swc2tif(file_path, tif_path, converted_path)


# filepath - the absolute path to the tif file
def tif2DProjection(filepath):
    tiff_folder = filepath.split('.tif')[0] + '_tif_2D_projection/'
    if not os.path.exists(tiff_folder):
        os.makedirs(tiff_folder)

    img = loadtiff3d(filepath)
    # print(np.max(img, axis=0).shape)
    yz = np.max(img, axis=0)
    writetiff2d(tiff_folder+ filepath.split('.tif')[0].split('/')[-1] + '_yz.tif', yz)
    xz = np.max(img, axis=1)
    writetiff2d(tiff_folder+ filepath.split('.tif')[0].split('/')[-1] + '_xz.tif', xz)
    xy = np.max(img, axis=2)
    writetiff2d(tiff_folder+ filepath.split('.tif')[0].split('/')[-1] + '_xy.tif', xy)


def tif2dprojection_single():
    file_path = '/home/heng/Gold166-JSON-meit/FLY-JANELIA/8.tif'
    tif2DProjection(file_path)

def tif2dProjection_groupoperation():
    count = 0
    goldenfolderpath = '/home/heng/Gold166-JSON-meit/'
    with open(goldenfolderpath + 'jsoninfo/detailedinfo.txt') as f:
        lines = f.readlines()
        for item in lines:

            if item.__contains__('.'):
                filename = item.split('\t')[0]
                if filename.split('/')[0] != 'FLY-JANELIA':
                    continue
                filepath = goldenfolderpath + filename
                print(str(count) + ': ' + filename + ' generating 2D projection')
                tif2DProjection(filepath)
                count += 1


def updatePlane(swc, row, plane, axis1_coordinate, axis2_coordinate, axis1_max, axis2_max, r, p, axis1, axis2):
    plane[max(0, int(axis1_coordinate - r)):min(int(axis1_max), int(axis1_coordinate + r)), max(0, int(axis2_coordinate - r)):min(int(axis2_max), int(axis2_coordinate + r))] = 1
    for search_parent in range(0, row):
        if swc[search_parent][0] == p:
            rr, cc, val = line_aa(int(axis1_coordinate), int(axis2_coordinate), int(swc[search_parent][axis1]), int(swc[search_parent][axis2]))
            plane[rr, cc] = val * 255
            break
    return plane


def swc2DProjection(filepath, tif_filepath):
    img = loadtiff3d(tif_filepath)

    swc = loadswc(filepath)
    print(swc.shape)
    print(img.shape)
    x_max = np.max(swc[:, 2])
    y_max = np.max(swc[:, 3])
    z_max = np.max(swc[:, 4])
    print((x_max, y_max, z_max))
    xy_plane = np.zeros(shape=(img.shape[0], img.shape[1]))
    yz_plane = np.zeros(shape=(img.shape[1], img.shape[2]))
    xz_plane = np.zeros(shape=(img.shape[0], img.shape[2]))
    for row in range(swc.shape[0]):
        x = swc[row][2]
        y = swc[row][3]
        z = swc[row][4]
        r = swc[row][-2]
        p = swc[row][-1]
        xy_plane = updatePlane(swc, row, xy_plane, x, y, x_max, y_max, r, p, 2, 3)
        yz_plane = updatePlane(swc, row, yz_plane, y, z, y_max, z_max, r, p, 3, 4)
        xz_plane = updatePlane(swc, row, xz_plane, x, z, x_max, z_max, r, p, 2, 4)
    swc_2d_folder = filepath.split('.swc')[0] + '_swc_2D_projection/'
    if not os.path.exists(swc_2d_folder):
        os.makedirs(swc_2d_folder)

    writetiff2d(swc_2d_folder + filepath.split('.swc')[0].split('/')[-1] + '_xy.tif', (xy_plane>0)*255)
    writetiff2d(swc_2d_folder + filepath.split('.swc')[0].split('/')[-1] + '_yz.tif', (yz_plane>0)*255)
    writetiff2d(swc_2d_folder + filepath.split('.swc')[0].split('/')[-1] + '_xz.tif', (xz_plane>0)*255)



def swc2DProjection_single():
    swc_file_path = '/home/heng/Gold166-JSON-meit/FLY-JANELIA/23.swc'
    tif_file_path = '/home/heng/Gold166-JSON-meit/FLY-JANELIA/23.tif'
    swc2DProjection(swc_file_path, tif_file_path)


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value

    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state

    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_logger(logdir):
    logger = logging.getLogger('ptsemseg')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger
