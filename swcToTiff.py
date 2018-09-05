import time
from .utils.io import *


goldenfolderpath = '/home/heng/Gold166-JSON-meit/'


def swc2tif(filepath, tif_filepath):
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
    writetiff3d(filepath.split('.swc')[0] + '_swc2tif.tif', output)

def swc2tif_single():
    swc_file_path = '/home/heng/Gold166-JSON-meit/FLY-JANELIA/23.swc'
    tif_file_path = '/home/heng/Gold166-JSON-meit/FLY-JANELIA/23.tif'
    a = time.time()
    swc2tif(swc_file_path, tif_file_path)
    b = time.time()
    print(b-a)

def swc2tif_groupoperation():
    count = 0
    with open(goldenfolderpath + 'jsoninfo/detailedinfo.txt') as f:
        lines = f.readlines()
        for item in lines:

            if item.__contains__('.'):
                filename = item.split('\t')[0]
                if filename.split('/')[0] != 'FLY-JANELIA':
                    continue
                filepath = goldenfolderpath + filename
                print(str(count) + ': ' + filename + ' swc->tif')
                swc2tif(filepath.split('.tif')[0]+'.swc',filepath)
                count += 1





