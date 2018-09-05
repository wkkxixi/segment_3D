from skimage.draw import line_aa
from .utils.io import *


goldenfolderpath = '/home/heng/Gold166-JSON-meit/'


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




