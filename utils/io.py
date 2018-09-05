import os
import numpy as np
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
                    # cells[2:5] = [c-1 for c in cells[2:5]]
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
    #a.close()

    return out

def writetiff2d(filepath, block):
    import tifffile as tiff
    try:
        os.remove(filepath)
    except OSError:
        pass
    with tiff.TiffWriter(filepath, bigtiff=False) as tif:
        # for y in range(block.shape[1]):
        #     tif.save((np.rot90(block[:, y])), compress=0)
        tif.save(np.rot90(block))
def writetiff3d(filepath, block):
    # from libtiff import TIFF
    import tifffile as tiff
    try:
        os.remove(filepath)
    except OSError:
        pass
    # tiff = TIFF.open(filepath, mode='w')
    # block = np.swapaxes(block, 0, 1)
    with tiff.TiffWriter(filepath, bigtiff=False) as tif:
        for z in range(block.shape[2]):
            # tif.save(np.flipud(block[:,:,z]), compress = 0)
            tif.save((np.rot90(block[:, :, z])), compress=0)
    # for z in range(block.shape[2]):
    #     tiff.write_image(np.flipud(block[:, :, z]), compression=None)
    # tiff.close()