DEBUG=True
def log(s):
    if DEBUG:
        print(s)

import sys, os
import torch
# import visdom
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from scipy.ndimage.interpolation import zoom

from ptsemseg.utils import *

# try:
#     import pydensecrf.densecrf as dcrf
# except:
#     print(
#         "Failed to import pydensecrf,\
#            CRF post-processing will not work"
#     )

def imgToTensor(img, device):
    # log('img before to tensor: {}'.format(img.shape))
    img = np.stack([img], axis=0)
    img = np.stack([img], axis=0)
    img = torch.from_numpy(img).float()
    # log('img after to tensor: {}'.format(img.size()))
    img = img.to(device)

    return img

# decode predicted result
def decoder(tensor):
    # log('tensor before to pred img: {}'.format(tensor.size()))
    pred = tensor.data.cpu().numpy()
    # log('pred img after to img: {}'.format(pred.shape))
    ret = (pred[0][0]<pred[0][1]).astype('int')
    # log('final return label map: {}'.format(ret.shape))
    return ret



def test(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file_name = os.path.split(args.model_path)[1]
    model_name = {'arch':model_file_name.split('/')[-1].split('_')[0]}
    log('model_name: {}'.format(model_name))

    # Setup image
    print("Read Input Image from : {}".format(args.img_path))
    img = loadtiff3d(args.img_path)
    log('img before zoom has shape: {}'.format(img.shape))
    ratioX = max(img.shape[0], 160) / img.shape[0]
    ratioY = max(img.shape[1], 160) / img.shape[1]
    ratioZ = max(img.shape[2], 8) / img.shape[2]
    img = zoom(img, (ratioX, ratioY, ratioZ))
    log('img after zoom has shape: {}'.format(img.shape))
    shapeX = img.shape[0]
    shapeY = img.shape[1]
    shapeZ = img.shape[2]

    # Setup Model
    log('set up model')
    n_classes = 2
    log('model name: {}'.format(model_name))
    model = get_model(model_name, n_classes, version=args.dataset)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    log('apply 160*160*8 network to the img')
    stack_alongX = None
    stack_alongY = None
    stack_alongZ = None
    overlapX = 0
    overlapY = 0
    overlapZ = 0
    x = 0
    y = 0
    z = 0
    while x < shapeX:
        while y < shapeY:
            while z < shapeZ:
                patch = img[x:x+160, y:y+160, z:z+8]
                patch = imgToTensor(patch, device)
                pred = model(patch)
                pred = decoder(pred)
                if overlapZ:
                    pred = pred[:,:,overlapZ:]
                    stack_alongZ = np.concatenate((stack_alongZ, pred), axis=2)
                    overlapZ = 0
                else:
                    if stack_alongZ is None:
                        stack_alongZ = pred
                    else:
                        stack_alongZ = np.concatenate((stack_alongZ, pred), axis=2)
                log('===>z ({}/{}) loop: stack_alongZ shape: {}'.format(z, shapeZ, stack_alongZ.shape))
                # residual
                if z+8 >= shapeZ:
                    overlapZ = 8 - (shapeZ - z - 8)
                    z = shapeZ - 8
                    log('overlapZ: {}'.format(overlapZ))
                    continue
                z += 8

            if overlapY:
                stack_alongZ = stack_alongZ[:,overlapY:,:]
                stack_alongY = np.concatenate((stack_alongY, stack_alongZ), axis=1)
                overlapY = 0
            else:
                if stack_alongY is None:
                    stack_alongY = stack_alongZ
                else:
                    stack_alongY = np.concatenate((stack_alongY, stack_alongZ), axis=1)
            log('==>y ({}/{}) loop: stack_alongY shape: {}'.format(y, shapeY, stack_alongY.shape))
            stack_alongZ = None
            # residual
            if y+160 >= shapeY:
                overlapY = 160 - (shapeY - y - 160)
                y = shapeY - 160
                log('overlapY: {}'.format(overlapY))
                continue
            y += 160

        if overlapX:
            stack_alongY = stack_alongY[overlapX:, :, :]
            stack_alongX = np.concatenate((stack_alongX, stack_alongY), axis=0)
            overlapX = 0
        else:
            if stack_alongX is None:
                stack_alongX = stack_alongY
            else:
                stack_alongX = np.concatenate((stack_alongX, stack_alongY), axis=0)
        log('=>x ({}/{}) loop: stack_alongX shape: {}'.format(x, shapeX, stack_alongX.shape))
        stack_alongY = None
        # residual
        if x+160 >= shapeX:
            overlapX = 160 - (shapeX - x - 160)
            x= shapeX - 160
            log('overlapX: {}'.format(overlapX))
            continue
        x += 160
    log('save prediction in path: {}'.format(args.out_path))
    writetiff3d(args.out_path, stack_alongX*255)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="/Users/wonh/y3s2/isbi/segment_3D/runs/fcn3d_fly/65461/fcn3dnet_flyJanelia_best_model.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="flyJanelia",
        help="Dataset to use ['flyJanelia, pascal, camvid, ade20k etc']",
    )

    # parser.add_argument(
    #     "--img_norm",
    #     dest="img_norm",
    #     action="store_true",
    #     help="Enable input image scales normalization [0, 1] \
    #                           | True by default",
    # )
    # parser.add_argument(
    #     "--no-img_norm",
    #     dest="img_norm",
    #     action="store_false",
    #     help="Disable input image scales normalization [0, 1] |\
    #                           True by default",
    # )
    # parser.set_defaults(img_norm=True)

    # parser.add_argument(
    #     "--dcrf",
    #     dest="dcrf",
    #     action="store_true",
    #     help="Enable DenseCRF based post-processing | \
    #                           False by default",
    # )
    # parser.add_argument(
    #     "--no-dcrf",
    #     dest="dcrf",
    #     action="store_false",
    #     help="Disable DenseCRF based post-processing | \
    #                           False by default",
    # )
    # parser.set_defaults(dcrf=False)

    parser.add_argument(
        "--img_path", nargs="?", type=str, default=None, help="Path of the input image"
    )
    parser.add_argument(
        "--out_path",
        nargs="?",
        type=str,
        default=None,
        help="Path of the output segmap",
    )
    args = parser.parse_args()
    test(args)
