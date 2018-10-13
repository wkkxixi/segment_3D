import torch
#
# x = torch.randn(2, 3)
# print(torch.cat((x,x,x), 0).size())
# print(torch.cat((x,x,x), 1).size())
# def foo():
#     assert 0 == 1
#
# foo()

import math
import numbers
# import random
import numpy as np

from PIL import Image, ImageOps

# zz = np.random.random((3,3))
# mask = zz.copy()
# mask = Image.fromarray(mask, mode="L")
# print(zz)
# img = Image.fromarray(zz, mode="RGB")
# print(img.size)
# print(np.array(img))
# print(np.array(mask, dtype=np.uint8))
# img_size = ('same', 'same')
# img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
# print(img_size)
# with open('/Users/wonh/Desktop/flyJanelia/datainfo/datainfo.txt') as f:
#     content = f.readlines()
# # you may also want to remove whitespace characters like `\n` at the end of each line
# content = [x.strip() for x in content]
# for c in content:
#     print(c.split()[0])

# im = Image.open('/Users/wonh/Desktop/flyJanelia/images/1.tif')
# print(np.array(im).shape)

from PIL import Image
from torchvision.transforms import ToTensor
from ptsemseg.utils import *
import numpy as np
from scipy.ndimage.interpolation import zoom
from os.path import join as pjoin
# from random import randint
from torchvision import transforms

# how to keep the image same after tensor
def np_tensor_np():
    image = loadtiff3d('/Users/wonh/Desktop/flyJanelia/images/1.tif') # x, y, z
    image = ToTensor()(image) # z, x, y
    image = image.numpy()*255
    image = image.astype('uint8') # important
    image = np.transpose(image, (1,2,0)) # z, x, y => x, y, z

def os_expand_user():
    root = '/Users/wonh/Desktop/flyJanelia/'
    path = os.path.expanduser(root)


def getInfoLists(root):
    nameList = []
    xList = []
    yList = []
    zList = []
    with open(root + '/datainfo/datainfo.txt') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for c in content:
        nameList.append(c.split()[0])
        xList.append(c.split()[1])
        yList.append(c.split()[2])
        zList.append(c.split()[3])
    return nameList, xList, yList, zList

# print(pjoin('/Users/wonh/Desktop/flyJanelia', 'images', '1.tif'))

def resize_small_img(filepath):
    img = loadtiff3d(filepath)
    x = max(img.shape[0], 160)/img.shape[0]
    y = max(img.shape[1], 160)/img.shape[1]
    z = max(img.shape[2], 8)/img.shape[2]
    print((x,y,z))
    img = zoom(img, (x,y,z))

    x = randint(0, img.shape[0]-160) if x == 1 else 0
    y = randint(0, img.shape[1]-160) if y == 1 else 0
    z = randint(0, img.shape[2] - 8) if z == 1 else 0
    print((x,y,z))
    img = img[x:x+160, y:y+160, z:z+8]
    print(img.shape)

    # img = np.resize(img, (x,y,z))
    writetiff3d('/Users/wonh/Desktop/test2.tif', img)

# resize_small_img('/Users/wonh/Desktop/flyJanelia/images/12.tif')

def transforms_tensor():
    tf = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])])
    img = loadtiff3d('/Users/wonh/Desktop/flyJanelia/labels/12.tif')

    img = tf(img)
    img = img.numpy() * 255
    img = img.astype('uint8')  # important
    img = np.transpose(img, (1, 2, 0))  # z, x, y => x, y, z
    writetiff3d('/Users/wonh/Desktop/test3.tif', img)
    print(img)

# transforms_tensor()
def tensor_array():
    a = np.ndarray([1,2,3])
    a = ToTensor()(a)
    print(a)
    a.view(1,1,2,3)
    print(a)
    # log('')

# tensor_array()

def usage_of_in():
    if 'a' in 'apple':
        print('yes')
# usage_of_in()

def usage_of_random_shuffle():
    from random import shuffle
    a = [1,2,3,4]
    b = a[:-1]
    shuffle(a)
    print(a)
    print(b)
    shuffle(b)
    print(b)
    print(a)

# usage_of_random_shuffle()

def usage_of_large_brackets():
    file_paths = 'file path'
    val_case_index = [1,2,3]
    case_index = [4,5,6]

    return {'file_paths': file_paths, 'val_case_index': val_case_index, 'case_index': case_index}

# a = usage_of_large_brackets()
# print(a)
# print(a['file_paths'])

def rotate3d(degree):
    degree += 40
    print('degree is {}'.format(degree))

def learn_yaml():
    import yaml
    with open('/Users/wonh/y3s2/isbi/segment_3D/configs/fcn3d_fly.yml') as fp:
        cfg = yaml.load(fp)
    dict = cfg['training'].get('augmentations', None)
    print(dict)
    train = cfg['training']
    for a, b in dict.items():
        if a == 'rotate3d':
            rotate3d(b)
        print('a: {}  b: {}'.format(a, b))
    # a = cfg['training']['patch_size']
    # a = a.split(',')
    # a = [int(tmp) for tmp in a]
    # print(a)
    # print(cfg['training'].get('patch_size'))
    # print(cfg['training'].get('patch_size', None))
    # print(cfg['training']['patch_size'].items())
    # patch_size = [para for axis, para in cfg['training']['patch_size'].items()]
    # print(patch_size)
    # if cfg['model']['arch'] == 'fcn3dnet':
    #     print('yeah')

    # print(cfg.get('data'))
# learn_yaml( )

 # def learn_seed():
 #     import random
 #     random.seed(1)
 #     print(random.randint(1, 10))
def learn_seed():
    import random
    random.seed(1)
    a = random.randint(1,10)
    print(a)
# learn_seed()

def learn_pjoin():
    root = '/Users/wonh/Desktop/flyJanelia'
    a = pjoin(root, 'images')
    print(a)
# learn_pjoin()

def learn_view():
    a = np.ndarray([3,3])
    # print(a.shape)
    a = ToTensor()(a)
    print(a)
    # a.view(1,3,3)
    # print(a)

# learn_view()

def learn_stack():
    a = np.ones(shape=(3,3,3))
    # print(a)
    b = np.ones(shape=(3, 3, 3))*99
    # a = np.stack([a], axis=3)
    a = np.stack([a,b], axis=0)
    print(a[1,:,:])
# learn_stack()

def learn_astype():
    lbl = np.zeros(shape=(3,3))
    lbl_background = (lbl == 0).astype('int')
    print(lbl_background)
    lbl = torch.from_numpy(lbl_background).long()
    print(lbl)
# learn_astype()

def test_label():
    lbl = loadtiff3d('/Users/wonh/Desktop/flyJanelia/labels/12.tif')
    lbl_background = (lbl == 0).astype('int')
    lbl_foreground = (lbl != 0).astype('int')
    lbl = np.stack([lbl_foreground, lbl_background], axis=3)
    print(lbl.shape)
    # print('lbl foreground sum: {} background sum: {}'.format(np.sum(lbl_background), np.sum(lbl_foreground)))
    # lbl = torch.from_numpy(lbl).long()

# test_label()

def test_matrix_add():
    a = np.asarray([1,2,3])
    a = a + 1
    print(a)
# test_matrix_add()

def test_zoom():
    a = np.ones(shape=(3,3,3))
    a[2][2][2] = 99
    print(a)
    a = torch.from_numpy(a).long()
    print(a.max())

# test_zoom()

def learn_makenp():
    from tensorboardX.summary import make_np
    scalar = [0.3]
    print(scalar)
    scalar = make_np(scalar)
    print(scalar.squeeze().ndim) # scalar.squeeze().ndim == 0

# learn_makenp()

def test_scope():
    for x in range(0,5,2):
        print('loop x: {}'.format(x))
    print(x)

# test_scope()

def test_max_along_axis():
    a = np.asarray([[1,2,3,],[4,5,6]])
    b = np.asarray([[3,2,1],[6,5,4]])
    c = np.stack((a, b), axis=0)
    print(c.shape)
    f = (c[0]>=c[1]).astype('int')
    print(f)
    # c = np.stack((a,b), axis=0)
    # print(c)
    # print(c.max(0))
# test_max_along_axis()

def test_indices():
    a = np.asarray([[1,2,3], [4,5,6]])
    print(a[:,2:])
# test_indices()

def test_range_loop():
    a = 0
    while a < 7:
        print(a)
        if (7-a)%6 != 0:
            print('!=0')
            a = 1
            continue
        a += 6

# test_range_loop()

def test_time():
    import time
    start = time.time()
    # for i in range(10000):
    #     a = 3
    time.sleep(3663)
    end = time.time()
    elapsed = end - start
    print(elapsed)
    hour = int(elapsed / 3600)
    left = elapsed % 3600
    minute = int(left / 60)
    seconds = left % 60
    print('The total time is: {} h {} m {} s'.format(hour, minute, seconds))
# test_time()

def learn_tensor():
    import torch
    a = np.zeros(shape=(2,2,2))
    a = torch.from_numpy(a)
    b = np.ones(shape=(2,2,2))
    b = torch.from_numpy(b)
    c = a+b
    c = c/2
    print(c)
# learn_tensor()


def learn_exit():
    print('start')
    exit(1)
    print('end')

# learn_exit()

def learn_operator():
    a = 99
    print(a//2)
    print(a/2)
# learn_operator()

def learn_pad():
    import torch.nn.functional as F
    t3d = torch.empty(3, 3, 1)
    print(t3d)
    out = F.pad(t3d, [1,1], "constant", 0)
    print(out)
# learn_pad()

def learn_raise_exception():
    if 1 == 1:
        raise Exception('THe image...')
    print('should not print this line')

# learn_raise_exception()

def learn_flatten():
    a = np.zeros(shape=(3,3,3))
    a = a.flatten()
    print(a)
# learn_flatten()

def learn_histogram2d():
    import matplotlib.pyplot as plt
    xedges = [0, 1, 3, 5]
    yedges = [0, 2, 3, 4, 6]
    # x = np.random.normal(2, 1, 100) # mean, dev, size
    # y = np.random.normal(1, 1, 100)
    x = np.asarray([0,2,3,4,5])
    y = np.asarray([0,2,3,4,5])
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    print(H)
    H = H.T
    print(H)
    print(H.shape, xedges.shape, yedges.shape)
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(132, title='pcolormesh: actual edges',aspect='equal')
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, H)
    plt.show()
# learn_histogram2d()

def swc_to_tiff():
    from ptsemseg.utils import swc2tif_operation
    folder = '/home/heng/Desktop/Research/isbi/flyJanelia'
    swc2tif_operation(folder)
# swc_to_tiff()

def test_extend():
    a = []
    a.extend([1,2,3])
    print(a)
    a.extend([i for i in range(4,7)])
    print(a)

# test_extend()

def test_array():
    a = np.ones(shape=(3,3,3))
    print(a[0][0][0])
    print(a[0,0,0])
# test_array()

def test_skfmm():
    import skfmm
    import numpy as np
    phi = np.ones((3, 3))
    phi[1, 1] = -1
    print(skfmm.distance(phi))
# test_skfmm()


# def test_dt():
#     from ptsemseg.utils import make_from_swc
#     make_from_swc()
# test_dt()
from ptsemseg.utils import *
def test_dt():
    folder = '/Users/wonh/Desktop/flyJanelia-'
    label_folder_name = 'labels_dt'
    swc2tif_operation(folder, label_folder_name, mode=1)
# test_dt()

def test_flip():
    img = loadtiff3d('/Users/wonh/Desktop/flyJanelia/images/2.tif')
    img1 = np.flipud(img) # up down
    writetiff3d('/Users/wonh/Desktop/flyJanelia/aug/2_0.tif', img1)
    img2 = np.fliplr(img) # left right
    writetiff3d('/Users/wonh/Desktop/flyJanelia/aug/2_1.tif', img2)
    img3 = np.flip(img, 2) # ?
    writetiff3d('/Users/wonh/Desktop/flyJanelia/aug/2_2.tif', img3)
    img3 = None
    writetiff3d('/Users/wonh/Desktop/flyJanelia/aug/2.tif', img)
# test_flip()


def test_random():
    import random
    print(random.random())
# test_random()

def test_rotate():
    from scipy.ndimage import rotate
    img = loadtiff3d('/Users/wonh/Desktop/flyJanelia/images/2.tif')
    # img1 =
    writetiff3d('/Users/wonh/Desktop/flyJanelia/aug/2_10_90.tif', rotate(img, 90, axes=[1,0]))
    writetiff3d('/Users/wonh/Desktop/flyJanelia/aug/2_10_90_original.tif', img)
# test_rotate()

def test_randomint():
    import random
    print(random.sample(range(0, 3), 2))
# test_randomint()

def test_array():
    a = np.asarray([1,2,3])
    b = np.asarray([True, False, True])
    print(a[b])
# test_array()
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# def dataset_generator(folder):
#     from os.path import join as pjoin
#     import shutil
#     import fnmatch
#     # img_folder_path = pjoin(folder, 'images')
#     # swc_folder_path = pjoin(folder, 'ground_truth')
#     # label_folder_path = pjoin(folder, 'labels')
#     # new_folder_maker(img_folder_path)
#     # new_folder_maker(swc_folder_path)
#     # new_folder_maker(label_folder_path)
#
#     for filename in os.listdir(folder):
#         if fnmatch.fnmatch(filename, '*.tif') and is_number(filename.split('.tif')[-2]):
#             print(filename)
# # dataset_generator('/Users/wonh/Desktop/FLY-TAIWAN')

def test_dataset_generator(folder):
    dataset_generator(folder)
# info_generator('/Users/wonh/Desktop/FLY-TAIWAN')
# test_dataset_generator('/home/heng/Desktop/Research/isbi/fly-dataset/janeliafly2')

# test_dataset_generator('/home/heng/Desktop/Research/isbi/fly-dataset/utokyofly')
def test_out_of_bound():
    a = np.asarray([1,2,3])
    a[-9:10] = 0
    print(a)
# test_out_of_bound()

def test_list(folder):
    from glob import glob
    for filename in os.listdir(folder):
        if filename.__contains__('fly'):
            print(filename)
            sub_dataset = pjoin(folder, filename)
            imgs = pjoin(sub_dataset, 'images')
            print(imgs)
            data_paths = glob(imgs + '/*.tif')
            print(data_paths)

# test_list('/Users/wonh/Desktop/fly-dataset/')

def test_empty_list():
    a = []
    print(a)
    a.extend([2,3,4])
    print(a)
    # stack = ['a', 'b']
    # stack.extend(['d', 'g'])
    # print(stack)
#test_empty_list()

def test_randint():
    from random import randint
    x = 1
    x = randint(0, 100) if x == 1 else 0
    print(x)

# test_randint()

def test_for():
    with open('/home/heng/Desktop/Research/isbi/log.txt', 'a') as f:
        f.writelines('a')
        f.writelines('b')
        f.writelines('c')
# test_for()

def test_meta(folder):
    from ptsemseg.utils import dataset_meta
    dataset_meta(folder)

# test_meta('/home/heng/Desktop/Research/isbi/fly-dataset/utokyofly')

def test_replace():
    a = 'good/images/2.tif'
    b = a.replace('images', 'pred')
    print(a)
    print(b)
    b = b[:-2]
    print(a)
    print(b)
# test_replace()

def test_isdir():
    pred_folder = '/home/heng/Desktop/Research/isbi/fly-dataset/flyJanelia/pred_testtest22'
    print(os.path.join(os.getcwd(), pred_folder))
    print(os.getcwd())
    if not os.path.isdir(pred_folder):
        os.mkdir(pred_folder)
    else:
        print(pred_folder + ' already exists')
# test_isdir()

def test_with_open():
    with open('/home/heng/Research/isbi/log_test3.txt', 'w') as f:
        pass
# test_with_open()

def test_add_to_txt():
    with open('/home/heng/Research/isbi/log_test2.txt', 'a') as f:
        f.write('a\n')
# test_add_to_txt()

def test_cwd():
    print(os.getcwd())
# test_cwd()

def test_dictionary():
    a = {'dict':{'a':1, 'b':2, 'c':3}}
    a_dict = a['dict']
    print(a_dict)
    b = {'a':99}
    a_dict.update(b)
    print(a_dict)
    print(a)
# test_dictionary()



def test_load_model(model_path=None):

    teacher_model = torch.load("/home/heng/Research/segment_3D/runs/final_1/65188/unet3dregTeacher_flyDataset_model_best.pkl")
    student_model = torch.load("/home/heng/Research/segment_3D/runs/final_2/9424/unet3dregSmartStudentRes_flyDataset_model_best.pkl")
    teacher_model_state = teacher_model['model_state']
    student_model_state = student_model['model_state']
    print('length of teacher {} \n length of student {}'.format(len(teacher_model_state), len(student_model_state)))
    # print(teacher_model_state)
    # print(student_model_state)
    for name, param in teacher_model_state.items():
        print(name + ': ' + str(param.size()))

    # teacher_model = torch.load(model_path)
    # # # pre_trained_model = torch.load("Path to the .pth file")
    # # # print(len(model_state))
    # student_model = torch.load('/home/heng/Research/segment_3D/runs/student_unet3d_regression_4/65046/unet3dregStudent_flyDataset_model_4.pkl')
    # student_model_state = student_model['model_state']
    # a = student_model_state.copy()
    # teacher_model_state = teacher_model['model_state']
    # print('teacher: {} student: {}'.format(len(teacher_model_state), len(student_model_state)))
    #
    #
    # pretrained_dict = {k: v for k, v in teacher_model_state.items() if k in student_model_state}
    # # 2. overwrite entries in the existing state dict
    # student_model_state.update(pretrained_dict)
    # b = student_model_state
    # # shared_items = {k: x[k] for k in x if k in y and torch.equal(x[k], z[k])}
    #
    # # print(len(shared_items))
    # # 3. load the new state dict
    # state = {
    #     "epoch": 100,
    #     "model_state": student_model_state,
    #     "optimizer_state": student_model['optimizer_state'],
    #     "scheduler_state": student_model['scheduler_state']
    # }
    #
    # torch.save(state, '/home/heng/Research/isbi/test.pkl')
    #
    # updated_student_model = torch.load('/home/heng/Research/isbi/test.pkl')
    # updated_model_state = updated_student_model['model_state']
    # print(len(updated_model_state))
    # print(len(pretrained_dict))
    # # for name, param in updated_model_state.items():
    #     # print(name + ': ' + str(param.type()))
    # x = updated_model_state
    # y = pretrained_dict
    # z = a
    # shared_items = {k: x[k] for k in x if k in y and torch.equal(x[k], y[k])}
    #
    # print(len(shared_items))
    # shared_items = {k: x[k] for k in x if k in z and torch.equal(x[k], z[k])}
    #
    # print(len(shared_items))
# test_load_model(model_path=None)
# test_load_model('/home/heng/Research/segment_3D/runs/teacher_unet3d_regression_4/89050/unet3dregTeacher_flyDataset_model_4.pkl')
# test_load_model('/home/heng/Research/segment_3D/runs/student_unet3d_regression_4/65046/unet3dregStudent_flyDataset_model_4.pkl')
def abcd():
    a = 1
    b = 2
    c = 3
    return a, b, c

def test_return_multiple_value():
    a = abcd()
    a = a[0]
    print(a)
# test_return_multiple_value()

def drawplot(xpoints, ypoints):
    import matplotlib.pyplot as plt
    # plotting the line 1 points
    plt.plot(xpoints, ypoints, label="line 1")

    # naming the x axis
    plt.xlabel('Recall - axis')
    # naming the y axis
    plt.ylabel('Precision - axis')
    # giving a title to my graph
    plt.title('Precision-recall for different threshold')

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.show()

def test_vstack():
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 4])
    c = np.vstack((a, b))
    print('a shape: {} c shape: {}'.format(a.shape, c.shape))
    d = np.mean(c, axis=0)
    print(d)
    print('d shape: {}'.format(d.shape))
    drawplot(d, d)
# test_vstack()

def test_numpy_append():
    # a = []
    # b = np.append(a, [1])
    # b = np.append(b, 2)
    # print(a)
    # print(b)
    a = list()
    a.append(1)
    a.append(2)
    a = np.mean(a, axis=0)
    print(a)
# test_numpy_append()

def create_new_ground_truth():
    dataset_generator('/home/heng/Research/isbi/fly-dataset/flyJanelia')
    # dataset_meta('/home/heng/Research/isbi/fly-dataset/flyJanelia', target='test')

# create_new_ground_truth()
# from train_test_comparison import *

# def test_compare():
#     compare_with_gt('/home/heng/Research/isbi/fly-dataset/flyJanelia', 65188)
# test_compare()

def test_dt():
    swc_path = '/home/heng/Research/isbi/fly-dataset/flyJanelia/swc/9.swc'
    tif_path = '/home/heng/Research/isbi/fly-dataset/flyJanelia/images/9.tif'
    output_path = '/home/heng/Research/isbi/fly-dataset/flyJanelia/labels/9.tif'
    output_path_2 = '/home/heng/Pictures/9.tif'
    output_path_3 = '/home/heng/Research/isbi/fly-dataset/flyJanelia/labels/9.tif'
    swc2tif_dt(swc_path, tif_path, output_path_3)
# test_dt()
from ptsemseg.models import *

def test_parameters(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file_name = os.path.split(model_path)[1]
    model_name = {'arch':model_file_name.split('/')[-1].split('_')[0]}


    # Setup Model
    log('set up model')
    n_classes = 1
    log('model name: {}'.format(model_name))
    model = get_model(model_name, n_classes, version='flyDataset')
    state = convert_state_dict(torch.load(model_path)["model_state"])
    model.load_state_dict(state)
    model.to(device)
    # print('model paramerters length: {}'.format(len(model.parameters())))
    c = 0
    for name, param in model.named_parameters():
        if True:#param.requires_grad:
            print(name, param.data)
            c += 1
    print(c)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

test_parameters('/home/heng/Research/segment_3D/runs/final_2/9424/unet3dregSmartStudentRes_flyDataset_model_best.pkl')