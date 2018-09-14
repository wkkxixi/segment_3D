from utils.io import *
# from swcToTiff import *
import os

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
def label_generator(folder):
    datainfo_path = folder + '/datainfo/datainfo.txt'
    with open(datainfo_path) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    print(content)


datafolder = '/Users/wonh/Desktop/flyJanelia'
info_generator(datafolder)

