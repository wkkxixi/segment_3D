# segment_3D

### Paper
[Deep learning segmentation of optical microscopy images improves 3D neuron reconstruction](https://drive.google.com/open?id=1jWykhjOMP_WqxO5i4IBBEHorPztfCqfv)

### 1. Setup the dependencies (MAC & Ubuntu)
To run ..., you need to install the following packages manually beforehand


* `scikit-image-0.12.3`
* `tifffile-0.9.0`
* `scipy`
* `Cython`
* `tensorboardX`
* `pyyaml`
* `tqdm`
```

(torch36)$ conda install scikit-image=0.12.3
(torch36)$ conda install -c conda-forge tifffile=0.9.0
(torch36)$ conda install -c anaconda scipy
(torch36)$ pip install Cython
(torch36)$ pip install tensorboardX
(torch36)$ pip install pyyaml
(torch36)$ conda install -c conda-forge tqdm
