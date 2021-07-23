! pip install pyyaml==5.2
! pip install scipy==1.1.0
! pip install torch==1.2.0 torchvision==0.4.0
! pip install pillow==6.2.2
import torch
print(torch.__version__)
import yaml, scipy
print(yaml.__version__)
print(scipy.__version__)

!rm -rf /kaggle/working/AlphaPose
os.chdir('/kaggle/working/')
!git clone https://github.com/MVIG-SJTU/AlphaPose.git
    
!python -m pip install cython
!apt-get install libyaml-dev

import os
os.chdir('/kaggle/working/AlphaPose')
print(os.getcwd())
! python setup.py build develop

import gdown
import os
for o1, o2 in [
    (
        '1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC',
        '/kaggle/working/AlphaPose/detector/yolo/data/yolov3-spp.weights',
    ),
    (
        '1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA',
        '/kaggle/working/AlphaPose/detector/tracker/data/JDE-1088x608-uncertainty',
    ),
    (
        '1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn',
        '/kaggle/working/AlphaPose/pretrained_models/fast_res50_256x192.pth'
    ),
]:
    os.makedirs(os.path.split(o2)[0], exist_ok=True)
    gdown.download(
        'https://drive.google.com/u/0/uc?export=download&confirm=f_Ix&id=%s' % o1,
        o2,
        quiet=False
    )

    
import os
os.chdir('/kaggle/working/AlphaPose')
! ls
! python3 scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/ --save_img
# result json and rendered images are saved here:
! ls examples/res/
! ls examples/res/vis
