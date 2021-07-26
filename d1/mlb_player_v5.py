import os

if os.system(r''' pip show alphapose''') != 0:
	t1 = r'''
	rm -fr /kaggle/working/AlphaPose
	pip install pyyaml==5.2
	pip install scipy==1.1.0
	git clone https://github.com/WildflowerSchools/AlphaPose
	python -m pip install cython gdown
	apt-get install libyaml-dev
	cd /kaggle/working/AlphaPose && python setup.py build develop
	'''

	for o in t1.splitlines():
		print(o)
		assert os.system(o) == 0

import os
#!git clone https://github.com/MVIG-SJTU/AlphaPose.git

import torch
print(torch.__version__)
import yaml, scipy
print(yaml.__version__)
print(scipy.__version__)
    
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


assert os.system(r'''
mkdir -p /kaggle/working/test-input && mkdir -p /kaggle/working/test-output && cp /kaggle/working/AlphaPose/examples/demo/*.jpg /kaggle/working/test-input
cd /kaggle/working/AlphaPose && python3 scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir /kaggle/working/test-input --outdir /kaggle/working/test-output --save_img
''') == 0
