import pprint
import cv2
import re
import json
import io
import glob
import xarray
import numpy
import json
import glob
import io
import os
import pandas
import pickle
import subprocess

def kernel_1():
    t4 = 'kernel_1-t3.dat'

    def preprocess(t4):
        t1 = '/kaggle/input/mlb-player-digital-engagement-forecasting'
        t2 = glob.glob(
            os.path.join(
                t1,
                '*.csv'
            )
        )

        t3 = {
            o : pandas.read_csv(o).to_xarray()
            for o in t2
        }

        with io.open(t4, 'wb') as f:
            pickle.dump(t3, f)

    if not os.path.exists(t4):
        preprocess(t4=t4)

    with io.open(t4, 'rb') as f:
        t3 = pickle.load(f)


    return dict(
        t3=t3,
    )

def kernel_2(
    o_1=None,
):
    t1 = {}

    for k in [
        'playerTwitterFollowers',
        'teamTwitterFollowers',
        'games',
        'events'
    ]:
        t4 = '%s.nc' % k
        if not os.path.exists(t4):
            print('started %s' % t4)
            t2 = '/kaggle/input/mlb-player-digital-engagement-forecasting/train.csv'
            t3 = pandas.DataFrame(
                sum(
                    [
                        json.loads(o)
                        for o in o_1['t3'][t2][k].values
                        if isinstance(o, str)
                    ],
                    []
                )
            ).to_xarray()
            t3.to_netcdf(t4)
            print('cached %s' % t4)

        if k == 'events':
            t5 = '%s-v2.nc' % k
            if not os.path.exists(t5):
                t2 = xarray.load_dataset(t4)
                t3 = t2.sel(
                    index=numpy.arange(
                        2017653 - 10 * 1000,
                        2017653 + 1
                    )
                )
                t3.to_netcdf(t5)
            t1[k] = xarray.load_dataset(t5)
            print('loaded %s' % t5)
        else:
            t1[k] = xarray.load_dataset(t4)
            print('loaded %s' % t4)


    return dict(
        t1=t1,
    )

def kernel_3(should_exist=None):
    if should_exist is None:
        should_exist = False

    t3 = [
        ('playerTwitterFollowers', None),
        ('teamTwitterFollowers', None),
        ('games', None),
        ('events', 'events-v2.nc'),
    ]

    o_1 = None
    o_2 = None

    t4 = '/kaggle/input/garbage'
    t5 = {}
    for k, v in t3:
        if v is None:
            t1 = os.path.join(
                t4,
                '%s.nc' % k,
            )
        else:
            t1 = os.path.join(
                t4,
                v,
            )

        if os.path.exists(t1):
            t2 = xarray.load_dataset(t1)
        else:
            if should_exist:
                pprint.pprint([k, v, t1])
                raise NotImplementedError

            if o_1 is None:
                o_1 = kernel_1()
            if o_2 is None:
                o_2 = kernel_2(
                    o_1=o_1
                )

            t2 = o_2['events']
        t5[k] = t2

    return dict(
        t5=t5,
    )

def kernel_4(
    o_3=None,
):
    [
        print(
            o_3['t5']['events'].to_dataframe().iloc[k].to_json(indent=4)
        )
        for k in range(-10, -1)
    ]

    [
        print(
            o_3['t5']['games'].to_dataframe().iloc[k].to_json(indent=4)
        )
        for k in range(-10, -1)
    ]


    t4 = 'https://www.youtube.com/watch?v=reaC7BHgL3M'

    r"""
        {
            "gamePk":634280,
            "gameType":"R",
            "season":2021,
            "gameDate":"2021-04-30",
            "gameTimeUTC":"2021-04-30T23:37:00Z",
            "resumeDate":"",
            "resumedFrom":"",
            "codedGameState":"F",
            "detailedGameState":"Final",
            "isTie":0.0,
            "gameNumber":1,
            "doubleHeader":"N",
            "dayNight":"night",
            "scheduledInnings":9,
            "gamesInSeries":3.0,
            "seriesDescription":"Regular Season",
            "homeId":141,
            "homeName":"Toronto Blue Jays",
            "homeAbbrev":"TOR",
            "homeWins":12,
            "homeLosses":12,
            "homeWinPct":0.5,
            "homeWinner":true,
            "homeScore":13.0,
            "awayId":144,
            "awayName":"Atlanta Braves",
            "awayAbbrev":"ATL",
            "awayWins":12.0,
            "awayLosses":14.0,
            "awayWinPct":0.462,
            "awayWinner":false,
            "awayScore":5.0
        }
    """

    t1 = numpy.where(o_3['t5']['events']['gamePk'] == 634280)[0]
    t5 = o_3['t5']['events'].index.data
    t6 = t5[t1]
    t2 = o_3['t5']['events'].sel(index=t6)
    t3 = o_3['t5']['games'].to_dataframe().iloc[-2].to_dict()
    pprint.pprint(t3)
    assert t3['gamePk'] == 634280

    t7 = 'https://www.youtube.com/watch?v=T0MUK91ZWys'

    return dict(
        t2=t2,
        t3=t3,
        t4=t4,
        t7=t7,
    )

def kernel_5(o_4):
    for o in [o_4['t4'], o_4['t7']]:
        subprocess.check_call(
            [
                'youtube-dl',
                '-f',
                '18',
                o,
            ]
        )

def kernel_12():
    import easyocr
    t6 = easyocr.Reader(['en'])

    return dict(
        t6=t6,
    )

def kernel_6(
    o_7=None,
    o_10=None,
    o_12=None,
    max_frames=None,
):
    if max_frames is None:
        max_frames = 10

    import tqdm
    import cv2

    t1 = glob.glob('*.mp4')

    t8 = []
    for o in t1:
        t7 = []
        t2 = None
        try:
            t2 = cv2.VideoCapture(o)
            for k in tqdm.tqdm(range(max_frames)):
                t3 = t2.read()
                assert t3[0]
                t4 = t3[1]
                if not o_12 is None:
                    t5 = o_12['t6'].readtext(t4)
                else:
                    t5 = None

                if not o_7 is None:
                    t10 = o_7['estimate_pose'](t4)
                else:
                    t10 = None
                if not o_10 is None:
                    t11 = o_10['model'](t4).pandas().xywhn
                else:
                    t11 = None

                t7.append(
                    dict(
                        frame_id=k,
                        t5=t5,
                        t10=t10,
                        t11=t11,
                    ),
                )
        finally:
            if not t2 is None:
                t2.release()
        t8.append(
            dict(
                video_path=o,
                frames=t7,
            )
        )

    t9 = []
    for o in t1:
        cap = None

        try:
            cap = cv2.VideoCapture(o)
            fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count/fps
        finally:
            if not cap is None:
                cap.release()

        t9.append(
            dict(
                video_path=o,
                fps=fps,
                frame_count=frame_count,
                duration=duration,
            )
        )


    return dict(
        t8=t8,
        t9=t9,
    )

def kernel_7(
    use_gpu=None,
):
    #!/usr/bin/env python
    # coding: utf-8

    #
    #
    # NOTE: Turn on Internet and GPU

    # The code hidden below handles all the imports and function definitions (the heavy lifting). If you're a beginner I'd advice you skip this for now. When you are able to understand the rest of the code, come back here and understand each function to get a deeper knowledge.

    # In[1]:


    # !/usr/bin/env python3
    # coding=utf-8
    # author=dave.fang@outlook.com
    # create=20171225

    import os
    import pprint
    import cv2
    import sys
    import math
    import time
    import tempfile
    import numpy as np
    import matplotlib.pyplot as plt

    import torch
    import torch.nn as nn
    import torch.nn.parallel
    import torch.backends.cudnn as cudnn
    import torch.optim as optim
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    import torchvision.models as models

    from torch.autograd import Variable

    from scipy.ndimage.filters import gaussian_filter

    #get_ipython().run_line_magic('matplotlib', 'inline')
    #get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

    # find connection in the specified sequence, center 29 is in the position 15
    limb_seq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
                [1, 16], [16, 18], [3, 17], [6, 18]]

    # the middle joints heatmap correpondence
    map_ids = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22],
               [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52],
               [55, 56], [37, 38], [45, 46]]

    # these are the colours for the 18 body points
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


    class PoseEstimation(nn.Module):
        def __init__(self, model_dict):
            super(PoseEstimation, self).__init__()

            self.model0 = model_dict['block_0']
            self.model1_1 = model_dict['block1_1']
            self.model2_1 = model_dict['block2_1']
            self.model3_1 = model_dict['block3_1']
            self.model4_1 = model_dict['block4_1']
            self.model5_1 = model_dict['block5_1']
            self.model6_1 = model_dict['block6_1']

            self.model1_2 = model_dict['block1_2']
            self.model2_2 = model_dict['block2_2']
            self.model3_2 = model_dict['block3_2']
            self.model4_2 = model_dict['block4_2']
            self.model5_2 = model_dict['block5_2']
            self.model6_2 = model_dict['block6_2']

        def forward(self, x):
            out1 = self.model0(x)

            out1_1 = self.model1_1(out1)
            out1_2 = self.model1_2(out1)
            out2 = torch.cat([out1_1, out1_2, out1], 1)

            out2_1 = self.model2_1(out2)
            out2_2 = self.model2_2(out2)
            out3 = torch.cat([out2_1, out2_2, out1], 1)

            out3_1 = self.model3_1(out3)
            out3_2 = self.model3_2(out3)
            out4 = torch.cat([out3_1, out3_2, out1], 1)

            out4_1 = self.model4_1(out4)
            out4_2 = self.model4_2(out4)
            out5 = torch.cat([out4_1, out4_2, out1], 1)

            out5_1 = self.model5_1(out5)
            out5_2 = self.model5_2(out5)
            out6 = torch.cat([out5_1, out5_2, out1], 1)

            out6_1 = self.model6_1(out6)
            out6_2 = self.model6_2(out6)

            return out6_1, out6_2


    def make_layers(layer_dict):
        layers = []

        for i in range(len(layer_dict) - 1):
            layer = layer_dict[i]
            for k in layer:
                v = layer[k]
                if 'pool' in k:
                    layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
                else:
                    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                    layers += [conv2d, nn.ReLU(inplace=True)]
        layer = list(layer_dict[-1].keys())
        k = layer[0]
        v = layer_dict[-1][k]

        conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
        layers += [conv2d]

        return nn.Sequential(*layers)


    def get_pose_model():
        blocks = {}

        block_0 = [{'conv1_1': [3, 64, 3, 1, 1]}, {'conv1_2': [64, 64, 3, 1, 1]}, {'pool1_stage1': [2, 2, 0]},
                   {'conv2_1': [64, 128, 3, 1, 1]}, {'conv2_2': [128, 128, 3, 1, 1]}, {'pool2_stage1': [2, 2, 0]},
                   {'conv3_1': [128, 256, 3, 1, 1]}, {'conv3_2': [256, 256, 3, 1, 1]}, {'conv3_3': [256, 256, 3, 1, 1]},
                   {'conv3_4': [256, 256, 3, 1, 1]}, {'pool3_stage1': [2, 2, 0]}, {'conv4_1': [256, 512, 3, 1, 1]},
                   {'conv4_2': [512, 512, 3, 1, 1]}, {'conv4_3_CPM': [512, 256, 3, 1, 1]},
                   {'conv4_4_CPM': [256, 128, 3, 1, 1]}]

        blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]}, {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                              {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]}, {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                              {'conv5_5_CPM_L1': [512, 38, 1, 1, 0]}]

        blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]}, {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                              {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]}, {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                              {'conv5_5_CPM_L2': [512, 19, 1, 1, 0]}]

        for i in range(2, 7):
            blocks['block%d_1' % i] = [{'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3]},
                                       {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
                                       {'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]}]
            blocks['block%d_2' % i] = [{'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3]},
                                       {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
                                       {'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]}]

        layers = []
        for block in block_0:
            # print(block)
            for key in block:
                v = block[key]
                if 'pool' in key:
                    layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
                else:
                    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                    layers += [conv2d, nn.ReLU(inplace=True)]

        models = {
            'block_0': nn.Sequential(*layers)
        }

        for k in blocks:
            v = blocks[k]
            models[k] = make_layers(v)

        return PoseEstimation(models)


    def get_paf_and_heatmap(model, img_raw, scale_search, param_stride=8, box_size=368):
        multiplier = [scale * box_size / img_raw.shape[0] for scale in scale_search]

        heatmap_avg = torch.zeros((len(multiplier), 19, img_raw.shape[0], img_raw.shape[1])).cuda()
        paf_avg = torch.zeros((len(multiplier), 38, img_raw.shape[0], img_raw.shape[1])).cuda()

        for i, scale in enumerate(multiplier):
            img_test = cv2.resize(img_raw, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            img_test_pad, pad = pad_right_down_corner(img_test, param_stride, param_stride)
            img_test_pad = np.transpose(np.float32(img_test_pad[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5

            feed = Variable(torch.from_numpy(img_test_pad)).cuda()
            output1, output2 = model(feed)

            #print(output1.size())
            #print(output2.size())

            heatmap = nn.UpsamplingBilinear2d((img_raw.shape[0], img_raw.shape[1])).cuda()(output2)

            paf = nn.UpsamplingBilinear2d((img_raw.shape[0], img_raw.shape[1])).cuda()(output1)

            heatmap_avg[i] = heatmap[0].data
            paf_avg[i] = paf[0].data

        heatmap_avg = torch.transpose(torch.transpose(torch.squeeze(torch.mean(heatmap_avg, 0)), 0, 1), 1, 2).cuda()
        heatmap_avg = heatmap_avg.cpu().numpy()

        paf_avg = torch.transpose(torch.transpose(torch.squeeze(torch.mean(paf_avg, 0)), 0, 1), 1, 2).cuda()
        paf_avg = paf_avg.cpu().numpy()

        return paf_avg, heatmap_avg


    def extract_heatmap_info(heatmap_avg, param_thre1=0.1):
        all_peaks = []
        peak_counter = 0

        for part in range(18):
            map_ori = heatmap_avg[:, :, part]
            map_gau = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map_gau.shape)
            map_left[1:, :] = map_gau[:-1, :]
            map_right = np.zeros(map_gau.shape)
            map_right[:-1, :] = map_gau[1:, :]
            map_up = np.zeros(map_gau.shape)
            map_up[:, 1:] = map_gau[:, :-1]
            map_down = np.zeros(map_gau.shape)
            map_down[:, :-1] = map_gau[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (map_gau >= map_left, map_gau >= map_right, map_gau >= map_up,
                 map_gau >= map_down, map_gau > param_thre1))

            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])  # note reverse
            peaks = list(peaks)
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            ids = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (ids[i],) for i in range(len(ids))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        return all_peaks


    def extract_paf_info(img_raw, paf_avg, all_peaks, param_thre2=0.05, param_thre3=0.5):
        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(map_ids)):
            score_mid = paf_avg[:, :, [x - 19 for x in map_ids[k]]]
            candA = all_peaks[limb_seq[k][0] - 1]
            candB = all_peaks[limb_seq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            if nA != 0 and nB != 0:
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        if norm < 1e-8:
                            raise ZeroDivisionError
                        vec = np.divide(vec, norm)

                        startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                       np.linspace(candA[i][1], candB[j][1], num=mid_num))
                        startend = list(startend)

                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0]
                                          for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1]
                                          for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts)
                        score_with_dist_prior += min(0.5 * img_raw.shape[0] / norm - 1, 0)

                        criterion1 = len(np.nonzero(score_midpts > param_thre2)[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if len(connection) >= min(nA, nB):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        return special_k, connection_all


    def get_subsets(connection_all, special_k, all_peaks):
        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(map_ids)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limb_seq[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if (subset[j][indexB] != partBs[i]):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        print("found = 2")
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])
        return subset, candidate


    def draw_key_point(subset, all_peaks, img_raw):
        del_ids = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                del_ids.append(i)
        subset = np.delete(subset, del_ids, axis=0)

        img_canvas = img_raw.copy()  # B,G,R order

        for i in range(18):
            for j in range(len(all_peaks[i])):
                cv2.circle(img_canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

        return subset, img_canvas


    def link_key_point(img_canvas, candidate, subset, stickwidth=4):
        for i in range(17):
            for n in range(len(subset)):
                index = subset[n][np.array(limb_seq[i]) - 1]
                if -1 in index:
                    continue
                cur_canvas = img_canvas.copy()
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                img_canvas = cv2.addWeighted(img_canvas, 0.4, cur_canvas, 0.6, 0)

        return img_canvas

    def pad_right_down_corner(img, stride, pad_value):
        h = img.shape[0]
        w = img.shape[1]

        pad = 4 * [None]
        pad[0] = 0  # up
        pad[1] = 0  # left
        pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
        pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

        img_padded = img
        pad_up = np.tile(img_padded[0:1, :, :] * 0 + pad_value, (pad[0], 1, 1))
        img_padded = np.concatenate((pad_up, img_padded), axis=0)
        pad_left = np.tile(img_padded[:, 0:1, :] * 0 + pad_value, (1, pad[1], 1))
        img_padded = np.concatenate((pad_left, img_padded), axis=1)
        pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + pad_value, (pad[2], 1, 1))
        img_padded = np.concatenate((img_padded, pad_down), axis=0)
        pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + pad_value, (1, pad[3], 1))
        img_padded = np.concatenate((img_padded, pad_right), axis=1)

        return img_padded, pad


    if __name__ == '__main__':
        print(get_pose_model())


    # First let's download the pre-trained model.

    # In[2]:


    # Using gdown to download the model directly from Google Drive

    #assert os.system(' conda install -y gdown') == 0
    import gdown


    # In[3]:


    model = 'coco_pose_iter_440000.pth.tar'
    if not os.path.exists(model):
        url = 'https://drive.google.com/u/0/uc?export=download&confirm=f_Ix&id=0B1asvDK18cu_MmY1ZkpaOUhhRHM'
        gdown.download(
            url,
            model,
            quiet=False
        )


    # In[4]:


    state_dict = torch.load(model)['state_dict']   # getting the pre-trained model's parameters
    # A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor.

    model_pose = get_pose_model()   # building the model (see fn. defn. above). To see the architecture, see below cell.
    model_pose.load_state_dict(state_dict)   # Loading the parameters (weights, biases) into the model.

    model_pose.float()   # I'm not sure why this is used. No difference if you remove it.

    if use_gpu is None:
        use_gpu = True

    if use_gpu:
        model_pose.cuda()
        model_pose = torch.nn.DataParallel(model_pose, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    def estimate_pose(
        img_ori,
        name=None,
        scale_param=None,
        display=None,
    ):
        if display is None:
            display = True

        if scale_param is None:
            scale_param = [0.5, 1.0, 1.5, 2.0]

        if display:
            if name is None:
                name = tempfile.mktemp(
                    dir='/kaggle/working',
                    suffix='.png',
                )
            pprint.pprint(
                ['estimate_pose', dict(name=name)],
            )

        # People might be at different scales in the image, perform inference at multiple scales to boost results

        # Predict Heatmaps for approximate joint position
        # Use Part Affinity Fields (PAF's) as guidance to link joints to form skeleton
        # PAF's are just unit vectors along the limb encoding the direction of the limb
        # A dot product of possible joint connection will be high if actual limb else low

        img_canvas = None
        img_points = None

        try:
            paf_info, heatmap_info = get_paf_and_heatmap(model_pose, img_ori, scale_param)
            peaks = extract_heatmap_info(heatmap_info)
            sp_k, con_all = extract_paf_info(img_ori, paf_info, peaks)

            subsets, candidates = get_subsets(con_all, sp_k, peaks)

            subsets, img_points = draw_key_point(subsets, peaks, img_ori)
            img_canvas = link_key_point(img_points, candidates, subsets)
        except ZeroDivisionError:
            pprint.pprint('zero de')

            img_points = img_ori.copy()
            img_canvas = img_ori.copy()

        # After predicting Heatmaps and PAF's, proceeed to link joints correctly
        if display:
            f = plt.figure(figsize=(15, 10))

            plt.subplot(1, 2, 1)
            plt.imshow(img_points[...,::-1])

            plt.subplot(1, 2, 2)
            plt.imshow(img_canvas[...,::-1])

            f.savefig(name)

        return dict(
            img_points=img_points,
            img_canvas=img_canvas,
        )


    # In[5]:

    return dict(
        cv2=cv2,
        estimate_pose=estimate_pose,
        model_pose=model_pose,
    )


def kernel_8(
    o_7,
):
    for i, o in enumerate([
    '../input/indonesian-traditional-dance/tgagrakanyar/tga_00%d0.jpg' % k
        for k in range(6)
    ]):
        arch_image = o
        img_ori = o_7['cv2'].imread(arch_image)
        o_7['estimate_pose'](img_ori)

def kernel_9_benchmark(
    o_7,
):
    import datetime

    t1 = o_7['cv2'].imread('../input/indonesian-traditional-dance/tgagrakanyar/tga_0000.jpg')
    t5 = 10
    t2 = datetime.datetime.now()
    for k in range(t5):
        o_7['estimate_pose'](
            img_ori=t1,
            scale_param=[1.0],
            display=False,
        )
    t3 = datetime.datetime.now()
    t4 = (t3 - t2).total_seconds() / t5
    pprint.pprint(
        ['kernel_9_benchmark', dict(t4=t4, t5=t5)]
    )

def kernel_10():
    import torch

    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5x, custom

    # Images
    img = 'https://ultralytics.com/images/zidane.jpg'  # or file, PIL, OpenCV, numpy, multiple

    # Inference
    results = model(img)

    # Results
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

    return dict(
        model=model,
    )

def kernel_11_benchmark(
    o_7,
    o_10,
):
    import datetime

    t1 = o_7['cv2'].imread('../input/indonesian-traditional-dance/tgagrakanyar/tga_0000.jpg')
    t5 = 10
    t2 = datetime.datetime.now()
    for k in range(t5):
        t6 = o_10['model'](t1)
        t7 = t6.pandas().xywhn

    t3 = datetime.datetime.now()
    t4 = (t3 - t2).total_seconds() / t5
    pprint.pprint(
        ['kernel_11_benchmark', dict(t4=t4, t5=t5)]
    )

def kernel_13(
    o_6=None,
):
    t2 = [
        '/kaggle/working',
        '/kaggle/input/garbage'
    ]

    t3 = [
        os.path.join(
            o,
            'kernel_13-object-detection.nc',
        )
        for o in t2
    ]

    t4 = [
        o
        for o in t3
        if os.path.exists(o)
    ]


    if not len(t4) > 0 or not o_6 is None:
        t1 = pandas.concat(
            sum(
                [
                    [
                        o2['t11'][0].assign(
                            frame_id=k,
                            video_path=o['video_path']
                        )
                        for k, o2 in enumerate(o['frames'])
                    ] for o in o_6['t8']
                ],
                []
            )
        ).to_xarray()
        t5 = t3[0]
        t1.to_netcdf(t5)
        del t1
    elif len(t4) > 0:
        t5 = t4[0]
    else:
        raise NotImplementedError

    t1 = xarray.load_dataset(t5)

    return dict(
        t1=t1,
    )

def kernel_14(
    skip_o_6=None,
    skip_o_7_10_12_13=None,
    run_benchmark=None,
):
    if skip_o_6 is None:
        skip_o_6 = True

    if skip_o_7_10_12_13 is None:
        skip_o_7_10_12_13 = True

    if run_benchmark is None:
        run_benchmark = False

    o_3 = kernel_3(should_exist=True)
    o_4 = kernel_4(o_3=o_3)
    o_5 = kernel_5(o_4=o_4)

    if not skip_o_7_10_12_13:
        o_7 = kernel_7()

        o_10 = kernel_10()
        o_12 = kernel_12()
    else:
        o_7 = None
        o_10 = None
        o_12 = None

    if not skip_o_6:
        o_6 = kernel_6(
            o_7=None,
            o_10=o_10,
            o_12=None,
            max_frames=10000
        )
    else:
        o_6 = None

    if not skip_o_7_10_12_13:
        o_13 = kernel_13(
            o_6=o_6,
        )
    else:
        o_13 = None

    if run_benchmark:
        o_11 = kernel_11_benchmark(o_7=o_7, o_10=o_10)
        o_9 = kernel_9_benchmark(o_7=o_7)
        o_8 = kernel_8(o_7=o_7)

    return dict(
        o_13=o_13,
        o_7=o_7,
    )

def kernel_15(
    o_14,
):
    t1 = pandas.DataFrame(
        numpy.unique(
            o_14['o_13']['t1']['name'].data,
            return_counts=True
        )
    ).T
    pprint.pprint(
        dict(
            t1=t1,
        )
    )

    t29 = 'output-gif'
    if not os.path.exists(t29):
        os.makedirs(t29, exist_ok=True)

    for t2 in [
        'baseball glove',
        'baseball bat',
        'sports ball',
        'person',
    ]:
        t28 = t2.replace(' ', '-')
        t3 = o_14['o_13']['t1']
        t4 = numpy.where(t3.name.data == t2)[0]

        t30 = 'output-png/%s' % t28
        if not os.path.exists(t30):
            os.makedirs(t30, exist_ok=True)

        numpy.random.seed(0)
        t22 = numpy.random.choice(t4, 10)
        pprint.pprint(t22)
        import tqdm
        t24 = []
        t27 = []
        for t5 in tqdm.tqdm(t22):
            t6 = t3.video_path.data[t5]
            t7 = t3.frame_id.data[t5]
            t8 = t3.to_dataframe().iloc[t5]
            #pprint.pprint([t6, t7])
            #pprint.pprint(t8)

            import cv2
            import matplotlib.pyplot

            t9 = cv2.VideoCapture(t6)
            t9.set(cv2.CAP_PROP_POS_FRAMES, t7)
            t10 = t9.read()
            t9.release()
            t11 = t10[1]
            t12 = cv2.cvtColor(t11, cv2.COLOR_BGR2RGB)
            t13 = t12.copy()
            t15 = numpy.array([t8.xcenter, t8.ycenter, t8.width, t8.height])
            t16 = numpy.array([t13.shape[1], t13.shape[0], t13.shape[1], t13.shape[0]])
            t17 = t15 * t16
            t18 = t17[:2] - t17[2:] / 2
            t19 = t17[:2] + t17[2:] / 2
            t20 = numpy.array([
                t18[0], t18[1],
                t19[0], t19[1],
            ])
            t21 = numpy.round(t20).astype(numpy.int32)
            t14 = cv2.rectangle(
                t13,
                tuple(t21[:2]),
                tuple(t21[2:]),
                (0, 255, 0),
                1,
            )

            if False:
                t32 = o_14['o_7']['estimate_pose'](
                    t12,
                    scale_param=[1.0],
                    display=False,
                )['img_canvas']
            else:
                t32 = kernel_16([t12])['t6'][0]

            f = matplotlib.pyplot.figure(figsize=(8, 12))
            f.suptitle(
                'name %s, frame_id %d\nvideo_path %s' % (
                    t8['name'],
                    t8.frame_id,
                    t8.video_path,
                )
            )
            matplotlib.pyplot.subplot(2, 1, 1)
            matplotlib.pyplot.title(
                'score %s' % (
                    t8.confidence,
                )
            )
            matplotlib.pyplot.imshow(t14)
            matplotlib.pyplot.subplot(2, 1, 2)
            matplotlib.pyplot.imshow(t32)
            t25 = os.path.join(
                t30,
                'kernel_15-%s-%05d.jpg' % (
                    t28,
                    t7,
                )
            )
            f.savefig(t25)
            t24.append(t25)
            matplotlib.pyplot.close(f)

            t27.append([t8, t21])
        pprint.pprint(
            pandas.concat([
                o[0]
                for o in t27
            ], axis=1).T
        )

        t23 = os.path.join(
            t29,
            'output-%s.gif' % t28
        )
        if os.path.exists(t23):
            subprocess.check_call(['rm', t23])

        subprocess.check_call(
            [
                'convert',
                '-delay',
                '100',
                '-loop',
                '0',
                *t24,
                t23,
            ]
        )

def kernel_16(images):
    assert isinstance(images, list)

    import cv2
    import subprocess
    import os

    t2 = '/kaggle/working/test-input'
    subprocess.check_call([
        'rm',
        '-fr',
        t2,
    ])

    subprocess.check_call([
        'mkdir',
        '-p',
        t2,
    ])

    t1 = []
    for i, o in enumerate(images):
        t5 = cv2.cvtColor(o, cv2.COLOR_RGB2BGR)
        t8 = 'image-%d.jpg' % i
        t3 = os.path.join(
            t2,
            t8,
        )
        cv2.imwrite(t3, t5)
        t1.append(
            dict(
                image_name=t8,
                image_path=t3,
                image_canvas=o,
                image_index=i,
            )
        )

    t4 = '/kaggle/working/test-output'
    subprocess.check_call([
        'rm',
        '-fr',
        t4,
    ])

    subprocess.check_call([
        'mkdir',
        '-p',
        t4,
    ])

    with subprocess.Popen('''
        cd /kaggle/working/AlphaPose &&
        python3 \
            scripts/demo_inference.py \
            --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
            --checkpoint pretrained_models/fast_res50_256x192.pth \
            --indir %s \
            --outdir %s \
            --save_img
    ''' % (t2, t4), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:

        pprint.pprint(p.communicate())
        p.wait()
        assert p.returncode == 0

    t6 = []
    with io.open(
        os.path.join(
            t4,
            'alphapose-results.json'
        ),
        'r'
    ) as f:
        t7 = json.load(f)
    for o in t1:
        t9 = os.path.join(
            t4,
            'vis',
            o['image_name']
        )
        assert os.path.exists(t9)
        t10 = cv2.imread(t9, cv2.IMREAD_COLOR)
        t11 = cv2.cvtColor(t10, cv2.COLOR_BGR2RGB)
        t6.append(t11)
    return dict(
        images=images,
        t1=t1,
        t6=t6,
        t7=t7,
    )

def kernel_17(
    o_14,
    max_images=None,
):
    if max_images is None:
        max_images = 10

    t50 = []
    for t2 in [
        'baseball glove',
        'baseball bat',
        'sports ball',
        'person',
    ]:
        t28 = t2.replace(' ', '-')
        t3 = o_14['o_13']['t1']
        t4 = numpy.where(t3.name.data == t2)[0]

        t30 = 'output-png/%s' % t28
        if not os.path.exists(t30):
            os.makedirs(t30, exist_ok=True)

        numpy.random.seed(0)
        t22 = numpy.random.choice(t4, max_images)
        pprint.pprint(t22)
        import tqdm
        t24 = []
        t27 = []
        for t5 in tqdm.tqdm(t22):
            t6 = t3.video_path.data[t5]
            t7 = t3.frame_id.data[t5]
            t8 = t3.to_dataframe().iloc[t5]
            #pprint.pprint([t6, t7])
            #pprint.pprint(t8)

            import cv2
            import matplotlib.pyplot

            t9 = cv2.VideoCapture(t6)
            t9.set(cv2.CAP_PROP_POS_FRAMES, t7)
            t10 = t9.read()
            t9.release()
            t11 = t10[1]
            t12 = cv2.cvtColor(t11, cv2.COLOR_BGR2RGB)
            t13 = t12.copy()

            t50.append(
                dict(
                    t2=t2,
                    t28=t28,
                    t6=t6,
                    t5=t5,
                    t7=t7,
                    t12=t12,
                )
            )

    return dict(
        t50=t50,
    )

def kernel_18(o_17):
    t1 = [o['t12'] for o in o_17['t50']]
    t2 = kernel_16(t1)

    return dict(
        t2=t2,
    )

def kernel_19(o_18):
    kernel_25(
        o_18['t2']['t6']
    )

def kernel_20(
    o_18,
    o_21=None,
):
    if o_21 is None:
        o_21 = kernel_21()

    import cv2
    import numpy
    import os


    t1 = numpy.array(o_18['t2']['t7'][0]['keypoints']).reshape(17, -1)
    t2 = o_18['t2']['t6'][0]
    t3 = o_18['t2']['t1'][0]['image_canvas'].copy()
    assert o_18['t2']['t7'][0]['image_id'] == os.path.split(o_18['t2']['t1'][0]['image_name'])[1]

    for i, o2 in enumerate(o_21['p_color']):
        if i >= 17:
            print('fuck %d' % i)
            continue
        o1 = t1[i, :]
        cv2.circle(t3, tuple(o1[:2].astype(numpy.int32)), 3, o2, -1)
    cv2.imwrite('output.jpg', cv2.cvtColor(t3, cv2.COLOR_RGB2BGR))
    cv2.imwrite('output-v2.jpg', cv2.cvtColor(t2, cv2.COLOR_RGB2BGR))

def kernel_21():
    l_pair = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (17, 11), (17, 12),  # Body
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]
    p_color = [
        # Nose, LEye, REye, LEar, REar
        (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
        # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
        (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
        # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle
        (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
        # Not Present
        ## Neck
        #(0, 255, 255),
    ]

    line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                  (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                  (77, 222, 255), (255, 156, 127),
                  (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
    #https://debuggercafe.com/wp-content/uploads/2020/10/keypoint_exmp.jpg
    # Seems like No Neck actually
    title = [
        'Nose',
        'LEye',
        'REye',
        'LEar',
        'REar',
        'LShoulder',
        'RShoulder',
        'LElbow',
        'RElbow',
        'LWrist',
        'RWrist',
        'LHip',
        'RHip',
        'LKnee',
        'RKnee',
        'LAnkle',
        'RAnkle',
        #'Neck',
    ]

    return dict(
        l_pair=l_pair,
        p_color=p_color,
        line_color=line_color,
    )

def kernel_22(o_18):

    t1 = o_18['t2']['t7']
    t2 = [
        numpy.array(o['keypoints']).reshape(17, 3)
        for o in t1
    ]

    o_31 = kernel_31(
        image_id=[
            o['image_id']
            for o in t1
        ],
        image_size=numpy.array([
            [
                list(o['image_canvas'].shape)
                for o in o_18['t2']['t1']
                if o['image_name'] == t1[i]['image_id']
            ][0]
            for i in range(len(t2))
        ]),
        keypoints=numpy.stack(t2, axis=0),
    )
    t12 = o_31['t12']

    return dict(
        t4=t12,
    )

def kernel_24(
    img,
    keypoints,
    o_21=None,
):
    t3 = img.copy()

    if o_21 is None:
        o_21 = kernel_21()

    import cv2
    import numpy
    import os

    for i, o2 in enumerate(o_21['p_color']):
        if i >= 17:
            print('fuck %d' % i)
            continue
        o1 = keypoints[i, :]
        cv2.circle(t3, tuple(o1[:2].astype(numpy.int32)), 3, o2, -1)

    return dict(
        t3=t3,
    )

def kernel_25(images):
    import tqdm
    import os
    import cv2
    import subprocess

    t3 = 'kernel_25-output.dir'
    os.makedirs(t3, exist_ok=True)
    t6 = []
    for i, o in tqdm.tqdm(enumerate(images)):
        t4 = 'image-%03d.jpg' % i
        t5 = os.path.join(t3, t4)
        t8 = cv2.cvtColor(o, cv2.COLOR_RGB2BGR)
        cv2.imwrite(t5, t8)
        t6.append(t5)

    t7 = os.path.join(t3, 'output.gif')

    subprocess.check_call(
        [
            'convert',
            '-delay',
            '100',
            '-loop',
            '0',
            *t6,
            t7,
        ]
    )

def kernel_26(o_18, image_name):
    t1 = [
        i
        for i, o in enumerate(o_18['t2']['t1'])
        if o['image_name'] == image_name
    ]
    assert len(t1) == 1
    return t1[0]

def kernel_23(o_18, o_22, ids=None):
    import numpy
    import tqdm

    if ids is None:
        ids = numpy.s_[:]

    t1 = numpy.arange(len(o_22['t4']))
    t2 = t1[ids]

    pprint.pprint(t2[:5])
    t7 = []
    for o in tqdm.tqdm(t2):
        t3 = o_22['t4'][o]
        t9 = kernel_26(o_18=o_18, image_name=t3['image_name'])
        t4 = o_18['t2']['t1'][t9]['image_canvas']
        t10 = o_18['t2']['t6'][t9]
        t4 = [
            o['image_canvas']
            for o in o_18['t2']['t1']
            if o['image_name'] == t3['image_name']
        ]
        assert len(t4) == 1
        t5 = t4[0]
        t6 = kernel_24(t5, t3['keypoints'])
        t7.append(t6['t3'])
        t7.append(t10)

    kernel_25(t7)

def kernel_27():
    import tqdm
    import os
    import subprocess
    import pprint

    t5 = '/kaggle/working/ATL AT TOR - April 19, 2015-T0MUK91ZWys.mp4'
    t3 = '/kaggle/working/kernel_27-output.dir'
    os.makedirs(t3, exist_ok=True)
    #R = int(round(30 / 5))
    FRAMERATE = 4
    SLICE_LENGTH = 5 * 6
    for i in tqdm.tqdm(range(100)):
        t1 = [SLICE_LENGTH * i, SLICE_LENGTH * (i + 1)]
        t2 = os.path.join(t3, 'slice-%d' % i)
        os.makedirs(t2, exist_ok=True)
        t4 = os.path.join(t2, 'output.mp4')
        if os.path.exists(t4):
            os.unlink(t4)
        with subprocess.Popen([
            'ffmpeg',
            '-i',
            '%s' % t5,
            '-filter:v',
            'fps=fps=%d' % FRAMERATE,
            '-ss',
            '%d' % t1[0],
            '-t',
            '%d' % (t1[1] - t1[0]),
            '%s' % t4,
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)  as p:
            if False:
                pprint.pprint(p.communicate())
            p.wait()
            assert p.returncode == 0

        t6 = '''
            cd /kaggle/working/AlphaPose && \
            python3 \
                scripts/demo_inference.py \
                --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
                --checkpoint pretrained_models/fast_res50_256x192.pth \
                --video %s \
                --outdir %s
        ''' % (t4, t2)
        if False:
            pprint.pprint([t4, t2, t6])
        with subprocess.Popen(t6, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  as p:
            if False:
                pprint.pprint(p.communicate())
            p.wait()
            assert p.returncode == 0

def kernel_28():
    import cv2
    import tqdm
    import os
    import subprocess
    import pprint

    t5 = '/kaggle/working/ATL AT TOR - April 19, 2015-T0MUK91ZWys.mp4'
    t3 = '/kaggle/working/kernel_28-output.dir'
    t13 = '/root/kernel_28-output.dir/tmp-slice'
    os.makedirs(t3, exist_ok=True)

    cap = None

    try:
        cap = cv2.VideoCapture(t5)
        fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps

        #R = int(round(30 / 5))
        FRAMERATE = 4
        SLICE_LENGTH = 5 * 6
        for i in tqdm.tqdm(range(100)):
            t1 = [SLICE_LENGTH * i, SLICE_LENGTH * (i + 1)]

            if os.path.exists(t13):
                subprocess.check_call(['rm', '-fr', t13])
            os.makedirs(t13, exist_ok=True)

            t8 = numpy.array([t1[0] * fps, t1[1] * fps]).astype(numpy.int32)
            cap.set(cv2.CAP_PROP_POS_FRAMES, t8[0])
            t12 = numpy.arange(t8[0], t8[1], SLICE_LENGTH * FRAMERATE)
            for k in range(t8[1] - t8[0]):
                ret, frame = cap.read()
                t11 = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if numpy.isin(t11, t12):
                    t10 = os.path.join(
                        t13,
                        'frame-%d.jpg' % t11,
                    )
                    cv2.imwrite(t10, frame)

            t2 = os.path.join(t3, 'slice-%d' % i)
            os.makedirs(t2, exist_ok=True)

            t6 = '''
                cd /kaggle/working/AlphaPose && \
                python3 \
                    scripts/demo_inference.py \
                    --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
                    --checkpoint pretrained_models/fast_res50_256x192.pth \
                    --indir %s \
                    --outdir %s
            ''' % (t13, t2)
            if False:
                pprint.pprint([t4, t2, t6])
            with subprocess.Popen(t6, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  as p:
                if False:
                    pprint.pprint(p.communicate())
                p.wait()
                assert p.returncode == 0
    finally:
        if not cap is None:
            cap.release()

def kernel_29():
    t4 = '/kaggle/working/kernel_29-poses.json'
    t6 = [
        t4,
        '/kaggle/input/garbage/kernel_29-poses.json',
    ]

    t7 = [
        o
        for o in t6
        if os.path.exists(o)
    ]

    if len(t7) == 0:

        t1 = [
            dict(
                data=json.load(
                    io.open(
                        o,
                        'r'
                    )
                ),
                input_path=o
            )
            for o in glob.glob(
                '/kaggle/working/kernel_28-output.dir/slice-*/*.json'
            )
        ]

        assert len(t1) > 0

        with io.open(t4, 'w') as f:
            f.write(json.dumps(t1))

        t7.append(t4)

    with io.open(t7[0], 'r') as f:
        t1 = json.load(f)

    t8 = sum([
        o['data']
        for o in t1
    ], [])
    t10 = re.compile('frame-(\d+)\.jpg')

    for i, o in enumerate(t8):
        o['frame_id'] = int(t10.match(o['image_id'])[1])

    t9 = sorted(
        t8,
        key=lambda o: o['frame_id']
    )

    t2 = pandas.DataFrame(t9)

    t5 = t2.to_xarray()
    t5['keypoints'] = xarray.DataArray(
        [
            numpy.array(o).reshape(17, 3)
            for o in t5['keypoints'].data
        ],
        dims=['index', 'joint', 'feature'],
    )
    o_31 = kernel_31(
        image_id=t5.image_id.data,
        image_size=[
            kernel_32('/kaggle/working/ATL AT TOR - April 19, 2015-T0MUK91ZWys.mp4')
        ] * t5.index.shape[0],
        keypoints=t5.keypoints.data,
    )
    for k, v in o_31.items():
        t5[k] = v

    return dict(
        t5=t5,
        t2=t2,
        t9=t9,
    )

def kernel_30(o_29, ids=None):
    t5 = o_29['t5']
    if ids is None:
        numpy.random.seed(0)
        t1 = numpy.random.choice(
            t5.index.shape[0],
            10
        )

    t7 = []

    for o in t1:
        t2 = t5.keypoints.data[o]
        t3 = t5.frame_id.data[o]

        cap = None
        try:
            cap = cv2.VideoCapture('/kaggle/working/ATL AT TOR - April 19, 2015-T0MUK91ZWys.mp4')
            cap.set(cv2.CAP_PROP_POS_FRAMES, t3)
            ret, frame = cap.read()
            t4 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t6 = kernel_24(t4, t2)['t3']
            t7.append(t6)
        finally:
            if not cap is None:
                cap.release()
    kernel_25(t7)

def kernel_31(image_id, image_size, keypoints):
    def get_angle(a,b):
        from math import sqrt, acos, degrees, atan, degrees
        #print(a)
        #print(b)
        del_y = a[1]-b[1]
        del_x = b[0]-a[0]
        if del_x == 0:
            del_x = 0.1
        #print("Del_X : "+str(del_x)+"-----Del_Y: "+str(del_y))
        angle = 0

        if del_x > 0 and del_y > 0:
            angle = degrees(atan(del_y / del_x))
        elif del_x < 0 and del_y > 0:
            angle = degrees(atan(del_y / del_x)) + 180

        return angle

    def angle_gor(a,b,c,d):
        import numpy as np
        from math import sqrt, acos, degrees, atan, degrees
        ab=[a[0]-b[0],a[1]-b[1]]
        ab1=[c[0]-d[0],c[1]-d[1]]
        cos = \
            abs(
                ab[0] * ab1[0] + ab[1] * ab1[1]
            ) / (
                sqrt(
                    ab[0] ** 2 + ab[1] ** 2
                ) * \
                sqrt(
                    ab1[0] ** 2 + ab1[1] ** 2
                ) + 1e-8
            )
        ang = acos(cos)
        return ang*180/np.pi


    def sit_ang(a,b,c,d):
        ang=angle_gor(a,b,c,d)
        s1=0
        if ang != None:
            #print("Angle",ang)
            if ang < 120 and ang>40:
                s1=1
        return s1

    def sit_rec(a,b,c,d):
        from math import sqrt, acos, degrees, atan, degrees

        ab = [a[0] - b[0], a[1] - b[1]]
        ab1 = [c[0] - d[0], c[1] - d[1]]
        l1=sqrt(ab[0]**2+ab[1]**2)
        l2=sqrt(ab1[0]**2+ab1[1]**2)
        s=0
        if l1!=0 and l2!=0:
            #print(l1,l2, "---------->>>")
            if l2/l1>=1.5:
                s=1
        return s

    t3 = []
    for i, o in enumerate(keypoints):
        t4 = numpy.min(o[:, 0])
        t5 = numpy.max(o[:, 0])
        t6 = numpy.min(o[:, 1])
        t7 = numpy.max(o[:, 1])

        t8 = (t5 - t4) * (t7 - t6)

        t9 = image_size[i]
        t10 = t9[0] * t9[1]
        t3.append(
            dict(
                t8=t8,
                t9=t9,
                t10=t10,
            )
        )
    t12 = [
        dict(
            square=t3[i]['t8'],
            total=t3[i]['t10'],
            portion=t3[i]['t8'] / (t3[i]['t10'] + 1e-8),
            keypoints=keypoints[i],
            image_name=image_id[i],
        )
        for i in range(len(t3))
    ]

    for i, o in enumerate(t12):
        t1 = o['keypoints']
        t2 = get_angle(
            t1[3, :2],
            t1[4, :2],
        )
        t3 = get_angle(
            t1[6, :2],
            t1[7, :2],
        )
        t4 = 0
        if 30 < t2 and t2 < 150:
            t4 = 1
        t5 = 0
        if 30 < t3 and t3 < 150:
            t5 = 1
        t6 = t4 + t6
        t7 = 0
        if t6 == 1:
            t7 = 1
        t8 = 0
        t8 += sit_rec(
            t1[9, :2],
            t1[10, :2],
            t1[10, :2],
            t1[11, :2],
        )
        t8 += sit_rec(
            t1[12, :2],
            t1[13, :2],
            t1[13, :2],
            t1[14, :2],
        )
        t9 = 0
        t9 += sit_ang(
            t1[9, :2],
            t1[10, :2],
            t1[10, :2],
            t1[11, :2],
        )
        t9 += sit_ang(
            t1[12, :2],
            t1[13, :2],
            t1[13, :2],
            t1[14, :2],
        )
        t10 = 0
        if t8 > 0 or t9 > 0:
            t10 = 1
        t11 = 0
        if t8 == 0 and t9 == 0:
            t11 = 1
        o.update(
            dict(
                t2=t2,
                t3=t3,
                t4=t4,
                t5=t5,
                t6=t6,
                t7=t7,
                t8=t8,
                t9=t9,
                t10=t10,
                t11=t11,
            )
        )

    t13 = pandas.DataFrame(t12).drop(columns=['keypoints']).to_xarray()

    return dict(
        t12=t12,
        t13=t13,
    )

def kernel_32(video_path):
    assert os.path.exists(video_path)

    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        return frame.shape
    finally:
        if not cap is None:
            cap.release()
