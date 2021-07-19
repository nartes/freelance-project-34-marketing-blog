import pprint
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

def kernel_6():
    import easyocr
    import cv2
    t6 = easyocr.Reader(['en'])

    t1 = glob.glob('*.mp4')

    t8 = []
    for o in t1:
        t7 = []
        t2 = None
        try:
            t2 = cv2.VideoCapture(o)
            for k in range(10):
                t3 = t2.read()
                assert t3[0]
                t4 = t3[1]
                t5 = t6.readtext(t4)
                t7.append(
                    t5
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
