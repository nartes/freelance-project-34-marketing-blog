import pprint
import xarray
import numpy
import json
import glob
import io
import os
import pandas
import pickle

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
