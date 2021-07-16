import xarray
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

        t1[k] = xarray.load_dataset(t4)
        print('loaded %s' % t4)

    return dict(
        t1=t1,
    )
