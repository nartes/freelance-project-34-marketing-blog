import glob
import pandas

def kernel_1():
    t1 = '/kaggle/input/mlb-player-digital-engagement-forecasting'
    t2 = glob.glob(t1, '*.csv')
    t3 = {
        o : pandas.read_csv(o)
        for o in t2
    }

    return dict(
        t1=t1,
        t2=t2,
        t3=t3,
    )
