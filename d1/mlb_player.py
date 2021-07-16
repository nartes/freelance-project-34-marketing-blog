# %% [markdown]
# <div>
#     <h1 align="center">MLB Player Digital Engagement Forecasting</h1>
#     <h1 align="center">LightGBM + CatBoost + ANN 2505f2</h1>
# </div>

# %% [markdown]
# <div class="alert alert-success">  
# </div>

# %% [markdown]
# <div class="alert alert-success">
#     <h1 align="center">If you find this work useful, please don't forget upvoting :)</h1>
# </div>

# %% [markdown]
# #### Thanks to: @lhagiimn   https://www.kaggle.com/lhagiimn/lightgbm-catboost-ann-2505f2
# 
# #### https://www.kaggle.com/columbia2131/mlb-lightgbm-starter-dataset-code-en-ja
# 
# #### https://www.kaggle.com/mlconsult/1-3816-lb-lbgm-descriptive-stats-param-tune
# 
# #### https://www.kaggle.com/batprem/lightgbm-ann-weight-with-love
# 
# #### https://www.kaggle.com/mlconsult/1-3816-lb-lbgm-descriptive-stats-param-tune
# 
# #### https://www.kaggle.com/ulrich07/mlb-ann-with-lags-tf-keras
# 

# %% [markdown]
# <div class="alert alert-success">  
# </div>

# %% [markdown]
# ## About Dataset

# %% [markdown]
# Train.csv is stored as a csv file with each column as follows.  
# 
# train.csvを以下のようにして各カラムをcsvファイルとして保管しています。

# %% [code] {"execution":{"iopub.status.busy":"2021-06-26T07:16:47.242749Z","iopub.execute_input":"2021-06-26T07:16:47.243324Z","iopub.status.idle":"2021-06-26T07:16:48.030215Z","shell.execute_reply.started":"2021-06-26T07:16:47.243266Z","shell.execute_reply":"2021-06-26T07:16:48.029Z"}}
import os

assert os.system(r'''cp ../input/fork-of-1-35-lightgbm-ann-2505f2-c4e96a/* .''') == 0

# %% [code] {"execution":{"iopub.status.busy":"2021-06-26T07:16:48.031858Z","iopub.execute_input":"2021-06-26T07:16:48.032396Z","iopub.status.idle":"2021-06-26T07:16:48.799514Z","shell.execute_reply.started":"2021-06-26T07:16:48.032357Z","shell.execute_reply":"2021-06-26T07:16:48.798628Z"}}
assert os.system(r'''ls''') == 0

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-26T07:16:48.801992Z","iopub.execute_input":"2021-06-26T07:16:48.802645Z","iopub.status.idle":"2021-06-26T07:16:48.813801Z","shell.execute_reply.started":"2021-06-26T07:16:48.802592Z","shell.execute_reply":"2021-06-26T07:16:48.812863Z"}}
#%%capture

"""
!pip install pandarallel 

import gc

import numpy as np
import pandas as pd
from pathlib import Path

from pandarallel import pandarallel
pandarallel.initialize()

BASE_DIR = Path('../input/mlb-player-digital-engagement-forecasting')
train = pd.read_csv(BASE_DIR / 'train.csv')

null = np.nan
true = True
false = False

for col in train.columns:

    if col == 'date': continue

    _index = train[col].notnull()
    train.loc[_index, col] = train.loc[_index, col].parallel_apply(lambda x: eval(x))

    outputs = []
    for index, date, record in train.loc[_index, ['date', col]].itertuples():
        _df = pd.DataFrame(record)
        _df['index'] = index
        _df['date'] = date
        outputs.append(_df)

    outputs = pd.concat(outputs).reset_index(drop=True)

    outputs.to_csv(f'{col}_train.csv', index=False)
    outputs.to_pickle(f'{col}_train.pkl')

    del outputs
    del train[col]
    gc.collect()
"""

# %% [markdown] {"execution":{"iopub.status.busy":"2021-06-16T09:14:33.869464Z","iopub.execute_input":"2021-06-16T09:14:33.869905Z","iopub.status.idle":"2021-06-16T09:14:33.874766Z","shell.execute_reply.started":"2021-06-16T09:14:33.869879Z","shell.execute_reply":"2021-06-16T09:14:33.873097Z"}}
# ## Training

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-26T07:16:48.81564Z","iopub.execute_input":"2021-06-26T07:16:48.816326Z","iopub.status.idle":"2021-06-26T07:16:50.081995Z","shell.execute_reply.started":"2021-06-26T07:16:48.816246Z","shell.execute_reply":"2021-06-26T07:16:50.080828Z"}}
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
from functools import reduce
from tqdm import tqdm
import lightgbm as lgbm
import mlb
import os

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-26T07:16:50.083534Z","iopub.execute_input":"2021-06-26T07:16:50.083899Z","iopub.status.idle":"2021-06-26T07:16:50.088159Z","shell.execute_reply.started":"2021-06-26T07:16:50.083863Z","shell.execute_reply":"2021-06-26T07:16:50.087357Z"}}
BASE_DIR = Path('../input/mlb-player-digital-engagement-forecasting')
TRAIN_DIR = Path('../input/mlb-pdef-train-dataset')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-26T07:16:50.08951Z","iopub.execute_input":"2021-06-26T07:16:50.090053Z","iopub.status.idle":"2021-06-26T07:16:54.221868Z","shell.execute_reply.started":"2021-06-26T07:16:50.090018Z","shell.execute_reply":"2021-06-26T07:16:54.220656Z"}}
players = pd.read_csv(BASE_DIR / 'players.csv')

rosters = pd.read_pickle(TRAIN_DIR / 'rosters_train.pkl')
targets = pd.read_pickle(TRAIN_DIR / 'nextDayPlayerEngagement_train.pkl')
scores = pd.read_pickle(TRAIN_DIR / 'playerBoxScores_train.pkl')
scores = scores.groupby(['playerId', 'date']).sum().reset_index()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-26T07:16:54.223547Z","iopub.execute_input":"2021-06-26T07:16:54.224Z","iopub.status.idle":"2021-06-26T07:16:54.243132Z","shell.execute_reply.started":"2021-06-26T07:16:54.22395Z","shell.execute_reply":"2021-06-26T07:16:54.242076Z"}}
targets_cols = ['playerId', 'target1', 'target2', 'target3', 'target4', 'date']
players_cols = ['playerId', 'primaryPositionName']
rosters_cols = ['playerId', 'teamId', 'status', 'date']
scores_cols = ['playerId', 'battingOrder', 'gamesPlayedBatting', 'flyOuts',
       'groundOuts', 'runsScored', 'doubles', 'triples', 'homeRuns',
       'strikeOuts', 'baseOnBalls', 'intentionalWalks', 'hits', 'hitByPitch',
       'atBats', 'caughtStealing', 'stolenBases', 'groundIntoDoublePlay',
       'groundIntoTriplePlay', 'plateAppearances', 'totalBases', 'rbi',
       'leftOnBase', 'sacBunts', 'sacFlies', 'catchersInterference',
       'pickoffs', 'gamesPlayedPitching', 'gamesStartedPitching',
       'completeGamesPitching', 'shutoutsPitching', 'winsPitching',
       'lossesPitching', 'flyOutsPitching', 'airOutsPitching',
       'groundOutsPitching', 'runsPitching', 'doublesPitching',
       'triplesPitching', 'homeRunsPitching', 'strikeOutsPitching',
       'baseOnBallsPitching', 'intentionalWalksPitching', 'hitsPitching',
       'hitByPitchPitching', 'atBatsPitching', 'caughtStealingPitching',
       'stolenBasesPitching', 'inningsPitched', 'saveOpportunities',
       'earnedRuns', 'battersFaced', 'outsPitching', 'pitchesThrown', 'balls',
       'strikes', 'hitBatsmen', 'balks', 'wildPitches', 'pickoffsPitching',
       'rbiPitching', 'gamesFinishedPitching', 'inheritedRunners',
       'inheritedRunnersScored', 'catchersInterferencePitching',
       'sacBuntsPitching', 'sacFliesPitching', 'saves', 'holds', 'blownSaves',
       'assists', 'putOuts', 'errors', 'chances', 'date']

feature_cols = ['label_playerId', 'label_primaryPositionName', 'label_teamId',
       'label_status', 'battingOrder', 'gamesPlayedBatting', 'flyOuts',
       'groundOuts', 'runsScored', 'doubles', 'triples', 'homeRuns',
       'strikeOuts', 'baseOnBalls', 'intentionalWalks', 'hits', 'hitByPitch',
       'atBats', 'caughtStealing', 'stolenBases', 'groundIntoDoublePlay',
       'groundIntoTriplePlay', 'plateAppearances', 'totalBases', 'rbi',
       'leftOnBase', 'sacBunts', 'sacFlies', 'catchersInterference',
       'pickoffs', 'gamesPlayedPitching', 'gamesStartedPitching',
       'completeGamesPitching', 'shutoutsPitching', 'winsPitching',
       'lossesPitching', 'flyOutsPitching', 'airOutsPitching',
       'groundOutsPitching', 'runsPitching', 'doublesPitching',
       'triplesPitching', 'homeRunsPitching', 'strikeOutsPitching',
       'baseOnBallsPitching', 'intentionalWalksPitching', 'hitsPitching',
       'hitByPitchPitching', 'atBatsPitching', 'caughtStealingPitching',
       'stolenBasesPitching', 'inningsPitched', 'saveOpportunities',
       'earnedRuns', 'battersFaced', 'outsPitching', 'pitchesThrown', 'balls',
       'strikes', 'hitBatsmen', 'balks', 'wildPitches', 'pickoffsPitching',
       'rbiPitching', 'gamesFinishedPitching', 'inheritedRunners',
       'inheritedRunnersScored', 'catchersInterferencePitching',
       'sacBuntsPitching', 'sacFliesPitching', 'saves', 'holds', 'blownSaves',
       'assists', 'putOuts', 'errors', 'chances','target1_mean',
 'target1_median',
 'target1_std',
 'target1_min',
 'target1_max',
 'target1_prob',
 'target2_mean',
 'target2_median',
 'target2_std',
 'target2_min',
 'target2_max',
 'target2_prob',
 'target3_mean',
 'target3_median',
 'target3_std',
 'target3_min',
 'target3_max',
 'target3_prob',
 'target4_mean',
 'target4_median',
 'target4_std',
 'target4_min',
 'target4_max',
 'target4_prob']
feature_cols2 = ['label_playerId', 'label_primaryPositionName', 'label_teamId',
       'label_status', 'battingOrder', 'gamesPlayedBatting', 'flyOuts',
       'groundOuts', 'runsScored', 'doubles', 'triples', 'homeRuns',
       'strikeOuts', 'baseOnBalls', 'intentionalWalks', 'hits', 'hitByPitch',
       'atBats', 'caughtStealing', 'stolenBases', 'groundIntoDoublePlay',
       'groundIntoTriplePlay', 'plateAppearances', 'totalBases', 'rbi',
       'leftOnBase', 'sacBunts', 'sacFlies', 'catchersInterference',
       'pickoffs', 'gamesPlayedPitching', 'gamesStartedPitching',
       'completeGamesPitching', 'shutoutsPitching', 'winsPitching',
       'lossesPitching', 'flyOutsPitching', 'airOutsPitching',
       'groundOutsPitching', 'runsPitching', 'doublesPitching',
       'triplesPitching', 'homeRunsPitching', 'strikeOutsPitching',
       'baseOnBallsPitching', 'intentionalWalksPitching', 'hitsPitching',
       'hitByPitchPitching', 'atBatsPitching', 'caughtStealingPitching',
       'stolenBasesPitching', 'inningsPitched', 'saveOpportunities',
       'earnedRuns', 'battersFaced', 'outsPitching', 'pitchesThrown', 'balls',
       'strikes', 'hitBatsmen', 'balks', 'wildPitches', 'pickoffsPitching',
       'rbiPitching', 'gamesFinishedPitching', 'inheritedRunners',
       'inheritedRunnersScored', 'catchersInterferencePitching',
       'sacBuntsPitching', 'sacFliesPitching', 'saves', 'holds', 'blownSaves',
       'assists', 'putOuts', 'errors', 'chances','target1_mean',
 'target1_median',
 'target1_std',
 'target1_min',
 'target1_max',
 'target1_prob',
 'target2_mean',
 'target2_median',
 'target2_std',
 'target2_min',
 'target2_max',
 'target2_prob',
 'target3_mean',
 'target3_median',
 'target3_std',
 'target3_min',
 'target3_max',
 'target3_prob',
 'target4_mean',
 'target4_median',
 'target4_std',
 'target4_min',
 'target4_max',
 'target4_prob',
    'target1']

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-26T07:16:54.244866Z","iopub.execute_input":"2021-06-26T07:16:54.24532Z","iopub.status.idle":"2021-06-26T07:16:54.296844Z","shell.execute_reply.started":"2021-06-26T07:16:54.245257Z","shell.execute_reply":"2021-06-26T07:16:54.295689Z"}}
player_target_stats = pd.read_csv("../input/player-target-stats/player_target_stats.csv")
data_names=player_target_stats.columns.values.tolist()
data_names

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-26T07:16:54.300157Z","iopub.execute_input":"2021-06-26T07:16:54.300622Z","iopub.status.idle":"2021-06-26T07:17:02.252208Z","shell.execute_reply.started":"2021-06-26T07:16:54.300578Z","shell.execute_reply":"2021-06-26T07:17:02.250423Z"}}
# creat dataset
train = targets[targets_cols].merge(players[players_cols], on=['playerId'], how='left')
train = train.merge(rosters[rosters_cols], on=['playerId', 'date'], how='left')
train = train.merge(scores[scores_cols], on=['playerId', 'date'], how='left')
train = train.merge(player_target_stats, how='inner', left_on=["playerId"],right_on=["playerId"])


# label encoding
player2num = {c: i for i, c in enumerate(train['playerId'].unique())}
position2num = {c: i for i, c in enumerate(train['primaryPositionName'].unique())}
teamid2num = {c: i for i, c in enumerate(train['teamId'].unique())}
status2num = {c: i for i, c in enumerate(train['status'].unique())}
train['label_playerId'] = train['playerId'].map(player2num)
train['label_primaryPositionName'] = train['primaryPositionName'].map(position2num)
train['label_teamId'] = train['teamId'].map(teamid2num)
train['label_status'] = train['status'].map(status2num)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-26T07:17:02.253453Z","iopub.status.idle":"2021-06-26T07:17:02.254076Z"}}
train_X = train[feature_cols]
train_y = train[['target1', 'target2', 'target3', 'target4']]

_index = (train['date'] < 20210401)
x_train1 = train_X.loc[_index].reset_index(drop=True)
y_train1 = train_y.loc[_index].reset_index(drop=True)
x_valid1 = train_X.loc[~_index].reset_index(drop=True)
y_valid1 = train_y.loc[~_index].reset_index(drop=True)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-26T07:17:02.255068Z","iopub.status.idle":"2021-06-26T07:17:02.255685Z"}}
train_X = train[feature_cols2]
train_y = train[['target1', 'target2', 'target3', 'target4']]

_index = (train['date'] < 20210401)
x_train2 = train_X.loc[_index].reset_index(drop=True)
y_train2 = train_y.loc[_index].reset_index(drop=True)
x_valid2 = train_X.loc[~_index].reset_index(drop=True)
y_valid2 = train_y.loc[~_index].reset_index(drop=True)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-26T07:17:02.256629Z","iopub.status.idle":"2021-06-26T07:17:02.257215Z"}}
train_X

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-26T07:17:02.258224Z","iopub.status.idle":"2021-06-26T07:17:02.258854Z"}}
def fit_lgbm(x_train, y_train, x_valid, y_valid, params: dict=None, verbose=100):
    oof_pred = np.zeros(len(y_valid), dtype=np.float32)
    model = lgbm.LGBMRegressor(**params)
    model.fit(x_train, y_train, 
        eval_set=[(x_valid, y_valid)],  
        early_stopping_rounds=verbose, 
        verbose=verbose)
    oof_pred = model.predict(x_valid)
    score = mean_absolute_error(oof_pred, y_valid)
    print('mae:', score)
    return oof_pred, model, score


# training lightgbm

params1 = {'objective':'mae',
           'reg_alpha': 0.14947461820098767, 
           'reg_lambda': 0.10185644384043743, 
           'n_estimators': 3633, 
           'learning_rate': 0.08046301304430488, 
           'num_leaves': 674, 
           'feature_fraction': 0.9101240539122566, 
           'bagging_fraction': 0.9884451442950513, 
           'bagging_freq': 8, 
           'min_child_samples': 51}

params2 = {
 'objective':'mae',
 'reg_alpha': 0.1,
 'reg_lambda': 0.1, 
 'n_estimators': 80,
 'learning_rate': 0.1,
 'random_state': 42,
 "num_leaves": 22
}

params4 = {'objective':'mae',
           'reg_alpha': 0.016468100279441976, 
           'reg_lambda': 0.09128335764019105, 
           'n_estimators': 9868, 
           'learning_rate': 0.10528150510326864, 
           'num_leaves': 157, 
           'feature_fraction': 0.5419185713426886, 
           'bagging_fraction': 0.2637405128936662, 
           'bagging_freq': 19, 
           'min_child_samples': 71}


params = {
 'objective':'mae',
 'reg_alpha': 0.1,
 'reg_lambda': 0.1, 
 'n_estimators': 10000,
 'learning_rate': 0.1,
 'random_state': 42,
 "num_leaves": 100
}


# Slow from this point !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

oof1, model1, score1 = fit_lgbm(
    x_train1, y_train1['target1'],
    x_valid1, y_valid1['target1'],
    params1
 )

oof2, model2, score2 = fit_lgbm(
    x_train2, y_train2['target2'],
    x_valid2, y_valid2['target2'],
    params2
)

oof3, model3, score3 = fit_lgbm(
    x_train2, y_train2['target3'],
    x_valid2, y_valid2['target3'],
   params
)

oof4, model4, score4 = fit_lgbm(
    x_train2, y_train2['target4'],
    x_valid2, y_valid2['target4'],
    params4
)

score = (score1+score2+score3+score4) / 4
print(f'score: {score}')

# %% [code]
import pickle
from catboost import CatBoostRegressor

def fit_lgbm(x_train, y_train, x_valid, y_valid, target, params: dict=None, verbose=100):
    oof_pred_lgb = np.zeros(len(y_valid), dtype=np.float32)
    oof_pred_cat = np.zeros(len(y_valid), dtype=np.float32)
    
    if os.path.isfile(f'../input/mlb-lgbm-and-catboost-models/model_lgb_{target}.pkl'):
        with open(f'../input/mlb-lgbm-and-catboost-models/model_lgb_{target}.pkl', 'rb') as fin:
            model = pickle.load(fin)
    else:
    
        model = lgbm.LGBMRegressor(**params)
        model.fit(x_train, y_train, 
            eval_set=[(x_valid, y_valid)],  
            early_stopping_rounds=verbose, 
            verbose=verbose)

        with open(f'model_lgb_{target}.pkl', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    oof_pred_lgb = model.predict(x_valid)
    score_lgb = mean_absolute_error(oof_pred_lgb, y_valid)
    print('mae:', score_lgb)
    
    if os.path.isfile(f'../input/mlb-lgbm-and-catboost-models/model_cb_{target}.pkl'):
        with open(f'../input/mlb-lgbm-and-catboost-models/model_cb_{target}.pkl', 'rb') as fin:
            model_cb = pickle.load(fin)
    else:
    
        model_cb = CatBoostRegressor(
                    n_estimators=2000,
                    learning_rate=0.05,
                    loss_function='MAE',
                    eval_metric='MAE',
                    max_bin=50,
                    subsample=0.9,
                    colsample_bylevel=0.5,
                    verbose=100)

        model_cb.fit(x_train, y_train, use_best_model=True,
                         eval_set=(x_valid, y_valid),
                         early_stopping_rounds=25)

        with open(f'model_cb_{target}.pkl', 'wb') as handle:
            pickle.dump(model_cb, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    oof_pred_cat = model_cb.predict(x_valid)
    score_cat = mean_absolute_error(oof_pred_cat, y_valid)
    print('mae:', score_cat)
    
    return oof_pred_lgb, model, oof_pred_cat, model_cb, score_lgb, score_cat


# training lightgbm
params = {
'boosting_type': 'gbdt',
'objective':'mae',
'subsample': 0.5,
'subsample_freq': 1,
'learning_rate': 0.03,
'num_leaves': 2**11-1,
'min_data_in_leaf': 2**12-1,
'feature_fraction': 0.5,
'max_bin': 100,
'n_estimators': 2500,
'boost_from_average': False,
"random_seed":42,
}

oof_pred_lgb2, model_lgb2, oof_pred_cat2, model_cb2, score_lgb2, score_cat2 = fit_lgbm(
    x_train1, y_train1['target2'],
    x_valid1, y_valid1['target2'],
    2, params
)

oof_pred_lgb1, model_lgb1, oof_pred_cat1, model_cb1, score_lgb1, score_cat1 = fit_lgbm(
    x_train1, y_train1['target1'],
    x_valid1, y_valid1['target1'],
    1, params
)

oof_pred_lgb3, model_lgb3, oof_pred_cat3, model_cb3, score_lgb3, score_cat3 = fit_lgbm(
    x_train1, y_train1['target3'],
    x_valid1, y_valid1['target3'],
    3, params
)
oof_pred_lgb4, model_lgb4, oof_pred_cat4, model_cb4, score_lgb4, score_cat4= fit_lgbm(
    x_train1, y_train1['target4'],
    x_valid1, y_valid1['target4'],
    4, params
)

score = (score_lgb1+score_lgb2+score_lgb3+score_lgb4) / 4
print(f'LightGBM score: {score}')

score = (score_cat1+score_cat2+score_cat3+score_cat4) / 4
print(f'Catboost score: {score}')

# %% [markdown]
# ## Inference

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-26T07:17:02.259872Z","iopub.status.idle":"2021-06-26T07:17:02.260506Z"}}
players_cols = ['playerId', 'primaryPositionName']
rosters_cols = ['playerId', 'teamId', 'status']
scores_cols = ['playerId', 'battingOrder', 'gamesPlayedBatting', 'flyOuts',
       'groundOuts', 'runsScored', 'doubles', 'triples', 'homeRuns',
       'strikeOuts', 'baseOnBalls', 'intentionalWalks', 'hits', 'hitByPitch',
       'atBats', 'caughtStealing', 'stolenBases', 'groundIntoDoublePlay',
       'groundIntoTriplePlay', 'plateAppearances', 'totalBases', 'rbi',
       'leftOnBase', 'sacBunts', 'sacFlies', 'catchersInterference',
       'pickoffs', 'gamesPlayedPitching', 'gamesStartedPitching',
       'completeGamesPitching', 'shutoutsPitching', 'winsPitching',
       'lossesPitching', 'flyOutsPitching', 'airOutsPitching',
       'groundOutsPitching', 'runsPitching', 'doublesPitching',
       'triplesPitching', 'homeRunsPitching', 'strikeOutsPitching',
       'baseOnBallsPitching', 'intentionalWalksPitching', 'hitsPitching',
       'hitByPitchPitching', 'atBatsPitching', 'caughtStealingPitching',
       'stolenBasesPitching', 'inningsPitched', 'saveOpportunities',
       'earnedRuns', 'battersFaced', 'outsPitching', 'pitchesThrown', 'balls',
       'strikes', 'hitBatsmen', 'balks', 'wildPitches', 'pickoffsPitching',
       'rbiPitching', 'gamesFinishedPitching', 'inheritedRunners',
       'inheritedRunnersScored', 'catchersInterferencePitching',
       'sacBuntsPitching', 'sacFliesPitching', 'saves', 'holds', 'blownSaves',
       'assists', 'putOuts', 'errors', 'chances']

null = np.nan
true = True
false = False

# %% [code] {"execution":{"iopub.status.busy":"2021-06-26T07:17:02.26162Z","iopub.status.idle":"2021-06-26T07:17:02.262287Z"}}
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import gc
from functools import reduce
from sklearn.model_selection import StratifiedKFold

ROOT_DIR = "../input/mlb-player-digital-engagement-forecasting"

#=======================#
def flatten(df, col):
    du = (df.pivot(index="playerId", columns="EvalDate", 
               values=col).add_prefix(f"{col}_").
      rename_axis(None, axis=1).reset_index())
    return du
#============================#
def reducer(left, right):
    return left.merge(right, on="playerId")
#========================

TGTCOLS = ["target1","target2","target3","target4"]
def train_lag(df, lag=1):
    dp = df[["playerId","EvalDate"]+TGTCOLS].copy()
    dp["EvalDate"]  =dp["EvalDate"] + timedelta(days=lag) 
    df = df.merge(dp, on=["playerId", "EvalDate"], suffixes=["",f"_{lag}"], how="left")
    return df
#=================================
def test_lag(sub):
    sub["playerId"] = sub["date_playerId"].apply(lambda s: int(  s.split("_")[1]  ) )
    assert sub.date.nunique() == 1
    dte = sub["date"].unique()[0]
    
    eval_dt = pd.to_datetime(dte, format="%Y%m%d")
    dtes = [eval_dt + timedelta(days=-k) for k in LAGS]
    mp_dtes = {eval_dt + timedelta(days=-k):k for k in LAGS}
    
    sl = LAST.loc[LAST.EvalDate.between(dtes[-1], dtes[0]), ["EvalDate","playerId"]+TGTCOLS].copy()
    sl["EvalDate"] = sl["EvalDate"].map(mp_dtes)
    du = [flatten(sl, col) for col in TGTCOLS]
    du = reduce(reducer, du)
    return du, eval_dt
    #
#===============

tr = pd.read_csv("../input/mlb-data/target.csv")
print(tr.shape)
gc.collect()

tr["EvalDate"] = pd.to_datetime(tr["EvalDate"])
tr["EvalDate"] = tr["EvalDate"] + timedelta(days=-1)
tr["EvalYear"] = tr["EvalDate"].dt.year

MED_DF = tr.groupby(["playerId","EvalYear"])[TGTCOLS].median().reset_index()
MEDCOLS = ["tgt1_med","tgt2_med", "tgt3_med", "tgt4_med"]
MED_DF.columns = ["playerId","EvalYear"] + MEDCOLS

LAGS = list(range(1,21))
FECOLS = [f"{col}_{lag}" for lag in reversed(LAGS) for col in TGTCOLS]

for lag in tqdm(LAGS):
    tr = train_lag(tr, lag=lag)
    gc.collect()
#===========
tr = tr.sort_values(by=["playerId", "EvalDate"])
print(tr.shape)
tr = tr.dropna()
print(tr.shape)
tr = tr.merge(MED_DF, on=["playerId","EvalYear"])
gc.collect()

X = tr[FECOLS+MEDCOLS].values
y = tr[TGTCOLS].values
cl = tr["playerId"].values

NFOLDS = 6
skf = StratifiedKFold(n_splits=NFOLDS)
folds = skf.split(X, cl)
folds = list(folds)

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

tf.random.set_seed(777)

def make_model(n_in):
    inp = L.Input(name="inputs", shape=(n_in,))
    x = L.Dense(50, activation="relu", name="d1")(inp)
    x = L.Dense(50, activation="relu", name="d2")(x)
    preds = L.Dense(4, activation="linear", name="preds")(x)
    
    model = M.Model(inp, preds, name="ANN")
    model.compile(loss="mean_absolute_error", optimizer="adam")
    return model

net = make_model(X.shape[1])
print(net.summary())

oof = np.zeros(y.shape)
nets = []
for idx in range(NFOLDS):
    print("FOLD:", idx)
    tr_idx, val_idx = folds[idx]
    ckpt = ModelCheckpoint(f"w{idx}.h5", monitor='val_loss', verbose=1, save_best_only=True,mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=3, min_lr=0.0005)
    es = EarlyStopping(monitor='val_loss', patience=6)
    reg = make_model(X.shape[1])
#     reg.fit(X[tr_idx], y[tr_idx], epochs=10, batch_size=35_000, validation_data=(X[val_idx], y[val_idx]),
#             verbose=1, callbacks=[ckpt, reduce_lr, es])
    reg.load_weights(f"w{idx}.h5")
    oof[val_idx] = reg.predict(X[val_idx], batch_size=50_000, verbose=1)
    nets.append(reg)
    gc.collect()
    #
#

mae = mean_absolute_error(y, oof)
mse = mean_squared_error(y, oof, squared=False)
print("mae:", mae)
print("mse:", mse)

# Historical information to use in prediction time
bound_dt = pd.to_datetime("2021-01-01")
LAST = tr.loc[tr.EvalDate>bound_dt].copy()

LAST_MED_DF = MED_DF.loc[MED_DF.EvalYear==2021].copy()
LAST_MED_DF.drop("EvalYear", axis=1, inplace=True)
del tr

#"""
import mlb
FE = []; SUB = [];

# %% [markdown]
# <div class="alert alert-success">  
# </div>

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-26T07:17:02.263332Z","iopub.status.idle":"2021-06-26T07:17:02.263974Z"}}
import copy

env = mlb.make_env() # initialize the environment
iter_test = env.iter_test() # iterator which loops over each date in test set

for (test_df, sample_prediction_df) in iter_test: # make predictions here
    
    sub = copy.deepcopy(sample_prediction_df.reset_index())
    sample_prediction_df = copy.deepcopy(sample_prediction_df.reset_index(drop=True))
    
    # LGBM summit
    # creat dataset
    sample_prediction_df['playerId'] = sample_prediction_df['date_playerId']\
                                        .map(lambda x: int(x.split('_')[1]))
    # Dealing with missing values
    if test_df['rosters'].iloc[0] == test_df['rosters'].iloc[0]:
        test_rosters = pd.DataFrame(eval(test_df['rosters'].iloc[0]))
    else:
        test_rosters = pd.DataFrame({'playerId': sample_prediction_df['playerId']})
        for col in rosters.columns:
            if col == 'playerId': continue
            test_rosters[col] = np.nan
            
    if test_df['playerBoxScores'].iloc[0] == test_df['playerBoxScores'].iloc[0]:
        test_scores = pd.DataFrame(eval(test_df['playerBoxScores'].iloc[0]))
    else:
        test_scores = pd.DataFrame({'playerId': sample_prediction_df['playerId']})
        for col in scores.columns:
            if col == 'playerId': continue
            test_scores[col] = np.nan
    test_scores = test_scores.groupby('playerId').sum().reset_index()
    test = sample_prediction_df[['playerId']].copy()
    test = test.merge(players[players_cols], on='playerId', how='left')
    test = test.merge(test_rosters[rosters_cols], on='playerId', how='left')
    test = test.merge(test_scores[scores_cols], on='playerId', how='left')
    test = test.merge(player_target_stats, how='inner', left_on=["playerId"],right_on=["playerId"])
    

    test['label_playerId'] = test['playerId'].map(player2num)
    test['label_primaryPositionName'] = test['primaryPositionName'].map(position2num)
    test['label_teamId'] = test['teamId'].map(teamid2num)
    test['label_status'] = test['status'].map(status2num)
    
    test_X = test[feature_cols]
    # predict
    pred1 = model1.predict(test_X)
    
    # predict
    pred_lgd1 = model_lgb1.predict(test_X)
    pred_lgd2 = model_lgb2.predict(test_X)
    pred_lgd3 = model_lgb3.predict(test_X)
    pred_lgd4 = model_lgb4.predict(test_X)
    
    pred_cat1 = model_cb1.predict(test_X)
    pred_cat2 = model_cb2.predict(test_X)
    pred_cat3 = model_cb3.predict(test_X)
    pred_cat4 = model_cb4.predict(test_X)
    
    test['target1'] = np.clip(pred1,0,100)
    test_X = test[feature_cols2]

    pred2 = model2.predict(test_X)
    pred3 = model3.predict(test_X)
    pred4 = model4.predict(test_X)
    
    # merge submission
    sample_prediction_df['target1'] = 0.65*np.clip(pred1, 0, 100)+0.25*np.clip(pred_lgd1, 0, 100)+0.10*np.clip(pred_cat1, 0, 100)
    sample_prediction_df['target2'] = 0.65*np.clip(pred2, 0, 100)+0.25*np.clip(pred_lgd2, 0, 100)+0.10*np.clip(pred_cat2, 0, 100)
    sample_prediction_df['target3'] = 0.65*np.clip(pred3, 0, 100)+0.25*np.clip(pred_lgd3, 0, 100)+0.10*np.clip(pred_cat3, 0, 100)
    sample_prediction_df['target4'] = 0.65*np.clip(pred4, 0, 100)+0.25*np.clip(pred_lgd4, 0, 100)+0.10*np.clip(pred_cat4, 0, 100)
    sample_prediction_df = sample_prediction_df.fillna(0.)
    del sample_prediction_df['playerId']
    # TF summit
    # Features computation at Evaluation Date
    sub_fe, eval_dt = test_lag(sub)
    sub_fe = sub_fe.merge(LAST_MED_DF, on="playerId", how="left")
    sub_fe = sub_fe.fillna(0.)
    
    _preds = 0.
    for reg in nets:
        _preds += reg.predict(sub_fe[FECOLS + MEDCOLS]) / NFOLDS
    sub_fe[TGTCOLS] = np.clip(_preds, 0, 100)
    sub.drop(["date"]+TGTCOLS, axis=1, inplace=True)
    sub = sub.merge(sub_fe[["playerId"]+TGTCOLS], on="playerId", how="left")
    sub.drop("playerId", axis=1, inplace=True)
    sub = sub.fillna(0.)
    # Blending
    blend = pd.concat(
        [sub[['date_playerId']],
        (0.35*sub.drop('date_playerId', axis=1) + 0.65*sample_prediction_df.drop('date_playerId', axis=1))],
        axis=1
    )
    env.predict(blend)
    # Update Available information
    sub_fe["EvalDate"] = eval_dt
    #sub_fe.drop(MEDCOLS, axis=1, inplace=True)
    LAST = LAST.append(sub_fe)
    LAST = LAST.drop_duplicates(subset=["EvalDate","playerId"], keep="last")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-26T07:17:02.264951Z","iopub.status.idle":"2021-06-26T07:17:02.265581Z"}}
pd.concat(
    [sub[['date_playerId']],
    (sub.drop('date_playerId', axis=1) + sample_prediction_df.drop('date_playerId', axis=1)) / 2],
    axis=1
)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-26T07:17:02.26657Z","iopub.status.idle":"2021-06-26T07:17:02.267169Z"}}
sample_prediction_df

# %% [markdown]
# <div class="alert alert-success">  
# </div>
