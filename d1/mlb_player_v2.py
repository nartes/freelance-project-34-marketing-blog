# %% [markdown] {"papermill":{"duration":0.099672,"end_time":"2021-06-11T18:42:32.853471","exception":false,"start_time":"2021-06-11T18:42:32.753799","status":"completed"},"tags":[]}
# # **MLB Player Digital Engagementコンペ概略**&#x1f600; 
# 
# ## ※ English page is here : https://www.kaggle.com/chumajin/eda-of-mlb-for-starter-english-ver
# 
# ## このコンペは、MLBのplayer idごとに、次の日(将来)にファンがデジタルコンテンツへのエンゲージメント（「反応」「行動」みたいなもの)をどれくらい起こすかというのを数値化したもの(target)を予測するコンペだと思います。targetは1～4で、それぞれ異なる指標で4つあって、0-100のスケールで数値化したものだそうです。
#    (コメントいただきました。ありがとうございます!!　たしかにサポーターなどのtwitterの書き込みとか、どこかのサイトへのアクセスなどそういうのを想像するとイメージしやすいですね。)
# 
# 

# %% [markdown] {"papermill":{"duration":0.10241,"end_time":"2021-06-11T18:42:33.052395","exception":false,"start_time":"2021-06-11T18:42:32.949985","status":"completed"},"tags":[]}
# ## もし、少しでもお役に立てば、**upvote**いただけたら嬉しいです！　他notebookでもupvoteいただけた方いつもありがとうございます。
# 
# ## また、基本的には、この事務局のスターターを見て、EDAを理解していきました(一部抜粋)。ありがとうございます。
# 
# ## こちらもupvoteお願いいたします。
# 
# https://www.kaggle.com/ryanholbrook/getting-started-with-mlb-player-digital-engagement

# %% [code] {"papermill":{"duration":1.060051,"end_time":"2021-06-11T18:42:34.209063","exception":false,"start_time":"2021-06-11T18:42:33.149012","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:06:08.741917Z","iopub.execute_input":"2021-06-14T09:06:08.742307Z","iopub.status.idle":"2021-06-14T09:06:09.566612Z","shell.execute_reply.started":"2021-06-14T09:06:08.742273Z","shell.execute_reply":"2021-06-14T09:06:09.565664Z"}}
import pprint
def display(*args, **kwargs):
    pprint.pprint(
        dict(
            args=args,
            kwargs=kwargs,
        ),
    )

import gc
import sys
import warnings
from pathlib import Path

import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
warnings.simplefilter("ignore")

# %% [markdown] {"papermill":{"duration":0.099753,"end_time":"2021-06-11T18:42:34.425178","exception":false,"start_time":"2021-06-11T18:42:34.325425","status":"completed"},"tags":[]}
# # 0. 何を予測するか (submissionファイルから見ちゃいます)

# %% [code] {"papermill":{"duration":0.148821,"end_time":"2021-06-11T18:42:34.687273","exception":false,"start_time":"2021-06-11T18:42:34.538452","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:06:13.75186Z","iopub.execute_input":"2021-06-14T09:06:13.752423Z","iopub.status.idle":"2021-06-14T09:06:13.804719Z","shell.execute_reply.started":"2021-06-14T09:06:13.752383Z","shell.execute_reply":"2021-06-14T09:06:13.803356Z"}}
example_sample_submission = pd.read_csv("../input/mlb-player-digital-engagement-forecasting/example_sample_submission.csv")
example_sample_submission

# %% [markdown] {"papermill":{"duration":0.092961,"end_time":"2021-06-11T18:42:34.873875","exception":false,"start_time":"2021-06-11T18:42:34.780914","status":"completed"},"tags":[]}
# playeridごとに、次の日(将来)にファンがデジタルコンテンツへのエンゲージメント（「反応」「行動」みたいなもの)をどれくらい起こすかというのを数値化したもの(target)を予測するコンペ。
# 
# targetは1～4で、それぞれ異なる指標で4つあって、0-100のスケールで数値化したものだそうです。
# 

# %% [markdown] {"papermill":{"duration":0.09515,"end_time":"2021-06-11T18:42:35.062356","exception":false,"start_time":"2021-06-11T18:42:34.967206","status":"completed"},"tags":[]}
# ## 0.1 どの情報から推測 ? (先にテストデータを見ちゃいます)

# %% [code] {"papermill":{"duration":0.895014,"end_time":"2021-06-11T18:42:36.051171","exception":false,"start_time":"2021-06-11T18:42:35.156157","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:07:01.149956Z","iopub.execute_input":"2021-06-14T09:07:01.150331Z","iopub.status.idle":"2021-06-14T09:07:02.04348Z","shell.execute_reply.started":"2021-06-14T09:07:01.1503Z","shell.execute_reply":"2021-06-14T09:07:02.042485Z"}}
example_test = pd.read_csv("../input/mlb-player-digital-engagement-forecasting/example_test.csv")
example_test

# %% [markdown] {"papermill":{"duration":0.093244,"end_time":"2021-06-11T18:42:36.239027","exception":false,"start_time":"2021-06-11T18:42:36.145783","status":"completed"},"tags":[]}
# パッと見て、submissionに出てくるplayer IDとかがすぐわかる感じではなさそう。json形式でいろいろな情報が入っていそう。
# 
# 
# テストデータは1日に1行のデータからなっている。
# 
# 
# 例えば、starterコードからの関数を使用すると、以下のように展開できる。

# %% [code] {"papermill":{"duration":0.105661,"end_time":"2021-06-11T18:42:36.437921","exception":false,"start_time":"2021-06-11T18:42:36.33226","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:07:46.275371Z","iopub.execute_input":"2021-06-14T09:07:46.275752Z","iopub.status.idle":"2021-06-14T09:07:46.280719Z","shell.execute_reply.started":"2021-06-14T09:07:46.275719Z","shell.execute_reply":"2021-06-14T09:07:46.279482Z"}}
# Helper function to unpack json found in daily data
def unpack_json(json_str):
    return np.nan if pd.isna(json_str) else pd.read_json(json_str)

# %% [code] {"papermill":{"duration":0.211983,"end_time":"2021-06-11T18:42:36.743198","exception":false,"start_time":"2021-06-11T18:42:36.531215","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:07:52.045349Z","iopub.execute_input":"2021-06-14T09:07:52.045719Z","iopub.status.idle":"2021-06-14T09:07:52.170803Z","shell.execute_reply.started":"2021-06-14T09:07:52.045686Z","shell.execute_reply":"2021-06-14T09:07:52.169701Z"}}
example_test.head(3)

# %% [markdown] {"papermill":{"duration":0.093974,"end_time":"2021-06-11T18:42:36.931685","exception":false,"start_time":"2021-06-11T18:42:36.837711","status":"completed"},"tags":[]}
# example_test["games"].iloc[0] の中身を見てみる

# %% [code] {"papermill":{"duration":0.165624,"end_time":"2021-06-11T18:42:37.192619","exception":false,"start_time":"2021-06-11T18:42:37.026995","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:08:00.433132Z","iopub.execute_input":"2021-06-14T09:08:00.43354Z","iopub.status.idle":"2021-06-14T09:08:00.50561Z","shell.execute_reply.started":"2021-06-14T09:08:00.433495Z","shell.execute_reply":"2021-06-14T09:08:00.50448Z"}}
unpack_json(example_test["games"].iloc[0])

# %% [markdown] {"papermill":{"duration":0.094383,"end_time":"2021-06-11T18:42:37.381562","exception":false,"start_time":"2021-06-11T18:42:37.287179","status":"completed"},"tags":[]}
# example_test["rosters"].iloc[0] の中身を見てみる

# %% [code] {"papermill":{"duration":0.126356,"end_time":"2021-06-11T18:42:37.603497","exception":false,"start_time":"2021-06-11T18:42:37.477141","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:09:08.225885Z","iopub.execute_input":"2021-06-14T09:09:08.226269Z","iopub.status.idle":"2021-06-14T09:09:08.255073Z","shell.execute_reply.started":"2021-06-14T09:09:08.226235Z","shell.execute_reply":"2021-06-14T09:09:08.254404Z"}}
unpack_json(example_test["rosters"].iloc[0])

# %% [markdown] {"papermill":{"duration":0.094364,"end_time":"2021-06-11T18:42:37.794102","exception":false,"start_time":"2021-06-11T18:42:37.699738","status":"completed"},"tags":[]}
# この辺の情報から、player idごとに次の日のtarget1～4という評価項目の期待値を推測するコンペだと思います。

# %% [code] {"papermill":{"duration":0.097753,"end_time":"2021-06-11T18:42:37.988036","exception":false,"start_time":"2021-06-11T18:42:37.890283","status":"completed"},"tags":[]}


# %% [markdown] {"papermill":{"duration":0.095565,"end_time":"2021-06-11T18:42:38.18146","exception":false,"start_time":"2021-06-11T18:42:38.085895","status":"completed"},"tags":[]}
# ---------以上を踏まえて、trainデータなど他のデータを見ていきます---------

# %% [markdown] {"papermill":{"duration":0.094539,"end_time":"2021-06-11T18:42:38.370792","exception":false,"start_time":"2021-06-11T18:42:38.276253","status":"completed"},"tags":[]}
# # 1. train.csv

# %% [code] {"papermill":{"duration":73.740436,"end_time":"2021-06-11T18:43:52.206935","exception":false,"start_time":"2021-06-11T18:42:38.466499","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:09:47.628029Z","iopub.execute_input":"2021-06-14T09:09:47.628644Z","iopub.status.idle":"2021-06-14T09:11:03.816877Z","shell.execute_reply.started":"2021-06-14T09:09:47.6286Z","shell.execute_reply":"2021-06-14T09:11:03.815822Z"}}
# 読み込みに少し時間かかります。
training = pd.read_csv("../input/mlb-player-digital-engagement-forecasting/train.csv")
training

# %% [code] {"papermill":{"duration":0.108401,"end_time":"2021-06-11T18:43:52.411026","exception":false,"start_time":"2021-06-11T18:43:52.302625","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:12:42.628153Z","iopub.execute_input":"2021-06-14T09:12:42.628536Z","iopub.status.idle":"2021-06-14T09:12:42.64072Z","shell.execute_reply.started":"2021-06-14T09:12:42.628496Z","shell.execute_reply":"2021-06-14T09:12:42.639545Z"}}
# dateはdatetimeに変換
training['date'] = pd.to_datetime(training['date'], format="%Y%m%d")

# %% [code] {"papermill":{"duration":0.124469,"end_time":"2021-06-11T18:43:52.63163","exception":false,"start_time":"2021-06-11T18:43:52.507161","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:12:46.690978Z","iopub.execute_input":"2021-06-14T09:12:46.691391Z","iopub.status.idle":"2021-06-14T09:12:46.716353Z","shell.execute_reply.started":"2021-06-14T09:12:46.691356Z","shell.execute_reply":"2021-06-14T09:12:46.715114Z"}}
training.info()

# %% [markdown] {"papermill":{"duration":0.096956,"end_time":"2021-06-11T18:43:52.824788","exception":false,"start_time":"2021-06-11T18:43:52.727832","status":"completed"},"tags":[]}
# 1216日分のデータ。nullデータは無し。nanデータがところどころにある。

# %% [markdown] {"papermill":{"duration":0.09861,"end_time":"2021-06-11T18:43:53.01957","exception":false,"start_time":"2021-06-11T18:43:52.92096","status":"completed"},"tags":[]}
# ---------------------------------------------------------------------

# %% [markdown] {"papermill":{"duration":0.095137,"end_time":"2021-06-11T18:43:53.213022","exception":false,"start_time":"2021-06-11T18:43:53.117885","status":"completed"},"tags":[]}
# ## ここから**カラムごとにデータがあるところのjsonを事例として1つ見てみます**。
# 
# 上述したように、train.csvの中身も1つのセルの中にjsonファイル形式で、dataframeがさらにそれぞれ入っているような複雑な形をしています。
# 
# (結果から言うと、全部で1216日分のデータの1日に対して、約11個(nanもあるのでも少し少ないですが)のDataFrameが情報量としてぶら下がっているイメージで、かなりの情報量です。
# 
# ので、ここから少し長いです。イメージだけつかんで、読み流しても良いかもです。)
# 

# %% [code] {"papermill":{"duration":0.105297,"end_time":"2021-06-11T18:43:53.414068","exception":false,"start_time":"2021-06-11T18:43:53.308771","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:13:37.94206Z","iopub.execute_input":"2021-06-14T09:13:37.942469Z","iopub.status.idle":"2021-06-14T09:13:37.94928Z","shell.execute_reply.started":"2021-06-14T09:13:37.942422Z","shell.execute_reply":"2021-06-14T09:13:37.948339Z"}}
training.columns

# %% [markdown] {"papermill":{"duration":0.098188,"end_time":"2021-06-11T18:43:53.612128","exception":false,"start_time":"2021-06-11T18:43:53.51394","status":"completed"},"tags":[]}
# 1つ1つ入力するのが、めんどくさいので、naを抜いて、n番目(0だと一番上)のサンプルをdataframeにしてcolumn名と中身を見る関数を作っちゃいます。

# %% [code] {"papermill":{"duration":0.1263,"end_time":"2021-06-11T18:43:53.851813","exception":false,"start_time":"2021-06-11T18:43:53.725513","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:14:01.593291Z","iopub.execute_input":"2021-06-14T09:14:01.593682Z","iopub.status.idle":"2021-06-14T09:14:01.598603Z","shell.execute_reply.started":"2021-06-14T09:14:01.593649Z","shell.execute_reply":"2021-06-14T09:14:01.597731Z"}}
def exshow(col,n):
    tmp = training[col]
    tmp = tmp.dropna()
    tmpdf = unpack_json(tmp.iloc[n])
    print(tmpdf.columns)
    return tmpdf

# %% [markdown] {"papermill":{"duration":0.112367,"end_time":"2021-06-11T18:43:54.094326","exception":false,"start_time":"2021-06-11T18:43:53.981959","status":"completed"},"tags":[]}
# ## 1.1 nextDayPlayerEngagement (train.csvのcolumn1番目)
# 翌日以降のすべてのモデリング ターゲットを含むネストされた JSON。

# %% [code] {"papermill":{"duration":0.13531,"end_time":"2021-06-11T18:43:54.391578","exception":false,"start_time":"2021-06-11T18:43:54.256268","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:14:10.526008Z","iopub.execute_input":"2021-06-14T09:14:10.526394Z","iopub.status.idle":"2021-06-14T09:14:10.552638Z","shell.execute_reply.started":"2021-06-14T09:14:10.526362Z","shell.execute_reply":"2021-06-14T09:14:10.551683Z"}}
training.head(3)

# %% [code] {"papermill":{"duration":0.15348,"end_time":"2021-06-11T18:43:54.656058","exception":false,"start_time":"2021-06-11T18:43:54.502578","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:14:19.888531Z","iopub.execute_input":"2021-06-14T09:14:19.888888Z","iopub.status.idle":"2021-06-14T09:14:19.928418Z","shell.execute_reply.started":"2021-06-14T09:14:19.888857Z","shell.execute_reply":"2021-06-14T09:14:19.927415Z"}}
tmpdf = exshow("nextDayPlayerEngagement",0)
tmpdf

# %% [markdown] {"papermill":{"duration":0.101315,"end_time":"2021-06-11T18:43:54.857283","exception":false,"start_time":"2021-06-11T18:43:54.755968","status":"completed"},"tags":[]}
# * engagementMetricsDate - 米国太平洋時間に基づくプレーヤーエンゲージメント指標の日付（前日のゲーム、名簿、フィールド統計、トランザクション、賞などと一致します）。
# * playerId
# * target1
# * target2
# * target3
# * target4
# 
# 
# target1-target4は、0から100のスケールでのデジタルエンゲージメントの毎日のインデックスです。

# %% [markdown] {"papermill":{"duration":0.099862,"end_time":"2021-06-11T18:43:55.056252","exception":false,"start_time":"2021-06-11T18:43:54.95639","status":"completed"},"tags":[]}
# ここから、plyaerIdと次の日以降のtarget1～4を抜くんですね。

# %% [markdown] {"papermill":{"duration":0.108076,"end_time":"2021-06-11T18:43:55.265858","exception":false,"start_time":"2021-06-11T18:43:55.157782","status":"completed"},"tags":[]}
# ## 1.2 games(train.csvのcolumn2番目)
# 特定の日のすべてのゲーム情報を含むネストされた JSON。レギュラー シーズン、ポストシーズン、オールスター ゲームに加えて、スプリング トレーニングとエキシビション ゲームが含まれています。

# %% [code] {"papermill":{"duration":0.138443,"end_time":"2021-06-11T18:43:55.51297","exception":false,"start_time":"2021-06-11T18:43:55.374527","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:14:48.471833Z","iopub.execute_input":"2021-06-14T09:14:48.472245Z","iopub.status.idle":"2021-06-14T09:14:48.498208Z","shell.execute_reply.started":"2021-06-14T09:14:48.472214Z","shell.execute_reply":"2021-06-14T09:14:48.497415Z"}}
training.head(3)

# %% [code] {"papermill":{"duration":0.153552,"end_time":"2021-06-11T18:43:55.769473","exception":false,"start_time":"2021-06-11T18:43:55.615921","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:14:56.122131Z","iopub.execute_input":"2021-06-14T09:14:56.122758Z","iopub.status.idle":"2021-06-14T09:14:56.169907Z","shell.execute_reply.started":"2021-06-14T09:14:56.122703Z","shell.execute_reply":"2021-06-14T09:14:56.168899Z"}}
tmpdf = exshow("games",1) # 0番目（1番上はデータが一行しかなかったので、1にしました。)
tmpdf

# %% [markdown] {"papermill":{"duration":0.105256,"end_time":"2021-06-11T18:43:55.985414","exception":false,"start_time":"2021-06-11T18:43:55.880158","status":"completed"},"tags":[]}
# カラムの意味の翻訳は↓を開いてください。(長いので、hideしています。)

# %% [markdown] {"_kg_hide-input":true,"papermill":{"duration":0.109485,"end_time":"2021-06-11T18:43:56.199254","exception":false,"start_time":"2021-06-11T18:43:56.089769","status":"completed"},"tags":[]}
# * gamePk  : ゲームの一意の識別子。
# * gameType  :   ゲームの種類、さまざまな種類がここにあります。
# * season : 
# * gameDate : 
# * gameTimeUTC  : UTCでの始球式。
# * resumeDate  :  タイムゲームが再開されました（放棄された場合、それ以外の場合はnull）。
# * resumedFrom  :  タイムゲームは元々放棄されていました（放棄された場合、それ以外の場合はnull）。
# * codedGameState  :  ゲームのステータスコード、さまざまなタイプがここにあります。
# * detailedGameState  :  ゲームのステータス、さまざまな種類がここにあります。
# * isTie  :  ブール値。ゲームが引き分けで終了した場合はtrue。
# * gameNumber  :  ダブルヘッダーを区別するためのゲーム番号フラグ
# * doubleHeader  :  YはDH、Nはシングルゲーム、Sはスプリット
# * dayNight  :  スケジュールされた開始時間の昼または夜のフラグ。
# * scheduledInnings  :  予定イニング数。
# * gamesInSeries  :  現在のシリーズのゲーム数。
# * seriesDescription  :  現在のシリーズのテキスト説明。
# * homeId  :  ホームチームの一意の識別子。
# * homeName  :  ホームチーム名。
# * homeAbbrev  :  ホームチームの略語。
# * homeWins  :  ホームチームのシーズンの現在の勝利数。
# * homeLosses  :  ホームチームのシーズンでの現在の損失数。
# * homeWinPct  :  ホームチームの現在の勝率。
# * homeWinner  :  ブール値。ホームチームが勝った場合はtrue。
# * homeScore  :  ホームチームが得点するラン。
# * awayId  :  アウェイチームの一意の識別子。
# * awayName  :  アウェイチームの一意の識別子。
# * awayAbbrev  :  アウェイチームの略。
# * awayWins  :  アウェイチームのシーズンの現在の勝利数。
# * awayLosses  :  アウェイチームのシーズン中の現在の敗北数。
# * awayWinPct  :  アウェイチームの現在の勝率。
# * awayWinner  :  ブール値。離れたチームが勝った場合はtrue。
# * awayScore  :  アウェイチームが得点したラン。

# %% [markdown] {"papermill":{"duration":0.099984,"end_time":"2021-06-11T18:43:56.401399","exception":false,"start_time":"2021-06-11T18:43:56.301415","status":"completed"},"tags":[]}
# ## 1.3 rosters(train.csvのcolumn3番目)
# 特定の日のすべての名簿情報を含むネストされた JSON。インシーズンとオフシーズンのチーム名簿が含まれます。

# %% [code] {"papermill":{"duration":0.133667,"end_time":"2021-06-11T18:43:56.635608","exception":false,"start_time":"2021-06-11T18:43:56.501941","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:15:42.681691Z","iopub.execute_input":"2021-06-14T09:15:42.682297Z","iopub.status.idle":"2021-06-14T09:15:42.710025Z","shell.execute_reply.started":"2021-06-14T09:15:42.682226Z","shell.execute_reply":"2021-06-14T09:15:42.708629Z"}}
training.head(3)

# %% [code] {"papermill":{"duration":0.136156,"end_time":"2021-06-11T18:43:56.876362","exception":false,"start_time":"2021-06-11T18:43:56.740206","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:15:47.818485Z","iopub.execute_input":"2021-06-14T09:15:47.818837Z","iopub.status.idle":"2021-06-14T09:15:47.849949Z","shell.execute_reply.started":"2021-06-14T09:15:47.818807Z","shell.execute_reply":"2021-06-14T09:15:47.848977Z"}}
tmpdf = exshow("rosters",0) 
tmpdf

# %% [markdown] {"papermill":{"duration":0.110068,"end_time":"2021-06-11T18:43:57.094794","exception":false,"start_time":"2021-06-11T18:43:56.984726","status":"completed"},"tags":[]}
# * playerId-プレーヤーの一意の識別子。
# * gameDate
# * teamId-そのプレーヤーがその日にいるteamId。
# * statusCode-名簿ステータスの略語。
# * status-説明的な名簿のステータス。

# %% [markdown] {"papermill":{"duration":0.10842,"end_time":"2021-06-11T18:43:57.309461","exception":false,"start_time":"2021-06-11T18:43:57.201041","status":"completed"},"tags":[]}
# ## 1.4 playerBoxScores(train.csvのcolumn4番目)
# 特定の日のプレイヤー ゲーム レベルで集計されたゲーム統計を含むネストされた JSON。レギュラーシーズン、ポストシーズン、オールスターゲームが含まれます。

# %% [code] {"papermill":{"duration":0.131819,"end_time":"2021-06-11T18:43:57.543572","exception":false,"start_time":"2021-06-11T18:43:57.411753","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:15:51.122412Z","iopub.execute_input":"2021-06-14T09:15:51.122822Z","iopub.status.idle":"2021-06-14T09:15:51.150918Z","shell.execute_reply.started":"2021-06-14T09:15:51.122791Z","shell.execute_reply":"2021-06-14T09:15:51.149868Z"}}
training.head(3)

# %% [code] {"papermill":{"duration":0.193827,"end_time":"2021-06-11T18:43:57.843961","exception":false,"start_time":"2021-06-11T18:43:57.650134","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:15:52.259031Z","iopub.execute_input":"2021-06-14T09:15:52.259431Z","iopub.status.idle":"2021-06-14T09:15:52.344774Z","shell.execute_reply.started":"2021-06-14T09:15:52.259394Z","shell.execute_reply":"2021-06-14T09:15:52.343636Z"}}
tmpdf = exshow("playerBoxScores",0) 
tmpdf.head(5)

# %% [markdown] {"papermill":{"duration":0.107718,"end_time":"2021-06-11T18:43:58.056077","exception":false,"start_time":"2021-06-11T18:43:57.948359","status":"completed"},"tags":[]}
# カラムの意味の翻訳は↓を開いてください。(長いので、hideしています。)

# %% [markdown] {"_kg_hide-input":true,"papermill":{"duration":0.106169,"end_time":"2021-06-11T18:43:58.272551","exception":false,"start_time":"2021-06-11T18:43:58.166382","status":"completed"},"tags":[]}
# * home  : バイナリ、ホームチームの場合は1、離れている場合は0。
# * gamePk  :   ゲームの一意の識別子。
# * gameDate : 
# * gameTimeUTC  : UTCでの始球式。
# * teamId  :   チームの一意の識別子。
# * teamName : 
# * playerId  : プレーヤーの一意の識別子。
# * playerName : 
# * jerseyNum : 
# * positionCode  : 番号の位置コード、詳細はこちらです。
# * positionName  :  テキスト位置の表示、詳細はこちらです。
# * positionType  :  ポジショングループ、詳細はこちらです。
# * battingOrder  :  形式：「###」。最初の桁は打順スポットを示し、次の2桁はそのプレーヤーがその打順スポットを占めた順序を示します。例：「300」は、打順の3番目のスポットのスターターを示します。 4人目（900、901、902以降）が打順9位を占めることを示す「903」。ゲームに登場した場合にのみ入力されます。
# * gamesPlayedBatting  :  プレーヤーが打者、ランナー、または野手としてゲームに参加した場合は1。
# * flyOuts  :  ゲームの合計フライアウト。
# * groundOuts  :  ゲームのトータルグラウンドアウト。
# * runsScored  :  ゲームの合計ランが記録されました。
# * doubles  :  ゲームの合計は2倍です。
# * triples  :  ゲームの合計トリプル。
# * homeRuns  :  ゲームの総本塁打。
# * strikeOuts  :  ゲームの合計三振。
# * baseOnBalls  :  ゲームの合計ウォーク。
# * intentionalWalks  :  ゲームの故意四球。
# * hits  :  ゲームの総ヒット数。
# * hitByPitch  :  ピッチによるゲームの合計ヒット。
# * atBats  :  でのゲーム合計
# * caughtStealing  :  ゲームの合計が盗塁をキャッチしました。
# * stolenBases  :  ゲームの盗塁総数。
# * groundIntoDoublePlay  :  ゲームの合計併殺はに基づいています。
# * groundIntoTriplePlay  :  ゲームの合計 3 回プレイが基礎になります。
# * plateAppearances  :  ゲームの総打席。
# * totalBases  :  ゲームの総拠点数。
# * rbi  :  ゲームの合計打点。
# * leftOnBase  :  ゲームの総ランナーはベースに残った。
# * sacBunts  :  ゲームの合計犠牲バント。
# * sacFlies  :  ゲームの総犠牲フライ。
# * catchersInterference  :  ゲームのトータルキャッチャーの干渉が発生しました。
# * pickoffs  :  ゲームの合計回数がベースから外れました。
# * gamesPlayedPitching :  バイナリ、プレーヤーが投手としてゲームに参加した場合は 1。
# * gamesStartedPitching :  バイナリ、プレーヤーがゲームの先発投手だった場合は1。
# * completeGamesPitching  :  バイナリ、完投でクレジットされている場合は1。
# * shutoutsPitching  :  バイナリ、完封でクレジットされている場合は1。
# * winsPitching  :  バイナリ、勝利でクレジットされている場合は 1。
# * lossesPitching  :  バイナリ、損失がクレジットされている場合は1。
# * flyOutsPitching  :  許可されたフライアウトのゲーム合計。
# * airOutsPitching  :  エアアウト（フライアウト+ポップアウト）のゲーム合計が許可されます。
# * groundOutsPitching  :  ゲームの合計グラウンドアウトが許可されます。
# * runsPitching  :  ゲームの合計実行が許可されます。
# * doublesPitching  :  ゲームの合計は2倍になります。
# * triplesPitching  :  ゲームの合計トリプルが許可されます。
# * homeRunsPitching  :  ゲームの合計ホームランが許可されます。
# * strikeOutsPitching  :  ゲームの合計三振が許可されます。
# * baseOnBallsPitching  :  ゲームの合計歩行が許可されます。
# * intentionalWalksPitching  :  ゲームの故意四球の合計が許可されます。
# * hitsPitching  :  許可されるゲームの合計ヒット数。
# * hitByPitchPitching  :  許可されたピッチによるゲームの合計ヒット。
# * atBatsPitching  :  でのゲーム合計
# * caughtStealingPitching  :  ゲームの合計は、盗みをキャッチしました。
# * stolenBasesPitching  :  ゲームの盗塁の合計は許可されます。
# * inningsPitched  :  ゲームの総投球回。
# * saveOpportunities  :  バイナリ、保存の機会がある場合は1。
# * earnedRuns  :  ゲームの合計自責点は許可されています。
# * battersFaced  :  直面したゲームの総打者。
# * outsPitching  :  ゲームの合計アウトが記録されました。
# * pitchesThrown  :  投げられた投球のゲーム総数。
# * balls  :  投げられたゲームの合計ボール。
# * strikes  :  スローされたゲームの合計ストライク。
# * hitBatsmen  :  ゲームの総死球打者。
# * balks  :  ゲームの合計はボークします。
# * wildPitches  :  投げられた暴投のゲーム総数。
# * pickoffsPitching  :  ゲームのピックオフの総数。
# * rbiPitching  :  打点のゲーム総数は許可されています。
# * inheritedRunners  :  継承されたランナーのゲーム合計を想定。
# * inheritedRunnersScored :  得点した継承されたランナーのゲーム合計。
# * catchersInterferencePitching  :  キャッチャーの干渉のゲーム合計はバッテリーによって発生しました。
# * sacBuntsPitching  :  ゲームの犠牲バントの合計が許可されます。
# * sacFliesPitching  :  ゲームの犠牲フライは許可されています。
# * saves  :  バイナリ、保存でクレジットされている場合は1。
# * holds  :  バイナリ、保留がクレジットされている場合は1。
# * blownSaves  :  バイナリ、ブローセーブでクレジットされている場合は1。
# * assists  :  ゲームのアシスト総数。
# * putOuts  :  ゲームの刺殺の総数。
# * errors  :  ゲームのエラーの総数。
# * chances  :  ゲームのトータルフィールディングチャンス。

# %% [code] {"papermill":{"duration":0.102554,"end_time":"2021-06-11T18:43:58.478581","exception":false,"start_time":"2021-06-11T18:43:58.376027","status":"completed"},"tags":[]}


# %% [markdown] {"papermill":{"duration":0.108051,"end_time":"2021-06-11T18:43:58.692413","exception":false,"start_time":"2021-06-11T18:43:58.584362","status":"completed"},"tags":[]}
# ## 1.5 teamBoxScores(train.csvのcolumn5番目)
# 特定の日のチーム ゲーム レベルで集計されたゲーム統計を含むネストされた JSON。レギュラーシーズン、ポストシーズン、オールスターゲームが含まれます。

# %% [code] {"papermill":{"duration":0.131738,"end_time":"2021-06-11T18:43:58.930844","exception":false,"start_time":"2021-06-11T18:43:58.799106","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:16:00.738588Z","iopub.execute_input":"2021-06-14T09:16:00.738972Z","iopub.status.idle":"2021-06-14T09:16:00.765648Z","shell.execute_reply.started":"2021-06-14T09:16:00.738941Z","shell.execute_reply":"2021-06-14T09:16:00.764551Z"}}
training.head(3)

# %% [code] {"papermill":{"duration":0.15405,"end_time":"2021-06-11T18:43:59.189049","exception":false,"start_time":"2021-06-11T18:43:59.034999","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:16:01.366547Z","iopub.execute_input":"2021-06-14T09:16:01.367082Z","iopub.status.idle":"2021-06-14T09:16:01.4172Z","shell.execute_reply.started":"2021-06-14T09:16:01.36705Z","shell.execute_reply":"2021-06-14T09:16:01.415972Z"}}
tmpdf = exshow("teamBoxScores",0) 
tmpdf.head(5)

# %% [markdown] {"papermill":{"duration":0.121084,"end_time":"2021-06-11T18:43:59.427565","exception":false,"start_time":"2021-06-11T18:43:59.306481","status":"completed"},"tags":[]}
# カラムの意味の翻訳は↓を開いてください。(長いので、hideしています。)

# %% [markdown] {"_kg_hide-input":true,"papermill":{"duration":0.120566,"end_time":"2021-06-11T18:43:59.660476","exception":false,"start_time":"2021-06-11T18:43:59.53991","status":"completed"},"tags":[]}
# * home  : バイナリ、ホームチームの場合は1、離れている場合は0。
# * teamId  :   チームの一意の識別子。
# * gamePk  :   ゲームの一意の識別子。
# * gameDate : 
# * gameTimeUTC  : UTCでの始球式。
# * flyOuts  :  ゲームの合計フライアウト。
# * groundOuts  :  ゲームのトータルグラウンドアウト。
# * runsScored  :  ゲームの合計ランが記録されました。
# * doubles  :  ゲームの合計は2倍です。
# * triples  :  ゲームの合計トリプル。
# * homeRuns  :  ゲームの総本塁打。
# * strikeOuts  :  ゲームの合計三振。
# * baseOnBalls  :  ゲームの合計ウォーク。
# * intentionalWalks  :  ゲームの故意四球。
# * hits  :  ゲームの総ヒット数。
# * hitByPitch  :  ピッチによるゲームの合計ヒット。
# * atBats  :  でのゲーム合計
# * caughtStealing  :  ゲームの合計が盗塁をキャッチしました。
# * stolenBases  :  ゲームの盗塁総数。
# * groundIntoDoublePlay  :  ゲームの合計併殺はに基づいています。
# * groundIntoTriplePlay  :  ゲームの合計 3 回プレイが基礎になります。
# * plateAppearances  :  ゲームの総打席。
# * totalBases  :  ゲームの総拠点数。
# * rbi  :  ゲームの合計打点。
# * leftOnBase  :  ゲームの総ランナーはベースに残った。
# * sacBunts  :  ゲームの合計犠牲バント。
# * sacFlies  :  ゲームの総犠牲フライ。
# * catchersInterference  :  ゲームのトータルキャッチャーの干渉が発生しました。
# * pickoffs  :  ゲームの合計回数がベースから外れました。
# * airOutsPitching  :  エアアウト（フライアウト+ポップアウト）のゲーム合計が許可されます。
# * groundOutsPitching  :  ゲームの合計グラウンドアウトが許可されます。
# * runsPitching  :  ゲームの合計実行が許可されます。
# * doublesPitching  :  ゲームの合計は2倍になります。
# * triplesPitching  :  ゲームの合計トリプルが許可されます。
# * homeRunsPitching  :  ゲームの合計ホームランが許可されます。
# * strikeOutsPitching  :  ゲームの合計三振が許可されます。
# * baseOnBallsPitching  :  ゲームの合計歩行が許可されます。
# * intentionalWalksPitching  :  ゲームの故意四球の合計が許可されます。
# * hitsPitching  :  許可されるゲームの合計ヒット数。
# * hitByPitchPitching  :  許可されたピッチによるゲームの合計ヒット。
# * atBatsPitching  :  でのゲーム合計
# * caughtStealingPitching  :  ゲームの合計は、盗みをキャッチしました。
# * stolenBasesPitching  :  ゲームの盗塁の合計は許可されます。
# * inningsPitched  :  ゲームの総投球回。
# * earnedRuns  :  ゲームの合計自責点は許可されています。
# * battersFaced  :  直面したゲームの総打者。
# * outsPitching  :  ゲームの合計アウトが記録されました。
# * hitBatsmen  :  ゲームの総死球打者。
# * balks  :  ゲームの合計はボークします。
# * wildPitches  :  投げられた暴投のゲーム総数。
# * pickoffsPitching  :  ゲームのピックオフの総数。
# * rbiPitching  :  打点のゲーム総数は許可されています。
# * inheritedRunners  :  継承されたランナーのゲーム合計を想定。
# * inheritedRunnersScored :  得点した継承されたランナーのゲーム合計。
# * catchersInterferencePitching  :  キャッチャーの干渉のゲーム合計はバッテリーによって発生しました。
# * sacBuntsPitching  :  ゲームの犠牲バントの合計が許可されます。
# * sacFliesPitching  :  ゲームの犠牲フライは許可されています。

# %% [code] {"papermill":{"duration":0.120236,"end_time":"2021-06-11T18:43:59.90294","exception":false,"start_time":"2021-06-11T18:43:59.782704","status":"completed"},"tags":[]}


# %% [markdown] {"papermill":{"duration":0.124656,"end_time":"2021-06-11T18:44:00.150824","exception":false,"start_time":"2021-06-11T18:44:00.026168","status":"completed"},"tags":[]}
# ## 1.6 transactions(train.csvのcolumn6番目)
# 特定の日の MLB チームに関係するすべてのトランザクション情報を含むネストされた JSON。

# %% [code] {"papermill":{"duration":0.271551,"end_time":"2021-06-11T18:44:00.569742","exception":false,"start_time":"2021-06-11T18:44:00.298191","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:16:04.232745Z","iopub.execute_input":"2021-06-14T09:16:04.233092Z","iopub.status.idle":"2021-06-14T09:16:04.26048Z","shell.execute_reply.started":"2021-06-14T09:16:04.233063Z","shell.execute_reply":"2021-06-14T09:16:04.25898Z"}}
training.head(3)

# %% [code] {"papermill":{"duration":0.164462,"end_time":"2021-06-11T18:44:00.857988","exception":false,"start_time":"2021-06-11T18:44:00.693526","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:16:04.801025Z","iopub.execute_input":"2021-06-14T09:16:04.801379Z","iopub.status.idle":"2021-06-14T09:16:04.834846Z","shell.execute_reply.started":"2021-06-14T09:16:04.801351Z","shell.execute_reply":"2021-06-14T09:16:04.833668Z"}}
tmpdf = exshow("transactions",1) 
tmpdf

# %% [markdown] {"papermill":{"duration":0.108288,"end_time":"2021-06-11T18:44:01.074241","exception":false,"start_time":"2021-06-11T18:44:00.965953","status":"completed"},"tags":[]}
# * transactionId  : トランザクションの一意の識別子。
# * playerId  :   プレーヤーの一意の識別子。
# * playerName : 
# * date : 
# * fromTeamId  : プレーヤーの出身チームの一意の識別子。
# * fromTeamName : 
# * toTeamId  :   プレーヤーが行くチームの一意の識別子。
# * toTeamName : 
# * effectiveDate : 
# * resolutionDate : 
# * typeCode  : トランザクションステータスの略語。
# * typeDesc  :   トランザクションステータスの説明。
# * description  :   トランザクションのテキスト説明。

# %% [markdown] {"papermill":{"duration":0.112168,"end_time":"2021-06-11T18:44:01.294478","exception":false,"start_time":"2021-06-11T18:44:01.18231","status":"completed"},"tags":[]}
# ## 1.7 standings(train.csvのcolumn7番目)
# 特定の日の MLB チームに関するすべての順位情報を含むネストされた JSON。

# %% [code] {"papermill":{"duration":0.140989,"end_time":"2021-06-11T18:44:01.541726","exception":false,"start_time":"2021-06-11T18:44:01.400737","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:16:06.386009Z","iopub.execute_input":"2021-06-14T09:16:06.386417Z","iopub.status.idle":"2021-06-14T09:16:06.415615Z","shell.execute_reply.started":"2021-06-14T09:16:06.386382Z","shell.execute_reply":"2021-06-14T09:16:06.414552Z"}}
training.head(3)

# %% [code] {"papermill":{"duration":0.164115,"end_time":"2021-06-11T18:44:01.812397","exception":false,"start_time":"2021-06-11T18:44:01.648282","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:16:07.341281Z","iopub.execute_input":"2021-06-14T09:16:07.341726Z","iopub.status.idle":"2021-06-14T09:16:07.391635Z","shell.execute_reply.started":"2021-06-14T09:16:07.34169Z","shell.execute_reply":"2021-06-14T09:16:07.390416Z"}}
tmpdf = exshow("standings",0) 
tmpdf.head(5)

# %% [markdown] {"papermill":{"duration":0.112066,"end_time":"2021-06-11T18:44:02.037524","exception":false,"start_time":"2021-06-11T18:44:01.925458","status":"completed"},"tags":[]}
# カラムの意味の翻訳は↓を開いてください。(長いので、hideしています。)

# %% [markdown] {"_kg_hide-input":true,"papermill":{"duration":0.106878,"end_time":"2021-06-11T18:44:02.255318","exception":false,"start_time":"2021-06-11T18:44:02.14844","status":"completed"},"tags":[]}
# * season : 
# * gameDate : 
# * divisionId  : このチームが所属する部門を表す一意識別子。
# * teamId  :   チームの一意の識別子。
# * teamName : 
# * streakCode  : チームの現在の勝ち負けの連続の略語。最初の文字は勝ち負けを示し、数字はゲームの数です。
# * divisionRank  :  チームの部門における現在のランク。
# * leagueRank  :  リーグでのチームの現在のランク。
# * wildCardRank  :  ワイルドカードバースのチームの現在のランク。
# * leagueGamesBack  :  ゲームはチームのリーグに戻ります。
# * sportGamesBack  :  MLBのすべてに戻ってゲーム。
# * divisionGamesBack  :  チームの部門にゲームが戻ってきました。
# * wins  :  現在の勝利。
# * losses  :  現在の損失。
# * pct  :  現在の勝率。
# * runsAllowed  :  シーズン中に許可された実行。
# * runsScored  :  シーズンに得点したラン。
# * divisionChamp  :  チームが部門タイトルを獲得した場合はtrue。
# * divisionLeader  :  チームがディビジョンレースをリードしている場合はtrue。
# * wildCardLeader  :  チームがワイルドカードリーダーの場合はtrue。
# * eliminationNumber  :  ディビジョンレースから排除されるまでのゲーム数（チームの敗北+対戦相手の勝利）。
# * wildCardEliminationNumber  :  ワイルドカードレースから排除されるまでのゲーム数（チームの敗北+対戦相手の勝利）。
# * homeWins  :  ホームはシーズンに勝ちます。
# * homeLosses  :  シーズン中のホームロス。
# * awayWins  :  アウェイはシーズンに勝ちます。
# * awayLosses  :  シーズンのアウェイロス。
# * lastTenWins  :  過去10試合で勝ちました。
# * lastTenLosses  :  過去10試合で負けました。
# * extraInningWins  :  シーズンの追加イニングで勝ちます。
# * extraInningLosses  :  シーズンの追加イニングでの損失。
# * oneRunWins  :  シーズン中に1ランで勝ちます。
# * oneRunLosses  :  シーズン中に1ランで負けます。
# * dayWins  :  デイゲームはシーズンに勝ちます。
# * dayLosses Day game losses on the season. : 
# * nightWins  : ナイトゲームはシーズンに勝ちます。
# * nightLosses  :   シーズン中のナイトゲームの敗北。
# * grassWins  :   芝生のフィールドがシーズンに勝ちます。
# * grassLosses  :   季節の草地の損失。
# * turfWins  :   芝フィールドはシーズンに勝ちます。
# * turfLosses  :   シーズン中の芝フィールドの損失。
# * divWins  :   シーズン中にディビジョンの対戦相手に勝ちます。
# * divLosses  :   シーズン中のディビジョンの対戦相手に対する敗北。
# * alWins  :   シーズン中にALチームに勝ちます。
# * alLosses  :   シーズン中のALチームに対する敗北。
# * nlWins  :   シーズン中にNLチームに勝ちます。
# * nlLosses  :   シーズン中のNLチームに対する敗北。
# * xWinLossPct  :   スコアリングおよび許可されたランに基づく予想勝率.

# %% [code] {"papermill":{"duration":0.106775,"end_time":"2021-06-11T18:44:02.471542","exception":false,"start_time":"2021-06-11T18:44:02.364767","status":"completed"},"tags":[]}


# %% [markdown] {"papermill":{"duration":0.107466,"end_time":"2021-06-11T18:44:02.687027","exception":false,"start_time":"2021-06-11T18:44:02.579561","status":"completed"},"tags":[]}
# ## 1.8 awards(train.csvのcolumn8番目)
# 特定の日に配られたすべての賞または栄誉を含むネストされた JSON。

# %% [code] {"papermill":{"duration":0.134686,"end_time":"2021-06-11T18:44:02.93004","exception":false,"start_time":"2021-06-11T18:44:02.795354","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:16:11.266915Z","iopub.execute_input":"2021-06-14T09:16:11.267266Z","iopub.status.idle":"2021-06-14T09:16:11.294052Z","shell.execute_reply.started":"2021-06-14T09:16:11.267232Z","shell.execute_reply":"2021-06-14T09:16:11.29274Z"}}
training.head(3)

# %% [code] {"papermill":{"duration":0.135331,"end_time":"2021-06-11T18:44:03.177168","exception":false,"start_time":"2021-06-11T18:44:03.041837","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:16:12.039564Z","iopub.execute_input":"2021-06-14T09:16:12.039924Z","iopub.status.idle":"2021-06-14T09:16:12.061189Z","shell.execute_reply.started":"2021-06-14T09:16:12.039892Z","shell.execute_reply":"2021-06-14T09:16:12.060285Z"}}
tmpdf = exshow("awards",0) 
tmpdf

# %% [markdown] {"papermill":{"duration":0.109992,"end_time":"2021-06-11T18:44:03.399011","exception":false,"start_time":"2021-06-11T18:44:03.289019","status":"completed"},"tags":[]}
# * awardId : 
# * awardName : 
# * awardDate  : 日付賞が与えられました。
# * awardSeason  :   シーズンアワードはからでした。
# * playerId  :   プレーヤーの一意の識別子。
# * playerName : 
# * awardPlayerTeamId : 

# %% [markdown] {"papermill":{"duration":0.1095,"end_time":"2021-06-11T18:44:03.617125","exception":false,"start_time":"2021-06-11T18:44:03.507625","status":"completed"},"tags":[]}
# ## 1.9 events(train.csvのcolumn9番目)
# 特定の日のすべてのオンフィールド ゲーム イベントを含むネストされた JSON。レギュラーシーズンとポストシーズンの試合が含まれます。

# %% [code] {"papermill":{"duration":0.137444,"end_time":"2021-06-11T18:44:03.868309","exception":false,"start_time":"2021-06-11T18:44:03.730865","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:16:15.023498Z","iopub.execute_input":"2021-06-14T09:16:15.023833Z","iopub.status.idle":"2021-06-14T09:16:15.049979Z","shell.execute_reply.started":"2021-06-14T09:16:15.023804Z","shell.execute_reply":"2021-06-14T09:16:15.048934Z"}}
training.head(3)

# %% [code] {"papermill":{"duration":0.464152,"end_time":"2021-06-11T18:44:04.44451","exception":false,"start_time":"2021-06-11T18:44:03.980358","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:16:15.920155Z","iopub.execute_input":"2021-06-14T09:16:15.920493Z","iopub.status.idle":"2021-06-14T09:16:16.20824Z","shell.execute_reply.started":"2021-06-14T09:16:15.920465Z","shell.execute_reply":"2021-06-14T09:16:16.206929Z"}}
tmpdf = exshow("events",0) 
tmpdf.head(5)

# %% [markdown] {"papermill":{"duration":0.110823,"end_time":"2021-06-11T18:44:04.666523","exception":false,"start_time":"2021-06-11T18:44:04.5557","status":"completed"},"tags":[]}
# カラムの意味の翻訳は↓を開いてください。(長いので、hideしています。)

# %% [markdown] {"_kg_hide-input":true,"papermill":{"duration":0.110798,"end_time":"2021-06-11T18:44:04.89031","exception":false,"start_time":"2021-06-11T18:44:04.779512","status":"completed"},"tags":[]}
# * gamePk  : ゲームの一意の識別子。
# * gameDate : 
# * gameTimeUTC  : UTCでの始球式。
# * season : 
# * gameType  : ゲームの種類、さまざまな種類がここにあります。
# * playId  :  スタットキャストのプレイガイド。
# * eventId : 
# * inning  : イニングABが発生しました。
# * halfInning :   「上」または「下」のイニングインジケーター。
# * homeScore  :   イベント開始時のホームスコア。
# * awayScore  :   イベント開始時のアウェイスコア。
# * menOnBase  :   走者がベースにいる場合に使用されるスプリット–すなわち（RISP、空）。
# * atBatIndex  :   で
# * atBatDesc  :   演奏する
# * atBatEvent  :   atBatのイベントタイプの結果。さまざまなタイプがここにあります。
# * hasOut  :   バイナリ、ランナーが場に出ている場合は1。
# * pitcherTeamId  :   ピッチングチームの一意の識別子。
# * isPitcherHome  :   バイナリ、投手がホームチームの場合は1。
# * pitcherTeam  :   ピッチングチームのチーム名。
# * hitterTeamId  :   打撃チームの一意の識別子。
# * hitterTeam  :   打撃チームのチーム名。
# * pitcherId : 
# * pitcherName : 
# * isStarter  : バイナリ、プレーヤーがゲームの先発投手だった場合は1。
# * pitcherHand  :   プレーヤーが手を投げる：「L」、「R」。
# * hitterId : 
# * hitterName : 
# * batSide  : プレーヤーのバット側：「L」、「R」。
# * pitchNumber  :  ABのピッチシーケンス番号。
# * balls  :  イベント後のボール数。
# * strikes  :  イベント後のストライクカウント。
# * isGB  :  バイナリ、打席がグラウンドボールの場合は1。
# * isLD  :  バイナリ、打席がラインドライブの場合は1。
# * isFB  :  バイナリ、打席が飛球の場合は1。
# * isPU  :  バイナリ、打席がポップアップの場合は1。
# * launchSpeed  :  打球の測定速度。
# * launchAngle  :  ヒットが開始された地平線に対する垂直角度。
# * totalDistance  :  ボールが移動した合計距離。
# * event  :  で発生する可能性のあるイベント
# * description  :  イベントのテキスト説明。
# * rbi  :  AB中に打点を打った回数。
# * pitchType  :  ピッチタイプ分類コード。さまざまなタイプがここにあります。
# * call  :  投球または投球の結果分類コード。さまざまなタイプがここにあります。
# * outs  :  ABの現在/最終アウト。
# * inPlay  :  ボールが場に出た場合は真/偽。
# * isPaOver  :  バイナリ、このイベントがプレートの外観の終わりである場合は1。
# * startSpeed  :  ホームプレートの前50フィートでのボールのMPHでの速度。
# * endSpeed  :  ボールがホームプレートの前端（x軸で0,0）を横切るときのボールのMPHでの速度。
# * nastyFactor  :  各ピッチのいくつかのプロパティを評価し、ピッチの「不快感」を0からのスケールで評価します
# * breakAngle  :  ピッチの平面が垂直から外れる時計回り（打者の視点）の角度。
# * breakLength  :  ピッチがピッチ開始とピッチ終了の間の直線から離れる最大距離。
# * breakY  :  ブレークが最大のホームプレートからの距離。
# * spinRate  :  ピッチャーによってRPMでリリースされた後のボールのスピン率。
# * spinDirection  :  スピンがボールの弾道にどのように影響するかを反映する角度として与えられる、リリース時のボールの回転軸。ピュアバック
# * pX  :  ボールがホームプレートの前軸と交差するときのボールのフィート単位の水平位置。
# * pZ  :  ボールがホームプレートの前軸と交差するときの、ボールのホームプレートからのフィート単位の垂直位置。
# * aX  :  z軸のボール加速度。
# * aY  :  y軸のボール加速度。
# * aZ  :  z 軸上のボールの加速度。
# * pfxX  :  インチ単位のボールの水平方向の動き。
# * pfxZ  :  インチ単位のボールの垂直方向の動き。
# * vX0  :  x軸からのボールの速度。
# * vY0  :  y軸からのボールの速度。 0,0,0 はバッターの後ろにあり、ボールはピッチャー マウンドから 0,0,0 に向かって移動するため、これは負です。
# * vZ0  :  z軸からのボールの速度。
# * x  :  ピッチがホームプレートの前を横切ったX座標。
# * y  :  ピッチがホームプレートの前面と交差するY座標。
# * x0  :  ピッチャーの手を離したときのボールの x 軸上の座標位置 (時間 = 0)。
# * y0  :  y軸上でピッチャーの手からボールがリリースされたポイントでのボールの座標位置（時間= 0）。
# * z0  :  z軸上でピッチャーの手からボールがリリースされたポイントでのボールの座標位置（時間= 0）。
# * type  :  「ピッチ」または「アクション」のいずれかのイベントのタイプ
# * zone  :  ゾーンロケーション番号.下を参照
# 
# ![image.png](attachment:1ad951bc-0f08-4424-83c4-6ff88a557d7d.png)
# 

# %% [markdown] {"papermill":{"duration":0.114746,"end_time":"2021-06-11T18:44:05.117226","exception":false,"start_time":"2021-06-11T18:44:05.00248","status":"completed"},"tags":[]}
# ## 1.10 playerTwitterFollowers(train.csvのcolumn10番目)
# その日の一部のプレイヤーの Twitter フォロワー数を含むネストされた JSON。

# %% [code] {"papermill":{"duration":0.142856,"end_time":"2021-06-11T18:44:05.374626","exception":false,"start_time":"2021-06-11T18:44:05.23177","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:16:19.415876Z","iopub.execute_input":"2021-06-14T09:16:19.416249Z","iopub.status.idle":"2021-06-14T09:16:19.445478Z","shell.execute_reply.started":"2021-06-14T09:16:19.416219Z","shell.execute_reply":"2021-06-14T09:16:19.444498Z"}}
training.head(3)

# %% [code] {"papermill":{"duration":0.148668,"end_time":"2021-06-11T18:44:05.64077","exception":false,"start_time":"2021-06-11T18:44:05.492102","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:16:20.69924Z","iopub.execute_input":"2021-06-14T09:16:20.699607Z","iopub.status.idle":"2021-06-14T09:16:20.734173Z","shell.execute_reply.started":"2021-06-14T09:16:20.699576Z","shell.execute_reply":"2021-06-14T09:16:20.732938Z"}}
tmpdf = exshow("playerTwitterFollowers",0) 
tmpdf.head(3)

# %% [markdown] {"papermill":{"duration":0.115258,"end_time":"2021-06-11T18:44:05.87703","exception":false,"start_time":"2021-06-11T18:44:05.761772","status":"completed"},"tags":[]}
# Twitterのフォローデータは、MLBによってメジャーリーグプレーヤーのTwitter APIから毎月1日に収集され、2018年1月1日までさかのぼります。 すべてのプレーヤーがTwitterアカウントを持っている/持っているわけではない、プレーヤーがランダムにアカウントを作成/削除/復元する、または特定の日にフォロワーデータを収集できないその他のシナリオがあるため、このデータセットはすべての月にわたってすべてのプレーヤーを網羅しているわけではありません。

# %% [markdown] {"papermill":{"duration":0.1152,"end_time":"2021-06-11T18:44:06.109077","exception":false,"start_time":"2021-06-11T18:44:05.993877","status":"completed"},"tags":[]}
# * date  : フォロワー数の日付。
# * playerId  :   プレーヤーの一意の識別子。
# * playerName : 
# * accountName  : プレイヤーのTwitterアカウントの名前。
# * twitterHandle  :   プレイヤーのツイッターハンドル。
# * numberOfFollowers  :   フォロワー数

# %% [markdown] {"papermill":{"duration":0.117931,"end_time":"2021-06-11T18:44:06.340356","exception":false,"start_time":"2021-06-11T18:44:06.222425","status":"completed"},"tags":[]}
# ## 1.11 teamTwitterFollowers(train.csvのcolumn11番目)
# その日の各チームの Twitter フォロワー数を含むネストされた JSON。

# %% [code] {"papermill":{"duration":0.144909,"end_time":"2021-06-11T18:44:06.602116","exception":false,"start_time":"2021-06-11T18:44:06.457207","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:17:20.07115Z","iopub.execute_input":"2021-06-14T09:17:20.071555Z","iopub.status.idle":"2021-06-14T09:17:20.098844Z","shell.execute_reply.started":"2021-06-14T09:17:20.071521Z","shell.execute_reply":"2021-06-14T09:17:20.097876Z"}}
training.head(3)

# %% [code] {"papermill":{"duration":0.140245,"end_time":"2021-06-11T18:44:06.858481","exception":false,"start_time":"2021-06-11T18:44:06.718236","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:17:21.031613Z","iopub.execute_input":"2021-06-14T09:17:21.03204Z","iopub.status.idle":"2021-06-14T09:17:21.057308Z","shell.execute_reply.started":"2021-06-14T09:17:21.032005Z","shell.execute_reply":"2021-06-14T09:17:21.056119Z"}}
tmpdf = exshow("teamTwitterFollowers",0) 
tmpdf.head(3)

# %% [markdown] {"papermill":{"duration":0.119312,"end_time":"2021-06-11T18:44:07.09097","exception":false,"start_time":"2021-06-11T18:44:06.971658","status":"completed"},"tags":[]}
# Twitterのフォローデータは、2018年1月1日までさかのぼって、毎月1日に、メジャーリーグの30チームすべてのTwitterAPIからMLBによって収集されました。

# %% [markdown] {"papermill":{"duration":0.11246,"end_time":"2021-06-11T18:44:07.317214","exception":false,"start_time":"2021-06-11T18:44:07.204754","status":"completed"},"tags":[]}
# * date  : フォロワー数の日付。
# * teamId  :   チームの一意の識別子。
# * teamName : 
# * accountName  : チームのTwitterアカウントの名前。
# * twitterHandle  :   チームのツイッターハンドル。

# %% [markdown] {"papermill":{"duration":0.119547,"end_time":"2021-06-11T18:44:07.551012","exception":false,"start_time":"2021-06-11T18:44:07.431465","status":"completed"},"tags":[]}
# やっとこ中身確認完了。おつかれさまでした。。。

# %% [code] {"papermill":{"duration":0.120035,"end_time":"2021-06-11T18:44:07.783273","exception":false,"start_time":"2021-06-11T18:44:07.663238","status":"completed"},"tags":[]}


# %% [markdown] {"papermill":{"duration":0.112525,"end_time":"2021-06-11T18:44:08.009126","exception":false,"start_time":"2021-06-11T18:44:07.896601","status":"completed"},"tags":[]}
# # 2. 他のadditional data ( awards.csv, players.csv, seasons.csv, teams.csv)

# %% [markdown] {"papermill":{"duration":0.12237,"end_time":"2021-06-11T18:44:08.250799","exception":false,"start_time":"2021-06-11T18:44:08.128429","status":"completed"},"tags":[]}
# ## 2.1 starterにあったwidgetの練習(こんなことできるんだーと思いましたので・・・)

# %% [code] {"papermill":{"duration":0.120605,"end_time":"2021-06-11T18:44:08.489884","exception":false,"start_time":"2021-06-11T18:44:08.369279","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:17:55.345428Z","iopub.execute_input":"2021-06-14T09:17:55.345843Z","iopub.status.idle":"2021-06-14T09:17:55.35036Z","shell.execute_reply.started":"2021-06-14T09:17:55.345812Z","shell.execute_reply":"2021-06-14T09:17:55.349032Z"}}
df_names = ['seasons', 'teams', 'players', 'awards']

path = "../input/mlb-player-digital-engagement-forecasting"

# %% [code] {"papermill":{"duration":0.155209,"end_time":"2021-06-11T18:44:08.759151","exception":false,"start_time":"2021-06-11T18:44:08.603942","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:17:58.414493Z","iopub.execute_input":"2021-06-14T09:17:58.414844Z","iopub.status.idle":"2021-06-14T09:17:58.448828Z","shell.execute_reply.started":"2021-06-14T09:17:58.414816Z","shell.execute_reply":"2021-06-14T09:17:58.447643Z"}}
kaggle_data_tabs = widgets.Tab()
# widgetsにそれぞれのDataFrameをchildrenの中にタブで表示
kaggle_data_tabs.children = list([widgets.Output() for df_name in df_names])

# %% [code] {"papermill":{"duration":0.295868,"end_time":"2021-06-11T18:44:09.169213","exception":false,"start_time":"2021-06-11T18:44:08.873345","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:17:59.564008Z","iopub.execute_input":"2021-06-14T09:17:59.564414Z","iopub.status.idle":"2021-06-14T09:17:59.734201Z","shell.execute_reply.started":"2021-06-14T09:17:59.564381Z","shell.execute_reply":"2021-06-14T09:17:59.733044Z"}}
for index in range(len(df_names)):
    # タブのタイトルを設定
    kaggle_data_tabs.set_title(index, df_names[index])
    
    df = pd.read_csv(os.path.join(path,df_names[index]) + ".csv")
    
    # それぞれのタブにDataFrameを埋め込む
    with kaggle_data_tabs.children[index]:
        display(df)

# %% [code] {"papermill":{"duration":0.127693,"end_time":"2021-06-11T18:44:09.419366","exception":false,"start_time":"2021-06-11T18:44:09.291673","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:18:01.219466Z","iopub.execute_input":"2021-06-14T09:18:01.221633Z","iopub.status.idle":"2021-06-14T09:18:01.229718Z","shell.execute_reply.started":"2021-06-14T09:18:01.22159Z","shell.execute_reply":"2021-06-14T09:18:01.228737Z"}}
display(kaggle_data_tabs)

# %% [markdown] {"papermill":{"duration":0.112987,"end_time":"2021-06-11T18:44:09.648047","exception":false,"start_time":"2021-06-11T18:44:09.53506","status":"completed"},"tags":[]}
# -----------細かく一つ一つ見ていきます-----------

# %% [markdown] {"papermill":{"duration":0.128132,"end_time":"2021-06-11T18:44:09.894007","exception":false,"start_time":"2021-06-11T18:44:09.765875","status":"completed"},"tags":[]}
# ## 2.2 Seasons.csv

# %% [code] {"papermill":{"duration":0.150781,"end_time":"2021-06-11T18:44:10.190127","exception":false,"start_time":"2021-06-11T18:44:10.039346","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:19:00.781853Z","iopub.execute_input":"2021-06-14T09:19:00.782327Z","iopub.status.idle":"2021-06-14T09:19:00.807238Z","shell.execute_reply.started":"2021-06-14T09:19:00.782296Z","shell.execute_reply":"2021-06-14T09:19:00.806039Z"}}
seasons = pd.read_csv("../input/mlb-player-digital-engagement-forecasting/seasons.csv")
seasons

# %% [markdown] {"papermill":{"duration":0.116735,"end_time":"2021-06-11T18:44:10.442337","exception":false,"start_time":"2021-06-11T18:44:10.325602","status":"completed"},"tags":[]}
# * seasonId : シーズンID
# * seasonStartDate : シーズンスタート日
# * seasonEndDate : シーズン終了日
# * preSeasonStartDate : 1つ前のシーズンスタート日
# * preSeasonEndDate : 1つ前のシーズンの終わりの日
# * regularSeasonStartDate : レギュラーシーズンのスタートの日
# * regularSeasonEndDate : レギュラーシーズンの終わりの日
# * lastDate1stHalf : 1st halfの最終日
# * allStarDate : オールスター戦の日付
# * firstDate2ndHalf : 2nd halfの始まり日
# * postSeasonStartDate : 次のシーズンのスタート日
# * postSeasonEndDate : 次のシーズンの終わり日

# %% [code] {"papermill":{"duration":0.118911,"end_time":"2021-06-11T18:44:10.677851","exception":false,"start_time":"2021-06-11T18:44:10.55894","status":"completed"},"tags":[]}


# %% [markdown] {"papermill":{"duration":0.11553,"end_time":"2021-06-11T18:44:10.912934","exception":false,"start_time":"2021-06-11T18:44:10.797404","status":"completed"},"tags":[]}
# ## 2.3 teams.csv

# %% [code] {"papermill":{"duration":0.138966,"end_time":"2021-06-11T18:44:11.167983","exception":false,"start_time":"2021-06-11T18:44:11.029017","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:19:09.180321Z","iopub.execute_input":"2021-06-14T09:19:09.180697Z","iopub.status.idle":"2021-06-14T09:19:09.204548Z","shell.execute_reply.started":"2021-06-14T09:19:09.180668Z","shell.execute_reply":"2021-06-14T09:19:09.203438Z"}}
teams = pd.read_csv("../input/mlb-player-digital-engagement-forecasting/teams.csv")
teams.head(3)

# %% [markdown] {"papermill":{"duration":0.125126,"end_time":"2021-06-11T18:44:11.428349","exception":false,"start_time":"2021-06-11T18:44:11.303223","status":"completed"},"tags":[]}
# ## teams.csv
# * id - チームID
# * name : 名前
# * teamName : チームの名前
# * teamCode : チームのコード
# * shortName : 短い名前
# * abbreviation : 略語
# * locationName : 場所の名前
# * leagueId : リーグのid
# * leagueName : リーグの名前
# * divisionId : 部門id
# * divisionName : 部門名
# * venueId : 会場id
# * venueName : 会場名

# %% [markdown] {"papermill":{"duration":0.115273,"end_time":"2021-06-11T18:44:11.660746","exception":false,"start_time":"2021-06-11T18:44:11.545473","status":"completed"},"tags":[]}
# ## 2.4 players.csv

# %% [code] {"papermill":{"duration":0.146866,"end_time":"2021-06-11T18:44:11.923004","exception":false,"start_time":"2021-06-11T18:44:11.776138","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:19:17.018473Z","iopub.execute_input":"2021-06-14T09:19:17.019073Z","iopub.status.idle":"2021-06-14T09:19:17.050961Z","shell.execute_reply.started":"2021-06-14T09:19:17.019024Z","shell.execute_reply":"2021-06-14T09:19:17.049924Z"}}
players = pd.read_csv("../input/mlb-player-digital-engagement-forecasting/players.csv")
players.head(3)

# %% [markdown] {"papermill":{"duration":0.116271,"end_time":"2021-06-11T18:44:12.158173","exception":false,"start_time":"2021-06-11T18:44:12.041902","status":"completed"},"tags":[]}
# * playerId - Unique identifier for a player. : プレーヤーID-プレーヤーの一意の識別子。
# * playerName : プレーヤの名前
# * DOB - Player’s date of birth. : DOB-プレーヤーの生年月日。
# * mlbDebutDate : MLBデビュー日
# * birthCity : 生まれた町
# * birthStateProvince : 出生州
# * birthCountry : 生まれた国
# * heightInches : 身長(inch)
# * weight : 体重
# * primaryPositionCode - Player’s primary position code : 主要ポジションコード
# * primaryPositionName - player’s primary position : 主要ポジション名
# * playerForTestSetAndFuturePreds - Boolean, true if player is among those for whom predictions are to be made in test data
# 
# : **ブール値、プレーヤーがテストデータで予測が行われる対象の1人である場合はtrue**

# %% [code] {"papermill":{"duration":0.11651,"end_time":"2021-06-11T18:44:12.393967","exception":false,"start_time":"2021-06-11T18:44:12.277457","status":"completed"},"tags":[]}


# %% [markdown] {"papermill":{"duration":0.118357,"end_time":"2021-06-11T18:44:12.629827","exception":false,"start_time":"2021-06-11T18:44:12.51147","status":"completed"},"tags":[]}
# ## 2.5 awards.csv

# %% [code] {"papermill":{"duration":0.148565,"end_time":"2021-06-11T18:44:12.897385","exception":false,"start_time":"2021-06-11T18:44:12.74882","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:19:57.113713Z","iopub.execute_input":"2021-06-14T09:19:57.114402Z","iopub.status.idle":"2021-06-14T09:19:57.152653Z","shell.execute_reply.started":"2021-06-14T09:19:57.114352Z","shell.execute_reply":"2021-06-14T09:19:57.151616Z"}}
awards = pd.read_csv("../input/mlb-player-digital-engagement-forecasting/awards.csv")
awards.head(3)

# %% [markdown] {"papermill":{"duration":0.118007,"end_time":"2021-06-11T18:44:13.133412","exception":false,"start_time":"2021-06-11T18:44:13.015405","status":"completed"},"tags":[]}
# このファイルには、日次データの開始前（つまり、2018年以前）にトレーニングセットのプレーヤーが獲得した賞が含まれています。
# 
# * awardDate - Date award was given. : 授与日 - 授与された日付。
# * awardSeason - Season award was from. : アワードシーズン-シーズンアワードはからでした。
# * awardId : アワードid
# * awardName : アワード名
# * playerId - Unique identifier for a player. : プレーヤーID-プレーヤーの一意の識別子。
# * playerName : プレーヤーの名前
# * awardPlayerTeamId : プレイヤーのチームID

# %% [code] {"papermill":{"duration":0.116296,"end_time":"2021-06-11T18:44:13.36787","exception":false,"start_time":"2021-06-11T18:44:13.251574","status":"completed"},"tags":[]}


# %% [markdown] {"papermill":{"duration":0.115578,"end_time":"2021-06-11T18:44:13.601453","exception":false,"start_time":"2021-06-11T18:44:13.485875","status":"completed"},"tags":[]}
# # 3. Data Merge

# %% [markdown] {"papermill":{"duration":0.120359,"end_time":"2021-06-11T18:44:13.841012","exception":false,"start_time":"2021-06-11T18:44:13.720653","status":"completed"},"tags":[]}
# とりあえず、スターターhttps://www.kaggle.com/ryanholbrook/getting-started-with-mlb-player-digital-engagement　
# 
# のコピーです。けっこう時間かかります。

# %% [code] {"papermill":{"duration":0.164068,"end_time":"2021-06-11T18:44:14.121247","exception":false,"start_time":"2021-06-11T18:44:13.957179","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:20:09.176809Z","iopub.execute_input":"2021-06-14T09:20:09.177183Z","iopub.status.idle":"2021-06-14T09:20:09.220929Z","shell.execute_reply.started":"2021-06-14T09:20:09.177151Z","shell.execute_reply":"2021-06-14T09:20:09.2202Z"}}
for name in df_names:
    globals()[name] = pd.read_csv(os.path.join(path,name)+ ".csv")

# %% [code] {"papermill":{"duration":306.845173,"end_time":"2021-06-11T18:49:21.088167","exception":false,"start_time":"2021-06-11T18:44:14.242994","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:20:10.410562Z","iopub.execute_input":"2021-06-14T09:20:10.411092Z","iopub.status.idle":"2021-06-14T09:25:36.408897Z","shell.execute_reply.started":"2021-06-14T09:20:10.411043Z","shell.execute_reply":"2021-06-14T09:25:36.40774Z"}}
#### Unnest various nested data within training (daily) data ####
daily_data_unnested_dfs = pd.DataFrame(data = {
  'dfName': training.drop('date', axis = 1).columns.values.tolist()
  })

# Slow from this point !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

daily_data_unnested_dfs['df'] = [pd.DataFrame() for row in 
  daily_data_unnested_dfs.iterrows()]

for df_index, df_row in daily_data_unnested_dfs.iterrows():
    nestedTableName = str(df_row['dfName'])
    
    date_nested_table = training[['date', nestedTableName]]
    
    date_nested_table = (date_nested_table[
      ~pd.isna(date_nested_table[nestedTableName])
      ].
      reset_index(drop = True)
      )
    
    daily_dfs_collection = []
    
    for date_index, date_row in date_nested_table.iterrows():
        daily_df = unpack_json(date_row[nestedTableName])
        
        daily_df['dailyDataDate'] = date_row['date']
        
        daily_dfs_collection = daily_dfs_collection + [daily_df]

    unnested_table = pd.concat(daily_dfs_collection,
      ignore_index = True).set_index('dailyDataDate').reset_index()

    # Creates 1 pandas df per unnested df from daily data read in, with same name
    globals()[df_row['dfName']] = unnested_table    
    
    daily_data_unnested_dfs['df'][df_index] = unnested_table

del training
gc.collect()



#### Get some information on each date in daily data (using season dates of interest) ####
dates = pd.DataFrame(data = 
  {'dailyDataDate': nextDayPlayerEngagement['dailyDataDate'].unique()})

dates['date'] = pd.to_datetime(dates['dailyDataDate'].astype(str))

dates['year'] = dates['date'].dt.year
dates['month'] = dates['date'].dt.month

dates_with_info = pd.merge(
  dates,
  seasons,
  left_on = 'year',
  right_on = 'seasonId'
  )

dates_with_info['inSeason'] = (
  dates_with_info['date'].between(
    dates_with_info['regularSeasonStartDate'],
    dates_with_info['postSeasonEndDate'],
    inclusive = True
    )
  )

dates_with_info['seasonPart'] = np.select(
  [
    dates_with_info['date'] < dates_with_info['preSeasonStartDate'], 
    dates_with_info['date'] < dates_with_info['regularSeasonStartDate'],
    dates_with_info['date'] <= dates_with_info['lastDate1stHalf'],
    dates_with_info['date'] < dates_with_info['firstDate2ndHalf'],
    dates_with_info['date'] <= dates_with_info['regularSeasonEndDate'],
    dates_with_info['date'] < dates_with_info['postSeasonStartDate'],
    dates_with_info['date'] <= dates_with_info['postSeasonEndDate'],
    dates_with_info['date'] > dates_with_info['postSeasonEndDate']
  ], 
  [
    'Offseason',
    'Preseason',
    'Reg Season 1st Half',
    'All-Star Break',
    'Reg Season 2nd Half',
    'Between Reg and Postseason',
    'Postseason',
    'Offseason'
  ], 
  default = np.nan
  )

#### Add some pitching stats/pieces of info to player game level stats ####

player_game_stats = (playerBoxScores.copy().
  # Change team Id/name to reflect these come from player game, not roster
  rename(columns = {'teamId': 'gameTeamId', 'teamName': 'gameTeamName'})
  )

# Adds in field for innings pitched as fraction (better for aggregation)
player_game_stats['inningsPitchedAsFrac'] = np.where(
  pd.isna(player_game_stats['inningsPitched']),
  np.nan,
  np.floor(player_game_stats['inningsPitched']) +
    (player_game_stats['inningsPitched'] -
      np.floor(player_game_stats['inningsPitched'])) * 10/3
  )

# Add in Tom Tango pitching game score (https://www.mlb.com/glossary/advanced-stats/game-score)
player_game_stats['pitchingGameScore'] = (40
#     + 2 * player_game_stats['outs']
    + 1 * player_game_stats['strikeOutsPitching']
    - 2 * player_game_stats['baseOnBallsPitching']
    - 2 * player_game_stats['hitsPitching']
    - 3 * player_game_stats['runsPitching']
    - 6 * player_game_stats['homeRunsPitching']
    )

# Add in criteria for no-hitter by pitcher (individual, not multiple pitchers)
player_game_stats['noHitter'] = np.where(
  (player_game_stats['gamesStartedPitching'] == 1) &
  (player_game_stats['inningsPitched'] >= 9) &
  (player_game_stats['hitsPitching'] == 0),
  1, 0
  )

player_date_stats_agg = pd.merge(
  (player_game_stats.
    groupby(['dailyDataDate', 'playerId'], as_index = False).
    # Some aggregations that are not simple sums
    agg(
      numGames = ('gamePk', 'nunique'),
      # Should be 1 team per player per day, but adding here for 1 exception:
      # playerId 518617 (Jake Diekman) had 2 games for different teams marked
      # as played on 5/19/19, due to resumption of game after he was traded
      numTeams = ('gameTeamId', 'nunique'),
      # Should be only 1 team for almost all player-dates, taking min to simplify
      gameTeamId = ('gameTeamId', 'min')
      )
    ),
  # Merge with a bunch of player stats that can be summed at date/player level
  (player_game_stats.
    groupby(['dailyDataDate', 'playerId'], as_index = False)
    [['runsScored', 'homeRuns', 'strikeOuts', 'baseOnBalls', 'hits',
      'hitByPitch', 'atBats', 'caughtStealing', 'stolenBases',
      'groundIntoDoublePlay', 'groundIntoTriplePlay', 'plateAppearances',
      'totalBases', 'rbi', 'leftOnBase', 'sacBunts', 'sacFlies',
      'gamesStartedPitching', 'runsPitching', 'homeRunsPitching', 
      'strikeOutsPitching', 'baseOnBallsPitching', 'hitsPitching',
      'inningsPitchedAsFrac', 'earnedRuns', 
      'battersFaced','saves', 'blownSaves', 'pitchingGameScore', 
      'noHitter'
      ]].
    sum()
    ),
  on = ['dailyDataDate', 'playerId'],
  how = 'inner'
  )

#### Turn games table into 1 row per team-game, then merge with team box scores ####
# Filter to regular or Postseason games w/ valid scores for this part
games_for_stats = games[
  np.isin(games['gameType'], ['R', 'F', 'D', 'L', 'W', 'C', 'P']) &
  ~pd.isna(games['homeScore']) &
  ~pd.isna(games['awayScore'])
  ]

# Get games table from home team perspective
games_home_perspective = games_for_stats.copy()

# Change column names so that "team" is "home", "opp" is "away"
games_home_perspective.columns = [
  col_value.replace('home', 'team').replace('away', 'opp') for 
    col_value in games_home_perspective.columns.values]

games_home_perspective['isHomeTeam'] = 1

# Get games table from away team perspective
games_away_perspective = games_for_stats.copy()

# Change column names so that "opp" is "home", "team" is "away"
games_away_perspective.columns = [
  col_value.replace('home', 'opp').replace('away', 'team') for 
    col_value in games_away_perspective.columns.values]

games_away_perspective['isHomeTeam'] = 0

# Put together games from home/away perspective to get df w/ 1 row per team game
team_games = (pd.concat([
  games_home_perspective,
  games_away_perspective
  ],
  ignore_index = True)
  )

# Copy over team box scores data to modify
team_game_stats = teamBoxScores.copy()

# Change column names to reflect these are all "team" stats - helps 
# to differentiate from individual player stats if/when joining later
team_game_stats.columns = [
  (col_value + 'Team') 
  if (col_value not in ['dailyDataDate', 'home', 'teamId', 'gamePk',
    'gameDate', 'gameTimeUTC'])
    else col_value
  for col_value in team_game_stats.columns.values
  ]

# Merge games table with team game stats
team_games_with_stats = pd.merge(
  team_games,
  team_game_stats.
    # Drop some fields that are already present in team_games table
    drop(['home', 'gameDate', 'gameTimeUTC'], axis = 1),
  on = ['dailyDataDate', 'gamePk', 'teamId'],
  # Doing this as 'inner' join excludes spring training games, postponed games,
  # etc. from original games table, but this may be fine for purposes here 
  how = 'inner'
  )

team_date_stats_agg = (team_games_with_stats.
  groupby(['dailyDataDate', 'teamId', 'gameType', 'oppId', 'oppName'], 
    as_index = False).
  agg(
    numGamesTeam = ('gamePk', 'nunique'),
    winsTeam = ('teamWinner', 'sum'),
    lossesTeam = ('oppWinner', 'sum'),
    runsScoredTeam = ('teamScore', 'sum'),
    runsAllowedTeam = ('oppScore', 'sum')
    )
   )

# Prepare standings table for merge w/ player digital engagement data
# Pick only certain fields of interest from standings for merge
standings_selected_fields = (standings[['dailyDataDate', 'teamId', 
  'streakCode', 'divisionRank', 'leagueRank', 'wildCardRank', 'pct'
  ]].
  rename(columns = {'pct': 'winPct'})
  )

# Change column names to reflect these are all "team" standings - helps 
# to differentiate from player-related fields if/when joining later
standings_selected_fields.columns = [
  (col_value + 'Team') 
  if (col_value not in ['dailyDataDate', 'teamId'])
    else col_value
  for col_value in standings_selected_fields.columns.values
  ]

standings_selected_fields['streakLengthTeam'] = (
  standings_selected_fields['streakCodeTeam'].
    str.replace('W', '').
    str.replace('L', '').
    astype(float)
    )

# Add fields to separate winning and losing streak from streak code
standings_selected_fields['winStreakTeam'] = np.where(
  standings_selected_fields['streakCodeTeam'].str[0] == 'W',
  standings_selected_fields['streakLengthTeam'],
  np.nan
  )

standings_selected_fields['lossStreakTeam'] = np.where(
  standings_selected_fields['streakCodeTeam'].str[0] == 'L',
  standings_selected_fields['streakLengthTeam'],
  np.nan
  )

standings_for_digital_engagement_merge = (pd.merge(
  standings_selected_fields,
  dates_with_info[['dailyDataDate', 'inSeason']],
  on = ['dailyDataDate'],
  how = 'left'
  ).
  # Limit down standings to only in season version
  query("inSeason").
  # Drop fields no longer necessary (in derived values, etc.)
  drop(['streakCodeTeam', 'streakLengthTeam', 'inSeason'], axis = 1).
  reset_index(drop = True)
  )

#### Merge together various data frames to add date, player, roster, and team info ####
# Copy over player engagement df to add various pieces to it
player_engagement_with_info = nextDayPlayerEngagement.copy()

# Take "row mean" across targets to add (helps with studying all 4 targets at once)
player_engagement_with_info['targetAvg'] = np.mean(
  player_engagement_with_info[['target1', 'target2', 'target3', 'target4']],
  axis = 1)

# Merge in date information
player_engagement_with_info = pd.merge(
  player_engagement_with_info,
  dates_with_info[['dailyDataDate', 'date', 'year', 'month', 'inSeason',
    'seasonPart']],
  on = ['dailyDataDate'],
  how = 'left'
  )

# Merge in some player information
player_engagement_with_info = pd.merge(
  player_engagement_with_info,
  players[['playerId', 'playerName', 'DOB', 'mlbDebutDate', 'birthCity',
    'birthStateProvince', 'birthCountry', 'primaryPositionName']],
   on = ['playerId'],
   how = 'left'
   )

# Merge in some player roster information by date
player_engagement_with_info = pd.merge(
  player_engagement_with_info,
  (rosters[['dailyDataDate', 'playerId', 'statusCode', 'status', 'teamId']].
    rename(columns = {
      'statusCode': 'rosterStatusCode',
      'status': 'rosterStatus',
      'teamId': 'rosterTeamId'
      })
    ),
  on = ['dailyDataDate', 'playerId'],
  how = 'left'
  )
    
# Merge in team name from player's roster team
player_engagement_with_info = pd.merge(
  player_engagement_with_info,
  (teams[['id', 'teamName']].
    rename(columns = {
      'id': 'rosterTeamId',
      'teamName': 'rosterTeamName'
      })
    ),
  on = ['rosterTeamId'],
  how = 'left'
  )

# Merge in some player game stats (previously aggregated) from that date
player_engagement_with_info = pd.merge(
  player_engagement_with_info,
  player_date_stats_agg,
  on = ['dailyDataDate', 'playerId'],
  how = 'left'
  )

# Merge in team name from player's game team
player_engagement_with_info = pd.merge(
  player_engagement_with_info,
  (teams[['id', 'teamName']].
    rename(columns = {
      'id': 'gameTeamId',
      'teamName': 'gameTeamName'
      })
    ),
  on = ['gameTeamId'],
  how = 'left'
  )

# Merge in some team game stats/results (previously aggregated) from that date
player_engagement_with_info = pd.merge(
  player_engagement_with_info,
  team_date_stats_agg.rename(columns = {'teamId': 'gameTeamId'}),
  on = ['dailyDataDate', 'gameTeamId'],
  how = 'left'
  )

# Merge in player transactions of note on that date
    
# Merge in some pieces of team standings (previously filter/processed) from that date
player_engagement_with_info = pd.merge(
  player_engagement_with_info,
  standings_for_digital_engagement_merge.
    rename(columns = {'teamId': 'gameTeamId'}),
  on = ['dailyDataDate', 'gameTeamId'],
  how = 'left'
  )

display(player_engagement_with_info)

# %% [code] {"papermill":{"duration":0.143732,"end_time":"2021-06-11T18:49:21.348227","exception":false,"start_time":"2021-06-11T18:49:21.204495","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:26:55.997662Z","iopub.execute_input":"2021-06-14T09:26:55.998073Z","iopub.status.idle":"2021-06-14T09:26:56.018325Z","shell.execute_reply.started":"2021-06-14T09:26:55.998038Z","shell.execute_reply":"2021-06-14T09:26:56.017469Z"}}
player_engagement_with_info.info()

# %% [markdown] {"papermill":{"duration":0.116904,"end_time":"2021-06-11T18:49:21.581773","exception":false,"start_time":"2021-06-11T18:49:21.464869","status":"completed"},"tags":[]}
# output結果をreferenceできるように、一応pickleで保存しておきます。

# %% [code] {"execution":{"iopub.execute_input":"2021-06-11T18:49:21.819366Z","iopub.status.busy":"2021-06-11T18:49:21.81832Z","iopub.status.idle":"2021-06-11T18:49:29.543521Z","shell.execute_reply":"2021-06-11T18:49:29.542493Z","shell.execute_reply.started":"2021-06-11T18:29:24.484735Z"},"papermill":{"duration":7.84534,"end_time":"2021-06-11T18:49:29.543677","exception":false,"start_time":"2021-06-11T18:49:21.698337","status":"completed"},"tags":[]}
player_engagement_with_info.to_pickle("player_engagement_with_info.pkl")

# %% [code] {"papermill":{"duration":0.134207,"end_time":"2021-06-11T18:49:29.803106","exception":false,"start_time":"2021-06-11T18:49:29.668899","status":"completed"},"tags":[]}


# %% [markdown] {"papermill":{"duration":0.136875,"end_time":"2021-06-11T18:49:30.071632","exception":false,"start_time":"2021-06-11T18:49:29.934757","status":"completed"},"tags":[]}
# #### スターターではここからkerasで簡単なモデル作成をしていますので、興味ある方はそちらをご覧ください。

# %% [code] {"papermill":{"duration":0.153007,"end_time":"2021-06-11T18:49:30.347614","exception":false,"start_time":"2021-06-11T18:49:30.194607","status":"completed"},"tags":[]}


# %% [markdown] {"papermill":{"duration":0.12281,"end_time":"2021-06-11T18:49:30.601655","exception":false,"start_time":"2021-06-11T18:49:30.478845","status":"completed"},"tags":[]}
# ### 以下、検証用として、target1～4をすべて中間値(スコア上がるため、v8でmeanからmedianに変更しました)でsubmitします。

# %% [code] {"papermill":{"duration":0.163708,"end_time":"2021-06-11T18:49:30.885597","exception":false,"start_time":"2021-06-11T18:49:30.721889","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:27:36.099392Z","iopub.execute_input":"2021-06-14T09:27:36.099805Z","iopub.status.idle":"2021-06-14T09:27:36.314269Z","shell.execute_reply.started":"2021-06-14T09:27:36.09976Z","shell.execute_reply":"2021-06-14T09:27:36.313164Z"}}
t1_median = player_engagement_with_info["target1"].median()
t2_median = player_engagement_with_info["target2"].median()
t3_median = player_engagement_with_info["target3"].median()
t4_median = player_engagement_with_info["target4"].median()

# %% [code] {"papermill":{"duration":0.131632,"end_time":"2021-06-11T18:49:31.142954","exception":false,"start_time":"2021-06-11T18:49:31.011322","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:27:37.123159Z","iopub.execute_input":"2021-06-14T09:27:37.12362Z","iopub.status.idle":"2021-06-14T09:27:37.130077Z","shell.execute_reply.started":"2021-06-14T09:27:37.123585Z","shell.execute_reply":"2021-06-14T09:27:37.128856Z"}}
print(t1_median,t2_median,t3_median,t4_median)

# %% [markdown] {"papermill":{"duration":0.11749,"end_time":"2021-06-11T18:49:33.163029","exception":false,"start_time":"2021-06-11T18:49:33.045539","status":"completed"},"tags":[]}
# # 4. submitの形式
# riiidの https://www.kaggle.com/chumajin/eda-for-biginner　で解説したのと同じく、1部のtest dataをget → 1部を予測　→　1部を提出　をどんどん繰り返していく方式です。
# 今回は1日分のtest data→次の日を予測提出、んで、次の日のtest data→その次の日を予測、提出　の流れです。
# 
# 
# ## **↓のmake_envは1回しか実行できません。**
# ## **失敗したら、データをrestart(上の方のFactory resetボタンを押す)して、再度やることになりますので、注意が必要です!**

# %% [markdown] {"papermill":{"duration":0.118225,"end_time":"2021-06-11T18:49:33.398072","exception":false,"start_time":"2021-06-11T18:49:33.279847","status":"completed"},"tags":[]}
# #### 最終形はこんな感じです(スターターから抜粋。解説用に少し細かくやっていきます

# %% [code] {"execution":{"iopub.execute_input":"2021-06-11T18:49:33.63826Z","iopub.status.busy":"2021-06-11T18:49:33.637334Z","iopub.status.idle":"2021-06-11T18:49:33.641655Z","shell.execute_reply":"2021-06-11T18:49:33.642107Z","shell.execute_reply.started":"2021-06-11T18:29:31.450494Z"},"papermill":{"duration":0.127157,"end_time":"2021-06-11T18:49:33.642278","exception":false,"start_time":"2021-06-11T18:49:33.515121","status":"completed"},"tags":[]}
"""
if 'kaggle_secrets' in sys.modules:  # only run while on Kaggle
    import mlb

    env = mlb.make_env()
    iter_test = env.iter_test()

    for (test_df, sample_prediction_df) in iter_test:
    
        # Example: unpack a dataframe from a json column
        today_games = unpack_json(test_df['games'].iloc[0])
    
        # Make your predictions for the next day's engagement
        sample_prediction_df['target1'] = 100.00
    
        # Submit your predictions 
        env.predict(sample_prediction_df)


"""

# %% [markdown] {"papermill":{"duration":0.118402,"end_time":"2021-06-11T18:49:33.878823","exception":false,"start_time":"2021-06-11T18:49:33.760421","status":"completed"},"tags":[]}
# #### ここから↑のサンプルコードを少し解説

# %% [markdown] {"papermill":{"duration":0.117862,"end_time":"2021-06-11T18:49:34.114753","exception":false,"start_time":"2021-06-11T18:49:33.996891","status":"completed"},"tags":[]}
# mlbのダウンロード

# %% [code] {"papermill":{"duration":0.157437,"end_time":"2021-06-11T18:49:34.38967","exception":false,"start_time":"2021-06-11T18:49:34.232233","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:28:19.253637Z","iopub.execute_input":"2021-06-14T09:28:19.254017Z","iopub.status.idle":"2021-06-14T09:28:19.298491Z","shell.execute_reply.started":"2021-06-14T09:28:19.253987Z","shell.execute_reply":"2021-06-14T09:28:19.297258Z"}}
if 'kaggle_secrets' in sys.modules:  # only run while on Kaggle
    import mlb

# %% [markdown] {"papermill":{"duration":0.124681,"end_time":"2021-06-11T18:49:34.711578","exception":false,"start_time":"2021-06-11T18:49:34.586897","status":"completed"},"tags":[]}
# envとiter_testの定義 (お決まりの作業と思ってもらえれば)

# %% [code] {"papermill":{"duration":0.125932,"end_time":"2021-06-11T18:49:34.955809","exception":false,"start_time":"2021-06-11T18:49:34.829877","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:28:24.550083Z","iopub.execute_input":"2021-06-14T09:28:24.550516Z","iopub.status.idle":"2021-06-14T09:28:24.55605Z","shell.execute_reply.started":"2021-06-14T09:28:24.550468Z","shell.execute_reply":"2021-06-14T09:28:24.554671Z"}}
env = mlb.make_env()
iter_test = env.iter_test()

# %% [markdown] {"papermill":{"duration":0.118437,"end_time":"2021-06-11T18:49:35.193711","exception":false,"start_time":"2021-06-11T18:49:35.075274","status":"completed"},"tags":[]}
# iter_testの中身を見てみる (とりあえずbreakで1個だけ見る。break外すとエラーでます。理由はそのあと解説しています)

# %% [code] {"papermill":{"duration":0.991035,"end_time":"2021-06-11T18:49:36.301953","exception":false,"start_time":"2021-06-11T18:49:35.310918","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:28:40.080548Z","iopub.execute_input":"2021-06-14T09:28:40.081101Z","iopub.status.idle":"2021-06-14T09:28:40.942286Z","shell.execute_reply.started":"2021-06-14T09:28:40.081052Z","shell.execute_reply":"2021-06-14T09:28:40.941273Z"}}
for (test_df, sample_prediction_df) in iter_test:
    display(test_df)
    display(sample_prediction_df)
    break

# %% [markdown] {"papermill":{"duration":0.121287,"end_time":"2021-06-11T18:49:36.544242","exception":false,"start_time":"2021-06-11T18:49:36.422955","status":"completed"},"tags":[]}
# 1日分のtest dataと、submissionファイルが出てくるのがわかる
# 
# 
# ここで、submissionファイルに予測値を記入して、提出しないと、次の日のtest dataを受け取ることができないというエラーが出る(以下のように、もう一度走らせると怒られる)

# %% [code] {"papermill":{"duration":0.226649,"end_time":"2021-06-11T18:49:36.890061","exception":false,"start_time":"2021-06-11T18:49:36.663412","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:29:31.022394Z","iopub.execute_input":"2021-06-14T09:29:31.022795Z","iopub.status.idle":"2021-06-14T09:29:31.080859Z","shell.execute_reply.started":"2021-06-14T09:29:31.022764Z","shell.execute_reply":"2021-06-14T09:29:31.079283Z"}}
for (test_df, sample_prediction_df) in iter_test:
    display(test_df)
    display(sample_prediction_df)
    break

# %% [markdown] {"papermill":{"duration":0.121178,"end_time":"2021-06-11T18:49:37.134545","exception":false,"start_time":"2021-06-11T18:49:37.013367","status":"completed"},"tags":[]}
# なので、一度submissionファイルに記入して提出してあげないといけません。ここではお試しに、先ほどのtargetのそれぞれの中間値を全部入れてみます。

# %% [code] {"papermill":{"duration":0.145229,"end_time":"2021-06-11T18:49:37.401468","exception":false,"start_time":"2021-06-11T18:49:37.256239","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:29:51.650122Z","iopub.execute_input":"2021-06-14T09:29:51.650494Z","iopub.status.idle":"2021-06-14T09:29:51.671854Z","shell.execute_reply.started":"2021-06-14T09:29:51.650465Z","shell.execute_reply":"2021-06-14T09:29:51.670792Z"}}
sample_prediction_df["target1"] = t1_median
sample_prediction_df["target2"] = t2_median
sample_prediction_df["target3"] = t3_median
sample_prediction_df["target4"] = t4_median


sample_prediction_df

# %% [markdown] {"papermill":{"duration":0.121948,"end_time":"2021-06-11T18:49:37.64384","exception":false,"start_time":"2021-06-11T18:49:37.521892","status":"completed"},"tags":[]}
# 予測値を入れたらこの時点で一度下記のコードでsubmitします

# %% [code] {"papermill":{"duration":0.129576,"end_time":"2021-06-11T18:49:37.895614","exception":false,"start_time":"2021-06-11T18:49:37.766038","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:29:59.007068Z","iopub.execute_input":"2021-06-14T09:29:59.007417Z","iopub.status.idle":"2021-06-14T09:29:59.012104Z","shell.execute_reply.started":"2021-06-14T09:29:59.00739Z","shell.execute_reply":"2021-06-14T09:29:59.010732Z"}}
env.predict(sample_prediction_df)

# %% [markdown] {"papermill":{"duration":0.123149,"end_time":"2021-06-11T18:49:38.142959","exception":false,"start_time":"2021-06-11T18:49:38.01981","status":"completed"},"tags":[]}
# そうすると、次の日のデータが受け取れるようになります。(以下のように先ほどと同じコードを流してもエラーで怒られません)

# %% [code] {"papermill":{"duration":0.383184,"end_time":"2021-06-11T18:49:38.648033","exception":false,"start_time":"2021-06-11T18:49:38.264849","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:30:09.450715Z","iopub.execute_input":"2021-06-14T09:30:09.45107Z","iopub.status.idle":"2021-06-14T09:30:09.707916Z","shell.execute_reply.started":"2021-06-14T09:30:09.451042Z","shell.execute_reply":"2021-06-14T09:30:09.707035Z"}}
for (test_df, sample_prediction_df) in iter_test:
    display(test_df)
    display(sample_prediction_df)
    break

# %% [markdown] {"papermill":{"duration":0.121984,"end_time":"2021-06-11T18:49:38.89173","exception":false,"start_time":"2021-06-11T18:49:38.769746","status":"completed"},"tags":[]}
# 以下のスターターのコードをもう一度みて見ると、for文の中でこれを繰り返してsubmitしていることがわかります。
# 
# 基本的には以下のfor文の中身を、test dfの前処理と、predictionして、sample_prediction_dfの書き換え、env.predictで提出していく流れですね。
# 
# riiidコンペの場合は、一つ前の情報の正解が流れてきてましたので、それを使って次の予測のためのデータとして使用していました(今回もおそらくそうなのかな??)。

# %% [code] {"papermill":{"duration":0.121709,"end_time":"2021-06-11T18:49:39.135494","exception":false,"start_time":"2021-06-11T18:49:39.013785","status":"completed"},"tags":[]}


# %% [code] {"execution":{"iopub.execute_input":"2021-06-11T18:49:39.386397Z","iopub.status.busy":"2021-06-11T18:49:39.385345Z","iopub.status.idle":"2021-06-11T18:49:39.390058Z","shell.execute_reply":"2021-06-11T18:49:39.389555Z"},"papermill":{"duration":0.133027,"end_time":"2021-06-11T18:49:39.390209","exception":false,"start_time":"2021-06-11T18:49:39.257182","status":"completed"},"tags":[]}
"""
if 'kaggle_secrets' in sys.modules:  # only run while on Kaggle
    import mlb

    env = mlb.make_env()
    iter_test = env.iter_test()

    for (test_df, sample_prediction_df) in iter_test:
    
        # Example: unpack a dataframe from a json column
        today_games = unpack_json(test_df['games'].iloc[0])
    
        # Make your predictions for the next day's engagement
        sample_prediction_df['target1'] = 100.00
    
        # Submit your predictions 
        env.predict(sample_prediction_df)


"""

# %% [markdown] {"papermill":{"duration":0.123008,"end_time":"2021-06-11T18:49:39.63599","exception":false,"start_time":"2021-06-11T18:49:39.512982","status":"completed"},"tags":[]}
# ## 今回はこのままsubmitしたいので、2回目の提出後にfor文で最後まで回します。

# %% [code] {"papermill":{"duration":0.132458,"end_time":"2021-06-11T18:49:39.894569","exception":false,"start_time":"2021-06-11T18:49:39.762111","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:30:26.736705Z","iopub.execute_input":"2021-06-14T09:30:26.737092Z","iopub.status.idle":"2021-06-14T09:30:26.745284Z","shell.execute_reply.started":"2021-06-14T09:30:26.737059Z","shell.execute_reply":"2021-06-14T09:30:26.743878Z"}}
# 2回目の提出

sample_prediction_df["target1"] = t1_median
sample_prediction_df["target2"] = t2_median
sample_prediction_df["target3"] = t3_median
sample_prediction_df["target4"] = t4_median
env.predict(sample_prediction_df)


# %% [code] {"papermill":{"duration":0.760212,"end_time":"2021-06-11T18:49:40.779803","exception":false,"start_time":"2021-06-11T18:49:40.019591","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-06-14T09:31:21.734487Z","iopub.execute_input":"2021-06-14T09:31:21.734853Z","iopub.status.idle":"2021-06-14T09:31:22.380877Z","shell.execute_reply.started":"2021-06-14T09:31:21.734822Z","shell.execute_reply":"2021-06-14T09:31:22.379995Z"}}
# 残り最後まで

for (test_df, sample_prediction_df) in iter_test:
    
        # Example: unpack a dataframe from a json column
        #today_games = unpack_json(test_df['games'].iloc[0])
    
        # Make your predictions for the next day's engagement
        sample_prediction_df["target1"] = t1_median
        sample_prediction_df["target2"] = t2_median
        sample_prediction_df["target3"] = t3_median
        sample_prediction_df["target4"] = t4_median
    
        # Submit your predictions 
        env.predict(sample_prediction_df)

# %% [markdown] {"papermill":{"duration":0.121297,"end_time":"2021-06-11T18:49:41.02355","exception":false,"start_time":"2021-06-11T18:49:40.902253","status":"completed"},"tags":[]}
# ご参考> コード要件
# 
# * これはコードコンペティションです
# * このコンテストへの提出は、ノートブックを通じて行う必要があります。コミット後に[送信]ボタンをアクティブにするには、次の条件が満たされている必要があります。
# * 
# * CPUノートブック<= 6時間の実行時間
# * GPUノートブック<= 6時間の実行時間
# * **インターネットアクセスが無効**
# * 事前にトレーニングされたモデルを含む、無料で公開されている外部データが許可されます
# * 提出は、mlbPythonモジュールを使用して行う必要があります

# %% [code] {"papermill":{"duration":0.135355,"end_time":"2021-06-11T18:49:41.280874","exception":false,"start_time":"2021-06-11T18:49:41.145519","status":"completed"},"tags":[]}


# %% [code] {"papermill":{"duration":0.122852,"end_time":"2021-06-11T18:49:41.541567","exception":false,"start_time":"2021-06-11T18:49:41.418715","status":"completed"},"tags":[]}


# %% [markdown] {"papermill":{"duration":0.122603,"end_time":"2021-06-11T18:49:41.786428","exception":false,"start_time":"2021-06-11T18:49:41.663825","status":"completed"},"tags":[]}
# # ここまで読んでいただいてありがとうございます！
# # お役にたてば、upvote/followいただけたら嬉しいです！
# # よろしくお願いいたします !!

# %% [code] {"papermill":{"duration":0.123659,"end_time":"2021-06-11T18:49:42.033425","exception":false,"start_time":"2021-06-11T18:49:41.909766","status":"completed"},"tags":[]}
