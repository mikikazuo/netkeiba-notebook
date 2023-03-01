# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=[]
# .dbではcudfでの並列読み込みが難しそうなので.csvにした
import datetime
import json
import re
from multiprocessing import Pool

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import cudf

pd.set_option("display.max_columns", None)


# %% tags=[]
class HorseProcessing:
    def changeType(self, colmns, type):
        for col in colmns:
            self.df[col] = self.df[col].astype(type)

    # 代入先cudf型のdataframeならcudfに自動変換して代入される
    def processForPandas(self, column, func):
        # map関数はpandasでないと使えないため一時的に変換
        return self.df[column].to_pandas().map(func)

    def __init__(self):
        self.df = cudf.read_csv("../csv_data/horse.csv")

        # 不要列の削除
        self.df = self.df.drop(
            ["race_name", "horse_name", "sell_price", "maker_name", "jockey", "reward"],
            axis=1,
        )
        # 不要行の削除
        self.df = self.df.dropna(subset=["order", "horse_weight", "pace"])

        self.df["birth_date"] = self.processForPandas(
            "birth_date", lambda x: datetime.datetime.strptime(x, "%Y年%m月%d日").month
        )
        self.df["race_date"] = self.processForPandas(
            "race_date", lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")
        )
        self.df["race_month"] = self.processForPandas("race_date", lambda x: x.month)
        self.df["venue"] = self.processForPandas(
            "venue", lambda x: re.sub(r"\d", "", x)
        )

        timeList = self.df["time"].str.split(":")
        timeList = timeList.to_pandas().map(lambda x: list(map(float, x)))
        self.df["time"] = timeList.map(lambda x: x[0] * 60 + x[1])

        cornerList = self.df["order_of_corners"].str.split("-")
        self.df["order_of_corners"] = cornerList.to_pandas().map(
            lambda x: list(map(int, x))
        )
        self.df["order_of_corners"] = (
            self.df["order_of_corners"].to_pandas().map(lambda x: x[0] - x[-1])
        )

        paceList = self.df["pace"].str.split("-")
        self.df = self.df.drop(["pace"], axis=1)
        paceList = paceList.to_pandas().map(lambda x: list(map(float, x)))
        self.df["pace_start"] = paceList.map(lambda x: x[0])
        self.df["pace_goal"] = paceList.map(lambda x: x[1])

        # TODO sell_priceが空のパターンがないことを確認したのち追加  reward  horssenumでorderを割る  rewardは現状すべて０
        self.changeType(
            [
                "birth_date",
                "race_month",
                "horse_num",
                "wakuban",
                "umaban",
                "popularity",
                "order",
                "add_horse_weight",
                "order_of_corners",
            ],
            "int8",
        )
        self.changeType(["length", "horse_weight"], "int16")

        # TODO rewardの桁数足りないが大丈夫か horseData.dfList.reward.map(float).max()
        # weightは端数(0.5)ありのためこっち
        self.changeType(
            [
                "odds",
                "time",
                "diff_from_top",
                "nobori",
                "weight",
                "pace_start",
                "pace_goal",
            ],
            "float64",
        )
        self.df["order_normalize"] = 1 - (self.df["order"] - 1) / (
            self.df["horse_num"] - 1
        ).astype("float64")
        self.changeType(
            ["from", "venue", "weather", "type", "condition", "maker_id", "jockey_id"],
            "category",
        )

        self.df = self.df.set_index(["horse_id", "race_id"])


# %% tags=[]
# （注意）cudf上でdf.loc[<horse_id>,<race_id>]で取得する場合、
# horseDataと違いマルチインデックスがhorse_idでまとめられていないため毎回行順序がバラバラ
# horseDataでも<horse_id>ではなく<race_id>で取得した場合は順序がバラバラ
class RaceProcessing:
    def changeType(self, colmns, type):
        for col in colmns:
            self.df[col] = self.df[col].astype(type)

    def __init__(self):
        # １０万件でおおよそ4年くらいのレースが対象になる　２０万が限界 2past
        self.df = cudf.read_csv("../csv_data/race.csv")[-100000:]

        # 不要列の削除
        self.df = self.df.drop(["owner", "trainer"], axis=1)

        timeList = self.df["start_time"].str.split(":")
        timeList = timeList.to_pandas().map(lambda x: list(map(float, x)))
        self.df["start_time"] = timeList.map(lambda x: x[0] * 60 + x[1])
        self.df["outside"] = timeList.map(lambda x: 1 if x == "True" else 0)
        self.changeType(["start_time"], "int16")

        self.changeType(["age", "outside"], "int8")

        self.changeType(["sex", "tresen", "trainer_id", "owner_id", "turn"], "category")
        self.df = self.df.set_index(["horse_id", "race_id"])


# %%
# pd.Naについてスクレイピング側で要検証
class PaybackProcessing:
    # def preProcessing(self):
    #     for col in self.df.columns:
    #         data = json.loads(self.df.iloc[0][col])
    #         if len(data['payback']):
    #             for key in ['payback','ninki']:
    #                 data[key] = [int(d.replace(',', '')) for d in data[key]] #paybackが4桁ある時にカンマが入るため ex.201709040602
    #             # 馬番専用
    #             data['umaban'] = [tuple(map(int, d.split(' - '))) for d in data['umaban']] if '-' in data['umaban'][
    #                 0] else list(map(int, data['umaban']))
    #             self.df.iloc[0][col] = data
    #         else:
    #             self.df.iloc[0][col] = None

    def __init__(self):
        self.df = pd.read_csv("../csv_data/payback.csv", index_col="race_id")



        self.df = self.df.set_index("race_id")


# %% tags=[]
# %% tags=[] jupyter={"outputs_hidden": true}
# %%time
horseData = HorseProcessing()  # pdに比べcudfは3倍の速度(csvの読み込み、編集など)
raceData = RaceProcessing()
paybackData = PaybackProcessing()


# %%
def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


# %%
# %%time
# cudfはlocによる参照が10倍ほど遅いのでここでpandasに変換
horseData.df = reduce_mem_usage(horseData.df.to_pandas())
raceData.df = reduce_mem_usage(raceData.df.to_pandas())
paybackData.df = reduce_mem_usage(paybackData.df.to_pandas())

# %% tags=[]
# %%time


# 過去のレース数
def set_race_count(id):
    horseData.df.loc[id[0], "race_cnt"] = list(reversed(range(0, id[1])))


# 過去の上位順位回数
def set_order_count(id):
    order_count = [[0], [0], [0]]
    df = horseData.df.loc[id[0], "order"]
    df_length = len(df)
    for idx, order in enumerate(df.iloc[::-1]):
        if idx == df_length - 1:
            break
        for order_idx in range(3):
            order_count[order_idx].insert(
                0,
                order_count[order_idx][0] + 1
                if order == order_idx + 1
                else order_count[order_idx][0],
            )

    for order_idx in range(3):
        horseData.df.loc[id[0], "order_" + str(order_idx + 1) + "_cnt"] = order_count[
            order_idx
        ]


count_data = [
    [_name, len(_df)] for _name, _df in tqdm(horseData.df.groupby("horse_id"))
]
horseData.df["race_cnt"] = 0
horseData.df["race_cnt"] = horseData.df["race_cnt"].astype("uint8")
for id in tqdm(count_data):
    set_race_count(id)
    set_order_count(id)

del count_data

# %% tags=[]
# %%time
for order_idx in range(3):
    horseData.df["order_" + str(order_idx + 1) + "_cnt_normalize"] = (
        horseData.df["order_" + str(order_idx + 1) + "_cnt"] / horseData.df["race_cnt"]
    ).astype("float16")
    horseData.df = horseData.df.drop("order_" + str(order_idx + 1) + "_cnt", axis=1)
    # horseData.df = horseData.df.drop('order_'+str(order_idx+1), axis=1)

# %%
# %%time
# del horseData
# del raceData

mergeined_df_base = raceData.df.merge(
    horseData.df, on=["horse_id", "race_id"], how="inner"
)

# %% tags=[]
# %%time
pastNum = 2
# 過去に関係なく一貫性のあるコラム 過去データとして連結しないカラム メモリが足りないためtarget_columnsで絞るため廃止
common_columns = ["birth_date", "from", "maker_id", "sex", "trainer_id", "tresen"]
common_columns.append("race_cnt")
for order_idx in range(3):
    # common_columns.append('order_'+str(order_idx+1)+'_cnt')
    common_columns.append("order_" + str(order_idx + 1) + "_cnt_normalize")

# ソートのためにtarget_columnsにrace_dateを含めること
target_columns = list(set(mergeined_df_base.columns) ^ set(common_columns))


def pastRaceIdList(horse_id):
    # cudfはlocに向いていない、また、マルチプロセスに非対応
    race_df = mergeined_df_base.loc[horse_id, target_columns].sort_values(
        "race_date", ascending=False
    )
    race_lenght = len(race_df)
    if race_lenght <= pastNum:
        return None
    past_df_list = []
    for i in range(pastNum + 1):
        data = race_df[i : i + race_lenght - pastNum]
        if i == 0:
            base_index = data.index
        else:
            # mergeはなぜか使えない 一致する自作カラムが最低一つ必要か？
            past_df_list.append(
                data.rename(
                    columns=dict(zip(race_df.columns, race_df.columns + "_" + str(i)))
                ).reset_index(drop=True)
            )
    merged_frame = pd.concat(past_df_list, axis=1)
    # ここでマルチインデックスを付け足す  参照段階で残す場合、括弧でくくるパターンloc[[horse_id],:]　は処理が重いので
    merged_frame.index = pd.MultiIndex.from_arrays(
        [[horse_id] * (race_lenght - pastNum), base_index],
        names=["horse_id", "race_id"],
    )
    return merged_frame


# 過去データcsv作成
if __name__ == "__main__":
    horse_id = [horse_id[0] for horse_id in mergeined_df_base.index]
    horse_id = set(horse_id)  # 馬idが重複して入っているのでsetを使う
    with Pool() as p:
        # クラスメソッドだとフリーズしてしまうため外にメソッド定義
        imap = p.imap(pastRaceIdList, horse_id)
        result_list = list(tqdm(imap, total=len(horse_id)))
    past_df = pd.concat(result_list)
    del result_list

# %%
mergeined_df = mergeined_df_base.merge(past_df, on=["horse_id", "race_id"], how="inner")
for i in range(pastNum):
    mergeined_df["race_date_" + str(i + 1)] = (
        (mergeined_df["race_date"] - mergeined_df["race_date_" + str(i + 1)])
        .map(lambda x: x.days)
        .astype("int16")
    )
    # mergeined_df['jockey_same_' + str(i+1)] = (mergeined_df['jockey_id'] == mergeined_df['jockey_id_' + str(i+1)]).astype('int8')
    # mergeined_df['owner_same_' + str(i+1)] = (mergeined_df['owner_id'] == mergeined_df['owner_id_' + str(i+1)]).astype('int8')


# %%
class LightGbm:
    testSize = 0.1

    def training(self):
        df_train, df_val = train_test_split(
            self.df[[self.col] + self.target_columns],
            shuffle=False,
            test_size=self.testSize,
        )
        train_y = df_train[self.col]
        train_x = df_train.drop(self.col, axis=1)

        self.val_y = df_val[self.col]
        self.val_x = df_val.drop(self.col, axis=1)

        trains = lgb.Dataset(train_x, train_y)
        valids = lgb.Dataset(self.val_x, self.val_y)

        params = {"objective": "regression", "metrics": "mae"}

        self.model = lgb.train(
            params,
            trains,
            valid_sets=valids,
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=500, verbose=True
                ),  # early_stopping用コールバック関数
                lgb.log_evaluation(0),
            ],  # コマンドライン出力用コールバック関数
        )
        self.test_predicted = self.model.predict(self.val_x)

    # 予測値を図で確認する関数の定義
    def predictionAccuracy(self, predicted_df):
        RMSE = np.sqrt(
            mean_squared_error(predicted_df["true"], predicted_df["predicted"])
        )
        plt.figure(figsize=(7, 7))
        ax = plt.subplot(111)
        ax.scatter("true", "predicted", data=predicted_df)
        ax.set_xlabel("True Price", fontsize=20)
        ax.set_ylabel("Predicted Price", fontsize=20)
        plt.tick_params(labelsize=15)
        x = np.linspace(0, 1, 2)
        y = x
        ax.plot(x, y, "r-")
        plt.text(
            0.1,
            0.9,
            "RMSE = {}".format(str(round(RMSE, 3))),
            transform=ax.transAxes,
            fontsize=15,
        )

    def protData(self):
        # テストデータを用いて予測精度を確認する)
        predicted_df = pd.concat(
            [self.val_y.reset_index(drop=True), pd.Series(self.test_predicted)], axis=1
        )
        predicted_df.columns = ["true", "predicted"]

        self.predictionAccuracy(predicted_df)

    # ペイバック計算用の予測値を付与したレースごとにまとめたデータリスト作成、他に必要なカラム(order, umabanなど)も追加
    def makePredictDataset(self):
        df_train, self.df_predict = train_test_split(
            self.df[self.target_columns + ["order", "odds"]],
            shuffle=False,
            test_size=self.testSize,
        )
        # lightgbmの予測値カラムの追加
        self.df_predict["predict"] = self.test_predicted
        df_val_list = []
        for _name, _df in self.df_predict.groupby("race_id"):
            # 1レースにおける全馬のデータが存在している場合だけ抽出、過去データが不足している馬がいる場合除外。
            if len(_df) == _df.iloc[0]["horse_num"]:
                df_val_list.append(_df)
        return df_val_list

    def __init__(self, df):
        # 目的変数カラム
        self.col = "order_normalize"

        # 答えになってしまうカラム（レース後にわかるデータ）
        answer_col = [self.col] + [
            "time",
            "diff_from_top",
            "nobori",
            "order",
            "pace_goal",
            "pace_start",
            "popularity",
            "odds",
            "order_of_corners",
        ]

        # 説明変数カラム
        self.target_columns = list(df.columns)
        for i in answer_col:
            self.target_columns.remove(i)

        # self.target_columns.remove('race_cnt')
        self.target_columns.remove("race_date")

        # 特例 order_normalizeがあるので不要
        for i in range(pastNum):
            self.target_columns.remove("order_" + str(i + 1))
            # self.target_columns.remove('odds_'+str(i+1))
        self.df = df

        self.training()


# %% tags=[]
gbm = LightGbm(mergeined_df)
df_predict = gbm.makePredictDataset()

# %% tags=[]
# 特徴量の重要度を確認
lgb.plot_importance(gbm.model, height=0.5, figsize=(8, 16), ignore_zero=False)
gbm.protData()


# %%
# TODO 辞書に変換されるようにキーをstr型でスクレイピング側で統一するようにする
class Monetize:
    def changeDict(self, df, col):
        data = self.paybackData.df.loc[df.index.get_level_values("race_id")[0]][col]
        return dict(zip(data["umaban"], data["payback"])) if data else None

    def changeTuple(self, df, col):
        data = self.paybackData.df.loc[df.index.get_level_values("race_id")[0]][col]
        print(data)
        print(type(data))
        print(data["umaban"])
        print(",".join(data["umaban"]))
        print(dict(zip(",".join(data["umaban"])), data["payback"]))
        return dict(zip(",".join(data["umaban"])), data["payback"]) if data else None

    # 上位のindexを取得
    def getmax_rev(self, df, topnum=3, getmin=False):
        return (
            df.nsmallest(topnum, "predict")
            if getmin
            else df.nlargest(topnum, "predict")
        )

    def printResult(self, sumRace, winCount, sumPayback):
        print(str(winCount) + "/" + str(sumRace), str(winCount / sumRace * 100) + "%")
        print(
            str(sumPayback) + "/" + str(sumRace * 100), str(sumPayback / sumRace) + "%"
        )

    # paybackBorderは実際、オッズを確認してから馬券購入するという流れを踏まえたもの
    def tanshou(self, ranker=0, paybackBorder=100):
        winCount = 0
        sumPayback = 0

        for df in tqdm(self.df_predict):
            data = self.changeDict(df, "tanshou")
            # 上位馬番の取得
            targetUmaban = self.getmax_rev(df)["umaban"].iloc[ranker]
            # どんな結果でも最低100払い戻しは保証されているらしい
            if targetUmaban in data and data[targetUmaban] > paybackBorder:
                sumPayback = sumPayback + data[targetUmaban]
                winCount = winCount + 1
        self.printResult(len(self.df_predict), winCount, sumPayback)

    # 3着以内に入る馬を1つ選択
    def fukushou(self, ranker=0, paybackBorder=100):
        winCount = 0
        sumPayback = 0
        for df in tqdm(self.df_predict):
            data = self.changeDict(df, "fukushou")
            # 上位馬番の取得
            targetUmaban = self.getmax_rev(df)["umaban"].iloc[ranker]
            if targetUmaban in data and data[targetUmaban] > paybackBorder:
                sumPayback = sumPayback + data[targetUmaban]
                winCount = winCount + 1
        self.printResult(len(self.df_predict), winCount, sumPayback)

    # 3着以内に入る馬を２つ選択
    def wide(self, ranker=[0, 1], paybackBorder=100):
        winCount = 0
        sumPayback = 0

        for df in tqdm(self.df_predict):
            data = self.changeTuple(df, "wide")
            if data is None:
                continue
            # 上位馬番の取得
            targetUmaban = {self.getmax_rev(df)["umaban"][index] for index in ranker}
            for tupleKey in data:
                if targetUmaban == set(tupleKey) and data[tupleKey] > paybackBorder:
                    sumPayback = sumPayback + data[tupleKey]
                    winCount = winCount + 1
        self.printResult(len(self.df_predict), winCount, sumPayback)
        self.printResult(len(self.df_predict), winCount, sumPayback)

    def __init__(self, df_predict, paybackData):
        self.df_predict = df_predict
        self.paybackData = paybackData


# %%
money = Monetize(df_predict, paybackData)

# %% tags=[]
# index -100000:
money.tanshou(0)
money.tanshou(1)
money.tanshou(2)
money.fukushou(0, 100)
money.fukushou(1, 100)
money.fukushou(2, 100)

# %%
# index -100000:
money.wide(0)

# %%
