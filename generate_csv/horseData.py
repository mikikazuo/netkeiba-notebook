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

# %%
import csv
import datetime
import os
import pathlib
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lxml.html
import pandas as pd
from tqdm import tqdm


# %%
class horse_scraper:
    lock = threading.Lock()

    def my_work(self, path):
        read = path.read_text(encoding="euc-jp")
        self.scrape_from_page(read, path.name)

    def scrape_from_page(self, html, filename):
        """
        1つのページからlxmlなどを使ってスクレイピングする。
        """

        html = lxml.html.fromstring(html)
        print("scraping", filename)

        # !!!xpathにtbodyふくむとうまくいかない　!!!
        result_table_rows = html.xpath(
            '//*[@id="contents"]/div[4]/div/table//tr[not(contains(@align, "center"))]'
        )
        horse_result_all = []
        for result_table_row in result_table_rows:
            horse_weight = result_table_row.xpath("td[24]")[0].text.split(r"(")

            # 馬体重で弾く　計測不可となっている
            # if len(horse_weight) < 2:
            #   continue
            # エラーで過去のレースすべてが取得できなくなるためここで、一部を弾く オッズをもとに弾く
            # ex ペースがないパターン2018105204　ペースがあるパターン2016102165 タイムがあるパターン 2017105649 生産者のハイパーリンクがない2012190003
            # if result_table_row.xpath("td[10]")[0].text == '\xa0':
            #   continue

            # 馬名が英語の場合がある ex 2012190002
            horseName = html.xpath(
                '//*[@id="db_main_box"]/div[1]/div[1]/div[1]/h1/text()'
            )
            horseName = (
                horseName[0].strip()
                if len(horseName)
                else html.xpath(
                    '//*[@id="db_main_box"]/div[1]/div[1]/div[1]/p[@class="eng_name"]/text()'
                )[0]
            )

            # 　どうしてもスクレイプうまくいかないhtmlはある。（netkeiba側でミスってるとき）そのためのtry-except
            sell_price = html.xpath('//*[text() = "セリ取引価格"]/following-sibling::td')[0]

            # 生産者名のタグ構造が異なるパターンがある ex2000190013
            makerName = html.xpath('//*[text() = "生産者"]/following-sibling::td/a')
            makerName = (
                makerName[0].text
                if len(makerName)
                else html.xpath('//*[text() = "生産者"]/following-sibling::td')[0].text
            )
            # 生産者のidが存在しないパターンがある、あとtdタグになってる ex 2012190003
            makerId = html.xpath('//*[text() = "生産者"]/following-sibling::td/a/@href')
            makerId = re.sub("\\D", "", makerId[0]) if makerId else None

            inputJokey = result_table_row.xpath("td[13]/a")
            jokey = (
                inputJokey[0].text
                if len(inputJokey)
                else result_table_row.xpath("td[13]")[0].text.strip()
            )
            horse_result = {
                "horse_id": filename.split(".")[0],
                "horse_name": horseName,  # stripを使う場合はtext()をxpathに埋め込む必要がある
                "birth_date": html.xpath(
                    '//*[@id="db_main_box"]/div[2]/div/div[2]/table//tr[1]/td'
                )[0].text,
                "maker_name": makerName,
                "maker_id": makerId,
                "from": html.xpath('//*[text() = "産地"]/following-sibling::td')[0].text,
                "sell_price": sell_price if "円" in sell_price else None,
                "race_date": result_table_row.xpath("td[1]/a")[0].text.replace(
                    "/", "-"
                ),
                "venue": result_table_row.xpath("td[2]/a")[0].text,
                "weather": result_table_row.xpath("td[3]")[0].text,
                "race_name": result_table_row.xpath("td[5]/a")[0].text,
                "race_id": re.sub(
                    "\\D", "", result_table_row.xpath("td[5]/a/@href")[0]
                ),  # 数字以外の文字を削除
                "horse_num": result_table_row.xpath("td[7]")[0].text,
                "wakuban": result_table_row.xpath("td[8]")[0].text,
                "umaban": result_table_row.xpath("td[9]")[0].text,
                "odds": result_table_row.xpath("td[10]")[0].text,
                "popularity": result_table_row.xpath("td[11]")[0].text,
                "order": result_table_row.xpath("td[12]")[
                    0
                ].text,  # TODO2018110126の「取消し」あり
                "jokey": jokey,
                "jokey_id": re.sub(
                    "\\D", "", result_table_row.xpath("td[13]/a/@href")[0]
                )
                if len(inputJokey)
                else None,
                "weight": result_table_row.xpath("td[14]")[0].text,
                "type": result_table_row.xpath("td[15]")[0].text[0],  # 障害レースは芝しかないっぽい
                "length": result_table_row.xpath("td[15]")[0].text[1:],
                "condition": result_table_row.xpath("td[16]")[0].text,
                "time": result_table_row.xpath("td[18]")[0].text,
                "diff_from_top": result_table_row.xpath("td[19]")[0].text,
                "order_of_corners": result_table_row.xpath("td[21]")[0].text,
                "pace": result_table_row.xpath("td[22]")[0].text,
                "nobori": result_table_row.xpath("td[23]")[0].text,
                "horse_weight": horse_weight[0] if len(horse_weight) >= 2 else None,
                "add_horse_weight": horse_weight[1][:-1]
                if len(horse_weight) >= 2
                else None,
                "reward": None
                if result_table_row.xpath("td[28]")[0].text == "\xa0"
                else result_table_row.xpath("td[28]")[0].text,
            }
            horse_result_all.append(horse_result)
        unique_key = ["horse_id", "race_id"]
        self.put_to_sqlite(
            horse_result_all, "horse", '"' + horse_result["horse_id"] + '"', unique_key
        )

    def put_to_sqlite(self, horse_result_all, file_name, table_name, unique_key):
        """
        sqlite3にデータを入れる

        Parameters
        ----------
        horse_result_all : dict

        Returns
        -------

        """

        db_name = file_name + ".db"

        create_query = ""
        for key in horse_result_all[0]:
            row = '"' + key + '"' + " varchar(30),"
            create_query = create_query + row

        row = "unique(" + ", ".join(unique_key) + "),"
        create_query = create_query + row

        create_query = create_query[:-1]

        create_query = "create table " + table_name + "(" + create_query + ")"

        with self.lock:
            conn = sqlite3.connect(db_name)
            c = conn.cursor()

            try:
                c.execute(create_query)
                # print(db_name, 'is created!')
            except:
                # print(db_name, 'is already exits!')
                pass

            for horse_result in horse_result_all:
                put_query = ""
                for value in horse_result.values():
                    put_query = (
                        put_query + '"' + str(value) + '"' + ","
                    )  # 2020110110のMakerName(Lemon's Mill)でシングルクォーテーションが使われているためダブルクォーテーションに変更

                put_query = put_query[:-1]  # 末尾の','とりのぞく

                put_query = (
                    "INSERT OR IGNORE INTO "
                    + table_name
                    + " VALUES "
                    + "("
                    + put_query
                    + ")"
                )

                c.execute(put_query)

            conn.commit()

            conn.close()


if __name__ == "__main__":
    # 時間がかかりすぎるためcsvを参考に分割 全部で4万件ほど
    dbname = "horse.db"
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    c.execute("select * from sqlite_master where type='table'")
    horseDbList = [row[1] for row in c.fetchall()]
    conn.close()

    horse_id_df = pd.read_csv("D:/netkeiba-scrapy/all_horse.csv")
    horse_id_list = horse_id_df["id"].values.tolist()

    horse_id_list = [str(i) for i in horse_id_list if str(i) not in horseDbList]
    path_iter = [
        Path("D:/netkeiba-scrapy/horse_html/" + x + ".html") for x in horse_id_list
    ]
    print(len(horseDbList))
    print("=======残り==========")
    print(len(horse_id_list))
    print("===================")

    args = sys.argv
    if 2 <= len(args):
        path_iter = path_iter[: int(args[1])]

    scraper = horse_scraper()

    start = time.time()
    with ThreadPoolExecutor() as executor:
        features = [executor.submit(scraper.my_work, path) for path in path_iter]
    # for path in path_iter:
    #     scraper.my_work(path)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

# %% tags=[]
path_to_htmldir = "../../html_data/horse_html/"
csv_path = r"../../csv_data/horse.csv"
# イテレータのままだと一度参照すると通り過ぎ、再度参照できないためリスト化する。globで自動でsortされている。
path_list = [path for path in pathlib.Path(path_to_htmldir).glob("20*")]

# %%
