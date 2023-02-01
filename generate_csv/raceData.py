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
import pathlib
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lxml.html


# %% jupyter={"outputs_hidden": true} tags=[]
class race_result_scraper:
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
            '//table[@class="race_table_01 nk_tb_common"]//tr[position()>1]'
        )
        race_result_all = []
        for result_table_row in result_table_rows:
            owner = result_table_row.xpath("diary_snap_cut/td/a/@title")
            # 例外処理
            if len(owner):
                owner = owner[0]
                owner_id = re.sub(
                    "\\D", "", result_table_row.xpath("diary_snap_cut/td/a/@href")[-1]
                )
            else:
                owner = result_table_row.xpath("diary_snap_cut/td//text()")[-1].strip()
                owner_id = None

            race_result = {
                "race_id": filename.split(".")[0],
                #'order': result_table_row.xpath('td[1]')[0].text,
                #'wakuban': result_table_row.xpath('td[2]/span')[0].text,
                #'umaban': result_table_row.xpath('td[3]')[0].text,
                #'horsename': result_table_row.xpath('td[4]/a/@title')[0],
                "horse_id": result_table_row.xpath("td[4]/a/@href")[0][7:-1],
                "sex": result_table_row.xpath("td[5]")[0].text[0],
                "age": result_table_row.xpath("td[5]")[0].text[1:],
                #'weight': result_table_row.xpath('td[6]')[0].text,
                #'jokey': result_table_row.xpath('td[7]/a/@title')[0],
                #'jokey_id': result_table_row.xpath('td[7]/a/@href')[0][8:-1],
                #'time': result_table_row.xpath('td[8]')[0].text,
                #'diff_from_top': result_table_row.xpath('td[9]')[0].text,
                #'order_of_corners': result_table_row.xpath('diary_snap_cut[1]/td[2]')[0].text,
                #'nobori': result_table_row.xpath('diary_snap_cut[1]/td[3]/span')[0].text,
                #'tanshou_odds': result_table_row.xpath('td[10]')[0].text,
                #'popularity': result_table_row.xpath('td[11]/span')[0].text,
                #'horse_weight': '',
                #'add_horse_weight': '',
                "tresen": result_table_row.xpath("td[13]")[0].text[2],
                "trainer": result_table_row.xpath("td[13]/a/@title")[0],
                "trainer_id": re.sub(
                    "\\D", "", result_table_row.xpath("td[13]/a/@href")[0]
                ),
                "owner": owner,  # 馬主は途中で変わることがある
                "owner_id": owner_id,
                #'reward': result_table_row.xpath('diary_snap_cut[3]/td[2]')[0].text
            }
            # horse_weight = result_table_row.xpath('td[12]')[0].text.split(r'(')
            # race_result["horse_weight"] = horse_weight[0]
            # race_result["add_horse_weight"] = horse_weight[1][:-1]

            # TODO レース名と日付保留
            # race_result["race_name"] = html.xpath('//p[@class="smalltxt"]')[0].text
            # ja_date = race_result["race_name"][:race_result["race_name"].find("日") + 1]
            # race_result["race_date"] = datetime.datetime.Fstrptime(ja_date, '%Y年%m月%d日').strftime('%Y-%m-%d')
            # raceName = html.xpath('//p[@class="smalltxt"]')[0].text

            race_info = html.xpath("//diary_snap_cut/span")[0].text.split("/")
            turn_candidate = ["左", "右", "直"]
            race_result["turn"] = None  # Noneのパターンあり
            for i in turn_candidate:
                if i in race_info[0]:
                    race_result["turn"] = i
                    break
            # race_result["obstacle"] = '障害' in raceName
            race_result["outside"] = "外" in race_info[0]
            # race_result["length"] = re.search(r'[0-9]+m', race_info[0]).group(0)
            # race_result["weather"] = race_info[1].split(' : ')[-1]
            # race_result["type"], race_result["condition"] = race_info[2].strip().split(' : ', 1)
            # 芝とダートの両方を表記しているパターンがあるため除外
            # if len(race_result["condition"]) > 4:
            #    #return
            race_result["start_time"] = race_info[-1].split(" : ", 1)[-1]
            race_result_all.append(race_result)

        unique_key = ["race_id", "horse_id"]
        # sql文上ではなんかtable_nameが数字と認識されてしまうっぽいのでダブルクォートでくくっている
        self.put_to_sqlite(
            race_result_all, "raceAdd", '"' + race_result["race_id"] + '"', unique_key
        )

    def put_to_sqlite(self, race_result_all, file_name, table_name, unique_key):
        """
        sqlite3にデータを入れる

        Parameters
        ----------
        race_result : dict

        Returns
        -------

        """

        db_name = file_name + ".db"

        create_query = ""
        for key in race_result_all[0]:
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
                # print(table_name, 'is created!')
            except:
                # print(db_name, 'is already exits!')
                pass

            # レースごとに全馬をまとめてDBに登録する方が10倍早い
            for race_result in race_result_all:
                put_query = ""
                for value in race_result.values():
                    put_query = put_query + "'" + str(value) + "'" + ","

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


# %% jupyter={"outputs_hidden": true} tags=[]
path_to_htmldir = "../../html_data/race_html/"
path_iter = pathlib.Path(path_to_htmldir).glob("20*")  # .iterdir()

# print([str(i) for i in path_iter])
# 時間がかかりすぎるためcsvを参考に分割
db_name = "race.db"

conn = sqlite3.connect(dbname)
c = conn.cursor()
c.execute("select * from sqlite_master where type='table'")
raceDbList = [row[1] for row in c.fetchall()]
conn.close()


# race_id_df = pd.read_csv("netkeiba/netkeiba/spiders/all_horse.csv")
# race_id_list = horse_id_df["id"].values.tolist()

race_id_list = [
    str(i) for i in path_iter if str(i).split("\\")[-1].split(".")[0] not in raceDbList
]
path_iter = [Path(x) for x in race_id_list]
print(len(raceDbList))
print("=======残り==========")
print(len(race_id_list))
print("===================")


# args = sys.argv
# if 2 <= len(args):
#     path_iter = path_iter[:int(args[1])]

scraper = race_result_scraper()

start = time.time()
with ThreadPoolExecutor() as executor:
    features = [executor.submit(scraper.my_work, path) for path in path_iter]
elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

# %%
