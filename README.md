# netkeiba-notebook
１．htmlから競馬データを抽出しCSVに出力（CSV_Generator.py）

２．LightGBMによるデータ分析（LightGBM.py）
 
# 経緯
当初はデータベース(SQLite)を利用していたが、cuDFでの並列処理での取得が難しいため簡単なCSVに移行した。
 
# 必要ライブラリ
* lightgbm 3.3.5
* lxml

# インストール方法
```bash
pip install lightgbm
pip install lxml
```
# 流れ
cuDFのDockerコンテナを作成する。（[コマンド作成器](https://rapids.ai/start.html#get-rapids)）

PowerShellに以下のコマンドを入力。ホスト側パス内で適宜フォルダを作り、クローンを行う。

```bash
docker create -v D:/netkeiba:/rapids/notebooks/netkeiba --name netkeiba --gpus all -it --shm-size=1g --ulimit memlock=-1 -p 8888:8888 -p 8787:8787 -p 8786:8786 nvcr.io/nvidia/rapidsai/rapidsai-core:22.10-cuda11.5-runtime-ubuntu20.04-py3.9
```

### -vオプションについて
ホスト側のフォルダをコンテナ内フォルダとリンクさせるためのもの。

`-v [ホスト側パス]:[コンテナ側パス]`

## Jupyter Labをもっと便利に
### jupytext
.ipynbではなく.pyでgitに保存させる。.ipynbではメタデータが多いため、gitと相性が悪い。
https://gammasoft.jp/blog/jupyterlab-desktop-install-extensions/

### black ＆ isort
コード整形とインポート順の自動整理。
https://pystyle.info/jupyterlab-recommend-extensions/#outline__4
 
# 注意
jupytextで.pyが生成されるタイミングが自動整形よりも前なので注意。
（対策　⇒　間隔をあけて２回保存ボタンを押す or 整形専用ショートカットを押してから保存）

cuDFのPython対応バージョンが3.8 or 3.9のため、3.9を使用している。

cuDFはlocによる参照が遅いので、必要であればto_pandas()でPandasに変換するとよい。.csvの読み込みはcuDFのほうが上。

メモリを5GB以上消費する。

馬データとレースデータのマージ後対象データが１０万件以下に抑えたり、deleteで変数を解放するとよい。

# 著者
* mikikazuo
* booster515ezweb@gmail.com
 
# License
非公開
