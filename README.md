# ObjectiveTextMiner-JP
感情語・推論語を除外して客観的な分析を行う日本語テキストマイニングツール
<br>
<br>

## 分析結果の例

このツールは、入力されたテキストから客観的な構造を抽出し、以下のようなレポートを生成します。

| 共起ネットワーク図（語句の関連性） | ワードクラウド（重要語） |
| :---: | :---: |
| <img src="https://github.com/trgr-karasutoragara/ObjectiveTextMiner-JP/blob/main/network_filtered.png" width="400"> | <img src="https://github.com/trgr-karasutoragara/ObjectiveTextMiner-JP/blob/main/wordcloud_filtered.png" width="400"> |

<br>
<br>


*This text mining tool is specifically designed for Japanese language processing, so the documentation is written in Japanese. However, the program structure and filtering approach may be applicable to your language. Feel free to use and adapt it.*
<br>
<br>


## 製作者より

このツールは、自分のエッセイの原稿をテキストマイニングする客観視の観点で生まれました。分析すると何が見えるだろうと。そして、日本語で研究する方に貢献できると考え、公開しました。これはあなたへのパスであり、あなたがご自身の領域でシュートを決めたり夢を叶えてくださると嬉しいです。
<br>
<br>

---
<br>
<br>


## 何ができるか

- テキストから「思う」「感じる」「らしい」等の主観的表現を除去
- 「信頼」「価値」「社会」等の客観的内容語のみを抽出
- 論争や議論の感情的要素を排除して核心的な内容を把握
- 分析結果をメールで自動送信
<br>
<br>

## 必要な環境
<br>

### 初回設定（技術者向け要件リスト）

**システム要件**：
- Python 3.8以上
- Ubuntu環境では仮想環境必須：`python3 -m venv myenv`
<br>
<br>

**設定手順**：
```bash
# 1. このツールをダウンロード

# 2. 仮想環境作成・有効化（Ubuntu必須）
# myenvは仮想環境の名前（任意の名前に変更可能）
python3 -m venv myenv
# 仮想環境を有効化
source myenv/bin/activate

# 3. 必要なライブラリをインストール
# requirements.txtは必要なソフトウェア部品のリストファイル
pip install -r requirements.txt

# 4. ディレクトリ作成
mkdir -p ~/Dropbox/text ~/Dropbox/processed ~/Dropbox/results
```

**仮想環境について**：
- `python3 -m venv 仮想環境名` で仮想環境を作成
- `myenv` は仮想環境の名前（変更可能）
- 作業時は毎回 `source myenv/bin/activate` で仮想環境を有効化する必要があります
<br>
<br>

**requirements.txt（必要なソフトウェア部品リストファイル）の内容**：
```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
janome>=0.4.2
mecab-python3>=1.0.5
gensim>=4.2.0
textstat>=0.7.3
wordcloud>=1.9.0
networkx>=2.8.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0
```

これらは、このツールを動作させるために必要なソフトウェア部品（ライブラリ）です。
<br>

**MeCabインストール（必須）**：
```bash
sudo apt-get install python3-dev build-essential libmecab-dev
```

<br>

**メール設定（必須）**：
- Gmailアプリパスワードの設定
- objective_text_miner.py内のメール設定を編集

<br>

**設定について**：
上記の技術要件を満たせる方に設定を依頼してください。

<br>
<br>

## 日常的な使い方（設定後）

### 1. テキストファイルを準備
- 分析したいテキストを `.txt` ファイルで保存。〔複数のファイルを一度に処理できます〕
- `~/Dropbox/text/` フォルダに置く

<br>
<br>


### 2. 分析実行
**Ubuntu環境での手順**：
```bash
# 仮想環境を有効化（毎回必要）
source myenv/bin/activate
# 分析実行
python objective_text_miner.py
```

<br>
<br>


### 3. 結果確認
- **分析結果本文**：メールで送信される
- **画像とHTMLファイル**：`~/Dropbox/results/` に生成
  - `network_filtered.png`：語彙関係図
  - `network_filtered_interactive.html`：語彙関係図（動的・ファイルが重すぎるので、お手数ですがダウンロードしてセキュリティ確認してからご覧下さい）→https://github.com/trgr-karasutoragara/ObjectiveTextMiner-JP/blob/main/network_filtered_interactive.html
  - `wordcloud_filtered.png`：重要語の可視化


<br>
<br>


## 分析例

<br>

**入力**：日本語テキスト
**出力**：客観的内容語のみを抽出

例：「信頼」(37回)、「価値」(27回)、「社会」(25回)

詳しくは→[【メールサンプル】高度テキストマイニング包括分析レポート](https://github.com/trgr-karasutoragara/ObjectiveTextMiner-JP/blob/main/%E3%80%90%E3%83%A1%E3%83%BC%E3%83%AB%E3%82%B5%E3%83%B3%E3%83%97%E3%83%AB%E3%80%91%E9%AB%98%E5%BA%A6%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E3%83%9E%E3%82%A4%E3%83%8B%E3%83%B3%E3%82%B0%E5%8C%85%E6%8B%AC%E5%88%86%E6%9E%90%E3%83%AC%E3%83%9D%E3%83%BC%E3%83%88txt)


<br>
<br>

## 対象ユーザー例

- **人文系研究者**：文献の客観的構造の分析
- **社会科学者**：論争構造の把握  
- **議論分析者**：感情を排除した構造の理解
- **学生**：レポート・論文の客観的構造分析
- **noteのみんな**：エッセイ書き溜めて定期的にテキストマイニングすると新たな視点が得られるかも

<br>

## 使用手順

1. **初回設定**：技術要件を満たせる方に依頼
2. **日常使用**：仮想環境有効化後、`python objective_text_miner.py` を実行
3. **複数ファイル**：一度に分析可能
4. **結果の活用**：客観的議論の材料として使用

---

設定完了後は、仮想環境を有効化し、 `python objective_text_miner.py` を実行すると動作します。お役に立ちますように!
設定は、詳しい方や、お友達や、AIさんに質問すると、どうにかなるので諦めずに、お試し下さい。

