# BOJ Meeting Similarity and Market Impact Analysis

![BOJ Minutes Analysis](boj_minutes_analysis.png)

![BOJ Minutes Analysis](market_indices_analysis.png)


## Overview
This project analyzes how changes in Bank of Japan (BOJ) monetary policy meeting content correlate with financial market movements. It combines text similarity analysis of meeting minutes with market statistics to identify patterns between policy language changes and market outcomes.

## Keyword Categories and Analysis

The project analyzes BOJ meeting minutes by tracking mentions of keywords across various economic and policy categories:

### Inflation and Price Categories
1. **Inflation-related (インフレ関連)**
   - Keywords: インフレ, 上昇圧力, 上昇率, 期待, インフレーション, 物価上昇, インフレ期待, インフレ予想, インフレターゲティング, CPI, 消費者物価

2. **Deflation-related (デフレ関連)**
   - Keywords: デフレ, 下落圧力, 懸念, デフレーション, 物価下落, 物価下押し, デフレスパイラル

3. **Price-related (物価関連)**
   - Keywords: 物価, 消費者, 生産者, 卸売, 国内企業, 価格, 商品, 製品, 資産, 土地, 住宅, 輸出, 輸入

### Financial Markets and Policy
4. **Exchange Rate-related (為替関連)**
   - Keywords: 為替, 円相場, 円レート, ドル円, 円ドル, ユーロ円, 円安, 円高

5. **Interest Rate-related (金利関連)**
   - Keywords: 金利, 利率, 利回り, イールド, 貸出, 預金, 政策金利, 短期, 長期, コールレート, 国債

6. **Financial (金融)**
   - Keywords: 金融, 政策, 量的, 質的, マネー, 通貨, マネタリーベース, ベースマネー

7. **Monetary Easing (金融緩和関連)**
   - Keywords: 金融緩和, 量的緩和, 質的緩和, 緩和的, 緩和

8. **Monetary Tightening (金融引き締め)**
   - Keywords: 金融引き締め, 量的縮小, 量的引き締め, 引締, 資金吸収

### Economic Conditions
9. **Economic Conditions (景気関連)**
   - Keywords: 景気動向, 景況, 経済情勢, 経済状況, 景気, 経済活動, 需給ギャップ, GDP

10. **Consumption-related (消費関連)**
    - Keywords: 消費, 個人消費, 家計, 消費者, マインド, 賃金

### Institutional Categories
11. **Central Bank (中央銀行)**
    - Keywords: 中央銀行

12. **Government (政府)**
    - Keywords: 政府, 財政

13. **Financial Institutions (金融機関)**
    - Keywords: 金融機関

### Business and Production
14. **Investment-related (投資関連)**
    - Keywords: 投資, 設備投資, 資本投資, 企業, 民間, 公共投資

15. **SME-related (中小企業関連)**
    - Keywords: 中小企業, 人手, 資金繰り, 事業継承, 生産性向上, 中小事業者

16. **Production-related (生産関連)**
    - Keywords: 生産, 鉱工業, 工業, 製造, 設備, 収益, 輸出

### Labor Market
17. **Employment-related (雇用関連)**
    - Keywords: 雇用, 正規, 非正規, 求人, 人口, 賃金

18. **Unemployment-related (失業関連)**
    - Keywords: 失業, 構造的, 摩擦的, 求職

### Market and Other Categories
19. **Stock Market (株式)**
    - Keywords: 株式, 株, 証券会社

20. **Forecast-related (予想関連)**
    - Keywords: 予想, 見通し, 見込み, 想定, 期待, 見通し, 期待

21. **Others (その他)**
    - Keywords: アンカー, ノミナルアンカー

## Market Indices Analysis

The project tracks and analyzes three key market indices to understand the impact of BOJ policy discussions:

### 1. Nikkei 225 Index (日経平均株価)
- **Data Source**: Yahoo Finance (^N225)
- **Metrics Tracked**:
  - Daily closing prices
  - Daily returns
  - 30-day rolling volatility
  - Mean values between meetings
  - Cumulative returns between meetings

### 2. USD/JPY Exchange Rate (米ドル/円レート)
- **Data Source**: FRED (DEXJPUS)
- **Metrics Tracked**:
  - Daily exchange rates
  - Daily returns
  - 30-day rolling volatility
  - Mean values between meetings
  - Cumulative returns between meetings

### 3. 10-year Japanese Government Bond Yield (10年国債利回り)
- **Data Source**: FRED (IRLTLT01JPM156N)
- **Metrics Tracked**:
  - Daily yield values
  - Daily changes
  - 30-day rolling volatility
  - Mean values between meetings
  - Cumulative changes between meetings

### Analysis Periods
- Market data is analyzed across different BOJ governor terms:
  - Masaru Hayami (1998-2003)
  - Toshihiko Fukui (2003-2008)
  - Masaaki Shirakawa (2008-2013)
  - Haruhiko Kuroda (2013-2023)
  - Kazuo Ueda (2023-present)

### Market Statistics
For each period between BOJ meetings, the following statistics are calculated:
- Mean values and returns
- Volatility measures
- Cumulative returns
- Correlation with text similarity scores
- Impact analysis of significant policy language changes

## Features
- Scrape and download BOJ monetary policy meeting dates and minutes
- Extract and analyze text from BOJ meeting minutes (with Japanese language support)
- Calculate text similarity between consecutive meetings
- Fetch financial market data (Nikkei 225, USD/JPY, JGB)
- Calculate market statistics between policy meetings
- Analyze relationships between text similarity and market outcomes
- Generate visualizations of findings

## Analysis Steps

1. **Get BOJ meeting dates**: Retrieves historical BOJ monetary policy meeting dates
2. **Download BOJ documents**: Downloads meeting minutes and press conference PDFs
3. **Extract text**: Uses OCR to extract Japanese text from PDFs
4. **Analyze text content**: Processes and analyzes language patterns in meeting minutes
5. **Fetch market data**: Retrieves financial market performance data around meeting dates
6. **Calculate statistics**: Computes market performance metrics between meetings
7. **Combine analyses**: Integrates text analysis with market performance
8. **Analyze relationships**: Studies correlation between language patterns and market movements

## Data Sources

- BOJ meeting minutes and press conferences: Bank of Japan official website
- Financial market data: Yahoo Finance (via yfinance) and FRED

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Setting up the environment

You have two options for setting up your Python environment for this project:

#### Option 1: Using a dedicated Conda environment (Recommended)

1. Create the project directory structure:

```bash
mkdir -p ~/BOJ_Analysis/pdfs ~/BOJ_Analysis/output
cd ~/BOJ_Analysis
# Place main.py in this directory
```

2. Create and activate a dedicated "textanalysis" Conda environment:

```bash
# Create the environment with Python 3.9
conda create -n textanalysis python=3.9
conda activate textanalysis
```

3. Install all required dependencies:

```bash
# Core scientific and data packages
conda install -c conda-forge numpy pandas matplotlib seaborn scikit-learn

# Web scraping and data fetching
conda install -c conda-forge requests beautifulsoup4 tqdm lxml
pip install yfinance pandas-datareader

# PDF processing
conda install -c conda-forge pypdf2 pdfminer.six
pip install pymupdf pikepdf pdf2image pytesseract

# Japanese text processing
pip install mecab-python3
```

4. Create an activation script for easier environment management:

```bash
# Save as ~/BOJ_Analysis/activate_textanalysis.sh
#!/bin/bash

# Activate the text analysis environment
conda activate textanalysis

# Print environment info
echo "Text Analysis environment activated"
echo "Python: $(python --version)"
echo "Working directory: $(pwd)"
```

Make it executable:
```bash
chmod +x ~/BOJ_Analysis/activate_textanalysis.sh
```

#### Option 2: Using a virtual environment

1. Create the project directory:

```bash
mkdir BOJ_Analysis
cd BOJ_Analysis
# Place main.py in this directory
```

2. Create and activate a virtual environment:

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tqdm requests beautifulsoup4 lxml yfinance pandas-datareader PyPDF2 pdfminer.six pymupdf pikepdf
```

4. For Japanese text processing (required for full functionality):

```bash
# For Ubuntu/Debian
sudo apt-get install mecab mecab-ipadic-utf8 libmecab-dev swig poppler-utils
pip install mecab-python3 pdf2image pytesseract

# For macOS (using Homebrew)
brew install mecab mecab-ipadic swig poppler
pip install mecab-python3 pdf2image pytesseract

# For Windows
# Install MeCab manually, then:
pip install mecab-python3 pdf2image pytesseract
```

5. Install Tesseract OCR for PDF processing:

```bash
# For Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-jpn

# For macOS
brew install tesseract tesseract-lang

# For Windows
# Download and install from https://github.com/UB-Mannheim/tesseract/wiki
# Add Tesseract to PATH and download Japanese language data
```

## Usage

### Option 1: Using the dedicated textanalysis environment

1. Navigate to your project directory and activate the environment:

```bash
cd ~/BOJ_Analysis
source ./activate_textanalysis.sh
```

2. Run the main script:

```bash
python main.py
```

### Option 2: Using a virtual environment

1. Navigate to your project directory and activate the environment:

```bash
cd ~/BOJ_Analysis
# For Windows
venv\Scripts\activate
# For macOS/Linux
source venv/bin/activate
```

2. Run the main script:

```bash
python main.py
```

### Option 3: Using Google Colab with main.ipynb

1. Upload the main.ipynb notebook to Google Colab
2. Mount your Google Drive (required for saving outputs):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. Install the required dependencies:
   ```python
   !pip install numpy pandas matplotlib seaborn scikit-learn tqdm requests beautifulsoup4 lxml yfinance pandas-datareader PyPDF2 pdfminer.six pymupdf pikepdf pdf2image pytesseract mecab-python3
   ```

4. Install system dependencies:
   ```python
   !apt-get update
   !apt-get install -y tesseract-ocr tesseract-ocr-jpn poppler-utils mecab libmecab-dev mecab-ipadic-utf8
   ```

5. Run the analysis code cells in the notebook

The notebook provides the same functionality as the script but allows for interactive exploration of the data and visualizations.

### What the script does

The script will guide you through the following steps:
   - Download BOJ meeting dates
   - Optionally download BOJ meeting PDFs
   - Analyze meeting texts and calculate text similarity
   - Fetch market data
   - Calculate market statistics between meetings
   - Combine text and market analyses
   - Analyze relationships between meeting similarity and market outcomes

The results will be saved in the `BOJ_Analysis/output` directory:
   - CSV files with analyzed data
   - Visualizations of key findings

## Data Directory Structure

The script will create the following directory structure:

```
BOJ_Analysis/
│
├── pdfs/                  # Meeting minutes PDFs
│
└── output/                # Analysis results
    ├── boj_meeting_dates.csv
    ├── boj_text_analysis.csv
    ├── market_data.csv
    ├── boj_market_stats.csv
    ├── boj_combined_analysis.csv
    └── [visualization images]
```

## Configuration Options

You can modify the following variables at the top of `main.py` to customize your analysis:

- `BASE_DIR`: Base directory for the project
- `PDF_DIR`: Directory for storing PDFs
- `OUTPUT_DIR`: Directory for output files

## Troubleshooting

### Missing Japanese Text Processing

If you see the warning "Japanese text processing modules not available", the script will skip text analysis but still continue with other parts. To enable full functionality, install the Japanese text processing packages as described in the installation section.

### PDF Download Issues

If you encounter problems downloading PDFs, you can manually download them from the [BOJ website](https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm/) and place them in the `pdfs` directory.

### Market Data Access

The script uses Yahoo Finance and FRED for market data. If you encounter connection issues, try running the script again or check your internet connection.

## Notes for Advanced Users

- By default, the script processes only 5 pages of each PDF. You can adjust the `max_pages` parameter for more thorough analysis.
- The text analysis uses TF-IDF and cosine similarity. You can modify these algorithms in the code for different analysis approaches.
- Market data starts from 1998 by default but can be adjusted by modifying the `start_date` parameter in the `fetch_market_data` function.
- When running in Google Colab, the script automatically detects the environment and configures paths accordingly.

## Output

The analysis produces various outputs including:
- Text similarity analysis between consecutive BOJ meetings
- Keyword trend analysis (inflation, deflation, FX, etc.)
- Market performance metrics around meeting dates
- Correlation analysis between communication patterns and market movements
- Visualizations of key relationships and trends

## License

This project is available for personal and educational use.

---

# 日本語版 / Japanese Version

# 日本銀行金融政策決定会合の分析

このプロジェクトは、日本銀行（日銀）の金融政策決定会合の議事録の内容を分析し、その内容と市場動向の関係を調査するものです。言語パターンの変化と市場結果の間のパターンを特定するために、会合議事録のテキスト類似性分析と市場統計を組み合わせます。

## 概要

分析プロセス：
1. 日銀の会合日程の取得
2. 日銀の議事録と記者会見のPDFをダウンロード
3. OCRとNLP技術を用いてテキスト内容の抽出と分析
4. 関連する金融市場データの取得
5. 会合間の市場統計の計算
6. テキスト分析と市場パフォーマンスデータの統合
7. 言語パターンと市場動向の関係性を分析

## 機能
- 日銀の金融政策決定会合の日程と議事録のスクレイピングとダウンロード
- 日銀会合議事録からのテキスト抽出と分析（日本語対応）
- 連続する会合間のテキスト類似性の計算
- 金融市場データの取得（日経平均、ドル円、日本国債）
- 政策会合間の市場統計の計算
- テキスト類似性と市場結果の関係の分析
- 分析結果の可視化

## インストール方法

### 前提条件
- Python 3.7+
- pip（Pythonパッケージインストーラー）

### 環境のセットアップ

#### 方法1: 専用のConda環境を使用（推奨）

1. プロジェクトディレクトリ構造を作成：

```bash
mkdir -p ~/BOJ_Analysis/pdfs ~/BOJ_Analysis/output
cd ~/BOJ_Analysis
# main.pyをこのディレクトリに配置
```

2. 専用の「textanalysis」Conda環境を作成して有効化：

```bash
# Python 3.9で環境を作成
conda create -n textanalysis python=3.9
conda activate textanalysis
```

3. 必要な依存関係をすべてインストール：

```bash
# コアの科学計算とデータパッケージ
conda install -c conda-forge numpy pandas matplotlib seaborn scikit-learn

# Webスクレイピングとデータ取得
conda install -c conda-forge requests beautifulsoup4 tqdm lxml
pip install yfinance pandas-datareader

# PDF処理
conda install -c conda-forge pypdf2 pdfminer.six
pip install pymupdf pikepdf pdf2image pytesseract

# 日本語テキスト処理
pip install mecab-python3
```

4. 日本語テキスト処理用のシステム依存性（完全な機能に必要）：

```bash
# Ubuntuの場合
sudo apt-get install mecab mecab-ipadic-utf8 libmecab-dev swig poppler-utils
sudo apt-get install tesseract-ocr tesseract-ocr-jpn

# macOSの場合（Homebrewを使用）
brew install mecab mecab-ipadic swig poppler
brew install tesseract tesseract-lang
```

5. 環境アクティベーションスクリプトの作成：

```bash
# ~/BOJ_Analysis/activate_textanalysis.shとして保存
#!/bin/bash

# テキスト分析環境を有効化
conda activate textanalysis

# 環境情報を表示
echo "テキスト分析環境が有効化されました"
echo "Python: $(python --version)"
echo "作業ディレクトリ: $(pwd)"
```

実行可能にする：
```bash
chmod +x ~/BOJ_Analysis/activate_textanalysis.sh
```

## 使用方法

### 方法1: 専用のtextanalysis環境を使用

1. プロジェクトディレクトリに移動し、環境を有効化：

```bash
cd ~/BOJ_Analysis
source ./activate_textanalysis.sh
```

2. メインスクリプトを実行：

```bash
python main.py
```

### 方法2: 仮想環境を使用

1. プロジェクトディレクトリに移動し、環境を有効化：

```bash
cd ~/BOJ_Analysis
# Windowsの場合
venv\Scripts\activate
# macOS/Linuxの場合
source venv/bin/activate
```

2. メインスクリプトを実行：

```bash
python main.py
```

### 方法3: Google Colabとmain.ipynbを使用

1. main.ipynbノートブックをGoogle Colabにアップロード
2. Google Driveをマウント（出力の保存に必要）：
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. 必要な依存関係をインストール：
   ```python
   !pip install numpy pandas matplotlib seaborn scikit-learn tqdm requests beautifulsoup4 lxml yfinance pandas-datareader PyPDF2 pdfminer.six pymupdf pikepdf pdf2image pytesseract mecab-python3
   ```

4. システム依存関係をインストール：
   ```python
   !apt-get update
   !apt-get install -y tesseract-ocr tesseract-ocr-jpn poppler-utils mecab libmecab-dev mecab-ipadic-utf8
   ```

5. ノートブック内の分析コードセルを実行

このノートブックはスクリプトと同じ機能を提供しますが、データと可視化のインタラクティブな探索が可能です。

## 上級ユーザー向けの注意点

- デフォルトでは、スクリプトは各PDFの最初の5ページのみを処理します。より詳細な分析のために`max_pages`パラメータを調整できます。
- テキスト分析にはTF-IDFとコサイン類似度を使用しています。異なる分析アプローチのためにコード内のこれらのアルゴリズムを変更できます。
- 市場データはデフォルトで1998年から開始しますが、`fetch_market_data`関数の`start_date`パラメータを変更して調整できます。

## 出力

分析により、以下のような様々な出力が生成されます：
- 連続する日銀会合間のテキスト類似性分析
- キーワードトレンド分析（インフレ、デフレ、為替など）
- 会合日周辺の市場パフォーマンス指標
- コミュニケーションパターンと市場動向の相関分析
- 主要な関係性とトレンドの可視化

## ライセンス

このプロジェクトは個人的および教育目的での使用が可能です。
