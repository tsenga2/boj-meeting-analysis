# BOJ Meeting Similarity and Market Impact Analysis

This project analyzes Bank of Japan (BOJ) monetary policy meeting minutes and press conferences to identify patterns and their relationship with market movements.

## Overview

The analysis pipeline:
1. Retrieves BOJ meeting dates
2. Downloads BOJ meeting minutes and press conference PDFs
3. Extracts and analyzes text content using OCR and NLP techniques
4. Fetches corresponding financial market data
5. Calculates market statistics between meetings
6. Combines text analysis with market performance data
7. Analyzes relationships between language patterns and market movements

## Requirements

### System dependencies:
- Tesseract OCR with Japanese language support
- MeCab (Japanese text segmentation)
- Poppler utilities (PDF processing)

### Python packages:
- pdf2image, pytesseract, mecab-python3
- yfinance, pandas-datareader
- scikit-learn, numpy, pandas
- matplotlib, seaborn
- requests, beautifulsoup4, lxml, tqdm

## Installation

```bash
# Install system dependencies
apt-get update
apt-get install -y tesseract-ocr tesseract-ocr-jpn poppler-utils mecab libmecab-dev mecab-ipadic-utf8

# Install Python packages
pip install pdf2image pytesseract mecab-python3 yfinance pandas-datareader scikit-learn numpy pandas matplotlib seaborn requests beautifulsoup4 lxml tqdm
```

## Usage

### Option 1: Run with Python script
```bash
python main.py
```

### Option 2: Run with Jupyter Notebook
The analysis can also be run using the provided `main.ipynb` notebook:

1. Mount Google Drive (if using Google Colab)
2. Clone the repository
3. Install required packages
4. Run the analysis pipeline with customizable parameters

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
- Financial market data: Yahoo Finance (via yfinance)

## Output

The analysis produces various outputs including:
- Text sentiment and similarity analysis of BOJ communications
- Market performance metrics around meeting dates
- Correlation analysis between communication patterns and market movements
- Visualizations of key relationships and trends

## License

[Specify your license here]

---

# 日本語版 / Japanese Version

# 日本銀行金融政策決定会合の分析

このプロジェクトは、日本銀行（日銀）の金融政策決定会合の議事録および記者会見の内容を分析し、その内容と市場動向の関係を調査するものです。

## 概要

分析プロセス：
1. 日銀の会合日程の取得
2. 日銀の議事録と記者会見のPDFをダウンロード
3. OCRとNLP技術を用いてテキスト内容の抽出と分析
4. 関連する金融市場データの取得
5. 会合間の市場統計の計算
6. テキスト分析と市場パフォーマンスデータの統合
7. 言語パターンと市場動向の関係性を分析

## 必要条件

### システム依存性：
- 日本語対応のTesseract OCR
- MeCab（日本語形態素解析）
- Popplerユーティリティ（PDF処理）

### Pythonパッケージ：
- pdf2image, pytesseract, mecab-python3
- yfinance, pandas-datareader
- scikit-learn, numpy, pandas
- matplotlib, seaborn
- requests, beautifulsoup4, lxml, tqdm

## インストール方法

```bash
# システム依存パッケージのインストール
apt-get update
apt-get install -y tesseract-ocr tesseract-ocr-jpn poppler-utils mecab libmecab-dev mecab-ipadic-utf8

# Pythonパッケージのインストール
pip install pdf2image pytesseract mecab-python3 yfinance pandas-datareader scikit-learn numpy pandas matplotlib seaborn requests beautifulsoup4 lxml tqdm
```

## 使用方法

### 方法1: Pythonスクリプトで実行
```bash
python main.py
```

### 方法2: Jupyter Notebookで実行
分析は提供されている`main.ipynb`ノートブックを使用して実行することもできます：

1. Google Driveをマウント（Google Colabを使用する場合）
2. リポジトリをクローン
3. 必要なパッケージをインストール
4. カスタマイズ可能なパラメータで分析パイプラインを実行

## 分析ステップ

1. **日銀会合日の取得**：日本銀行の金融政策決定会合の過去の日程を取得
2. **日銀文書のダウンロード**：会合議事録と記者会見のPDFをダウンロード
3. **テキスト抽出**：OCRを使用してPDFから日本語テキストを抽出
4. **テキスト内容の分析**：会合議事録の言語パターンを処理・分析
5. **市場データの取得**：会合日周辺の金融市場パフォーマンスデータを取得
6. **統計の計算**：会合間の市場パフォーマンス指標を計算
7. **分析の統合**：テキスト分析と市場パフォーマンスを統合
8. **関係性の分析**：言語パターンと市場動向の相関関係を研究

## データソース

- 日銀の議事録と記者会見：日本銀行公式ウェブサイト
- 金融市場データ：Yahoo Finance（yfinance経由）

## 出力結果

分析により、以下のような様々な出力が生成されます：
- 日銀のコミュニケーションのテキストセンチメントと類似性分析
- 会合日周辺の市場パフォーマンス指標
- コミュニケーションパターンと市場動向の相関分析
- 主要な関係性とトレンドの可視化

