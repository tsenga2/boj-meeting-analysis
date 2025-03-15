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
