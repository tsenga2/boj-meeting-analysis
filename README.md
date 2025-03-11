# BOJ Meeting Similarity and Market Impact Analysis

This repository analyzes how the similarity between consecutive Bank of Japan (BOJ) policy meetings affects market outcomes.

## Project Overview

The analysis combines text mining of BOJ meeting minutes with financial market data to:

1. Measure text similarity between consecutive monetary policy meetings
2. Extract key economic term mentions (inflation, deflation, FX, etc.)
3. Calculate market statistics between meetings
4. Analyze relationships between policy meeting similarity and market behavior

## Repository Structure

```
├── main.py                           # Main analysis script
├── boj_past_meeting_similarity_ocr.py  # Text similarity analysis 
├── boj_mpm_marketstats.py            # Market statistics
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Setup and Usage

The project can be run either locally or on Google Colab.

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/tsenga2/boj-meeting-analysis.git
   cd boj-meeting-analysis
   ```

2. Install system dependencies:
   - **Ubuntu/Debian:**
     ```bash
     sudo apt-get update
     sudo apt-get install -y tesseract-ocr tesseract-ocr-jpn poppler-utils mecab libmecab-dev mecab-ipadic-utf8
     ```
   - **macOS:**
     ```bash
     brew install tesseract tesseract-lang poppler mecab mecab-ipadic
     ```
   - **Windows:**
     - Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) with Japanese language support
     - Install [MeCab](https://taku910.github.io/mecab/) for Windows
     - Install [Poppler](https://blog.alivate.com.au/poppler-windows/)
     - Add all binaries to your PATH

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the analysis:
   ```bash
   python main.py
   ```

### Google Colab Setup

1. Create a new Colab notebook

2. Add the following code blocks and run them sequentially:

   ```python
   # 1. Mount Google Drive
   from google.colab import drive
   drive.mount('/content/drive')
   ```

   ```python
   # 2. Clone the repository
   !git clone https://github.com/tsenga2/boj-meeting-analysis.git
   %cd boj-meeting-analysis
   ```

   ```python
   # 3. Install required packages
   !apt-get update
   !apt-get install -y tesseract-ocr tesseract-ocr-jpn poppler-utils mecab libmecab-dev mecab-ipadic-utf8
   !pip install pdf2image pytesseract mecab-python3 yfinance pandas-datareader scikit-learn numpy pandas matplotlib seaborn requests beautifulsoup4 lxml tqdm
   ```

   ```python
   # 4. Run the analysis (import from main.py and set Google Drive paths)
   import os
   import sys
   sys.path.append('.')
   from main import get_boj_meeting_dates, analyze_meeting_texts, fetch_market_data, calculate_inter_meeting_stats, combine_text_and_market_analysis, analyze_similarity_market_relationship

   # Set paths to Google Drive
   BASE_DIR = '/content/drive/MyDrive/BOJ_Analysis'
   PDF_DIR = os.path.join(BASE_DIR, 'pdfs')
   OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

   # Create directories
   for directory in [BASE_DIR, PDF_DIR, OUTPUT_DIR]:
       os.makedirs(directory, exist_ok=True)

   # Run analysis steps
   print("Step 1: Get BOJ meeting dates")
   meeting_dates = get_boj_meeting_dates()
   
   print("Step 2: Analyze meeting texts")
   text_df = analyze_meeting_texts(pdf_dir=PDF_DIR)
   
   print("Step 3: Fetch market data")
   market_data = fetch_market_data()
   
   print("Step 4: Calculate market statistics")
   market_stats = calculate_inter_meeting_stats(market_data, meeting_dates)
   
   print("Step 5: Combine analyses")
   combined_df = combine_text_and_market_analysis(text_df, market_stats)
   
   print("Step 6: Analyze relationship")
   correlation_df, group_stats = analyze_similarity_market_relationship(combined_df)
   
   print("Analysis complete!")
   ```

## Output

The analysis generates the following in the `BOJ_Analysis/output` directory:

1. `boj_meeting_dates.csv` - List of all BOJ meeting dates
2. `boj_text_analysis.csv` - Text similarity analysis results
3. `market_data.csv` - Market data
4. `boj_combined_analysis.csv` - Combined analysis
5. Visualization files:
   - `similarity_market_correlation.png`
   - `similarity_market_scatterplots.png`
   - `similarity_group_analysis.png`