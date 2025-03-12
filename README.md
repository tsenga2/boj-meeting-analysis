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
├── main.ipynb                        # Jupyter notebook version of the analysis
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
   
   Or run the Jupyter notebook:
   ```bash
   jupyter notebook main.ipynb
   ```

### Google Colab Setup

1. Create a new Colab notebook or open the existing `main.ipynb` directly in Google Colab

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
   from main import get_boj_meeting_dates, download_boj_pdfs, analyze_meeting_texts, fetch_market_data, calculate_inter_meeting_stats, combine_text_and_market_analysis, analyze_similarity_market_relationship

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
   
   print("Step 2: Download BOJ meeting PDFs")
   download_boj_pdfs(years=range(1998, 2015))  # For PDF downloads the website structure changes after 2014
   
   print("Step 3: Analyze meeting texts")
   text_df = analyze_meeting_texts(pdf_dir=PDF_DIR, max_pages=5)  # Limit to 5 pages per document for faster processing
   
   print("Step 4: Fetch market data")
   market_data = fetch_market_data()
   
   print("Step 5: Calculate market statistics")
   market_stats = calculate_inter_meeting_stats(market_data, meeting_dates)
   
   print("Step 6: Combine analyses")
   combined_df = combine_text_and_market_analysis(text_df, market_stats)
   
   print("Step 7: Analyze relationship")
   correlation_df, group_stats = analyze_similarity_market_relationship(combined_df)
   
   print("Analysis complete!")
   ```

## Analysis Pipeline

1. **Meeting Date Collection**
   - Scrape and parse BOJ monetary policy meeting dates from the BOJ website

2. **Meeting PDF Documents**
   - Download PDF files of BOJ meeting minutes (from years 1998-2014)
   - **This step is critical** - without the PDFs, the text analysis cannot proceed
   - Note: The website structure for BOJ documents changes after 2014

3. **Text Processing and Similarity Analysis**
   - Convert PDFs to images for OCR processing (limited to 5 pages per document by default)
   - Extract text using Tesseract OCR with Japanese language support
   - Process text with MeCab for Japanese tokenization
   - Extract key economic term mentions (inflation, deflation, interest rates, etc.)
   - Calculate text similarity between consecutive meetings using TF-IDF and cosine similarity

4. **Market Data Analysis**
   - Fetch financial market data (Nikkei 225, USD/JPY, JGB yields)
   - Calculate market statistics between consecutive BOJ meetings
   - Measure returns, volatility, and trend metrics

5. **Combined Analysis**
   - Merge text similarity data with market statistics
   - Calculate correlations between text similarity and market metrics
   - Group meetings by similarity level and analyze market behavior patterns

## Output

The analysis generates the following in the `BOJ_Analysis/output` directory:

1. `boj_meeting_dates.csv` - List of all BOJ meeting dates
2. `boj_text_analysis.csv` - Text similarity analysis results
3. `market_data.csv` - Raw market data
4. `boj_market_stats.csv` - Inter-meeting market statistics
5. `boj_combined_analysis.csv` - Combined text and market data
6. Visualization files:
   - `text_similarity_analysis.png` - Text similarity between meetings and key term frequencies
   - `similarity_market_correlation.png` - Correlation between text similarity and market metrics
   - `similarity_market_scatterplots.png` - Scatter plots of similarity vs. market outcomes
   - `similarity_group_analysis.png` - Market statistics grouped by similarity level

## Notes

- Japanese text processing requires specialized libraries and configurations
- The PDF download step is essential - without the PDFs, text analysis cannot proceed
- For best results, ensure Tesseract OCR is properly configured for Japanese language
- The BOJ website structure changes after 2014, so the PDF download function is configured to work with years 1998-2014
- Processing is limited to 5 pages per document by default to improve performance; adjust the `max_pages` parameter in `analyze_meeting_texts()` if needed
