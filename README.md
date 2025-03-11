# BOJ Meeting Similarity and Market Impact Analysis

This repository analyzes how the similarity between consecutive Bank of Japan (BOJ) policy meetings affects market outcomes in the periods between meetings.

## Project Overview

The analysis combines text mining of BOJ meeting minutes with financial market data to:

1. Measure text similarity between consecutive monetary policy meetings
2. Extract key economic term mentions (inflation, deflation, FX, etc.)
3. Calculate market statistics (returns, volatility) between meetings
4. Analyze relationships between policy meeting similarity and market behavior

## Repository Structure

```
├── main.py                 # Main script that integrates all components
├── boj_past_meeting_similarity_ocr.py    # Text similarity analysis script
├── boj_mpm_marketstats.py  # Market statistics script
├── requirements.txt        # Python dependencies
├── README.md               # This readme file
└── output/                 # Generated analysis and visualizations (gitignored)
    ├── boj_meeting_dates.csv
    ├── boj_text_analysis.csv
    ├── market_data.csv
    ├── boj_combined_analysis.csv
    ├── similarity_market_correlation.png
    ├── similarity_market_scatterplots.png
    └── similarity_group_analysis.png
```

## Requirements

- Python 3.7+
- Japanese language support packages (MeCab, tesseract-ocr-jpn)
- Financial data packages (yfinance, pandas_datareader)
- Text processing packages (pdf2image, pytesseract)
- Data analysis packages (pandas, numpy, scikit-learn)
- Visualization packages (matplotlib, seaborn)

## Installation

### Local Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/boj-meeting-analysis.git
   cd boj-meeting-analysis
   ```

2. Install dependencies:
   
   **Ubuntu/Debian:**
   ```bash
   # Install system dependencies
   sudo apt-get update
   sudo apt-get install -y tesseract-ocr tesseract-ocr-jpn poppler-utils mecab libmecab-dev mecab-ipadic-utf8
   
   # Install Python dependencies
   pip install -r requirements.txt
   ```
   
   **macOS:**
   ```bash
   # Install system dependencies
   brew install tesseract tesseract-lang poppler mecab mecab-ipadic
   
   # Install Python dependencies
   pip install -r requirements.txt
   ```
   
   **Windows:**
   - Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) with Japanese language support
   - Install [MeCab](https://taku910.github.io/mecab/) for Windows
   - Install [Poppler](https://blog.alivate.com.au/poppler-windows/)
   - Add all binaries to your PATH
   - Install Python dependencies: `pip install -r requirements.txt`

### Google Colab Setup

1. Upload the scripts to your Google Drive or clone the repo directly:

   ```python
   !git clone https://github.com/yourusername/boj-meeting-analysis.git
   ```

2. Install dependencies in Colab:

   ```python
   !apt-get update
   !apt-get install -y tesseract-ocr tesseract-ocr-jpn poppler-utils mecab libmecab-dev mecab-ipadic-utf8
   !pip install pdf2image pytesseract mecab-python3 pdfminer.six yfinance pandas-datareader scikit-learn
   ```

3. Mount your Google Drive to save results:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. Update the base directory paths in the script to use Google Drive:

   ```python
   BASE_DIR = '/content/drive/MyDrive/BOJ_Analysis'
   ```

## Usage

### Running Locally

1. Execute the main script:

   ```bash
   python main.py
   ```

2. For specific components only:

   ```bash
   # Text similarity analysis only
   python boj_past_meeting_similarity_ocr.py
   
   # Market statistics only
   python boj_mpm_marketstats.py
   ```

### Running on Google Colab

1. Upload the notebook version of the script to Google Colab or open it from GitHub.

2. Execute the notebook cells sequentially.

3. Alternatively, you can run the Python script directly in Colab:

   ```python
   %cd /content/boj-meeting-analysis
   !python main.py
   ```

## Output Files

The analysis generates the following outputs in the `output` directory:

1. `boj_meeting_dates.csv` - List of all BOJ meeting dates
2. `boj_text_analysis.csv` - Text similarity analysis and keyword mentions
3. `market_data.csv` - Raw market data for Nikkei, USD/JPY, and JGBs
4. `boj_combined_analysis.csv` - Combined text and market statistics
5. Visualization files:
   - `similarity_market_correlation.png` - Correlation coefficients
   - `similarity_market_scatterplots.png` - Scatter plots of relationships
   - `similarity_group_analysis.png` - Bar charts of market metrics by similarity group

## Customization

You can customize the analysis by modifying parameters in the main script:

- `MAX_PAGES`: Number of PDF pages to process per meeting (default=5)
- `YEARS`: Years of BOJ meetings to analyze (default=1998-2014)
- Market metrics: You can modify or add additional market indicators

## Troubleshooting

### Common Issues

1. **OCR Issues**: If you encounter poor text extraction quality:
   - Ensure Japanese language data is installed for Tesseract
   - Try increasing DPI in the `convert_from_path` function

2. **Market Data Issues**: If market data fails to download:
   - Check internet connectivity
   - FRED API may have rate limits or require an API key
   - Try alternative sources in the `fetch_market_data` function

3. **Memory Errors**: For large datasets:
   - Process fewer years at a time
   - Reduce `MAX_PAGES` parameter
   - Use a machine with more memory

### Getting Help

If you encounter any issues or have questions, please open an issue in the GitHub repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
