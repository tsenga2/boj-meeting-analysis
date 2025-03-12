#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOJ Meeting Similarity and Market Impact Analysis

This script combines text similarity analysis of BOJ meeting minutes with 
market statistics to analyze how changes in policy meeting content affect
financial markets before the next meeting.
"""

import os
import sys
import re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas_datareader.data as pdr

# Check if running in Colab or locally
try:
    import google.colab
    IN_COLAB = True
    print("Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("Running locally")

# Configure paths based on environment
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = '/content/drive/MyDrive/BOJ_Analysis'
else:
    BASE_DIR = './BOJ_Analysis'

PDF_DIR = os.path.join(BASE_DIR, 'pdfs')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Create directories if they don't exist
for directory in [BASE_DIR, PDF_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure Japanese text processing
try:
    import MeCab
    from pdf2image import convert_from_path
    import pytesseract
    JAPANESE_TEXT_AVAILABLE = True
    os.environ['MECABRC'] = "/etc/mecabrc"  # Default path
except ImportError:
    JAPANESE_TEXT_AVAILABLE = False
    print("Warning: Japanese text processing modules not available.")
    print("Text similarity analysis will be skipped.")

#############################################################
# Part 1: BOJ Meeting Dates and Document Collection
#############################################################

def get_boj_meeting_dates(start_year=1998, end_year=2025, output_dir="."):
    """
    Scrape BOJ Monetary Policy Meeting dates from yearly minutes pages
    
    Parameters:
    -----------
    start_year : int
        First year to fetch meeting dates
    end_year : int
        Last year to fetch meeting dates
    output_dir : str
        Directory to save output CSV file
        
    Returns:
    --------
    list
        List of datetime objects representing meeting dates
    """
    print(f"Fetching BOJ meeting dates from {start_year} to {end_year}...")
    all_dates = []
    base_url = "https://www.boj.or.jp"
    
    for year in range(start_year, end_year + 1):
        print(f"Fetching dates for {year}...")
        url = f"{base_url}/en/mopo/mpmsche_minu/minu_{year}/index.htm"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Find text containing meeting dates
            meeting_texts = []
            
            # Look for meeting date patterns in the page
            for element in soup.find_all(['td', 'li', 'p']):
                text = element.get_text().strip()
                if text.startswith("Meeting on ") or text.startswith("Meetingon "):
                    meeting_texts.append(text)
            
            year_dates = []
            for text in meeting_texts:
                try:
                    # Clean up the text
                    # Strip "Meeting on " prefix and remove PDF info
                    clean_text = text.replace("Meeting on ", "").replace("Meetingon ", "")
                    clean_text = re.sub(r'\[PDF.*?\]', '', clean_text).strip()
                    
                    # Handle dates with "and" (e.g., "January 21 and 22, 2020")
                    if " and " in clean_text:
                        parts = clean_text.split(" and ")
                        # Get the last day mentioned (after "and")
                        second_part = parts[1]
                        
                        # If the second part only contains day and potentially year
                        if len(second_part.split()) <= 2:
                            # Use the month from the first part
                            month = parts[0].split()[0]
                            day = second_part.split()[0].replace(",", "")
                            
                            # Check if year is present in second part
                            if len(second_part.split()) == 2:
                                year_str = second_part.split()[1]
                            else:
                                # If not, assume it's at the end of the string
                                year_str = clean_text.split(",")[-1].strip()
                                if not year_str.isdigit():
                                    year_str = str(year)
                        else:
                            # Full date after "and"
                            month = second_part.split()[0]
                            day = second_part.split()[1].replace(",", "")
                            
                            if len(second_part.split()) > 2:
                                year_str = second_part.split()[-1]
                            else:
                                year_str = str(year)
                    else:
                        # Regular date format
                        parts = clean_text.split()
                        month = parts[0]
                        day = parts[1].replace(",", "")
                        
                        if len(parts) > 2:
                            year_str = parts[-1]
                        else:
                            year_str = str(year)
                    
                    # Ensure year is a 4-digit number
                    if len(year_str) < 4 and year_str.isdigit():
                        year_str = f"20{year_str}" if int(year_str) < 50 else f"19{year_str}"
                    
                    # Construct the date string
                    date_str = f"{month} {day} {year_str}"
                    meeting_date = pd.to_datetime(date_str)
                    
                    # Verify the year is correct (in case it wasn't in the text)
                    if meeting_date.year != year and year_str == str(year):
                        # Try the next year (for December meetings referring to January dates)
                        date_str = f"{month} {day} {year+1}"
                        meeting_date = pd.to_datetime(date_str)
                        
                        # If still not matching, revert to the original year from the URL
                        if meeting_date.year != year:
                            date_str = f"{month} {day} {year}"
                            meeting_date = pd.to_datetime(date_str)
                    
                    year_dates.append(meeting_date)
                    
                except Exception as e:
                    print(f"Error parsing date '{text}': {e}")
                    # Add more detailed debugging if needed
                    continue
            
            if year_dates:
                print(f"Found {len(year_dates)} meetings in {year}")
                all_dates.extend(year_dates)
            else:
                print(f"No meetings found for {year}")
                
        except Exception as e:
            print(f"Error fetching {year}: {e}")
            continue
    
    # Sort dates and remove duplicates
    all_dates = sorted(list(set(all_dates)))
    
    # Create DataFrame and save to CSV
    df_dates = pd.DataFrame({'meeting_date': all_dates})
    output_path = os.path.join(output_dir, 'boj_meeting_dates.csv')
    df_dates.to_csv(output_path, index=False)
    
    print(f"Found {len(all_dates)} BOJ meeting dates. Saved to '{output_path}'")
    return all_dates

def download_boj_pdfs(years=None):
    """
    Download PDFs of BOJ meeting minutes for specified years
    """
    if not years:
        years = range(1998, 2015)  # Default range
    
    session = requests.Session()
    base_url = "https://www.boj.or.jp"
    
    print(f"Downloading PDFs for years {min(years)}-{max(years)}...")
    total_downloaded = 0
    
    for year in years:
        print(f"Checking year {year}...")
        url = f"{base_url}/mopo/mpmsche_minu/record_{year}/index.htm"
        
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Debug: Print all links on the page
            #print(f"--- All links on page for year {year} ---")
            #for link in soup.find_all('a'):
            #    href = link.get('href', '')
            #    text = link.get_text()
            #    print(f"Text: {text} | Href: {href}")
            
            # Use the improved regex pattern
            pdf_links = soup.find_all('a', href=re.compile(r'(?:/mopo/mpmsche_minu/record_\d{4}/|)g(?:jrk|irk)\d{6}a\.pdf$'))
                        
            
            if not pdf_links:
                print(f"No PDFs found for year {year}")
                continue
                
            print(f"Found {len(pdf_links)} PDFs for year {year}")
            year_downloaded = 0
            
            for link in tqdm(pdf_links, desc=f"Year {year}"):
                pdf_url = f"{base_url}{link['href']}"
                pdf_name = os.path.basename(link['href'])
                filename = os.path.join(PDF_DIR, f"{year}_{pdf_name}")
                
                if os.path.exists(filename):
                    continue
                    
                try:
                    response = session.get(pdf_url, timeout=30)
                    response.raise_for_status()
                    
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    
                    year_downloaded += 1
                    total_downloaded += 1
                        
                except Exception as e:
                    print(f"Error downloading {pdf_url}: {e}")
            
            print(f"Downloaded {year_downloaded} new PDFs for {year}")
                
        except Exception as e:
            print(f"Error processing year {year}: {e}")
    
    print(f"Download complete! Total new PDFs downloaded: {total_downloaded}")
    return total_downloaded

#############################################################
# Part 2: Text Processing and Similarity Analysis
#############################################################

def process_pdf_with_ocr(pdf_path, max_pages=5):
    """
    Convert PDF to images and perform OCR with Japanese language support.
    """
    if not JAPANESE_TEXT_AVAILABLE:
        print("Japanese text processing modules not available. Skipping OCR.")
        return ""
        
    try:
        print(f"Processing {os.path.basename(pdf_path)} with OCR...")
        images = convert_from_path(pdf_path)
        total_pages = len(images)
        pages_to_process = min(max_pages, total_pages)
        
        if pages_to_process < total_pages:
            print(f"Processing first {pages_to_process} of {total_pages} pages")

        full_text = []
        for i, image in enumerate(images[:pages_to_process], 1):
            sys.stdout.write(f"\rProcessing page {i}/{pages_to_process}")
            sys.stdout.flush()
            
            text = pytesseract.image_to_string(image, lang='jpn')
            full_text.append(text)
        
        print()  # New line after progress output
        return '\n'.join(full_text)

    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        return ""

def analyze_meeting_texts(pdf_dir=PDF_DIR, max_pages=5):
    """
    Process BOJ meeting PDFs and calculate text similarities
    """
    if not JAPANESE_TEXT_AVAILABLE:
        print("Japanese text processing modules not available. Skipping text analysis.")
        return pd.DataFrame()
        
    data = []
    
    # Get list of PDFs
    pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.endswith('.pdf')])
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        print("Please run download_boj_pdfs() first or check the directory path.")
        return pd.DataFrame()
        
    print(f"Found {len(pdf_files)} PDF files")
    print(f"Processing {max_pages} pages per document")
    
    # Process each PDF with progress bar
    for i, filename in enumerate(pdf_files):
        print(f"\nProcessing file {i+1}/{len(pdf_files)}: {filename}")
        pdf_path = os.path.join(pdf_dir, filename)

        # Extract date from filename
        date_match = re.search(r'(gj|gi)rk(\d{6})a', filename)
        if not date_match:
            print(f"No date match found in filename: {filename}")
            continue
        date_str = date_match.group(2)
        try:
            date = datetime.strptime(date_str, "%y%m%d").date()

            # Extract text using OCR (limited pages)
            text = process_pdf_with_ocr(pdf_path, max_pages=max_pages)

            if not text.strip():
                print(f"Warning: No text extracted from {filename}")
                continue

            # Process text with MeCab
            mecab = MeCab.Tagger("")
            processed_text = []
            node = mecab.parseToNode(text)
            while node:
                features = node.feature.split(',')
                if features[0] in ['名詞', '動詞', '形容詞']:
                    if features[0] == '動詞' and len(features) > 6:
                        processed_text.append(features[6])
                    else:
                        processed_text.append(node.surface)
                node = node.next

            processed_text = ' '.join(processed_text)

            # Extract metrics
            metrics = {
                'inflation_mentions': len(re.findall('インフレ|物価上昇', text)),
                'deflation_mentions': len(re.findall('デフレ|物価下落', text)),
                'fx_mentions': len(re.findall('為替|円相場|ドル円|ユーロ円', text)),
                'interest_rate_mentions': len(re.findall('金利|利率|利回り', text)),
                'economy_mentions': len(re.findall('景気|経済情勢|経済状況', text))
            }

            data.append({
                'date': date,
                'filename': filename,
                'text': processed_text,
                'pages_processed': min(max_pages, len(convert_from_path(pdf_path))),
                **metrics
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    # Create DataFrame
    if not data:
        print("No data was collected from PDFs")
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    df = df.sort_values('date')
    
    # Calculate text similarity
    if len(df) > 1:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['text'])
        similarity_matrix = cosine_similarity(tfidf_matrix)
        df['similarity_with_previous'] = [0] + [similarity_matrix[i, i-1] for i in range(1, len(df))]
    
    # Visualize similarity and key terms
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(df['date'], df['similarity_with_previous'], marker='o')
    plt.title("Cosine Similarity Between Consecutive Monetary Policy Meetings")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(df['date'], df['inflation_mentions'], label='Inflation', marker='o')
    plt.plot(df['date'], df['deflation_mentions'], label='Deflation', marker='*')
    plt.plot(df['date'], df['fx_mentions'], label='FX', marker='^')
    plt.title("Mention Frequency of Key Economic Indicators")
    plt.xlabel("Date")
    plt.ylabel("Mention Frequency")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "text_similarity_analysis.png"))
    
    # Save results
    df.to_csv(os.path.join(OUTPUT_DIR, 'boj_text_analysis.csv'), index=False)
    
    return df

#############################################################
# Part 3: Market Data Analysis
#############################################################

def fetch_market_data(start_date='1998-01-01', end_date=None):
    """
    Fetch market data from Yahoo Finance and FRED
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print("Fetching market data...")
    print(f"- Date range: {start_date} to {end_date}")
    
    try:
        # Nikkei 225
        print("- Downloading Nikkei 225 data...")
        nikkei = yf.download('^N225', start=start_date, end=end_date, progress=False)
        nikkei = nikkei['Adj Close'].squeeze() if 'Adj Close' in nikkei.columns else nikkei['Close'].squeeze()
        print(f"  Got {len(nikkei)} data points for Nikkei 225")

        # USD/JPY and JGB from FRED
        print("- Downloading USD/JPY and JGB data from FRED...")
        fred_data = pdr.get_data_fred(['DEXJPUS', 'IRLTLT01JPM156N'], start=start_date, end=end_date)
        print(f"  Got {len(fred_data)} data points from FRED")

        # Combine all data
        market_data = pd.DataFrame({
            'nikkei': nikkei,
            'usdjpy': fred_data['DEXJPUS'],
            'jgb': fred_data['IRLTLT01JPM156N']
        })

        # Forward fill missing values (weekends and holidays)
        market_data = market_data.fillna(method='ffill')
        
        # Calculate daily returns
        for col in ['nikkei', 'usdjpy', 'jgb']:
            market_data[f'{col}_return'] = market_data[col].pct_change()
        
        # Remove rows with NaN returns
        market_data = market_data.dropna()
        
        print(f"Final dataset: {len(market_data)} rows with complete data")
        
        # Save market data
        market_data.to_csv(os.path.join(OUTPUT_DIR, 'market_data.csv'))
        
        return market_data
        
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return pd.DataFrame()

def calculate_inter_meeting_stats(market_data, meeting_dates):
    """
    Calculate market statistics between consecutive meetings
    """
    if market_data.empty:
        print("No market data available. Cannot calculate statistics.")
        return pd.DataFrame()
        
    # Ensure meeting_dates is sorted
    meeting_dates = sorted(pd.to_datetime(meeting_dates))
    
    stats = []
    
    print(f"Calculating statistics between {len(meeting_dates)-1} meeting pairs...")
    
    # For each meeting except the last one
    for i in range(len(meeting_dates) - 1):
        start_date = meeting_dates[i]
        end_date = meeting_dates[i + 1]
        
        # Get market data between meetings
        mask = (market_data.index >= start_date) & (market_data.index < end_date)
        window_data = market_data[mask]
        
        if len(window_data) < 5:
            print(f"Warning: Insufficient data between {start_date} and {end_date}")
            continue
            
        # Calculate statistics
        stats_dict = {
            'meeting_date': start_date,
            'next_meeting_date': end_date,
            'days_between': (end_date - start_date).days,
            
            # Mean values
            'nikkei_mean': window_data['nikkei'].mean(),
            'usdjpy_mean': window_data['usdjpy'].mean(),
            'jgb_mean': window_data['jgb'].mean(),
            
            # Mean returns
            'nikkei_return_mean': window_data['nikkei_return'].mean() * 100,  # Convert to percentage
            'usdjpy_return_mean': window_data['usdjpy_return'].mean() * 100,
            'jgb_return_mean': window_data['jgb_return'].mean() * 100,
            
            # Volatilities
            'nikkei_vol': window_data['nikkei_return'].std() * 100,  # Convert to percentage
            'usdjpy_vol': window_data['usdjpy_return'].std() * 100,
            'jgb_vol': window_data['jgb_return'].std() * 100,
            
            # Cumulative returns
            'nikkei_cum_return': ((1 + window_data['nikkei_return']).cumprod().iloc[-1] - 1) * 100,
            'usdjpy_cum_return': ((1 + window_data['usdjpy_return']).cumprod().iloc[-1] - 1) * 100,
            'jgb_cum_return': ((1 + window_data['jgb_return']).cumprod().iloc[-1] - 1) * 100
        }
        
        stats.append(stats_dict)
    
    stats_df = pd.DataFrame(stats)
    
    # Save to CSV
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'boj_market_stats.csv'), index=False)
    
    return stats_df

#############################################################
# Part 4: Combined Analysis
#############################################################

def combine_text_and_market_analysis(text_df, market_stats_df):
    """
    Combine text similarity and market statistics for analysis
    """
    if text_df.empty or market_stats_df.empty:
        print("Missing data for combined analysis")
        if text_df.empty:
            print("- Text analysis data is empty")
        if market_stats_df.empty:
            print("- Market statistics data is empty")
        return pd.DataFrame()
    
    # Convert dates to datetime for proper merging
    text_df['date'] = pd.to_datetime(text_df['date'])
    market_stats_df['meeting_date'] = pd.to_datetime(market_stats_df['meeting_date'])
    
    # Merge on meeting date
    combined_df = pd.merge(
        text_df,
        market_stats_df,
        left_on='date',
        right_on='meeting_date',
        how='inner'
    )
    
    if combined_df.empty:
        print("No matching dates between text analysis and market statistics")
        return pd.DataFrame()
        
    print(f"Combined analysis has {len(combined_df)} meetings with both text and market data")
    
    # Save combined data
    combined_df.to_csv(os.path.join(OUTPUT_DIR, 'boj_combined_analysis.csv'), index=False)
    
    return combined_df

def analyze_similarity_market_relationship(combined_df):
    """
    Analyze the relationship between meeting similarity and market outcomes
    """
    if combined_df.empty:
        print("No data for similarity-market relationship analysis")
        return None, None
    
    # Calculate correlations between similarity and market metrics
    market_metrics = [
        'nikkei_return_mean', 'usdjpy_return_mean', 'jgb_return_mean',
        'nikkei_vol', 'usdjpy_vol', 'jgb_vol',
        'nikkei_cum_return', 'usdjpy_cum_return', 'jgb_cum_return'
    ]
    
    # Make sure we have the similarity column
    if 'similarity_with_previous' not in combined_df.columns:
        print("Error: 'similarity_with_previous' column missing from combined data")
        return None, None
    
    correlation_df = pd.DataFrame(index=['similarity_with_previous'])
    
    for metric in market_metrics:
        if metric in combined_df.columns:
            correlation_df[metric] = [combined_df['similarity_with_previous'].corr(combined_df[metric])]
        else:
            print(f"Warning: Metric '{metric}' not found in combined data")
    
    # Create visualization of correlations
    plt.figure(figsize=(12, 6))
    correlation_df.T.plot(kind='bar', color='skyblue')
    plt.title('Correlation between Meeting Similarity and Market Outcomes')
    plt.xlabel('Market Metric')
    plt.ylabel('Correlation Coefficient')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'similarity_market_correlation.png'))
    
    # Create scatter plots for key relationships
    available_metrics = [m for m in market_metrics if m in combined_df.columns]
    num_metrics = len(available_metrics)
    
    if num_metrics > 0:
        rows = (num_metrics + 2) // 3  # Ceiling division to get number of rows
        fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 5))
        fig.suptitle('Meeting Similarity vs Market Outcomes', fontsize=16)
        
        # Make axes iterable even if there's only one row
        if rows == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            row, col = i // 3, i % 3
            axes[row][col].scatter(combined_df['similarity_with_previous'], combined_df[metric], alpha=0.7)
            axes[row][col].set_title(f'Similarity vs {metric}')
            axes[row][col].set_xlabel('Text Similarity with Previous Meeting')
            axes[row][col].set_ylabel(metric)
            axes[row][col].grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(combined_df['similarity_with_previous'], combined_df[metric], 1)
            p = np.poly1d(z)
            axes[row][col].plot(
                combined_df['similarity_with_previous'], 
                p(combined_df['similarity_with_previous']), 
                "r--", alpha=0.7
            )
            
            # Add correlation coefficient
            corr = combined_df['similarity_with_previous'].corr(combined_df[metric])
            axes[row][col].annotate(f'r = {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction')
        
        # Hide unused subplots
        for i in range(num_metrics, rows * 3):
            row, col = i // 3, i % 3
            axes[row][col].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(os.path.join(OUTPUT_DIR, 'similarity_market_scatterplots.png'))
    
    # Calculate group statistics by similarity level
    if len(combined_df) >= 4:  # Need at least 4 rows for quartiles
        combined_df['similarity_group'] = pd.qcut(
            combined_df['similarity_with_previous'], 
            q=4, 
            labels=['Low', 'Medium-Low', 'Medium-High', 'High']
        )
        
        group_stats = combined_df.groupby('similarity_group')[available_metrics].mean()
        
        # Plot group statistics
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        
        # Returns
        return_metrics = [m for m in ['nikkei_return_mean', 'usdjpy_return_mean', 'jgb_return_mean'] 
                          if m in available_metrics]
        if return_metrics:
            group_stats[return_metrics].plot(kind='bar', ax=axes[0], colormap='viridis')
            axes[0].set_title('Average Returns by Meeting Similarity Group')
            axes[0].set_ylabel('Mean Return (%)')
            axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Volatility
        vol_metrics = [m for m in ['nikkei_vol', 'usdjpy_vol', 'jgb_vol'] 
                       if m in available_metrics]
        if vol_metrics:
            group_stats[vol_metrics].plot(kind='bar', ax=axes[1], colormap='plasma')
            axes[1].set_title('Average Volatility by Meeting Similarity Group')
            axes[1].set_ylabel('Volatility (%)')
            axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Cumulative Returns
        cum_return_metrics = [m for m in ['nikkei_cum_return', 'usdjpy_cum_return', 'jgb_cum_return'] 
                             if m in available_metrics]
        if cum_return_metrics:
            group_stats[cum_return_metrics].plot(kind='bar', ax=axes[2], colormap='cividis')
            axes[2].set_title('Average Cumulative Returns by Meeting Similarity Group')
            axes[2].set_ylabel('Cumulative Return (%)')
            axes[2].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'similarity_group_analysis.png'))
    else:
        group_stats = pd.DataFrame()
        print("Not enough data for similarity group analysis (need at least 4 meetings)")
    
    return correlation_df, group_stats

#############################################################
# Main Execution
#############################################################

def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("BOJ Meeting Similarity and Market Impact Analysis")
    print("=" * 60)
    
    # Step 1: Get BOJ meeting dates
    print("\nSTEP 1: Getting BOJ meeting dates")
    meeting_dates = get_boj_meeting_dates()
    if not meeting_dates:
        print("Error: Failed to retrieve BOJ meeting dates. Exiting.")
        return
    
    # Step 2: Ask user if they want to download PDFs
    if os.path.exists(PDF_DIR) and any(f.endswith('.pdf') for f in os.listdir(PDF_DIR)):
        pdf_count = len([f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')])
        print(f"\nFound {pdf_count} existing PDFs in {PDF_DIR}")
        
        if not IN_COLAB:  # Only ask if running locally
            download_choice = input("Do you want to download more BOJ meeting PDFs? (y/n): ").strip().lower()
            if download_choice == 'y':
                start_year = int(input("Enter start year (1998-2014): "))
                end_year = int(input("Enter end year (1998-2014): "))
                download_boj_pdfs(years=range(start_year, end_year + 1))
    else:
        print("\nNo existing PDFs found.")
        if not IN_COLAB:  # Only ask if running locally
            download_choice = input("Do you want to download BOJ meeting PDFs? (y/n): ").strip().lower()
            if download_choice == 'y':
                download_boj_pdfs()  # Use default years
    
    # Step 3: Analyze meeting texts and calculate similarities if Japanese text processing is available
    print("\nSTEP 3: Analyzing meeting texts")
    if JAPANESE_TEXT_AVAILABLE:
        text_df = analyze_meeting_texts(max_pages=5)  # Adjust max_pages as needed
    else:
        print("Skipping text analysis (Japanese text processing modules not available)")
        
        # Check if existing analysis is available
        text_analysis_path = os.path.join(OUTPUT_DIR, 'boj_text_analysis.csv')
        if os.path.exists(text_analysis_path):
            print(f"Loading existing text analysis from {text_analysis_path}")
            text_df = pd.read_csv(text_analysis_path)
            text_df['date'] = pd.to_datetime(text_df['date'])
        else:
            print("No existing text analysis found")
            text_df = pd.DataFrame()
    
    # Step 4: Fetch market data
    print("\nSTEP 4: Fetching market data")
    market_data = fetch_market_data()
    
    # Step 5: Calculate market statistics between meetings
    print("\nSTEP 5: Calculating inter-meeting market statistics")
    market_stats = calculate_inter_meeting_stats(market_data, meeting_dates)
    
    # Step 6: Combine text analysis and market statistics if both are available
    print("\nSTEP 6: Combining text and market analyses")
    if not text_df.empty and not market_stats.empty:
        combined_df = combine_text_and_market_analysis(text_df, market_stats)
        
        # Step 7: Analyze relationship between meeting similarity and market outcomes
        if not combined_df.empty:
            print("\nSTEP 7: Analyzing relationship between meeting similarity and market outcomes")
            correlation_df, group_stats = analyze_similarity_market_relationship(combined_df)
            
            # Display summary results
            if correlation_df is not None:
                print("\nCorrelation between Meeting Similarity and Market Outcomes:")
                print(correlation_df)
            
            if group_stats is not None and not group_stats.empty:
                print("\nMarket Statistics by Similarity Group:")
                print(group_stats)
        else:
            print("Skipping relationship analysis (no combined data available)")
    else:
        print("Skipping combined analysis (missing either text or market data)")
    
    print(f"\nAnalysis complete! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
