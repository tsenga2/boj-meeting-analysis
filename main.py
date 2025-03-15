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
        #url = f"{base_url}/mopo/mpmsche_minu/record_{year}/index.htm"
        url = f"{base_url}/mopo/mpmsche_minu/minu_{year}"
        
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
            #pdf_links = soup.find_all('a', href=re.compile(r'(?:/mopo/mpmsche_minu/record_\d{4}/|)g(?:jrk|irk)\d{6}a\.pdf$'))
            pdf_links = soup.find_all('a', href=re.compile(r'/mopo/mpmsche_minu/minu_\d{4}/g\d+\.pdf$'))
            
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

def download_boj_press_conferences(start_year, end_year):
    """
    Download BOJ press conference PDFs for specified years.
    """
    base_url = "https://www.boj.or.jp"
    url = f"{base_url}/mopo/mpmsche_minu/past.htm"
    url_2024 = f"{base_url}/mopo/mpmsche_minu/index.htm"

    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        pdfs_downloaded = 0

        # Find all links that match the pattern for press conference PDFs
        # Press Conference
        #pdf_links = soup.find_all('a', href=re.compile(r'/about/press/kaiken_\d{4}/kk\d+[ab]\.pdf$'))
        # Minutes
        pdf_links_g = soup.find_all('a', href=re.compile(r'/mopo/mpmsche_minu/minu_\d{4}/g\d+\.pdf$'))
        # 2024-
        pdf_links_2024 = soup.find_all('a', href=re.compile(r'/mopo/mpmsche_minu/minu_\d{4}/g\d+\.pdf$'))

        #print(f"Found {len(pdf_links)} potential PDF links for press conferences")
        print(f"Found {len(pdf_links_g)} potential PDF links for minutes")


        #for link in pdf_links:
        #    pdf_url = f"{base_url}{link['href']}"
        #    year_match = re.search(r'kaiken_(\d{4})', pdf_url)

        #    if year_match:
        #        year = int(year_match.group(1))

                # Only download PDFs within the specified year range
        #        if start_year <= year <= end_year:
        #            filename = os.path.join(PDF_DIR, f"press_conference_{os.path.basename(pdf_url)}")
        #            if download_pdf(pdf_url, filename):
        #                pdfs_downloaded += 1
        
        for link in pdf_links_2024:
            pdf_url = f"{base_url}{link['href']}"
            year_match = re.search(r'kaiken_(\d{4})', pdf_url)

            if year_match:
                year = int(year_match.group(1))

                # Only download PDFs within the specified year range
                if start_year <= year <= end_year:
                    filename = os.path.join(PDF_DIR, f"minutes_{os.path.basename(pdf_url)}")
                    if download_pdf(pdf_url, filename):
                        pdfs_downloaded += 1

        for link in pdf_links_g:
            pdf_url = f"{base_url}{link['href']}"
            year_match = re.search(r'minu_(\d{4})', pdf_url)

            if year_match:
                year = int(year_match.group(1))

                # Only download PDFs within the specified year range
                if start_year <= year <= end_year:
                    filename = os.path.join(PDF_DIR, f"minutes_{os.path.basename(pdf_url)}")
                    if download_pdf(pdf_url, filename):
                        pdfs_downloaded += 1

        print(f"\nTotal PDFs successfully downloaded: {pdfs_downloaded}")

    except requests.RequestException as e:
        print(f"Error fetching the webpage: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def download_pdf(pdf_url, filename):
    """
    Helper function to download a PDF from a URL.
    """
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
        return True

    except Exception as e:
        print(f"Error downloading {pdf_url}: {e}")
        return False


#############################################################
# Part 2: Text Processing and Similarity Analysis
#############################################################
def extract_boj_minutes_text(pdf_dir=PDF_DIR, output_dir=OUTPUT_DIR):
    """
    Specialized function for extracting text from BOJ minutes PDFs with
    comprehensive approach to handle encoding issues and various PDF formats.
    
    Parameters:
    -----------
    pdf_dir : str
        Directory containing the PDF files
    output_dir : str
        Directory to save output files
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing text analysis results
    """
    # Try to install required packages
    try:
        import subprocess
        import sys
        
        # Install required packages if not already available
        required_packages = ['pdfminer.six', 'pymupdf', 'pikepdf', 'PyPDF2']
        for package in required_packages:
            try:
                __import__(package.split('.')[0])
                print(f"{package} is already installed.")
            except ImportError:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Successfully installed {package}")
    except Exception as e:
        print(f"Warning: Package installation failed: {e}")
        print("Continuing with available packages...")
    
    # Get list of PDFs
    pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.endswith('.pdf')])
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return pd.DataFrame()
        
    print(f"Found {len(pdf_files)} PDF files")
    
    # Initialize results list
    data = []
    
    # Define Japanese keyword patterns to look for
    japanese_keywords = {
        'inflation': ['インフレ', '物価上昇', '物価', '上昇率', 'CPI', '消費者物価'],
        'deflation': ['デフレ', '物価下落', '物価下押し'],
        'fx': ['為替', '円相場', 'ドル円', 'ユーロ円', '円レート', '円安', '円高'],
        'interest_rate': ['金利', '利率', '利回り', '政策金利', '長期金利', '短期金利'],
        'economy': ['景気', '経済情勢', '経済状況', '景気回復', '景気拡大', '景気後退', '経済活動', 'GDP']
    }
    
    # Process each PDF with progress bar
    for i, filename in enumerate(pdf_files):
        print(f"\nProcessing file {i+1}/{len(pdf_files)}: {filename}")
        pdf_path = os.path.join(pdf_dir, filename)
        
        # Extract date from filename
        date_match = None
        
        # Try multiple date formats in the filename
        if re.search(r'(gj|gi)rk(\d{6})a', filename):
            date_str = re.search(r'(gj|gi)rk(\d{6})a', filename).group(2)
            date_format = "%y%m%d"
            date_match = True
        elif re.search(r'\d{4}_g\d{6}', filename):
            date_str = re.search(r'_g(\d{6})', filename).group(1)
            date_format = "%y%m%d"
            date_match = True
        elif re.search(r'minutes_g\d{6}', filename):
            date_str = re.search(r'_g(\d{6})', filename).group(1)
            date_format = "%y%m%d"
            date_match = True
        
        if not date_match:
            print(f"No date match found in filename: {filename}")
            # Try to use file modification time as a fallback
            try:
                file_mtime = os.path.getmtime(pdf_path)
                file_date = datetime.fromtimestamp(file_mtime).date()
                print(f"Using file modification date: {file_date}")
                date = file_date
            except:
                print(f"Skipping file due to date extraction failure: {filename}")
                continue
        else:
            try:
                date = datetime.strptime(date_str, date_format).date()
            except Exception as e:
                print(f"Date parsing error: {e}")
                continue
        
        # Initialize text variable
        text = ""
        
        # METHOD 1: PyMuPDF (fitz) - Usually best for Japanese text
        try:
            import fitz
            print("Trying PyMuPDF extraction...")
            
            doc = fitz.open(pdf_path)
            pages_processed = len(doc)
            
            # Get text from each page
            all_text = []
            for page_num in range(pages_processed):
                page = doc[page_num]
                page_text = page.get_text("text")
                all_text.append(page_text)
                
            text = "\n".join(all_text)
            if text.strip():
                print(f"Successfully extracted {len(text)} characters with PyMuPDF")
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")
        
        # METHOD 2: pdfminer.six with Japanese encoding
        if not text.strip():
            try:
                from pdfminer.high_level import extract_text
                from pdfminer.layout import LAParams
                
                print("Trying pdfminer.six extraction with Japanese encoding...")
                
                # Try with different encodings
                for encoding in ['utf-8', 'euc-jp', 'shift_jis', 'iso2022_jp']:
                    try:
                        laparams = LAParams()
                        text = extract_text(pdf_path, laparams=laparams, codec=encoding)
                        if text.strip():
                            print(f"Successfully extracted {len(text)} characters with pdfminer using {encoding}")
                            break
                    except:
                        continue
            except Exception as e:
                print(f"pdfminer extraction failed: {e}")
        
        # METHOD 3: PyPDF2
        if not text.strip():
            try:
                import PyPDF2
                print("Trying PyPDF2 extraction...")
                
                reader = PyPDF2.PdfReader(pdf_path)
                pages_processed = len(reader.pages)
                
                # Get text from each page
                all_text = []
                for page_num in range(pages_processed):
                    try:
                        page_text = reader.pages[page_num].extract_text()
                        all_text.append(page_text)
                    except:
                        continue
                        
                text = "\n".join(all_text)
                if text.strip():
                    print(f"Successfully extracted {len(text)} characters with PyPDF2")
            except Exception as e:
                print(f"PyPDF2 extraction failed: {e}")
        
        # METHOD 4: pikepdf (can sometimes access text when others can't)
        if not text.strip():
            try:
                import pikepdf
                import io
                from pdfminer.high_level import extract_text_to_fp
                
                print("Trying pikepdf extraction...")
                
                pdf = pikepdf.open(pdf_path)
                output_string = io.StringIO()
                
                # Extract text using pdfminer through pikepdf
                with io.BytesIO() as in_file:
                    pdf.save(in_file)
                    in_file.seek(0)
                    extract_text_to_fp(in_file, output_string)
                
                text = output_string.getvalue()
                if text.strip():
                    print(f"Successfully extracted {len(text)} characters with pikepdf")
            except Exception as e:
                print(f"pikepdf extraction failed: {e}")
        
        # METHOD 5: OCR as last resort
        if not text.strip() and JAPANESE_TEXT_AVAILABLE:
            try:
                print("Trying OCR extraction...")
                text = process_pdf_with_ocr(pdf_path, max_pages=5)
                if text.strip():
                    print(f"Successfully extracted text with OCR")
            except Exception as e:
                print(f"OCR extraction failed: {e}")
        
        # If all extraction methods failed
        if not text.strip():
            print(f"Warning: All extraction methods failed for {filename}")
            continue
        
        # Count keyword occurrences
        metrics = {}
        for category, keywords in japanese_keywords.items():
            count = 0
            for keyword in keywords:
                count += text.count(keyword)
            metrics[f"{category}_mentions"] = count
        
        # Save the full text for examination
        text_filename = os.path.splitext(filename)[0] + ".txt"
        text_path = os.path.join(output_dir, text_filename)
        try:
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Saved extracted text to {text_path}")
        except Exception as e:
            print(f"Error saving text file: {e}")
        
        # Determine page count
        try:
            import fitz
            doc = fitz.open(pdf_path)
            page_count = len(doc)
        except:
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(pdf_path)
                page_count = len(reader.pages)
            except:
                page_count = 0  # Unknown
        
        # Add data to results
        data.append({
            'date': date,
            'filename': filename,
            'text': text,
            'pages_processed': page_count,
            **metrics
        })
    
    # Create DataFrame
    if not data:
        print("No data was collected from PDFs")
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    df = df.sort_values('date')
    
    # Calculate text similarity if possible
    if len(df) > 1:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectorizer = TfidfVectorizer(max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(df['text'])
            similarity_matrix = cosine_similarity(tfidf_matrix)
            df['similarity_with_previous'] = [0] + [similarity_matrix[i, i-1] for i in range(1, len(df))]
            print("Successfully calculated text similarities")
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            df['similarity_with_previous'] = 0
    
    # Save results to CSV
    output_path = os.path.join(output_dir, 'boj_text_analysis_improved.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved analysis results to {output_path}")
    
    # Create visualizations
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set up the visualization style
        sns.set(style="whitegrid")
        
        # Define BOJ governors and their terms
        governors = [
            {"name": "Masaru Hayami", "start": "1998-03-20", "end": "2003-03-19", "color": "lightblue"},
            {"name": "Toshihiko Fukui", "start": "2003-03-20", "end": "2008-03-19", "color": "lightgreen"},
            {"name": "Masaaki Shirakawa", "start": "2008-04-09", "end": "2013-03-19", "color": "lightyellow"},
            {"name": "Haruhiko Kuroda", "start": "2013-03-20", "end": "2023-04-08", "color": "lightpink"},
            {"name": "Kazuo Ueda", "start": "2023-04-09", "end": "2028-04-08", "color": "lavender"}  # Current governor
        ]
        
        # Create trend visualization
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Similarity
        if 'similarity_with_previous' in df.columns:
            plt.subplot(2, 1, 1)
            
            # Convert df['date'] to pandas datetime objects for consistent comparison
            df['date'] = pd.to_datetime(df['date'])
            
            # Add background colors for governor terms
            for gov in governors:
                start_date = pd.to_datetime(gov['start'])
                end_date = pd.to_datetime(gov['end'])
                
                # Only show terms that overlap with our data period
                if start_date <= df['date'].max() and end_date >= df['date'].min():
                    plt.axvspan(
                        max(start_date, df['date'].min()), 
                        min(end_date, df['date'].max()), 
                        alpha=0.3, 
                        color=gov['color'], 
                        label=gov['name']
                    )
            
            plt.plot(df['date'], df['similarity_with_previous'], marker='o', linestyle='-', color='blue')
            plt.title("Text Similarity Between Consecutive BOJ Minutes", fontsize=14)
            plt.ylabel("Cosine Similarity")
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        # Plot 2: Keyword mentions
        plt.subplot(2, 1, 2)
        mentions_cols = [col for col in df.columns if col.endswith('_mentions')]
        
        # Add background colors for governor terms here too
        for gov in governors:
            start_date = pd.to_datetime(gov['start'])
            end_date = pd.to_datetime(gov['end'])
            
            # Only show terms that overlap with our data period
            if start_date <= df['date'].max() and end_date >= df['date'].min():
                plt.axvspan(
                    max(start_date, df['date'].min()), 
                    min(end_date, df['date'].max()), 
                    alpha=0.3, 
                    color=gov['color']
                )
        
        for col in mentions_cols:
            label = col.replace('_mentions', '').capitalize()
            plt.plot(df['date'], df[col], marker='o', linestyle='-', label=label)
        
        plt.title("Keyword Mentions in BOJ Minutes", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Number of Mentions")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "boj_minutes_analysis.png"))
        print(f"Saved visualization to {os.path.join(output_dir, 'boj_minutes_analysis.png')}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    return df


def find_key_meeting_changes(df, threshold=0.5):
    """
    Identify meetings with significant changes in content from previous meetings.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with BOJ meeting analysis
    threshold : float
        Similarity threshold below which meetings are considered significantly different
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with key meeting changes
    """
    if 'similarity_with_previous' not in df.columns:
        print("Similarity data not available in the DataFrame")
        return pd.DataFrame()
    
    # Make a copy of the dataframe with reset index to avoid index issues
    df_reset = df.reset_index(drop=True)
    
    # Find meetings with similarity below threshold (skip the first meeting which has similarity=0)
    key_changes = df_reset[(df_reset['similarity_with_previous'] < threshold) & 
                           (df_reset.index > 0)].copy()
    
    if key_changes.empty:
        print(f"No meetings found with similarity below {threshold}")
        return key_changes
    
    # Add previous meeting date for reference
    key_changes['previous_meeting_date'] = pd.NaT  # Initialize with NaT
    mention_cols = [col for col in df_reset.columns if col.endswith('_mentions')]
    
    # Calculate changes for each row in key_changes
    for idx, row in key_changes.iterrows():
        # Find the previous meeting (should exist since we filtered for index > 0)
        prev_idx = idx - 1
        prev_row = df_reset.iloc[prev_idx]
        
        # Set the previous meeting date
        key_changes.at[idx, 'previous_meeting_date'] = prev_row['date']
        
        # Calculate keyword changes
        for col in mention_cols:
            change_col = f'{col}_change'
            if change_col not in key_changes.columns:
                key_changes[change_col] = 0
            key_changes.at[idx, change_col] = row[col] - prev_row[col]
    
    print(f"Found {len(key_changes)} meetings with significant content changes")
    
    # Sort by date
    key_changes = key_changes.sort_values('date')
    
    # Add formatted output for easier reading
    key_changes['summary'] = key_changes.apply(
        lambda row: f"Meeting on {row['date']} (vs {row['previous_meeting_date']}): " + 
                   ", ".join([f"{col.split('_')[0]} {'+' if row[f'{col}_change'] > 0 else ''}{row[f'{col}_change']}" 
                             for col in mention_cols if abs(row[f'{col}_change']) > 0]),
        axis=1
    )
    
    return key_changes

def analyze_keyword_trends(df, window_size=3):
    """
    Analyze trends in keyword mentions over time.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with BOJ meeting analysis
    window_size : int
        Size of the rolling window for trend analysis
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with trend analysis
    """
    if df.empty:
        print("Empty DataFrame provided")
        return pd.DataFrame()
    
    # Ensure the data is sorted by date
    df = df.sort_values('date').copy()
    
    # Get keyword columns
    keyword_cols = [col for col in df.columns if col.endswith('_mentions')]
    
    # Calculate moving averages
    for col in keyword_cols:
        df[f'{col}_ma'] = df[col].rolling(window=window_size, min_periods=1).mean()
    
    # Calculate trend direction (increasing, decreasing, or stable)
    for col in keyword_cols:
        # Calculate rate of change
        df[f'{col}_trend'] = df[f'{col}_ma'].diff()
        
        # Convert to categorical trend
        conditions = [
            df[f'{col}_trend'] > 0.5,  # Increasing
            df[f'{col}_trend'] < -0.5,  # Decreasing
        ]
        choices = ['Increasing', 'Decreasing']
        default = 'Stable'
        df[f'{col}_trend_dir'] = np.select(conditions, choices, default=default)
    
    # Create an overall summary
    df['period'] = pd.to_datetime(df['date']).dt.to_period('Q')
    
    # Create quarterly summary
    quarterly_summary = df.groupby('period')[keyword_cols].mean().reset_index()
    quarterly_summary['period_str'] = quarterly_summary['period'].astype(str)
    
    # Save visualizations
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        sns.set(style="whitegrid")
        
        # Create trend visualization
        plt.figure(figsize=(14, 10))
        
        # Plot each keyword's quarterly average
        for col in keyword_cols:
            keyword = col.split('_')[0].capitalize()
            plt.plot(quarterly_summary['period_str'], quarterly_summary[col], 
                     marker='o', linestyle='-', label=keyword)
        
        plt.title("Quarterly Average Keyword Mentions in BOJ Minutes", fontsize=14)
        plt.xlabel("Quarter")
        plt.ylabel("Average Mentions")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "boj_keyword_trends.png"))
        
    except Exception as e:
        print(f"Error creating trend visualization: {e}")
    
    return df

def analyze_meeting_texts_all_formats(pdf_dir=PDF_DIR, max_pages=5, use_ocr=True):
    """
    Process BOJ meeting PDFs and calculate text similarities, handling different file naming formats
    
    Parameters:
    -----------
    pdf_dir : str
        Directory containing the PDF files
    max_pages : int
        Maximum number of pages to process per PDF
    use_ocr : bool
        Whether to use OCR for text extraction (set to False if PDFs are text-based)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing text analysis results
    """
    if use_ocr and not JAPANESE_TEXT_AVAILABLE:
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
    
    # Try installing necessary libraries if needed
    if not use_ocr:
        try:
            import pip
            pip.main(['install', 'PyPDF2'])
            print("Successfully installed PyPDF2")
        except:
            print("Failed to install PyPDF2. Will use alternative methods.")
            
        try:
            import pip
            pip.main(['install', 'pdfminer.six'])
            print("Successfully installed pdfminer.six")
        except:
            print("Failed to install pdfminer.six. Will use alternative methods.")
    
    # Process each PDF with progress bar
    for i, filename in enumerate(pdf_files):
        print(f"\nProcessing file {i+1}/{len(pdf_files)}: {filename}")
        pdf_path = os.path.join(pdf_dir, filename)

        # Extract date from filename using multiple patterns
        # Pattern 1: Original format like '2009_g091218.pdf' or with (gj|gi)rk(\d{6})a
        # Pattern 2: New format like 'minutes_g100126.pdf'
        date_match = None
        
        # Try the original pattern first
        original_pattern = re.search(r'(gj|gi)rk(\d{6})a', filename)
        if original_pattern:
            date_str = original_pattern.group(2)
            date_format = "%y%m%d"
            date_match = True
        
        # Try the format like '2009_g091218.pdf'
        elif re.search(r'\d{4}_g\d{6}', filename):
            date_str = re.search(r'_g(\d{6})', filename).group(1)
            date_format = "%y%m%d"
            date_match = True
            
        # Try the new format like 'minutes_g100126.pdf'
        elif re.search(r'minutes_g\d{6}', filename):
            date_str = re.search(r'_g(\d{6})', filename).group(1)
            date_format = "%y%m%d"
            date_match = True
            
        if not date_match:
            print(f"No date match found in filename: {filename}")
            continue
            
        try:
            date = datetime.strptime(date_str, date_format).date()

            # Try multiple text extraction methods
            text = ""
            
            # Method 1: Try PyPDF2
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(pdf_path)
                pages_to_process = min(max_pages, len(reader.pages))
                print(f"Using PyPDF2 to extract {pages_to_process} pages")
                for page_num in range(pages_to_process):
                    page_text = reader.pages[page_num].extract_text()
                    if page_text:
                        text += page_text + "\n"
                if text.strip():
                    print(f"Successfully extracted text with PyPDF2: {len(text)} characters")
            except Exception as e:
                print(f"PyPDF2 extraction failed: {e}")
                
            # Method 2: Try pdfminer if PyPDF2 failed
            if not text.strip():
                try:
                    from pdfminer.high_level import extract_text as pdfminer_extract
                    print("Using pdfminer.six for extraction")
                    text = pdfminer_extract(pdf_path, page_numbers=list(range(max_pages)))
                    if text.strip():
                        print(f"Successfully extracted text with pdfminer: {len(text)} characters")
                except Exception as e:
                    print(f"pdfminer extraction failed: {e}")
            
            # Method 3: Enhanced OCR with preprocessing if other methods failed
            if not text.strip() and use_ocr:
                print(f"Using enhanced OCR with preprocessing")
                try:
                    from PIL import Image, ImageEnhance
                    
                    images = convert_from_path(pdf_path, dpi=300)  # Higher DPI for better quality
                    pages_to_process = min(max_pages, len(images))
                    
                    for i, image in enumerate(images[:pages_to_process]):
                        print(f"OCR processing page {i+1}/{pages_to_process} with enhancements")
                        
                        # Preprocessing to improve OCR results
                        # Convert to grayscale
                        image = image.convert('L')
                        
                        # Enhance contrast
                        enhancer = ImageEnhance.Contrast(image)
                        image = enhancer.enhance(2.0)
                        
                        # Enhance sharpness
                        enhancer = ImageEnhance.Sharpness(image)
                        image = enhancer.enhance(2.0)
                        
                        # Save debug image if needed
                        # image.save(f"debug_page_{i}.png")
                        
                        # OCR with additional configuration
                        page_text = pytesseract.image_to_string(
                            image, 
                            lang='jpn', 
                            config='--psm 1 --oem 3'  # Page segmentation mode 1 (auto) and LSTM OCR engine
                        )
                        
                        text += page_text + "\n"
                    
                    if text.strip():
                        print(f"Successfully extracted text with enhanced OCR: {len(text)} characters")
                    else:
                        print("Enhanced OCR failed to extract text")
                        
                except Exception as e:
                    print(f"Enhanced OCR failed: {e}")

            # If still no text, try basic OCR as last resort
            if not text.strip() and use_ocr:
                print("Trying basic OCR as last resort")
                try:
                    text = process_pdf_with_ocr(pdf_path, max_pages=max_pages)
                    if text.strip():
                        print(f"Successfully extracted text with basic OCR: {len(text)} characters")
                except Exception as e:
                    print(f"Basic OCR failed: {e}")

            if not text.strip():
                print(f"Warning: All extraction methods failed for {filename}")
                # Optionally skip this file and continue with the next one
                # continue
                
                # Or use a placeholder for analysis
                text = f"NO_TEXT_EXTRACTED_{filename}"

            # Process text with MeCab if available
            if JAPANESE_TEXT_AVAILABLE:
                try:
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
                    print(f"Successfully processed with MeCab: {len(processed_text)} characters")
                except Exception as e:
                    print(f"MeCab processing failed: {e}")
                    processed_text = text
            else:
                # Simple tokenization for non-Japanese environment
                processed_text = text
                
            # Extract metrics (keep the same as original function)
            metrics = {
                'inflation_mentions': len(re.findall('インフレ|物価上昇', text)),
                'deflation_mentions': len(re.findall('デフレ|物価下落', text)),
                'fx_mentions': len(re.findall('為替|円相場|ドル円|ユーロ円', text)),
                'interest_rate_mentions': len(re.findall('金利|利率|利回り', text)),
                'economy_mentions': len(re.findall('景気|経済情勢|経済状況', text))
            }

            # Get page count
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(pdf_path)
                page_count = min(max_pages, len(reader.pages))
            except:
                try:
                    page_count = min(max_pages, len(convert_from_path(pdf_path)))
                except:
                    page_count = max_pages

            data.append({
                'date': date,
                'filename': filename,
                'text': processed_text,
                'pages_processed': page_count,
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
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(df['text'])
            similarity_matrix = cosine_similarity(tfidf_matrix)
            df['similarity_with_previous'] = [0] + [similarity_matrix[i, i-1] for i in range(1, len(df))]
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            df['similarity_with_previous'] = 0
    
    # Visualize similarity and key terms (keep the same as original function)
    try:
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
        plt.savefig(os.path.join(OUTPUT_DIR, "text_similarity_analysis_all_formats.png"))
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    # Save results
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, 'boj_text_analysis_all_formats.csv'), index=False)
    except Exception as e:
        print(f"Error saving CSV: {e}")
    
    return df

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
    meeting_dates = get_boj_meeting_dates(output_dir=OUTPUT_DIR)
    if not meeting_dates:
        print("Error: Failed to retrieve BOJ meeting dates. Exiting.")
        return
    
    # Step 2: Download BOJ meeting PDFs
    print("\nSTEP 2: Downloading BOJ meeting PDFs")
    # Use specific year range as in the notebook (adjust as needed)
    download_boj_pdfs(range(2006, 2010))
    
    # Step 3: Download BOJ press conferences
    print("\nSTEP 3: Downloading BOJ press conferences")
    download_boj_press_conferences(2010, 2024)
    
    # Step 4: Analyze meeting texts using extract_boj_minutes_text
    print("\nSTEP 4: Analyzing meeting texts")
    text_df = extract_boj_minutes_text(pdf_dir=PDF_DIR, output_dir=OUTPUT_DIR)
    
    # Step 5: Fetch market data
    print("\nSTEP 5: Fetching market data")
    market_data = fetch_market_data()
    
    # Step 6: Calculate market statistics between meetings
    print("\nSTEP 6: Calculating inter-meeting market statistics")
    market_stats = calculate_inter_meeting_stats(market_data, meeting_dates)
    
    # Step 7: Combine text analysis and market statistics if both are available
    print("\nSTEP 7: Combining text and market analyses")
    if not text_df.empty and not market_stats.empty:
        combined_df = combine_text_and_market_analysis(text_df, market_stats)
        
        # Step 8: Analyze relationship between meeting similarity and market outcomes
        if not combined_df.empty:
            print("\nSTEP 8: Analyzing relationship between meeting similarity and market outcomes")
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
