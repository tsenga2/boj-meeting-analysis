# -*- coding: utf-8 -*-
"""BOJ_MPM_MarketStats.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bMx-GxByBqIPL0CXiYdW5oG9j6OpC25C
"""

# First cell - Install required packages
!pip install yfinance pandas_datareader requests beautifulsoup4 lxml

# Second cell - Import libraries
import pandas as pd
import yfinance as yf
# Instead of directly importing data as pdr, keep pandas_datareader as it is
import pandas_datareader
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup

# Use pandas_datareader.data.DataReader instead of yf.pdr

# Third cell - Updated BOJ meeting dates function
def get_boj_meeting_dates(start_year=1998, end_year=2024, output_dir="."):
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

def fetch_market_data(start_date='1998-01-01', end_date=None):
    """
    Fetch market data from Yahoo Finance and FRED
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print("Fetching Nikkei 225 data...")
    nikkei = yf.download('^N225', start=start_date, end=end_date)
    # Check if 'Adj Close' column exists, if not, try 'Close'
    # Ensure nikkei is a 1-dimensional Series by squeezing it
    nikkei = nikkei['Adj Close'].squeeze() if 'Adj Close' in nikkei.columns else nikkei['Close'].squeeze()

    print("Fetching USD/JPY and JGB data...")
    # Import pandas_datareader.data as pdr
    import pandas_datareader.data as pdr
    fred_data = pdr.get_data_fred(['DEXJPUS', 'IRLTLT01JPM156N'], start=start_date, end=end_date)

    # Combine all data
    market_data = pd.DataFrame({
        'nikkei': nikkei,
        'usdjpy': fred_data['DEXJPUS'],
        'jgb': fred_data['IRLTLT01JPM156N']
    })

    # Forward fill missing values (weekends and holidays)
    market_data = market_data.fillna(method='ffill')

    return market_data

def calculate_stats(data, meeting_dates, window=30):
    """
    Calculate both volatility and mean for each market variable around meeting dates
    """
    stats = []

    for meeting_date in meeting_dates:
        try:
            if isinstance(meeting_date, str):
                meeting_date = pd.to_datetime(meeting_date)

            end_date = meeting_date + timedelta(days=window)
            mask = (data.index >= meeting_date) & (data.index <= end_date)
            window_data = data[mask]

            if len(window_data) < 5:
                print(f"Warning: Insufficient data for meeting date {meeting_date}")
                continue

            stats_dict = {
                'meeting_date': meeting_date,
                # Mean values
                'nikkei_mean': window_data['nikkei'].mean(),
                'usdjpy_mean': window_data['usdjpy'].mean(),
                'jgb_mean': window_data['jgb'].mean(),
                # Standard deviations (volatilities)
                'nikkei_vol': window_data['nikkei'].std(),
                'usdjpy_vol': window_data['usdjpy'].std(),
                'jgb_vol': window_data['jgb'].std()
            }

            stats.append(stats_dict)

        except Exception as e:
            print(f"Error processing meeting date {meeting_date}: {e}")

    return pd.DataFrame(stats)

def plot_market_stats(stats):
    """
    Plot both means and volatilities for each market variable
    """
    # Use 'seaborn-v0_8-whitegrid' instead of 'seaborn'
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create 2x3 subplots (means and volatilities for each variable)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Market Statistics around BOJ Meetings (30-day windows)', fontsize=16)

    # Plot means
    axes[0,0].plot(stats['meeting_date'], stats['nikkei_mean'])
    axes[0,0].set_title('Nikkei 225 Mean')
    axes[0,0].set_ylabel('Index Value')
    axes[0,0].tick_params(axis='x', rotation=45)

    axes[0,1].plot(stats['meeting_date'], stats['usdjpy_mean'])
    axes[0,1].set_title('USD/JPY Mean')
    axes[0,1].set_ylabel('Exchange Rate')
    axes[0,1].tick_params(axis='x', rotation=45)

    axes[0,2].plot(stats['meeting_date'], stats['jgb_mean'])
    axes[0,2].set_title('10-year JGB Mean')
    axes[0,2].set_ylabel('Yield (%)')
    axes[0,2].tick_params(axis='x', rotation=45)

    # Plot volatilities
    axes[1,0].plot(stats['meeting_date'], stats['nikkei_vol'])
    axes[1,0].set_title('Nikkei 225 Volatility')
    axes[1,0].set_ylabel('Volatility (%)')
    axes[1,0].tick_params(axis='x', rotation=45)

    axes[1,1].plot(stats['meeting_date'], stats['usdjpy_vol'])
    axes[1,1].set_title('USD/JPY Volatility')
    axes[1,1].set_ylabel('Volatility (%)')
    axes[1,1].tick_params(axis='x', rotation=45)

    axes[1,2].plot(stats['meeting_date'], stats['jgb_vol'])
    axes[1,2].set_title('10-year JGB Volatility')
    axes[1,2].set_ylabel('Volatility (%)')
    axes[1,2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

# Modified analysis cell
# Get BOJ meeting dates
meeting_dates = get_boj_meeting_dates()
print(f"Found {len(meeting_dates)} BOJ meeting dates")

# Fetch market data
market_data = fetch_market_data()

# Calculate statistics
market_stats = calculate_stats(market_data, meeting_dates)

# Display summary statistics
print("\nSummary Statistics:")
print(market_stats.describe())

# Plot results
plot_market_stats(market_stats)

# Save to CSV
market_stats.to_csv('boj_market_stats.csv', index=False)

# Correlation analysis
# Now including both means and volatilities
correlation_vars = ['nikkei_mean', 'usdjpy_mean', 'jgb_mean',
                   'nikkei_vol', 'usdjpy_vol', 'jgb_vol']

correlation_matrix = market_stats[correlation_vars].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
            vmin=-1, vmax=1, center=0)
plt.title('Correlation between Market Means and Volatilities')
plt.tight_layout()
plt.show()