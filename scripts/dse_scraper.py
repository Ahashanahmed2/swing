import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime

def get_trading_codes():
    """Get all trading codes from the main page"""
    url = "https://www.dsebd.org/latest_share_price_scroll_by_ltp.php"
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the table with trading codes
        table = soup.find('table', {'class': 'table table-bordered background-white shares-table fixedHeader'})
        trading_codes = []
        
        if table:
            rows = table.find_all('tr')[1:]  # Skip header row
            for row in rows:
                cols = row.find_all('td')
                if len(cols) > 1:
                    trading_code = cols[1].text.strip()
                    trading_codes.append(trading_code)
        
        return trading_codes
    except Exception as e:
        print(f"Error fetching trading codes: {e}")
        return []

def get_available_years(dividend_dict, eps_dict):
    """Get available years dynamically from the data"""
    all_years = set()
    
    # Add years from dividends
    for year_str in dividend_dict.keys():
        try:
            all_years.add(int(year_str))
        except ValueError:
            continue
    
    # Add years from EPS
    for year_str in eps_dict.keys():
        try:
            all_years.add(int(year_str))
        except ValueError:
            continue
    
    if not all_years:
        # If no years found, use current year and previous 5 years
        current_year = datetime.now().year
        all_years = set(range(current_year - 5, current_year + 1))
    
    return sorted(all_years, reverse=True)

def scrape_company_data(trading_code):
    """Scrape data for a specific company"""
    url = f"https://www.dsebd.org/displayCompany.php?name={trading_code}"
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract last trading price
        ltp = None
        ltp_th = soup.find('th', string='Last Trading Price')
        if ltp_th:
            ltp_td = ltp_th.find_next('td')
            if ltp_td:
                ltp = ltp_td.text.strip().replace(',', '')
        
        # Extract sector
        sector = None
        sector_th = soup.find('th', string='Sector')
        if sector_th:
            sector_td = sector_th.find_next('td')
            if sector_td:
                sector = sector_td.text.strip()
        
        # Extract cash dividends - dynamic years
        dividends = {}
        cash_div_th = soup.find('th', string=re.compile('Cash Dividend'))
        if cash_div_th:
            cash_div_td = cash_div_th.find_next('td')
            if cash_div_td:
                dividend_text = cash_div_td.text.strip()
                # Extract dividend percentages and years
                dividend_pattern = r'(\d+)%\s*(\d{4})'
                dividend_matches = re.findall(dividend_pattern, dividend_text)
                
                # Convert to dictionary
                for percentage, year in dividend_matches:
                    dividends[year] = percentage
        
        # Extract EPS data - dynamic years
        eps_data = {}
        # Look for Financial Performance section
        audited_header = soup.find('h2', string=re.compile('Financial Performance'))
        if not audited_header:
            # Try alternative header patterns
            audited_header = soup.find('h2', string=re.compile('Audited Financial Statements'))
        
        if audited_header:
            audited_table = audited_header.find_next('table')
            if audited_table:
                # Find all rows with year data
                for row in audited_table.find_all('tr'):
                    first_td = row.find('td')
                    if first_td and first_td.get('colspan') == '2':
                        year_match = re.search(r'\b(20\d{2})\b', first_td.text.strip())
                        if year_match:
                            year = year_match.group(1)
                            # Find EPS - look for numeric values in subsequent columns
                            tds = row.find_all('td')
                            for td in tds[1:]:  # Skip the year column
                                eps_text = td.text.strip()
                                # Look for numeric values (including decimals and negatives)
                                eps_match = re.search(r'(-?\d+\.?\d*)', eps_text)
                                if eps_match:
                                    try:
                                        eps_value = float(eps_match.group(1))
                                        eps_data[year] = eps_value
                                        break  # Use first numeric value found
                                    except ValueError:
                                        continue

        return {
            'Code': trading_code,
            'LTP': ltp,
            'Sector': sector,
            'Dividends': dividends,
            'EPS': eps_data
        }
    except Exception as e:
        print(f"Error scraping data for {trading_code}: {e}")
        return None

def has_valid_data(data):
    """Check if the company has valid dividend and EPS data for 5 consecutive years"""
    # Get available years dynamically
    available_years = get_available_years(data['Dividends'], data['EPS'])
    
    if len(available_years) < 5:
        return False
    
    # Check for 5 consecutive years with positive values in both dividends and EPS
    available_years.sort(reverse=True)
    
    for i in range(len(available_years) - 4):
        # Check if we have 5 consecutive years
        consecutive_years = available_years[i:i+5]
        if consecutive_years[0] - consecutive_years[4] == 4:  # Check if they're consecutive
            # Verify all years have positive dividends and EPS
            all_valid = True
            for year in consecutive_years:
                year_str = str(year)
                
                # Check dividend (must be positive)
                dividend = data['Dividends'].get(year_str)
                if not dividend or float(dividend) <= 0:
                    all_valid = False
                    break
                
                # Check EPS (must be positive)
                eps = data['EPS'].get(year_str)
                if eps is None or eps <= 0:
                    all_valid = False
                    break
            
            if all_valid:
                return True
    
    return False

def main():
    print("Starting DSE data scraping...")
    
    # Get all trading codes
    trading_codes = get_trading_codes()
    print(f"Found {len(trading_codes)} trading codes")
    
    if not trading_codes:
        print("No trading codes found. Please check the website structure.")
        return
    
    # For testing, limit to first 50 codes
    trading_codes = trading_codes[:50]
    
    all_data = []
    
    # Scrape data for each trading code
    for i, code in enumerate(trading_codes):
        print(f"Scraping data for {code} ({i+1}/{len(trading_codes)})")
        data = scrape_company_data(code)
        if data:
            # Check if the company has valid data
            if has_valid_data(data):
                all_data.append(data)
                print(f"Successfully scraped and validated data for {code}")
                # Print the years found for debugging
                div_years = sorted(data['Dividends'].keys(), reverse=True)
                eps_years = sorted(data['EPS'].keys(), reverse=True)
                print(f"  Dividend years: {div_years}")
                print(f"  EPS years: {eps_years}")
            else:
                print(f"Skipping {code} as it doesn't meet the criteria")
        else:
            print(f"Failed to scrape data for {code}")
        time.sleep(1)  # Be polite to the server
    
    # Convert to DataFrame and save as CSV
    if all_data:
        # Get all unique years from the collected data
        all_years = set()
        for item in all_data:
            all_years.update(item['Dividends'].keys())
            all_years.update(item['EPS'].keys())
        
        # Convert to sorted list (most recent first)
        sorted_years = sorted([int(year) for year in all_years], reverse=True)
        sorted_years_str = [str(year) for year in sorted_years]
        
        # Create a flattened structure for CSV
        flattened_data = []
        
        for item in all_data:
            flat_item = {
                'Code': item['Code'],
                'LTP': item['LTP'],
                'Sector': item['Sector']
            }
            
            # Add dividends for available years (using 'Div' instead of 'Dividend')
            for year in sorted_years_str:
                flat_item[f'Div {year}'] = item['Dividends'].get(year, '')
            
            # Add EPS for available years
            for year in sorted_years_str:
                flat_item[f'EPS {year}'] = item['EPS'].get(year, '')
            
            flattened_data.append(flat_item)
        
        df = pd.DataFrame(flattened_data)
        
        # Create column order: Code, LTP, Sector, then Div years, then EPS years
        columns = ['Code', 'LTP', 'Sector']
        for year in sorted_years_str:
            columns.append(f'Div {year}')
        for year in sorted_years_str:
            columns.append(f'EPS {year}')
        
        # Only keep columns that exist in the DataFrame
        existing_columns = [col for col in columns if col in df.columns]
        df = df[existing_columns]
        
        # Save to CSV
        df.to_csv('./../csv/dse_financial_data.csv', index=False, encoding='utf-8')
        print("Data saved to dse_financial_data.csv")
        print(f"Found {len(all_data)} companies that meet the criteria")
        
        # Show a preview of the data
        print("\nPreview of the data:")
        print(df.head())
        
        # Print column names to verify
        print("\nColumn names in CSV:")
        print(df.columns.tolist())
        
    else:
        print("No data was scraped that meets the criteria")

if __name__ == "__main__":
    main()