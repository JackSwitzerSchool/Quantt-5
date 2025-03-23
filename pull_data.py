import yfinance as yf
import pandas as pd
import os

# -----------------------------
# Configuration
# -----------------------------
ticker = "NQ=F"          # E-mini Nasdaq-100 Future
interval = "1h"         # 15-minute data
start_date = "2025-03-01"
end_date   = "2025-04-01"  # Up to (but not including) Apr 1 => all March data

# -----------------------------
# Download & Save
# -----------------------------
try:
    # Download 15-minute data for the entire March
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    if data.empty:
        print(f"No data returned for {ticker} from {start_date} to {end_date}.")
    else:
        # Determine output directory
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()
        
        output_dir = os.path.join(base_dir, "data")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save a single CSV with all March data
        csv_filename = f"{ticker.replace('=', '_')}_March2025_1h.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        data.to_csv(csv_path)
        print(f"Saved full March 2025 data to: {csv_path}")
except Exception as e:
    print(f"Error downloading data: {e}")
    print("Verify ticker symbol, date range, or interval.")
