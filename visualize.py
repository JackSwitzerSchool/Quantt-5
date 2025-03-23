import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import mplfinance as mpf

###############################################
# 1) Load CSV (works for either 15m or 1h)
###############################################

def load_csv(csv_path):
    """
    Loads a CSV with columns: Datetime,Close,High,Low,Open,Volume.
    Renames 'Datetime'->'date', sets as a datetime index, and sorts by date.
    Works regardless of whether the data is in 15m or 1h resolution.
    """
    df = pd.read_csv(csv_path)
    
    # Rename if needed
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "date"}, inplace=True)
    elif "Date" in df.columns:
        df.rename(columns={"Date": "date"}, inplace=True)
    
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["Close", "High", "Low", "Open"], inplace=True)
    
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    return df

###############################################
# 2) Detect Simple 2-Candle Gaps
###############################################

def detect_simple_gaps(df):
    """
    Properly detects Fair Value Gaps (FVG) between non-sequential candles:
      - Bullish FVG: When High(i-2) < Low(i) - gap between candles i-2 and i
      - Bearish FVG: When Low(i-2) > High(i) - gap between candles i-2 and i
    """
    df = df.copy()
    
    # We need at least 3 candles to detect an FVG
    if len(df) < 3:
        print("Warning: Not enough data to detect FVGs. Need at least 3 candles.")
        return df
    
    # Condition for bullish FVG: High(i-2) < Low(i)
    df["bullish_fvg"] = False
    df.loc[df.index[2:], "bullish_fvg"] = df["High"].shift(2)[2:] < df["Low"][2:]
    
    # Condition for bearish FVG: Low(i-2) > High(i)
    df["bearish_fvg"] = False
    df.loc[df.index[2:], "bearish_fvg"] = df["Low"].shift(2)[2:] > df["High"][2:]
    
    # Mark the top/bottom using the correct candles (i and i-2)
    df["bullish_fvg_top"] = np.where(df["bullish_fvg"], df["Low"], np.nan)
    df["bullish_fvg_bottom"] = np.where(df["bullish_fvg"], df["High"].shift(2), np.nan)
    df["bearish_fvg_top"] = np.where(df["bearish_fvg"], df["Low"].shift(2), np.nan)
    df["bearish_fvg_bottom"] = np.where(df["bearish_fvg"], df["High"], np.nan)
    
    # Fill status (optional)
    df["bullish_fvg_filled"] = False
    df["bearish_fvg_filled"] = False
    
    return df

###############################################
# 3) Optionally Extend the Gap
###############################################

def extend_simple_gaps(df, bars_to_extend=8):
    """
    Extends each FVG forward for 'bars_to_extend' bars.
    If a future candle 'fills' the gap, the corresponding bars are marked as filled.
    """
    df = df.copy()
    df.sort_index(inplace=True)
    
    # Track which rows have been processed to avoid redundant processing
    processed_bullish = set()
    processed_bearish = set()
    
    for idx, row in df.iterrows():
        if pd.isna(row.get("bullish_fvg_top")) and pd.isna(row.get("bearish_fvg_top")):
            continue  # Skip rows without FVG data
        
        row_i = df.index.get_loc(idx)
        
        # Process bullish FVG
        if row["bullish_fvg"] and idx not in processed_bullish:
            processed_bullish.add(idx)
            top = row["bullish_fvg_top"]      # Low of candle i
            bottom = row["bullish_fvg_bottom"]  # High of candle i-2
            
            # Ensure the gap has positive height
            if top <= bottom:
                continue
                
            for j in range(1, min(bars_to_extend+1, len(df)-row_i)):
                f_idx = df.index[row_i + j]
                # Extend gap boundaries forward if not already set
                if pd.isna(df.at[f_idx, "bullish_fvg_top"]):
                    df.at[f_idx, "bullish_fvg_top"] = top
                if pd.isna(df.at[f_idx, "bullish_fvg_bottom"]):
                    df.at[f_idx, "bullish_fvg_bottom"] = bottom
                # Mark as filled if future candle's Low intrudes into the gap
                if df.at[f_idx, "Low"] <= top and df.at[f_idx, "Low"] >= bottom:
                    for k in range(row_i, row_i + j + 1):
                        try:
                            df.at[df.index[k], "bullish_fvg_filled"] = True
                        except:
                            pass  # Skip if we can't update this row
        
        # Process bearish FVG
        if row["bearish_fvg"] and idx not in processed_bearish:
            processed_bearish.add(idx)
            top = row["bearish_fvg_top"]      # Low of candle i-2
            bottom = row["bearish_fvg_bottom"]  # High of candle i
            
            # Ensure the gap has positive height
            if top <= bottom:
                continue
                
            for j in range(1, min(bars_to_extend+1, len(df)-row_i)):
                f_idx = df.index[row_i + j]
                if pd.isna(df.at[f_idx, "bearish_fvg_top"]):
                    df.at[f_idx, "bearish_fvg_top"] = top
                if pd.isna(df.at[f_idx, "bearish_fvg_bottom"]):
                    df.at[f_idx, "bearish_fvg_bottom"] = bottom
                # Mark as filled if future candle's High intrudes into the gap
                if df.at[f_idx, "High"] >= bottom and df.at[f_idx, "High"] <= top:
                    for k in range(row_i, row_i + j + 1):
                        try:
                            df.at[df.index[k], "bearish_fvg_filled"] = True
                        except:
                            pass  # Skip if we can't update this row
                        
    return df

###############################################
# 4) Plot Gaps (NO returnfig, just returnfig)
###############################################

def plot_gaps(df, title="Fair Value Gaps", rectangle_hours=2):
    """
    Plots the OHLC data with bullish/bearish gap rectangles overlaid.
    
    Parameters:
      df : DataFrame with OHLC data and gap flags.
      title : Title of the chart.
      rectangle_hours : The width (in hours) of the gap rectangle.
    """
    if df.empty:
        print(f"No data to plot for {title}")
        return
    
    # Prepare data for mplfinance
    ohlc_data = df[["Open", "High", "Low", "Close"]].copy()
    
    # Ensure no NaN values in OHLC data
    ohlc_data = ohlc_data.dropna()
    
    # Define custom style with distinct candle colors
    mc = mpf.make_marketcolors(
        up='#228B22',              # Forest green for up candles
        down='#8B0000',            # Dark red for down candles
        edge='inherit',            # Use same color for edge as the candle
        wick={'up':'#228B22', 'down':'#8B0000'},  # Match wick color to candle color
        volume='#6060a8'
    )
    
    custom_style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle=':',
        y_on_right=False,
        gridcolor='#e6e6e6',
        facecolor='white',
        figcolor='white',
        gridaxis='both',
        rc={'axes.edgecolor':'#c8c8c8'}
    )
    
    # Plot the candlestick chart
    fig, axes = mpf.plot(
        ohlc_data,
        type="candle",
        style=custom_style,
        returnfig=True,
        figsize=(14, 7),
        linecolor='black',
        tight_layout=True,
        show_nontrading=True
    )
    ax = axes[0]  # Main Axes is the first element in the axes list
    
    # Calculate appropriate width based on timeframe
    if len(df) > 1:
        date_diffs = [(df.index[i+1] - df.index[i]).total_seconds() for i in range(len(df)-1)]
        avg_seconds = sum(date_diffs) / len(date_diffs)
        # Convert average seconds to days (mplfinance x-axis unit)
        time_unit_in_days = avg_seconds / 86400
    else:
        time_unit_in_days = 1.0 / 24.0  # Default: 1 hour in days
    
    # Calculate rectangle width in chart units (days)
    rectangle_width = rectangle_hours / 24.0
    
    # Track plotted gaps to avoid duplicates
    bullish_fvg_plotted = set()
    bearish_fvg_plotted = set()
    
    # Use original dataframe (with potential NaN values) for FVG detection
    for idx, row in df.iterrows():
        if idx not in ohlc_data.index:
            continue  # Skip this row if it's not in the cleaned data
            
        x = mdates.date2num(idx)
            
        # Process bullish FVGs
        if row.get("bullish_fvg", False) and idx not in bullish_fvg_plotted:
            # Start rectangle from the i-2 candle position
            x_start = x - 2 * time_unit_in_days
            
            # Get the correct gap boundaries
            top = row["bullish_fvg_top"]      # Low of candle i
            bottom = row["bullish_fvg_bottom"]  # High of candle i-2
            
            # Ensure the gap has positive height
            if pd.isna(top) or pd.isna(bottom) or top <= bottom:
                continue
                
            filled = row.get("bullish_fvg_filled", False)
            color = '#00FFFF'  # Cyan for bullish gap
            alpha = 0.4
            hatch = "///" if filled else None
            
            rect = patches.Rectangle(
                (x_start, bottom),  # Start from candle i-2
                rectangle_width,    # Use specified width 
                top - bottom,       # Height = gap between High(i-2) and Low(i)
                linewidth=1.5,
                edgecolor='#008B8B',
                facecolor=color,
                alpha=alpha,
                hatch=hatch
            )
            ax.add_patch(rect)
            bullish_fvg_plotted.add(idx)
        
        # Process bearish FVGs
        if row.get("bearish_fvg", False) and idx not in bearish_fvg_plotted:
            # Start rectangle from the i-2 candle position
            x_start = x - 2 * time_unit_in_days
            
            # Get the correct gap boundaries
            top = row["bearish_fvg_top"]      # Low of candle i-2
            bottom = row["bearish_fvg_bottom"]  # High of candle i
            
            # Ensure the gap has positive height
            if pd.isna(top) or pd.isna(bottom) or top <= bottom:
                continue
                
            filled = row.get("bearish_fvg_filled", False)
            color = '#FF00FF'  # Magenta for bearish gap
            alpha = 0.4
            hatch = "///" if filled else None
            
            rect = patches.Rectangle(
                (x_start, bottom),  # Start from candle i-2
                rectangle_width,    # Use specified width
                top - bottom,       # Height = gap between Low(i-2) and High(i)
                linewidth=1.5,
                edgecolor='#8B008B',
                facecolor=color,
                alpha=alpha,
                hatch=hatch
            )
            ax.add_patch(rect)
            bearish_fvg_plotted.add(idx)
    
    # Set axis limits to show all data
    start_date = mdates.date2num(ohlc_data.index.min())
    end_date = mdates.date2num(ohlc_data.index.max())
    ax.set_xlim(start_date - time_unit_in_days, end_date + time_unit_in_days)
    
    # Count gap occurrences and display in the console
    total_bullish = df["bullish_fvg"].sum()
    total_bearish = df["bearish_fvg"].sum()
    print(f"{title}: Found {total_bullish} bullish FVG(s) and {total_bearish} bearish FVG(s).")
    
    # Build a legend with examples for gap colors and candles
    bull_patch = patches.Patch(color='#00FFFF', alpha=0.4, label="Bullish FVG (Unfilled)")
    bear_patch = patches.Patch(color='#FF00FF', alpha=0.4, label="Bearish FVG (Unfilled)")
    bull_filled_patch = patches.Patch(color='#00FFFF', alpha=0.4, hatch="///", label="Bullish FVG (Filled)")
    bear_filled_patch = patches.Patch(color='#FF00FF', alpha=0.4, hatch="///", label="Bearish FVG (Filled)")
    
    up_candle = patches.Rectangle((0, 0), 1, 1, color='#228B22', label='Up Candle')
    down_candle = patches.Rectangle((0, 0), 1, 1, color='#8B0000', label='Down Candle')
    
    ax.legend(handles=[bull_patch, bear_patch, bull_filled_patch, bear_filled_patch, up_candle, down_candle], loc="upper left")
    
    ax.set_title(title)
    plt.show()

###############################################
# Main
###############################################

if __name__ == "__main__":
    # Process 15-minute data
    csv_file_path_15m = "NQ_F_March2025_15m.csv"
    df_15m = load_csv(csv_file_path_15m)
    print(f"Loaded {len(df_15m)} rows from {df_15m.index.min()} to {df_15m.index.max()} for 15m data.")

    # Handle potential missing data before processing
    df_15m = df_15m.fillna(method='ffill').fillna(method='bfill')
    
    df_15m_gap = detect_simple_gaps(df_15m)
    # Extend gaps for 2 hours (8 bars on 15m data)
    df_15m_gap = extend_simple_gaps(df_15m_gap, bars_to_extend=8)
    plot_gaps(df_15m_gap, title="NQ March 2025 - 15m Fair Value Gaps", rectangle_hours=2)
    
    # Process 1-hour data
    csv_file_path_1h = "NQ_F_March2025_1h.csv"
    df_1h = load_csv(csv_file_path_1h)
    print(f"Loaded {len(df_1h)} rows from {df_1h.index.min()} to {df_1h.index.max()} for 1h data.")

    # Handle potential missing data before processing
    df_1h = df_1h.fillna(method='ffill').fillna(method='bfill')
    
    df_1h_gap = detect_simple_gaps(df_1h)
    # Extend gaps for 2 hours (2 bars on 1h data)
    df_1h_gap = extend_simple_gaps(df_1h_gap, bars_to_extend=2)
    plot_gaps(df_1h_gap, title="NQ March 2025 - 1h Fair Value Gaps", rectangle_hours=2)
