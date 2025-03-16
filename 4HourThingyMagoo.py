import pandas as pd
import yfinance as yf
import ta  # For RSI indicator
import matplotlib.pyplot as plt
from datetime import datetime, time

def fetch_stock_data(tickers, interval='15m', period='60d'):
    """Fetch historical stock data for given tickers."""
    data = {}
    for ticker in tickers:
        #print(f"Fetching data for {ticker}...")
        stock = yf.Ticker(ticker)
        df = stock.history(interval=interval, period=period)
        df.reset_index(inplace=True)
        df['Datetime'] = df['Datetime'].dt.tz_convert('America/New_York')
        data[ticker] = df
    return data

def apply_po3_strategy(df):
    """Apply PO3 trading strategy with improved conditions."""
    df['4h_high'] = df['High'].shift(16).rolling(window=16).max()
    df['4h_low'] = df['Low'].shift(16).rolling(window=16).min()
    df['4h_open'] = df['Open'].shift(16)
    df['4h_close'] = df['Close'].shift(16)
    df['Hour'] = df['Datetime'].dt.hour
    df['Trading_Hours'] = (df['Hour'] >= 10) & (df['Hour'] <= 14)
    
    # Adjust FVG conditions to allow more valid trades
    df['FVG_Long'] = (df['Low'] < df['4h_low'] * 1.002) & df['Trading_Hours']  # 0.2% buffer
    df['FVG_Short'] = (df['High'] > df['4h_high'] * 0.998) & df['Trading_Hours']  # 0.2% buffer
    
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    
    # Adjust RSI thresholds for better trade filtering
    df['Long_Entry'] = df['FVG_Long'] & (df['RSI'] <= 30)
    df['Short_Entry'] = df['FVG_Short'] & (df['RSI'] >= 70)
    
    # Adjust risk-reward settings
    df['Stop_Loss_Long'] = df['Close'] - 2 #* df['ATR']
    df['Take_Profit_Long'] = df['Close'] + 4 #* df['ATR']
    df['Stop_Loss_Short'] = df['Close'] + 2 #* df['ATR']
    df['Take_Profit_Short'] = df['Close'] - 4 #* df['ATR']
    
    return df

def backtest_strategy(df, initial_balance=10000, risk_per_trade=0.02):
    """Backtest the PO3 strategy and compare with buy & hold."""
    balance_po3 = initial_balance
    balance_hold = initial_balance
    position = 0
    buy_price = df['Close'].iloc[0]
    shares_hold = initial_balance / buy_price
    balances_po3 = []
    balances_hold = []
    in_trade = False
    trade_type = None
    entry_price, stop_loss, take_profit = 0, 0, 0
    
    for _, row in df.iterrows():
        trade_size = balance_po3 * risk_per_trade  # Risk per trade
        if row['ATR'] == 0:
            continue # Skip rows where ATR is 0

        # Enter trade
        if not in_trade:
            if row['Long_Entry']:
                position = trade_size / row['Close']
                entry_price = row['Close']
                stop_loss = row['Stop_Loss_Long']
                take_profit = row['Take_Profit_Long']
                in_trade = True
                trade_type = 'Long'
                #print(f"Long Entry at {entry_price}, SL: {stop_loss}, TP: {take_profit}")
            elif row['Short_Entry']:
                position = -trade_size / row['Close']
                entry_price = row['Close']
                stop_loss = row['Stop_Loss_Short']
                take_profit = row['Take_Profit_Short']
                in_trade = True
                trade_type = 'Short'
                #print(f"Short Entry at {entry_price}, SL: {stop_loss}, TP: {take_profit}")
        
        # Manage trade exits
        if in_trade:
            if trade_type == 'Long':
                if row['Low'] <= stop_loss:
                    balance_po3 -= trade_size  # Loss
                    #print("Stopped out - Long")
                    in_trade = False
                elif row['High'] >= take_profit:
                    balance_po3 += trade_size * 2  # Profit
                    #print("Take Profit Hit - Long")
                    in_trade = False
            elif trade_type == 'Short':
                if row['High'] >= stop_loss:
                    balance_po3 -= trade_size  # Loss
                    #print("Stopped out - Short")
                    in_trade = False
                elif row['Low'] <= take_profit:
                    balance_po3 += trade_size * 2  # Profit
                    #print("Take Profit Hit - Short")
                    in_trade = False
        
        balances_po3.append(balance_po3 if balance_po3 > 0 else position * row['Close'])
        balances_hold.append(shares_hold * row['Close'])
    
    return balances_po3, balances_hold

def process_data_with_po3(data):
    """Run PO3 strategy and compare with buy & hold."""
    results = []
    for ticker, df in data.items():
        df = apply_po3_strategy(df)
        df.to_csv(f"{ticker}_po3.csv", index=False)
        #print(f"Processed and saved {ticker}_po3.csv")
        balances_po3, balances_hold = backtest_strategy(df)
        results.append([ticker, balances_po3[-1], balances_hold[-1]])
        
        # Enhanced visualization
        # plt.figure(figsize=(12, 6))
        # plt.plot(balances_po3, label='PO3 Strategy', linestyle='dashed', color='blue')
        # plt.plot(balances_hold, label='Buy & Hold', linestyle='solid', color='green')
        # plt.title(f"{ticker} Trading Strategy Performance")
        # plt.xlabel("Time Steps")
        # plt.ylabel("Portfolio Value ($)")
        # plt.legend()
        # plt.grid()
        # plt.show()
    
    print("\nTrading Performance Comparison:")
    print("{:<10} {:<15} {:<15}".format("Ticker", "PO3 Strategy", "Buy & Hold"))
    print("-" * 40)
    for row in results:
        print("{:<10} {:<15.2f} {:<15.2f}".format(row[0], row[1], row[2]))

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'BTC-USD', 'GDXJ']
    stock_data = fetch_stock_data(tickers)
    process_data_with_po3(stock_data)
