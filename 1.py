import yfinance as yf
import pandas as pd
import numpy as np

# --- Настройки пользователя ---
ticker = "SPY"
period = "6mo"
interval = "1d"

user_inputs = {
    "global_trend": "Moderate uptrend",      # Выбрать из справочника
    "local_trend": "No trend / Sideways",    # Выбрать из справочника
    "news_sentiment": 0                      # От -100 (очень плохо) до 100 (очень хорошо)
}

# --- Справочники вероятностей ---
trend_probabilities = {
    "Strong uptrend":     [80, 10, 10],
    "Moderate uptrend":   [65, 20, 15],
    "No trend / Sideways":[33, 33, 34],
    "Moderate downtrend": [25, 60, 15],
    "Strong downtrend":   [10, 80, 10]
}

sp500_distance_probabilities = {
    "New high":        [10, 80, 10],
    "Near high":       [25, 60, 15],
    "5–10% below":     [40, 40, 20],
    "10–15% below":    [50, 30, 20],
    "15–20% below":    [60, 25, 15],
    "20–30% below":    [70, 20, 10],
    ">30% below":      [80, 10, 10]
}

rsi_probabilities = {
    "< 30":      [75, 10, 15],
    "30–45":     [65, 20, 15],
    "45–55":     [50, 25, 25],
    "55–70":     [40, 40, 20],
    "70–80":     [30, 55, 15],
    "> 80":      [20, 70, 10]
}

volume_probabilities = {
    "High (on growth)": [70, 20, 10],
    "High (on decline)": [20, 70, 10],
    "Average": [40, 30, 30],
    "Low": [25, 25, 50]
}

macd_probabilities = {
    "MACD up (above 0)": [70, 20, 10],
    "MACD up (below 0)": [55, 30, 15],
    "MACD flat": [33, 33, 34],
    "MACD down (above 0)": [45, 45, 10],
    "MACD down (below 0)": [20, 70, 10]
}

# --- Загрузка данных ---
df = yf.download(ticker, period=period, interval=interval)
df.dropna(inplace=True)

# --- RSI ---
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))
current_rsi = rsi.dropna().iloc[-1].item()  # <--- исправлено
print("Current RSI:", round(current_rsi, 2))

def get_rsi_range(val):
    if val < 30:
        return "< 30"
    elif val < 45:
        return "30–45"
    elif val < 55:
        return "45–55"
    elif val < 70:
        return "55–70"
    elif val < 80:
        return "70–80"
    else:
        return "> 80"

rsi_range = get_rsi_range(current_rsi)

# --- % от хая ---
max_price = df['Close'].max().item()
current_price = df['Close'].iloc[-1].item()
print("max_price", max_price)
print("current_price", current_price)
distance_pct = (max_price - current_price) / max_price * 100

print(type(distance_pct))
print(distance_pct)

def get_sp500_range(pct):
    if pct < 1:
        return "New high"
    elif pct < 5:
        return "Near high"
    elif pct < 10:
        return "5–10% below"
    elif pct < 15:
        return "10–15% below"
    elif pct < 20:
        return "15–20% below"
    elif pct < 30:
        return "20–30% below"
    else:
        return ">30% below"

sp500_range = get_sp500_range(distance_pct)

# --- Объём ---
vol_ma = df['Volume'].rolling(20).mean()
vol_ratio = df['Volume'].iloc[-1] / vol_ma.iloc[-1].item()

vol_current = df['Volume'].iloc[-1]
vol_ma_last = vol_ma.iloc[-1]

if isinstance(vol_current, pd.Series):
    vol_current = vol_current.item()

if isinstance(vol_ma_last, pd.Series):
    vol_ma_last = vol_ma_last.item()

vol_ratio = vol_current / vol_ma_last



if vol_ratio > 1.3:
    volume_category = "High (on growth)" if df['Close'].iloc[-1] > df['Open'].iloc[-1] else "High (on decline)"
elif vol_ratio < 0.7:
    volume_category = "Low"
else:
    volume_category = "Average"

# --- MACD ---
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
macd = exp1 - exp2
signal = macd.ewm(span=9, adjust=False).mean()

macd_val = macd.iloc[-1]
signal_val = signal.iloc[-1]

# Если macd_val или signal_val - Series, взять скаляр
if isinstance(macd_val, pd.Series):
    macd_val = macd_val.item()
if isinstance(signal_val, pd.Series):
    signal_val = signal_val.item()

macd_above_zero = macd_val > 0

if abs(macd_val - signal_val) < 0.1:
    macd_status = "MACD flat"
elif macd_val > signal_val:
    macd_status = "MACD up (above 0)" if macd_above_zero else "MACD up (below 0)"
else:
    macd_status = "MACD down (above 0)" if macd_above_zero else "MACD down (below 0)"

# --- News sentiment ---
sent = user_inputs["news_sentiment"]
p_growth_sent = (sent + 100) / 200 * 100
p_decline_sent = (100 - sent) / 200 * 100
p_sideways_sent = (100 - abs(sent)) / 100 * 100

# --- Сбор всех вероятностей ---
components = [
    trend_probabilities[user_inputs["global_trend"]],
    sp500_distance_probabilities[sp500_range],
    rsi_probabilities[rsi_range],
    trend_probabilities[user_inputs["local_trend"]],
    volume_probabilities[volume_category],
    macd_probabilities[macd_status],
    [p_growth_sent, p_decline_sent, p_sideways_sent]
]

# --- Финальный расчёт ---
weights = [1] * len(components)
final = np.average(components, axis=0, weights=weights)

print(f"\nTicker: {ticker}")
print(f"RSI: {current_rsi:.2f} ({rsi_range})")
print(f"Distance from high: {distance_pct:.2f}% ({sp500_range})")
print(f"Volume category: {volume_category}")
print(f"MACD status: {macd_status}")
print("\n📈 Probabilities:")
print(f"Growth:  {final[0]:.2f}%")
print(f"Decline: {final[1]:.2f}%")
print(f"Sideways: {final[2]:.2f}%")
