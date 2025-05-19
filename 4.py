import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Настройки пользователя ---
ticker = "SPY"
period = "6mo"
interval = "1d"

user_inputs = {
    "global_trend": "Moderate uptrend",
    "local_trend": "No trend / Sideways",
    "news_sentiment": 0  # От -100 до 100
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
current_rsi = rsi.dropna().iloc[-1].item()

def get_rsi_range(val):
    if val < 30: return "< 30"
    elif val < 45: return "30–45"
    elif val < 55: return "45–55"
    elif val < 70: return "55–70"
    elif val < 80: return "70–80"
    else: return "> 80"

rsi_range = get_rsi_range(current_rsi)
rsi_prob = rsi_probabilities[rsi_range]
print(f"Current RSI: {round(current_rsi)} Probabilities Growth: {rsi_prob[0]}% Decline: {rsi_prob[1]}% Sideways: {rsi_prob[2]}%")

# --- % от хая ---
max_price = df['Close'].max().item()
current_price = df['Close'].iloc[-1].item()
distance_pct = (max_price - current_price) / max_price * 100

def get_sp500_range(pct):
    if pct < 1: return "New high"
    elif pct < 5: return "Near high"
    elif pct < 10: return "5–10% below"
    elif pct < 15: return "10–15% below"
    elif pct < 20: return "15–20% below"
    elif pct < 30: return "20–30% below"
    else: return ">30% below"

sp500_range = get_sp500_range(distance_pct)
sp500_prob = sp500_distance_probabilities[sp500_range]
print(f"Distance from high: {distance_pct:.2f}% ({sp500_range}) Probabilities Growth: {sp500_prob[0]}% Decline: {sp500_prob[1]}% Sideways: {sp500_prob[2]}%")

# --- Объём ---
vol_ma = df['Volume'].rolling(20).mean()
# vol_ratio = df['Volume'].iloc[-1] / vol_ma.iloc[-1].item()

vol_current = df['Volume'].iloc[-1]
vol_ma_last = vol_ma.iloc[-1]

if isinstance(vol_current, pd.Series):
    vol_current = vol_current.item()

if isinstance(vol_ma_last, pd.Series):
    vol_ma_last = vol_ma_last.item()
vol_ratio = vol_current / vol_ma_last

volume_category = "Average"
if vol_ratio > 1.3:
    volume_category = "High (on growth)" if df['Close'].iloc[-1] > df['Open'].iloc[-1] else "High (on decline)"
elif vol_ratio < 0.7:
    volume_category = "Low"
vol_prob = volume_probabilities[volume_category]
print(f"Volume: {volume_category} Probabilities Growth: {vol_prob[0]}% Decline: {vol_prob[1]}% Sideways: {vol_prob[2]}%")

# --- MACD ---
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
macd = exp1 - exp2
signal = macd.ewm(span=9, adjust=False).mean()
macd_val = macd.iloc[-1].item()
signal_val = signal.iloc[-1].item()
macd_above_zero = macd_val > 0

if abs(macd_val - signal_val) < 0.1:
    macd_status = "MACD flat"
elif macd_val > signal_val:
    macd_status = "MACD up (above 0)" if macd_above_zero else "MACD up (below 0)"
else:
    macd_status = "MACD down (above 0)" if macd_above_zero else "MACD down (below 0)"

macd_prob = macd_probabilities[macd_status]
print(f"MACD: {macd_status} Probabilities Growth: {macd_prob[0]}% Decline: {macd_prob[1]}% Sideways: {macd_prob[2]}%")

# --- News Sentiment ---
sent = user_inputs["news_sentiment"]
p_growth_sent = (sent + 100) / 200 * 100
p_decline_sent = (100 - sent) / 200 * 100
p_sideways_sent = (100 - abs(sent)) / 100 * 100
news_prob = [p_growth_sent, p_decline_sent, p_sideways_sent]
print(f"News Sentiment: {sent} Probabilities Growth: {news_prob[0]:.2f}% Decline: {news_prob[1]:.2f}% Sideways: {news_prob[2]:.2f}%")

# --- Тренды ---
global_prob = trend_probabilities[user_inputs["global_trend"]]
print(f"Global trend: {user_inputs['global_trend']} Probabilities Growth: {global_prob[0]}% Decline: {global_prob[1]}% Sideways: {global_prob[2]}%")

local_prob = trend_probabilities[user_inputs["local_trend"]]
print(f"Local trend: {user_inputs['local_trend']} Probabilities Growth: {local_prob[0]}% Decline: {local_prob[1]}% Sideways: {local_prob[2]}%")

# --- Сбор вероятностей ---
components = [
    global_prob,
    sp500_prob,
    rsi_prob,
    local_prob,
    vol_prob,
    macd_prob,
    news_prob
]

# --- Весовые коэффициенты ---
weights = [1.5, 1.2, 1.5, 1.0, 1.0, 1.2, 0.8]

# --- Финальный расчёт ---
final = np.average(components, axis=0, weights=weights)

# --- Вывод ---
print(f"\n📊 Итог по тикеру: {ticker}")
print(f"RSI: {current_rsi:.2f} ({rsi_range})")
print(f"Distance from high: {distance_pct:.2f}% ({sp500_range})")
print(f"Volume category: {volume_category}")
print(f"MACD status: {macd_status}")

print("\n📈 Финальные вероятности:")
print(f"Growth:   {final[0]:.2f}%")
print(f"Decline:  {final[1]:.2f}%")
print(f"Sideways: {final[2]:.2f}%")

# --- Визуализация ---
labels = ['Growth', 'Decline', 'Sideways']
colors = ['green', 'red', 'gray']
plt.bar(labels, final, color=colors)
plt.title(f'{ticker} Market Direction Probabilities')
plt.ylabel('Probability (%)')
plt.ylim(0, 100)
plt.grid(axis='y')
plt.show()
