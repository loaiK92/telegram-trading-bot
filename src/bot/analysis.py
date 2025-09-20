# src/bot/analysis.py

import pandas as pd
import httpx
import logging
import json
import urllib.parse
import talib
import time # Import the time module
from typing import List, Dict, Tuple
from ..core.config import ALLTICK_API_KEY
from .charting import create_elite_chart
from src.core.translations import get_text

# Set up logging to see errors in the console instead of sending them to the user
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_market_data(api_key: str, timeframe: str, num_candles: int) -> pd.DataFrame:
    kline_type_map = {'1H': '5', '15M': '3'}
    api_url = f"https://quote.tradeswitcher.com/quote-b-api/kline?token={api_key}&query=" + urllib.parse.quote(json.dumps({
        "data": {"code": "XAUUSD", "kline_type": kline_type_map[timeframe], "query_kline_num": str(num_candles)}
    }))
    response = httpx.get(api_url, timeout=15)
    response.raise_for_status()
    data = response.json().get('data', {}).get('kline_list', [])
    if not data: raise ValueError(f"API returned no k-line data for {timeframe}.")
    
    df = pd.DataFrame(data)
    df.rename(columns={'open_price': 'open', 'high_price': 'high', 'low_price': 'low', 'close_price': 'close'}, inplace=True)
    for col in ['open', 'high', 'low', 'close', 'timestamp', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open', 'high', 'low', 'close', 'timestamp'], inplace=True)
    df['volume'] = df['volume'].fillna(0)
    if df.empty: raise ValueError("DataFrame is empty after cleaning.")
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('datetime')
    return df

def calculate_volume_profile(df: pd.DataFrame, bins=50) -> float:
    """Calculates the Volume Profile and returns the Point of Control (POC)."""
    price_range = df['high'].max() - df['low'].min()
    if price_range == 0: return df['close'].iloc[-1]
    bin_size = price_range / bins
    binned_volume = df.groupby(pd.cut(df['close'], bins=bins, labels=False))['volume'].sum()
    poc_bin_index = binned_volume.idxmax()
    poc = df['low'].min() + poc_bin_index * bin_size
    return poc

def refine_zone_if_needed(zone: Dict, df: pd.DataFrame) -> Dict:
    """Checks if a zone is too large and refines it."""
    zone_height = zone['top'] - zone['bottom']
    atr = df['atr'].iloc[-1]
    
    if zone_height > (atr * 2.0):
        refinement_size = zone_height * 0.3
        if zone['type'] == 'Demand':
            zone['top'] = zone['bottom'] + refinement_size
        else: # Supply
            zone['bottom'] = zone['top'] - refinement_size
        zone['is_refined'] = True
    else:
        zone['is_refined'] = False
    return zone

def find_premium_zones(df: pd.DataFrame, bias: bool, poc: float, current_price: float, chart_view_range: tuple, num_zones=3) -> List[Dict]:
    """
    Uses a hybrid approach to find zones from both price velocity and significant swing points,
    then filters them to be relevant to the current chart view.
    """
    potential_zones = []

    # --- METHOD 1: Find zones from strong price velocity (imbalances) ---
    df['body'] = (df['close'] - df['open']).abs()
    threshold = df['body'].quantile(0.90)
    strong_move_indices = df[df['body'] > threshold].index
    for idx in strong_move_indices:
        i = df.index.get_loc(idx)
        if i < 1: continue
        base_candle, move_candle = df.iloc[i-1], df.iloc[i]

        if move_candle['close'] < base_candle['open']:
            zone_type, top, bottom = 'Supply', base_candle['high'], base_candle['low']
        elif move_candle['close'] > base_candle['open']:
            zone_type, top, bottom = 'Demand', base_candle['high'], base_candle['low']
        else: continue
        potential_zones.append({'type': zone_type, 'top': top, 'bottom': bottom, 'timestamp': str(base_candle.name)})

    # --- METHOD 2: Find zones from recent significant swing points ---
    recent_data = df.tail(120) # Look at the same range as the chart
    swing_high_idx = recent_data['high'].idxmax()
    swing_low_idx = recent_data['low'].idxmin()
    
    swing_high_candle = df.loc[swing_high_idx]
    potential_zones.append({'type': 'Supply', 'top': swing_high_candle['high'], 'bottom': swing_high_candle['low'], 'timestamp': str(swing_high_candle.name)})
    
    swing_low_candle = df.loc[swing_low_idx]
    potential_zones.append({'type': 'Demand', 'top': swing_low_candle['high'], 'bottom': swing_low_candle['low'], 'timestamp': str(swing_low_candle.name)})

    # De-duplicate zones that are too close to each other
    unique_zones = []
    sorted_by_time = sorted(potential_zones, key=lambda x: x['timestamp'], reverse=True)
    for zone in sorted_by_time:
        is_duplicate = any(abs(zone['top'] - uz['top']) < (df['atr'].mean() * 0.5) for uz in unique_zones)
        if not is_duplicate:
            unique_zones.append(zone)

    # --- NEW: Filter for relevance to the chart view ---
    min_visible, max_visible = chart_view_range
    relevant_zones = [z for z in unique_zones if z['top'] > min_visible and z['bottom'] < max_visible]

    # Refine, score, and sort the final relevant zones
    atr = df['atr'].iloc[-1]
    for zone in relevant_zones:
        zone = refine_zone_if_needed(zone, df)
        score = 0
        reasons = []
        if (zone['type'] == 'Demand' and bias is True) or (zone['type'] == 'Supply' and bias is False):
            score += 3
            reasons.append("Trend-Aligned")
        if min(abs(current_price - zone['top']), abs(current_price - zone['bottom'])) < (atr * 8):
            score += 2
            reasons.append("Near Price")
        if abs(zone['bottom'] - poc) < atr or abs(zone['top'] - poc) < atr:
            score += 1
            reasons.append("POC Confluence")
        zone['score'] = score
        zone['reasons'] = reasons if reasons else ["Price Action"]

    sorted_zones = sorted(relevant_zones, key=lambda x: (x['score'], x['timestamp']), reverse=True)
    
    final_zones = []
    for zone in sorted_zones:
        if len(final_zones) >= num_zones: break
        is_overlapping = any(not (zone['top'] < fz['bottom'] or zone['bottom'] > fz['top']) for fz in final_zones)
        if not is_overlapping:
            final_zones.append(zone)
            
    return final_zones

def calculate_trade_parameters(zone: Dict, df: pd.DataFrame) -> Dict:
    """Calculates SL and TP for a given zone."""
    atr = df['atr'].iloc[-1]
    if zone['type'] == 'Supply':
        stop_loss, risk = (zone['top'] + atr), ((zone['top'] + atr) - zone['bottom'])
        tp1, tp2 = (zone['bottom'] - risk), (zone['bottom'] - (risk * 2.0))
    else: # Demand
        stop_loss, risk = (zone['bottom'] - atr), (zone['top'] - (zone['bottom'] - atr))
        tp1, tp2 = (zone['top'] + risk), (zone['top'] + (risk * 2.0))
    zone.update({'sl': stop_loss, 'tp1': tp1, 'tp2': tp2})
    return zone

def generate_elite_report(df_htf: pd.DataFrame, df_ltf: pd.DataFrame, lang: str = 'en') -> Tuple[str, List[Dict]]:
    df_htf['atr'] = talib.ATR(df_htf['high'], df_htf['low'], df_htf['close'], 14)
    df_htf['EMA50'] = talib.EMA(df_htf['close'], 50)
    df_htf.dropna(inplace=True)
    
    current_price = df_ltf['close'].iloc[-1]
    bias_is_bullish = current_price > df_htf['EMA50'].iloc[-1]
    bias_text = get_text('bullish', lang) if bias_is_bullish else get_text('bearish', lang)
    
    poc = calculate_volume_profile(df_htf.tail(500))
    chart_df = df_htf.tail(120)
    chart_view_range = (chart_df['low'].min() * 0.997, chart_df['high'].max() * 1.003)
    
    zones = find_premium_zones(df_htf, bias=bias_is_bullish, poc=poc, current_price=current_price, chart_view_range=chart_view_range, num_zones=3)
    
    if not zones:
        return get_text('no_zones_found', lang), []

    trade_setups = [calculate_trade_parameters(zone, df_htf) for zone in zones]
    
    report = f"{get_text('elite_report_title', lang)}\n\n"
    report += f"**{get_text('current_price', lang)}:** `${current_price:,.2f}`\n"
    report += f"**{get_text('htf_trend', lang)}:** {bias_text}\n"
    report += f"**{get_text('poc', lang)}:** `${poc:,.2f}`\n\n"
    report += f"{get_text('zones_header', lang)}\n"
    report += "-----------------------------------\n"
    
    for i, setup in enumerate(trade_setups):
        trade_type = "SELL" if setup['type'] == 'Supply' else "BUY"
        refined_text = get_text('zone_refined', lang) if setup.get('is_refined') else ""
        
        report += f"**{get_text('zone_header', lang).format(i+1, setup['type'], refined_text)}**\n"
        report += f"   - **{get_text('setup', lang)}:** {get_text('setup_sell' if trade_type == 'SELL' else 'setup_buy', lang)}\n"
        report += f"   - **{get_text('entry_zone', lang)}:** `${setup['bottom']:,.2f} - ${setup['top']:,.2f}`\n"
        report += f"   - **{get_text('stop_loss', lang)}:** `${setup['sl']:,.2f}`\n"
        report += f"   - **{get_text('tp_1', lang)}:** `${setup['tp1']:,.2f}`\n"
        report += f"   - **{get_text('tp_2', lang)}:** `${setup['tp2']:,.2f}`\n\n"
        
    report += f"{get_text('disclaimer_elite', lang)}"
    
    return report, trade_setups