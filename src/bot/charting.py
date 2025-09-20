import pandas as pd
import mplfinance as mpf
import io
from typing import List, Dict
from matplotlib.patches import Rectangle

def create_elite_chart(df: pd.DataFrame, zones: List[Dict], trend_ema: pd.Series) -> io.BytesIO:
    """
    Generates a professional candlestick chart with a clean, white theme,
    and intelligently placed labels to avoid overlapping.
    """
    plot_df = df.tail(120).copy()
    
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', inherit=True)
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc, gridstyle='--', facecolor='white', edgecolor='#333333')
    
    addplots = [mpf.make_addplot(trend_ema.tail(120), color='#00b7ff', width=1.0)]

    fig, axes = mpf.plot(
        plot_df,
        type='candle',
        style=s,
        title='XAUUSD Analysis HTF',
        ylabel='Price ($)',
        addplot=addplots,
        figsize=(15, 8),
        returnfig=True,
        ylim=(plot_df['low'].min() * 0.997, plot_df['high'].max() * 1.003)
    )
    
    axes[0].grid(False, axis='x')
    axes[0].set_xticklabels([])
    axes[0].set_xlabel('')
    
    text_color = '#333333' # Dark gray

    for i, zone in enumerate(zones):
        zone_color = 'red' if zone['type'] == 'Supply' else 'blue'
        
        rect = Rectangle(
            (0, zone['bottom']),
            len(plot_df),
            zone['top'] - zone['bottom'],
            facecolor=zone_color,
            edgecolor=zone_color,
            alpha=0.1,
            linewidth=1.5,
            zorder=0
        )
        axes[0].add_patch(rect)
        
        x_pos = len(plot_df) * (0.98 - (i * 0.09))

        zone_label = f"Zone {i+1}"

        # --- CHANGE: Smaller font size for zone type text ---
        axes[0].text(x_pos, (zone['top'] + zone['bottom']) / 2, f"{zone['type']} {zone_label}", 
                     color='black', ha='right', va='center', fontsize=9, style='italic', weight='bold') # Fontsize changed from 10 to 9
        
        line_width = 0.5
        
        # --- CHANGE: Smaller font size for SL/TP labels ---
        axes[0].axhline(zone['sl'], color='#ff3333', linestyle='--', linewidth=line_width)
        axes[0].text(x_pos, zone['sl'], f"  {zone_label} SL",
                     color=text_color, ha='right', va='center', fontsize=8) # Fontsize changed from 9 to 8
                     
        axes[0].axhline(zone['tp1'], color='#00b300', linestyle='--', linewidth=line_width)
        axes[0].text(x_pos, zone['tp1'], f"  {zone_label} TP 1",
                     color=text_color, ha='right', va='center', fontsize=8) # Fontsize changed from 9 to 8
                     
        axes[0].axhline(zone['tp2'], color='#00b300', linestyle='--', linewidth=line_width)
        axes[0].text(x_pos, zone['tp2'], f"  {zone_label} TP 2",
                     color=text_color, ha='right', va='center', fontsize=8) # Fontsize changed from 9 to 8
        # ----------------------------------------------------

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf
