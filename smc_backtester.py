"""
SMC PRO BACKTESTER v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Walk-forward backtester for SMC Pro v4.0

HOW IT WORKS:
  1. Downloads full historical OHLCV for each pair (4H / 1H / 15M)
  2. Slides a window forward one candle at a time (no lookahead)
  3. Runs the EXACT same analyse() logic as the live bot
  4. Simulates TP1/TP2/TP3/SL hits on future candles
  5. Reports full stats + exports trades.csv

USAGE:
  python smc_backtester.py

SETTINGS (bottom of file):
  SYMBOLS      — list of pairs to test
  LOOKBACK_DAYS— how many days of history to use
  TIMEFRAME    — resolution for the walk (default '1h')
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import logging
import csv
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════
#  COPY-PASTE YOUR EXACT SETTINGS FROM THE BOT
# ═══════════════════════════════════════════════
MAX_SIGNALS_PER_SCAN  = 6
MIN_SCORE             = 75
MIN_VOLUME_24H        = 5_000_000
OB_TOLERANCE_PCT      = 0.008
OB_IMPULSE_ATR_MULT   = 1.0
STRUCTURE_LOOKBACK    = 20
HH_LL_LOOKBACK        = 10
HH_LL_BONUS           = 8

# ═══════════════════════════════════════════════
#  BACKTEST SETTINGS
# ═══════════════════════════════════════════════
LOOKBACK_DAYS   = 90        # days of history to test
WALK_STEP       = 1         # advance N 1H candles per step (1 = every candle)
MAX_TRADE_BARS  = 48        # max 48x1H bars to resolve a trade (=48h)
WARM_UP_BARS_1H = 100       # candles needed before first signal attempt
DEDUPE_HOURS    = 4         # ignore same symbol signals within N hours

OUTPUT_CSV = "backtest_trades.csv"
OUTPUT_DIR = "/mnt/user-data/outputs"


# ══════════════════════════════════════════════════════════════
#  INDICATORS  (identical to bot)
# ══════════════════════════════════════════════════════════════

def add_indicators(df):
    if len(df) < 55:
        return df
    try:
        df = df.copy()
        df['ema_21']  = ta.trend.EMAIndicator(df['close'], 21).ema_indicator()
        df['ema_50']  = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], min(200, len(df)-1)).ema_indicator()
        df['rsi']     = ta.momentum.RSIIndicator(df['close'], 14).rsi()

        macd = ta.trend.MACD(df['close'])
        df['macd']        = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist']   = macd.macd_diff()

        stoch = ta.momentum.StochRSIIndicator(df['close'])
        df['srsi_k'] = stoch.stochrsi_k()
        df['srsi_d'] = stoch.stochrsi_d()

        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

        bb = ta.volatility.BollingerBands(df['close'], 20, 2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_pband']  = bb.bollinger_pband()

        adx_i = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx']    = adx_i.adx()
        df['di_pos'] = adx_i.adx_pos()
        df['di_neg'] = adx_i.adx_neg()

        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
        df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()

        df['vol_sma']   = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_sma'].replace(0, np.nan)

        tp = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()

        body = (df['close'] - df['open']).abs()
        uw   = df['high'] - df[['open','close']].max(axis=1)
        lw   = df[['open','close']].min(axis=1) - df['low']

        df['bull_engulf'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] > df['open']) &
            (df['close'] > df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1))
        ).astype(int)

        df['bear_engulf'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'] < df['open']) &
            (df['close'] < df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1))
        ).astype(int)

        df['bull_pin'] = (
            (lw > body * 2.5) & (lw > uw * 2) & (df['close'] > df['open'])
        ).astype(int)

        df['bear_pin'] = (
            (uw > body * 2.5) & (uw > lw * 2) & (df['close'] < df['open'])
        ).astype(int)

        df['hammer'] = (
            (lw > body * 2.0) & (lw > uw * 1.5)
        ).astype(int)

        df['shooting_star'] = (
            (uw > body * 2.0) & (uw > lw * 1.5)
        ).astype(int)

    except Exception as e:
        logger.error(f"Indicator error: {e}")
    return df


# ══════════════════════════════════════════════════════════════
#  SMC ENGINE  (identical to bot)
# ══════════════════════════════════════════════════════════════

class SMCEngine:

    def swing_highs_lows(self, df, left=4, right=4):
        highs, lows = [], []
        n = len(df)
        for i in range(left, n - right):
            hi = df['high'].iloc[i]
            lo = df['low'].iloc[i]
            if all(hi >= df['high'].iloc[i-left:i]) and all(hi >= df['high'].iloc[i+1:i+right+1]):
                highs.append({'i': i, 'price': hi})
            if all(lo <= df['low'].iloc[i-left:i]) and all(lo <= df['low'].iloc[i+1:i+right+1]):
                lows.append({'i': i, 'price': lo})
        return highs, lows

    def check_4h_hh_ll(self, df_4h, direction, lookback=HH_LL_LOOKBACK):
        n = len(df_4h)
        if n < lookback * 2:
            return False, "⚠️ Not enough 4H data"
        recent = df_4h.iloc[-lookback:]
        prior  = df_4h.iloc[-(lookback * 2):-lookback]
        if direction == 'LONG':
            rh, ph = recent['high'].max(), prior['high'].max()
            return (rh > ph), f"4H HH: {ph:.5f}→{rh:.5f}"
        else:
            rl, pl = recent['low'].min(), prior['low'].min()
            return (rl < pl), f"4H LL: {pl:.5f}→{rl:.5f}"

    def detect_structure_break(self, df, highs, lows, lookback=STRUCTURE_LOOKBACK):
        events = []
        close = df['close']
        n = len(df)
        start = max(0, n - lookback - 15)
        for k in range(1, len(highs)):
            ph = highs[k-1]; ch = highs[k]
            if ch['i'] < start: continue
            level = ph['price']
            for j in range(ch['i'], min(ch['i'] + 10, n)):
                if close.iloc[j] > level:
                    kind = 'BOS_BULL' if ch['price'] > ph['price'] else 'MSS_BULL'
                    events.append({'kind': kind, 'level': level, 'bar': j})
                    break
        for k in range(1, len(lows)):
            pl = lows[k-1]; cl = lows[k]
            if cl['i'] < start: continue
            level = pl['price']
            for j in range(cl['i'], min(cl['i'] + 10, n)):
                if close.iloc[j] < level:
                    kind = 'BOS_BEAR' if cl['price'] < pl['price'] else 'MSS_BEAR'
                    events.append({'kind': kind, 'level': level, 'bar': j})
                    break
        if not events:
            return None
        latest = sorted(events, key=lambda x: x['bar'])[-1]
        if latest['bar'] < n - lookback:
            return None
        return latest

    def find_order_blocks(self, df, direction, lookback=60):
        obs = []
        n = len(df)
        start = max(2, n - lookback)
        for i in range(start, n - 3):
            c = df.iloc[i]
            atr_local = df['atr'].iloc[i] if 'atr' in df.columns and not pd.isna(df['atr'].iloc[i]) else (c['high'] - c['low'])
            min_impulse = atr_local * OB_IMPULSE_ATR_MULT
            if direction == 'LONG':
                if c['close'] >= c['open']: continue
                fwd_high = df['high'].iloc[i+1:min(i+5, n)].max()
                if fwd_high - c['low'] < min_impulse: continue
                ob = {'top': max(c['open'], c['close']), 'bottom': c['low'],
                      'mid': (max(c['open'], c['close']) + c['low']) / 2, 'bar': i}
                ob_50 = (ob['top'] + ob['bottom']) / 2
                if (df['close'].iloc[i+1:n] < ob_50).any(): continue
                obs.append(ob)
            else:
                if c['close'] <= c['open']: continue
                fwd_low = df['low'].iloc[i+1:min(i+5, n)].min()
                if c['high'] - fwd_low < min_impulse: continue
                ob = {'top': c['high'], 'bottom': min(c['open'], c['close']),
                      'mid': (c['high'] + min(c['open'], c['close'])) / 2, 'bar': i}
                ob_50 = (ob['top'] + ob['bottom']) / 2
                if (df['close'].iloc[i+1:n] > ob_50).any(): continue
                obs.append(ob)
        obs.sort(key=lambda x: x['bar'], reverse=True)
        return obs

    def price_in_ob(self, price, ob, tolerance_pct=OB_TOLERANCE_PCT):
        tol = ob['top'] * tolerance_pct
        return (ob['bottom'] - tol) <= price <= (ob['top'] + tol)

    def find_fvg(self, df, direction, lookback=25):
        fvgs = []
        n = len(df)
        for i in range(max(1, n - lookback), n - 1):
            prev = df.iloc[i-1]; nxt = df.iloc[i+1]
            if direction == 'LONG' and prev['high'] < nxt['low']:
                fvgs.append({'top': nxt['low'], 'bottom': prev['high'],
                             'mid': (nxt['low'] + prev['high']) / 2, 'bar': i})
            elif direction == 'SHORT' and prev['low'] > nxt['high']:
                fvgs.append({'top': prev['low'], 'bottom': nxt['high'],
                             'mid': (prev['low'] + nxt['high']) / 2, 'bar': i})
        return fvgs

    def recent_liquidity_sweep(self, df, direction, highs, lows, lookback=25):
        n = len(df)
        start = n - lookback
        if direction == 'LONG':
            for sl in reversed(lows):
                if sl['i'] < start: continue
                level = sl['price']
                for j in range(sl['i'] + 1, min(sl['i'] + 8, n)):
                    c = df.iloc[j]
                    if c['low'] < level and c['close'] > level:
                        return {'level': level, 'bar': j, 'type': 'SWEEP_LOW'}
        else:
            for sh in reversed(highs):
                if sh['i'] < start: continue
                level = sh['price']
                for j in range(sh['i'] + 1, min(sh['i'] + 8, n)):
                    c = df.iloc[j]
                    if c['high'] > level and c['close'] < level:
                        return {'level': level, 'bar': j, 'type': 'SWEEP_HIGH'}
        return None

    def pd_zone(self, df_4h, price):
        hi = df_4h['high'].iloc[-50:].max()
        lo = df_4h['low'].iloc[-50:].min()
        rang = hi - lo
        if rang == 0: return 'NEUTRAL', 0.5
        pos = (price - lo) / rang
        if pos < 0.40:   return 'DISCOUNT', pos
        elif pos > 0.60: return 'PREMIUM',  pos
        return 'NEUTRAL', pos


# ══════════════════════════════════════════════════════════════
#  SCORER  (identical to bot)
# ══════════════════════════════════════════════════════════════

def score_setup(direction, ob, structure, sweep, fvg_near,
                df_1h, df_15m, df_4h, pd_label, hh_ll_confirmed):
    score = 0
    reasons = []
    failed = []

    l1  = df_1h.iloc[-1]
    p1  = df_1h.iloc[-2]
    l15 = df_15m.iloc[-1]
    l4  = df_4h.iloc[-1]

    # 1. Structure
    if structure:
        if 'MSS' in structure['kind']:
            score += 20; reasons.append(f"MSS ({structure['kind']})")
        else:
            score += 14; reasons.append(f"BOS ({structure['kind']})")
    else:
        failed.append("No BOS/MSS")

    # 2. OB quality
    if ob:
        ob_size_pct = (ob['top'] - ob['bottom']) / ob['bottom'] * 100
        if ob_size_pct < 0.8:
            score += 20; reasons.append(f"Tight OB {ob_size_pct:.2f}%")
        elif ob_size_pct < 2.0:
            score += 13; reasons.append(f"OB {ob_size_pct:.2f}%")
        else:
            score += 7;  reasons.append(f"Wide OB {ob_size_pct:.2f}%")
    else:
        failed.append("No OB")

    # 3. 4H Trend
    e21 = l4.get('ema_21', 0); e50 = l4.get('ema_50', 0); e200 = l4.get('ema_200', 0)
    if direction == 'LONG':
        if e21 > e50 > e200:     score += 15; reasons.append("4H Triple EMA Bull")
        elif e21 > e50:          score += 10; reasons.append("4H EMA Bull")
        elif pd_label == 'DISCOUNT': score += 6; reasons.append("Discount zone")
        else: failed.append("4H trend weak")
    else:
        if e21 < e50 < e200:     score += 15; reasons.append("4H Triple EMA Bear")
        elif e21 < e50:          score += 10; reasons.append("4H EMA Bear")
        elif pd_label == 'PREMIUM': score += 6; reasons.append("Premium zone")
        else: failed.append("4H trend weak")

    # 4. HH/LL bonus
    if hh_ll_confirmed:
        score += HH_LL_BONUS; reasons.append(f"4H HH/LL +{HH_LL_BONUS}")

    # 5. 1H Trigger
    trigger = False
    if direction == 'LONG':
        if l1.get('bull_engulf', 0) == 1:   score += 25; trigger = True; reasons.append("1H BullEngulf")
        elif l1.get('bull_pin', 0) == 1:    score += 22; trigger = True; reasons.append("1H BullPin")
        elif l1.get('hammer', 0) == 1:      score += 18; trigger = True; reasons.append("1H Hammer")
        elif p1.get('bull_engulf', 0) == 1: score += 14; trigger = True; reasons.append("1H BullEngulf(prev)")
        elif p1.get('bull_pin', 0) == 1:    score += 11; trigger = True; reasons.append("1H BullPin(prev)")
        elif p1.get('hammer', 0) == 1:      score += 9;  trigger = True; reasons.append("1H Hammer(prev)")
    else:
        if l1.get('bear_engulf', 0) == 1:   score += 25; trigger = True; reasons.append("1H BearEngulf")
        elif l1.get('bear_pin', 0) == 1:    score += 22; trigger = True; reasons.append("1H BearPin")
        elif l1.get('shooting_star', 0) == 1: score += 18; trigger = True; reasons.append("1H ShootingStar")
        elif p1.get('bear_engulf', 0) == 1: score += 14; trigger = True; reasons.append("1H BearEngulf(prev)")
        elif p1.get('bear_pin', 0) == 1:    score += 11; trigger = True; reasons.append("1H BearPin(prev)")
        elif p1.get('shooting_star', 0) == 1: score += 9; trigger = True; reasons.append("1H SS(prev)")

    if not trigger:
        score -= 12; failed.append("No 1H trigger")

    # 6. Momentum
    rsi1 = l1.get('rsi', 50)
    macd1 = l1.get('macd', 0); ms1 = l1.get('macd_signal', 0)
    pm1 = p1.get('macd', 0); pms1 = p1.get('macd_signal', 0)
    sk1 = l1.get('srsi_k', 0.5); sd1 = l1.get('srsi_d', 0.5)

    if direction == 'LONG':
        if 28 <= rsi1 <= 55:     score += 4; reasons.append(f"RSI reset {rsi1:.0f}")
        elif rsi1 < 28:          score += 3; reasons.append(f"RSI oversold {rsi1:.0f}")
        if macd1 > ms1 and pm1 <= pms1: score += 5; reasons.append("MACD bull X")
        elif macd1 > ms1:        score += 2; reasons.append("MACD bull")
        if sk1 < 0.3 and sk1 > sd1: score += 3; reasons.append("StochRSI bull X")
    else:
        if 45 <= rsi1 <= 72:     score += 4; reasons.append(f"RSI OB zone {rsi1:.0f}")
        elif rsi1 > 72:          score += 3; reasons.append(f"RSI overbought {rsi1:.0f}")
        if macd1 < ms1 and pm1 >= pms1: score += 5; reasons.append("MACD bear X")
        elif macd1 < ms1:        score += 2; reasons.append("MACD bear")
        if sk1 > 0.7 and sk1 < sd1: score += 3; reasons.append("StochRSI bear X")

    # 7. Extras
    extras = 0
    if sweep:      extras += 4; reasons.append("LiqSweep")
    if fvg_near:   extras += 3; reasons.append("FVG+OB")
    vr15 = l15.get('vol_ratio', 1.0)
    if   vr15 >= 2.5: extras += 3; reasons.append(f"15M vol {vr15:.1f}x")
    elif vr15 >= 1.5: extras += 1; reasons.append(f"15M vol {vr15:.1f}x")
    close1 = l1.get('close', 0); vwap1 = l1.get('vwap', 0)
    if direction == 'LONG' and close1 < vwap1:    extras += 1; reasons.append("Below VWAP")
    elif direction == 'SHORT' and close1 > vwap1: extras += 1; reasons.append("Above VWAP")
    score += min(extras, 10)

    return max(0, min(int(score), 100)), reasons, failed


# ══════════════════════════════════════════════════════════════
#  ANALYSE  (identical to bot — operates on sliced history)
# ══════════════════════════════════════════════════════════════

smc = SMCEngine()

def analyse_slice(df4, df1, df15, symbol):
    """Run the exact same gate+score logic on sliced historical data."""
    try:
        if len(df1) < 80 or len(df15) < 40:
            return None

        price = df1['close'].iloc[-1]

        # Gate 1: 4H Bias
        l4 = df4.iloc[-1]
        e21 = l4.get('ema_21', 0); e50 = l4.get('ema_50', 0)
        if   e21 > e50: bias = 'LONG'
        elif e21 < e50: bias = 'SHORT'
        else: return None

        # HH/LL bonus
        hh_ll_ok, _ = smc.check_4h_hh_ll(df4, bias, HH_LL_LOOKBACK)

        # Gate 2: PD Zone
        pd_label, pd_pos = smc.pd_zone(df4, price)
        if bias == 'LONG'  and pd_label == 'PREMIUM':  return None
        if bias == 'SHORT' and pd_label == 'DISCOUNT': return None

        # Gate 3: 1H Structure
        highs1, lows1 = smc.swing_highs_lows(df1, left=4, right=4)
        structure = smc.detect_structure_break(df1, highs1, lows1, lookback=STRUCTURE_LOOKBACK)
        if structure:
            if bias == 'LONG'  and 'BEAR' in structure['kind']: return None
            if bias == 'SHORT' and 'BULL' in structure['kind']: return None

        # Gate 4: 1H OB (hard gate)
        obs = smc.find_order_blocks(df1, bias, lookback=60)
        if not obs: return None

        active_ob = None
        for ob in obs:
            if smc.price_in_ob(price, ob, OB_TOLERANCE_PCT):
                active_ob = ob; break
        if not active_ob: return None

        # FVG + Sweep
        fvgs = smc.find_fvg(df1, bias, lookback=25)
        fvg_near = next((f for f in fvgs
                         if f['bottom'] < active_ob['top'] and f['top'] > active_ob['bottom']), None)
        sweep = smc.recent_liquidity_sweep(df1, bias, highs1, lows1, lookback=20)

        # Score
        score, reasons, _ = score_setup(
            bias, active_ob, structure, sweep, fvg_near,
            df1, df15, df4, pd_label, hh_ll_ok
        )

        if score < MIN_SCORE:
            return None

        # Build trade levels
        atr1  = df1['atr'].iloc[-1]
        entry = price

        if bias == 'LONG':
            sl = active_ob['bottom'] - atr1 * 0.2
            sl = min(sl, entry - atr1 * 0.6)
        else:
            sl = active_ob['top'] + atr1 * 0.2
            sl = max(sl, entry + atr1 * 0.6)

        risk = abs(entry - sl)
        if risk < entry * 0.001:
            return None

        if bias == 'LONG':
            tps = [entry + risk*1.5, entry + risk*2.5, entry + risk*4.0]
        else:
            tps = [entry - risk*1.5, entry - risk*2.5, entry - risk*4.0]

        if   score >= 92: quality = 'ELITE'
        elif score >= 85: quality = 'PREMIUM'
        else:             quality = 'HIGH'

        return {
            'symbol':    symbol,
            'bias':      bias,
            'quality':   quality,
            'score':     score,
            'entry':     entry,
            'sl':        sl,
            'tp1':       tps[0],
            'tp2':       tps[1],
            'tp3':       tps[2],
            'risk_pct':  risk / entry * 100,
            'rr1':       1.5,
            'rr2':       2.5,
            'rr3':       4.0,
            'pd_zone':   pd_label,
            'hh_ll':     hh_ll_ok,
            'structure': structure['kind'] if structure else 'NONE',
            'reasons':   ' | '.join(reasons[:6]),
        }
    except Exception as e:
        logger.debug(f"analyse_slice error: {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  TRADE RESOLUTION  (simulate future price action)
# ══════════════════════════════════════════════════════════════

def resolve_trade(sig, future_df1h):
    """
    Walk future 1H candles to find what hit first: TP1/TP2/TP3/SL/TIMEOUT.
    Returns outcome dict.
    """
    entry = sig['entry']
    sl    = sig['sl']
    tp1, tp2, tp3 = sig['tp1'], sig['tp2'], sig['tp3']
    direction = sig['bias']

    tp1_hit = tp2_hit = tp3_hit = sl_hit = False
    tp1_bar = tp2_bar = tp3_bar = sl_bar = None
    exit_price = entry
    exit_bar   = len(future_df1h) - 1  # default = timeout

    for i, row in future_df1h.iterrows():
        bar_idx = future_df1h.index.get_loc(i)
        hi = row['high']; lo = row['low']

        if direction == 'LONG':
            # Check SL first (worst case for same candle)
            if lo <= sl and not sl_hit and not tp1_hit:
                sl_hit  = True
                sl_bar  = bar_idx
                exit_price = sl
                break
            if hi >= tp1 and not tp1_hit:
                tp1_hit = True; tp1_bar = bar_idx
            if hi >= tp2 and tp1_hit and not tp2_hit:
                tp2_hit = True; tp2_bar = bar_idx
            if hi >= tp3 and tp2_hit and not tp3_hit:
                tp3_hit = True; tp3_bar = bar_idx
                exit_price = tp3; exit_bar = bar_idx; break
            if lo <= sl and tp1_hit and not tp2_hit:
                # SL after TP1 but before TP2 → partial win
                sl_hit = True; sl_bar = bar_idx
                exit_price = sl; exit_bar = bar_idx; break
        else:
            if hi >= sl and not sl_hit and not tp1_hit:
                sl_hit  = True
                sl_bar  = bar_idx
                exit_price = sl
                break
            if lo <= tp1 and not tp1_hit:
                tp1_hit = True; tp1_bar = bar_idx
            if lo <= tp2 and tp1_hit and not tp2_hit:
                tp2_hit = True; tp2_bar = bar_idx
            if lo <= tp3 and tp2_hit and not tp3_hit:
                tp3_hit = True; tp3_bar = bar_idx
                exit_price = tp3; exit_bar = bar_idx; break
            if hi >= sl and tp1_hit and not tp2_hit:
                sl_hit = True; sl_bar = bar_idx
                exit_price = sl; exit_bar = bar_idx; break

    # Determine outcome label + P&L (assuming equal 1/3 splits at each TP)
    # TP1=50%, TP2=30%, TP3=20% (per bot's exit plan → approximate with 1/3 each for RR calc)
    pnl_r = 0.0   # in units of R (risk)

    if sl_hit and not tp1_hit:
        outcome = 'SL'
        pnl_r   = -1.0
    elif tp3_hit:
        outcome = 'TP3'
        pnl_r   = (1.5 + 2.5 + 4.0) / 3   # avg of 3 TPs
    elif tp2_hit:
        outcome = 'TP2'
        pnl_r   = (1.5 + 2.5) / 2 - 0.15  # partial exit + trail SL near BE
    elif tp1_hit and sl_hit:
        outcome = 'TP1+SL'
        pnl_r   = (1.5 - 1.0) / 2          # split: half won at TP1, half lost at SL
    elif tp1_hit:
        outcome = 'TP1'
        pnl_r   = 1.5 / 2                   # only half closed at TP1 → trailing
    else:
        outcome = 'TIMEOUT'
        pnl_r   = (exit_price - entry) / (abs(entry - sl)) * (1 if direction == 'LONG' else -1)

    bars_held = exit_bar + 1

    return {
        'outcome':    outcome,
        'pnl_r':      round(pnl_r, 3),
        'bars_held':  bars_held,
        'tp1_hit':    tp1_hit,
        'tp2_hit':    tp2_hit,
        'tp3_hit':    tp3_hit,
        'sl_hit':     sl_hit,
        'tp1_bar':    tp1_bar,
        'tp2_bar':    tp2_bar,
        'tp3_bar':    tp3_bar,
        'sl_bar':     sl_bar,
        'exit_price': round(exit_price, 8),
    }


# ══════════════════════════════════════════════════════════════
#  DATA FETCHER
# ══════════════════════════════════════════════════════════════

async def fetch_full_history(exchange, symbol, days=LOOKBACK_DAYS):
    """
    Fetch full OHLCV for 4H / 1H / 15M going back `days` days.
    Returns dict of DataFrames or None on failure.
    """
    since = int((datetime.utcnow() - timedelta(days=days + 5)).timestamp() * 1000)
    result = {}
    try:
        for tf, limit in [('4h', 500), ('1h', days * 24 + 100), ('15m', days * 96 + 200)]:
            all_ohlcv = []
            fetch_since = since
            while True:
                batch = await exchange.fetch_ohlcv(symbol, tf, since=fetch_since, limit=1000)
                if not batch:
                    break
                all_ohlcv += batch
                if len(batch) < 1000:
                    break
                fetch_since = batch[-1][0] + 1
                await asyncio.sleep(0.05)

            df = pd.DataFrame(all_ohlcv, columns=['ts','open','high','low','close','volume'])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            df = df.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
            result[tf] = df
            await asyncio.sleep(0.1)

        logger.info(f"  {symbol}: 4H={len(result['4h'])} 1H={len(result['1h'])} 15M={len(result['15m'])}")
        return result
    except Exception as e:
        logger.error(f"fetch_full_history {symbol}: {e}")
        return None


def align_slice(df_full, ts_1h, tf, n_bars):
    """Return the last n_bars of tf-data up to (and including) the candle at ts_1h."""
    # Find the closest candle at or before ts_1h
    mask = df_full['ts'] <= ts_1h
    sub  = df_full[mask].tail(n_bars).reset_index(drop=True)
    return sub


# ══════════════════════════════════════════════════════════════
#  WALK-FORWARD ENGINE
# ══════════════════════════════════════════════════════════════

async def backtest_symbol(exchange, symbol, days=LOOKBACK_DAYS):
    """
    Walk forward through 1H candles for one symbol.
    Returns list of trade result dicts.
    """
    logger.info(f"📊 Backtesting {symbol} ({days}d)...")
    data = await fetch_full_history(exchange, symbol, days)
    if not data:
        return []

    df4  = data['4h']
    df1  = data['1h']
    df15 = data['15m']

    # Warm up — need enough bars for indicators
    warm_1h  = WARM_UP_BARS_1H
    warm_4h  = 60
    warm_15m = 60

    trades = []
    last_signal_time = {}   # symbol→datetime  for dedup

    total_steps = len(df1) - warm_1h - MAX_TRADE_BARS
    if total_steps <= 0:
        logger.warning(f"  {symbol}: not enough 1H data ({len(df1)} bars)")
        return []

    for step in range(0, total_steps, WALK_STEP):
        bar_idx = warm_1h + step
        ts_now  = df1['ts'].iloc[bar_idx]

        # Dedup: skip if signal already fired recently for this symbol
        last = last_signal_time.get(symbol)
        if last and (ts_now - last).total_seconds() / 3600 < DEDUPE_HOURS:
            continue

        # Slice data up to current bar (no lookahead)
        slice_1h  = df1.iloc[:bar_idx + 1].copy()
        slice_4h  = align_slice(df4,  ts_now, '4h', warm_4h + 50)
        slice_15m = align_slice(df15, ts_now, '15m', warm_15m + 50)

        if len(slice_4h) < warm_4h or len(slice_15m) < warm_15m:
            continue

        # Add indicators to slices
        slice_1h  = add_indicators(slice_1h)
        slice_4h  = add_indicators(slice_4h)
        slice_15m = add_indicators(slice_15m)

        sig = analyse_slice(slice_4h, slice_1h, slice_15m, symbol)
        if sig is None:
            continue

        # Signal fired — resolve on future candles
        future_start = bar_idx + 1
        future_end   = min(future_start + MAX_TRADE_BARS, len(df1))
        future_df    = df1.iloc[future_start:future_end].reset_index(drop=True)

        if len(future_df) < 3:
            continue

        result = resolve_trade(sig, future_df)

        trade = {
            **sig,
            **result,
            'entry_time': ts_now.strftime('%Y-%m-%d %H:%M'),
            'symbol_clean': symbol.replace('/USDT:USDT', ''),
        }
        trades.append(trade)
        last_signal_time[symbol] = ts_now

        logger.info(f"  ✅ {symbol} {sig['bias']} {sig['quality']} sc={sig['score']} → {result['outcome']} {result['pnl_r']:+.2f}R")

    return trades


# ══════════════════════════════════════════════════════════════
#  STATS REPORT
# ══════════════════════════════════════════════════════════════

def print_report(all_trades):
    if not all_trades:
        print("\n❌ No trades found in backtest period.")
        return

    df = pd.DataFrame(all_trades)

    total   = len(df)
    wins    = len(df[df['pnl_r'] > 0])
    losses  = len(df[df['pnl_r'] <= 0])
    wr      = wins / total * 100
    total_r = df['pnl_r'].sum()
    avg_r   = df['pnl_r'].mean()
    avg_r_w = df[df['pnl_r'] > 0]['pnl_r'].mean() if wins else 0
    avg_r_l = df[df['pnl_r'] <= 0]['pnl_r'].mean() if losses else 0
    best    = df['pnl_r'].max()
    worst   = df['pnl_r'].min()

    # Outcome breakdown
    oc = df['outcome'].value_counts()

    # By quality
    by_q = df.groupby('quality').agg(
        trades=('pnl_r','count'),
        wr=('pnl_r', lambda x: (x > 0).mean() * 100),
        avg_r=('pnl_r','mean'),
        total_r=('pnl_r','sum')
    ).round(2)

    # By direction
    by_d = df.groupby('bias').agg(
        trades=('pnl_r','count'),
        wr=('pnl_r', lambda x: (x > 0).mean() * 100),
        avg_r=('pnl_r','mean')
    ).round(2)

    # By score bucket
    df['score_bucket'] = pd.cut(df['score'], bins=[74,79,84,89,94,100],
                                labels=['75-79','80-84','85-89','90-94','95-100'])
    by_score = df.groupby('score_bucket').agg(
        trades=('pnl_r','count'),
        wr=('pnl_r', lambda x: (x > 0).mean() * 100),
        avg_r=('pnl_r','mean')
    ).round(2)

    # Equity curve (running P&L)
    df = df.sort_values('entry_time').reset_index(drop=True)
    df['cumulative_r'] = df['pnl_r'].cumsum()
    max_dd = 0
    peak   = 0
    for r in df['cumulative_r']:
        if r > peak: peak = r
        dd = peak - r
        if dd > max_dd: max_dd = dd

    sep = "─" * 52

    print(f"\n{'═'*52}")
    print(f"   SMC PRO v4.0 — BACKTEST RESULTS")
    print(f"   {LOOKBACK_DAYS}d lookback | MIN_SCORE={MIN_SCORE} | {len(df['symbol_clean'].unique())} pairs")
    print(f"{'═'*52}")
    print(f"\n📊 OVERVIEW")
    print(sep)
    print(f"  Total trades      : {total}")
    print(f"  Wins / Losses     : {wins} / {losses}")
    print(f"  Win Rate          : {wr:.1f}%")
    print(f"  Total P&L (R)     : {total_r:+.2f}R")
    print(f"  Avg P&L / trade   : {avg_r:+.3f}R")
    print(f"  Avg Win           : {avg_r_w:+.3f}R")
    print(f"  Avg Loss          : {avg_r_l:+.3f}R")
    print(f"  Best trade        : {best:+.3f}R")
    print(f"  Worst trade       : {worst:+.3f}R")
    print(f"  Max Drawdown      : -{max_dd:.2f}R")

    print(f"\n📋 OUTCOME BREAKDOWN")
    print(sep)
    for o, c in oc.items():
        pct = c / total * 100
        print(f"  {o:<12} : {c:>4} ({pct:>5.1f}%)")

    print(f"\n🏆 BY QUALITY")
    print(sep)
    print(by_q.to_string())

    print(f"\n📍 BY DIRECTION")
    print(sep)
    print(by_d.to_string())

    print(f"\n📐 BY SCORE BUCKET")
    print(sep)
    print(by_score.to_string())

    # Top symbols
    by_sym = df.groupby('symbol_clean').agg(
        trades=('pnl_r','count'),
        wr=('pnl_r', lambda x: (x>0).mean()*100),
        total_r=('pnl_r','sum')
    ).sort_values('total_r', ascending=False)
    print(f"\n📈 TOP SYMBOLS (by total R)")
    print(sep)
    print(by_sym.head(10).round(2).to_string())
    print(f"\n📉 BOTTOM SYMBOLS (by total R)")
    print(sep)
    print(by_sym.tail(5).round(2).to_string())

    print(f"\n{'═'*52}\n")

    return df


def save_csv(all_trades, path):
    if not all_trades:
        return
    cols = [
        'entry_time','symbol_clean','bias','quality','score',
        'entry','sl','tp1','tp2','tp3','risk_pct',
        'outcome','pnl_r','bars_held',
        'tp1_hit','tp2_hit','tp3_hit','sl_hit',
        'pd_zone','hh_ll','structure','reasons'
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_trades)
    print(f"💾 Trades saved → {path}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

# ─── CONFIGURE YOUR BACKTEST HERE ───────────────────────────
SYMBOLS = [
    'BTC/USDT:USDT',
    'ETH/USDT:USDT',
    'SOL/USDT:USDT',
    'BNB/USDT:USDT',
    'XRP/USDT:USDT',
    'DOGE/USDT:USDT',
    'AVAX/USDT:USDT',
    'LINK/USDT:USDT',
    'ARB/USDT:USDT',
    'OP/USDT:USDT',
    'SUI/USDT:USDT',
    'TIA/USDT:USDT',
    'INJ/USDT:USDT',
    'WLD/USDT:USDT',
    'APT/USDT:USDT',
]
# ────────────────────────────────────────────────────────────


async def main():
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })

    print(f"\n🚀 SMC PRO v4.0 BACKTESTER")
    print(f"   Pairs     : {len(SYMBOLS)}")
    print(f"   Lookback  : {LOOKBACK_DAYS} days")
    print(f"   Min Score : {MIN_SCORE}")
    print(f"   Max Hold  : {MAX_TRADE_BARS}H")
    print(f"   Dedup     : {DEDUPE_HOURS}H\n")

    all_trades = []

    for symbol in SYMBOLS:
        try:
            trades = await backtest_symbol(exchange, symbol, LOOKBACK_DAYS)
            all_trades.extend(trades)
            print(f"  ✅ {symbol.replace('/USDT:USDT','')} — {len(trades)} trades found")
            await asyncio.sleep(1.0)   # be nice to the exchange
        except Exception as e:
            logger.error(f"  ❌ {symbol}: {e}")

    await exchange.close()

    print(f"\n📦 Total raw signals: {len(all_trades)}")
    df = print_report(all_trades)

    csv_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV)
    save_csv(all_trades, csv_path)

    if df is not None:
        # Also save equity curve
        eq_path = os.path.join(OUTPUT_DIR, "equity_curve.csv")
        df[['entry_time','symbol_clean','bias','score','outcome','pnl_r','cumulative_r']].to_csv(eq_path, index=False)
        print(f"📈 Equity curve → {eq_path}")


if __name__ == "__main__":
    asyncio.run(main())
