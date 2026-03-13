"""
╔══════════════════════════════════════════════════════════════════════════╗
║        SMC PRO v4.0 — FULL HISTORICAL BACKTESTER                        ║
║                                                                          ║
║  Mirrors your exact bot logic (OB, BOS/MSS, score, gates)               ║
║  Walk-forward over 4H / 1H / 15M — no lookahead bias                   ║
║  Outputs: CSV trades + detailed HTML report + console summary            ║
╚══════════════════════════════════════════════════════════════════════════╝

HOW IT WORKS
────────────
1. Downloads historical OHLCV for every USDT pair (or a curated list)
2. Walks forward in 1H steps (simulating your 30-min scan)
3. At each step it slices data UP TO that bar — zero lookahead
4. Runs the same analyse() logic from your live bot
5. When a signal fires it looks FORWARD in the 1H data to see
   which of TP1 / TP2 / TP3 / SL is hit first (realistic order)
6. Records every trade + all debug fields
7. Generates an HTML report with every metric you need to tune

SETUP (run once)
────────────────
  pip install ccxt pandas numpy ta

USAGE
─────
  python smc_backtester.py

CONFIG SECTION is at the top — edit PAIRS, DATE_FROM, DATE_TO, etc.
"""

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import ta
import warnings
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

# ─── try importing optional pretty libs ───────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track as rich_track
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ══════════════════════════════════════════════════════════════════════════
#  ██████╗ ██████╗ ███╗   ██╗███████╗██╗ ██████╗
#  ██╔════╝██╔═══██╗████╗  ██║██╔════╝██║██╔════╝
#  ██║     ██║   ██║██╔██╗ ██║█████╗  ██║██║  ███╗
#  ██║     ██║   ██║██║╚██╗██║██╔══╝  ██║██║   ██║
#  ╚██████╗╚██████╔╝██║ ╚████║██║     ██║╚██████╔╝
#   ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝
# ══════════════════════════════════════════════════════════════════════════

# ── Date range ────────────────────────────────────────────────────────────
DATE_FROM = "2024-01-01"   # start of backtest
DATE_TO   = "2025-03-01"   # end   of backtest

# ── Pairs to test (None = auto-fetch top-volume perps) ────────────────────
# Set to None for full market scan, or specify a list like below
PAIRS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT",
    "ADA/USDT:USDT", "DOT/USDT:USDT",  "POL/USDT:USDT", "LTC/USDT:USDT",
    "UNI/USDT:USDT", "ATOM/USDT:USDT", "FIL/USDT:USDT",  "NEAR/USDT:USDT",
    "APT/USDT:USDT", "ARB/USDT:USDT",  "OP/USDT:USDT",   "SUI/USDT:USDT",
    "INJ/USDT:USDT", "TIA/USDT:USDT",  "WLD/USDT:USDT",  "JTO/USDT:USDT",
    "PYTH/USDT:USDT","ORDI/USDT:USDT", "STX/USDT:USDT",  "SEI/USDT:USDT",
    "FET/USDT:USDT", "RUNE/USDT:USDT",
]

# ── Bot settings (must match your live bot exactly) ────────────────────────
MIN_SCORE          = 75
OB_TOLERANCE_PCT   = 0.008
OB_IMPULSE_ATR_MULT= 1.0
STRUCTURE_LOOKBACK = 20
HH_LL_LOOKBACK     = 10
HH_LL_BONUS        = 8
MAX_SIGNALS_PER_SCAN = 6

# ── Backtester settings ────────────────────────────────────────────────────
WALK_STEP_BARS_1H  = 1          # walk forward 1 bar at a time (1H)
MIN_WARMUP_BARS    = 150        # bars needed before first signal allowed
MAX_TRADE_BARS     = 48         # 48 × 1H = 48H max trade duration
CACHE_DIR          = Path("./bt_cache")   # raw OHLCV cache folder
RESULTS_DIR        = Path("./bt_results") # outputs folder

# ── Exchange (public data only — no key needed) ────────────────────────────
EXCHANGE_ID        = "binance"
EXCHANGE_TYPE      = "future"   # perps

# ══════════════════════════════════════════════════════════════════════════
#  INDICATORS  (identical to your live bot)
# ══════════════════════════════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
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

        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close']).average_true_range()

        bb = ta.volatility.BollingerBands(df['close'], 20, 2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_pband'] = bb.bollinger_pband()

        adx_i = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx']    = adx_i.adx()
        df['di_pos'] = adx_i.adx_pos()
        df['di_neg'] = adx_i.adx_neg()

        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
            df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
        df['mfi'] = ta.volume.MFIIndicator(
            df['high'], df['low'], df['close'], df['volume']).money_flow_index()

        df['vol_sma']   = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_sma'].replace(0, np.nan)

        tp_col = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (tp_col * df['volume']).cumsum() / df['volume'].cumsum()

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

        df['bull_pin']     = ((lw > body*2.5) & (lw > uw*2)   & (df['close'] > df['open'])).astype(int)
        df['bear_pin']     = ((uw > body*2.5) & (uw > lw*2)   & (df['close'] < df['open'])).astype(int)
        df['hammer']       = ((lw > body*2.0) & (lw > uw*1.5)).astype(int)
        df['shooting_star']= ((uw > body*2.0) & (uw > lw*1.5)).astype(int)

    except Exception as e:
        pass
    return df


# ══════════════════════════════════════════════════════════════════════════
#  SMC ENGINE  (identical to your live bot)
# ══════════════════════════════════════════════════════════════════════════

class SMCEngine:

    def swing_highs_lows(self, df, left=4, right=4):
        highs, lows = [], []
        n = len(df)
        for i in range(left, n - right):
            hi = df['high'].iloc[i]; lo = df['low'].iloc[i]
            if all(hi >= df['high'].iloc[i-left:i]) and all(hi >= df['high'].iloc[i+1:i+right+1]):
                highs.append({'i': i, 'price': hi})
            if all(lo <= df['low'].iloc[i-left:i]) and all(lo <= df['low'].iloc[i+1:i+right+1]):
                lows.append({'i': i, 'price': lo})
        return highs, lows

    def check_4h_hh_ll(self, df_4h, direction, lookback=HH_LL_LOOKBACK):
        n = len(df_4h)
        if n < lookback * 2:
            return False, "Not enough 4H data"
        recent = df_4h.iloc[-lookback:]
        prior  = df_4h.iloc[-(lookback*2):-lookback]
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
                    events.append({'kind': kind, 'level': level, 'bar': j}); break

        for k in range(1, len(lows)):
            pl = lows[k-1]; cl = lows[k]
            if cl['i'] < start: continue
            level = pl['price']
            for j in range(cl['i'], min(cl['i'] + 10, n)):
                if close.iloc[j] < level:
                    kind = 'BOS_BEAR' if cl['price'] < pl['price'] else 'MSS_BEAR'
                    events.append({'kind': kind, 'level': level, 'bar': j}); break

        if not events: return None
        latest = sorted(events, key=lambda x: x['bar'])[-1]
        if latest['bar'] < n - lookback: return None
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

    def price_in_ob(self, price, ob, tol=OB_TOLERANCE_PCT):
        t = ob['top'] * tol
        return (ob['bottom'] - t) <= price <= (ob['top'] + t)

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


# ══════════════════════════════════════════════════════════════════════════
#  SCORER  (identical to your live bot)
# ══════════════════════════════════════════════════════════════════════════

def score_setup(direction, ob, structure, sweep, fvg_near,
                df_1h, df_15m, df_4h, pd_label, hh_ll_confirmed):
    score = 0
    reasons = []
    failed  = []

    l1  = df_1h.iloc[-1]
    p1  = df_1h.iloc[-2]
    l15 = df_15m.iloc[-1]
    l4  = df_4h.iloc[-1]

    # 1. Structure (20)
    if structure:
        if 'MSS' in structure['kind']:
            score += 20; reasons.append(f"MSS({structure['kind']})")
        else:
            score += 14; reasons.append(f"BOS({structure['kind']})")
    else:
        failed.append("No BOS/MSS")

    # 2. OB quality (20)
    if ob:
        ob_size_pct = (ob['top'] - ob['bottom']) / ob['bottom'] * 100
        if ob_size_pct < 0.8:
            score += 20; reasons.append(f"TightOB({ob_size_pct:.2f}%)")
        elif ob_size_pct < 2.0:
            score += 13; reasons.append(f"OB({ob_size_pct:.2f}%)")
        else:
            score += 7;  reasons.append(f"WideOB({ob_size_pct:.2f}%)")
    else:
        failed.append("No OB")

    # 3. 4H Trend (15)
    e21 = l4.get('ema_21', 0); e50 = l4.get('ema_50', 0); e200 = l4.get('ema_200', 0)
    if direction == 'LONG':
        if e21 > e50 > e200:   score += 15; reasons.append("4H_TripleBull")
        elif e21 > e50:         score += 10; reasons.append("4H_Bull21>50")
        elif pd_label == 'DISCOUNT': score += 6; reasons.append("4H_Discount")
        else: failed.append("4H_WeakLong")
    else:
        if e21 < e50 < e200:   score += 15; reasons.append("4H_TripleBear")
        elif e21 < e50:         score += 10; reasons.append("4H_Bear21<50")
        elif pd_label == 'PREMIUM': score += 6; reasons.append("4H_Premium")
        else: failed.append("4H_WeakShort")

    # 4. HH/LL bonus (8)
    if hh_ll_confirmed:
        score += HH_LL_BONUS; reasons.append("HH_LL_OK")
    else:
        failed.append("No_HH_LL")

    # 5. 1H Trigger (25)
    trigger = False; trigger_label = ""; trigger_pts = 0
    if direction == 'LONG':
        if l1.get('bull_engulf', 0):  pts=25; lbl="1H_BullEngulf_cur"
        elif l1.get('bull_pin',0):     pts=22; lbl="1H_BullPin_cur"
        elif l1.get('hammer',0):       pts=18; lbl="1H_Hammer_cur"
        elif p1.get('bull_engulf',0):  pts=14; lbl="1H_BullEngulf_prev"
        elif p1.get('bull_pin',0):     pts=11; lbl="1H_BullPin_prev"
        elif p1.get('hammer',0):       pts=9;  lbl="1H_Hammer_prev"
        else: pts=0; lbl=""
    else:
        if l1.get('bear_engulf', 0):   pts=25; lbl="1H_BearEngulf_cur"
        elif l1.get('bear_pin',0):      pts=22; lbl="1H_BearPin_cur"
        elif l1.get('shooting_star',0): pts=18; lbl="1H_ShootingStar_cur"
        elif p1.get('bear_engulf',0):   pts=14; lbl="1H_BearEngulf_prev"
        elif p1.get('bear_pin',0):      pts=11; lbl="1H_BearPin_prev"
        elif p1.get('shooting_star',0): pts=9;  lbl="1H_ShootStar_prev"
        else: pts=0; lbl=""

    if pts > 0:
        score += pts; trigger = True; trigger_label = lbl; trigger_pts = pts
        reasons.append(lbl)
    else:
        score -= 12; failed.append("No_1H_Trigger")

    # 6. Momentum (12)
    rsi1  = l1.get('rsi', 50)
    macd1 = l1.get('macd', 0);  ms1  = l1.get('macd_signal', 0)
    pm1   = p1.get('macd', 0);  pms1 = p1.get('macd_signal', 0)
    sk1   = l1.get('srsi_k', 0.5); sd1 = l1.get('srsi_d', 0.5)
    if direction == 'LONG':
        if 28 <= rsi1 <= 55:     score += 4; reasons.append(f"RSI_reset({rsi1:.0f})")
        elif rsi1 < 28:           score += 3; reasons.append(f"RSI_OS({rsi1:.0f})")
        if macd1 > ms1 and pm1 <= pms1: score += 5; reasons.append("MACD_BCross")
        elif macd1 > ms1:                score += 2; reasons.append("MACD_Bull")
        if sk1 < 0.3 and sk1 > sd1:     score += 3; reasons.append("Stoch_BCross")
    else:
        if 45 <= rsi1 <= 72:     score += 4; reasons.append(f"RSI_OB_zone({rsi1:.0f})")
        elif rsi1 > 72:           score += 3; reasons.append(f"RSI_OB({rsi1:.0f})")
        if macd1 < ms1 and pm1 >= pms1: score += 5; reasons.append("MACD_BearCross")
        elif macd1 < ms1:                score += 2; reasons.append("MACD_Bear")
        if sk1 > 0.7 and sk1 < sd1:     score += 3; reasons.append("Stoch_BearCross")

    # 7. Extras (10 max)
    extras = 0
    if sweep:  extras += 4; reasons.append("Sweep")
    if fvg_near: extras += 3; reasons.append("FVG_OB_overlap")
    vr15 = l15.get('vol_ratio', 1.0)
    if   vr15 >= 2.5: extras += 3; reasons.append(f"15M_VolSpike({vr15:.1f}x)")
    elif vr15 >= 1.5: extras += 1; reasons.append(f"15M_Vol({vr15:.1f}x)")
    close1 = l1.get('close', 0); vwap1 = l1.get('vwap', 0)
    if direction == 'LONG' and close1 < vwap1:   extras += 1; reasons.append("BelowVWAP")
    elif direction == 'SHORT' and close1 > vwap1: extras += 1; reasons.append("AboveVWAP")
    score += min(extras, 10)

    return max(0, min(int(score), 100)), reasons, failed, trigger_label, trigger_pts


# ══════════════════════════════════════════════════════════════════════════
#  ANALYSE  (identical logic to your live bot, returns richer debug dict)
# ══════════════════════════════════════════════════════════════════════════

_smc = SMCEngine()

def analyse_slice(data: dict, symbol: str):
    """
    data = {'4h': df, '1h': df, '15m': df}  — sliced up to current bar
    Returns (signal_dict | None, debug_dict)
    """
    debug = {'symbol': symbol, 'score': 0, 'bias': '?', 'reject_reason': '', 'gates': []}

    try:
        df4  = data['4h']
        df1  = data['1h']
        df15 = data['15m']

        if len(df1) < MIN_WARMUP_BARS or len(df15) < 40 or len(df4) < 55:
            debug['reject_reason'] = 'not_enough_data'
            return None, debug

        price = df1['close'].iloc[-1]

        # Gate 1: 4H bias
        l4  = df4.iloc[-1]
        e21 = l4.get('ema_21', 0); e50 = l4.get('ema_50', 0)
        if e21 > e50:      bias = 'LONG'
        elif e21 < e50:    bias = 'SHORT'
        else:
            debug['reject_reason'] = 'flat_4h_ema'
            return None, debug
        debug['bias'] = bias

        hh_ll_ok, _ = _smc.check_4h_hh_ll(df4, bias, HH_LL_LOOKBACK)

        # Gate 2: PD zone
        pd_label, pd_pos = _smc.pd_zone(df4, price)
        if bias == 'LONG' and pd_label == 'PREMIUM':
            debug['reject_reason'] = 'pd_premium_no_long'
            return None, debug
        if bias == 'SHORT' and pd_label == 'DISCOUNT':
            debug['reject_reason'] = 'pd_discount_no_short'
            return None, debug

        # Gate 3: 1H Structure
        highs1, lows1 = _smc.swing_highs_lows(df1, 4, 4)
        structure = _smc.detect_structure_break(df1, highs1, lows1, STRUCTURE_LOOKBACK)
        if structure:
            if bias == 'LONG' and 'BEAR' in structure['kind']:
                debug['reject_reason'] = 'structure_opposes_long'
                return None, debug
            if bias == 'SHORT' and 'BULL' in structure['kind']:
                debug['reject_reason'] = 'structure_opposes_short'
                return None, debug

        # Gate 4: 1H OB
        obs = _smc.find_order_blocks(df1, bias, lookback=60)
        if not obs:
            debug['reject_reason'] = 'no_ob_found'
            return None, debug

        active_ob = None
        for ob in obs:
            if _smc.price_in_ob(price, ob, OB_TOLERANCE_PCT):
                active_ob = ob; break
        if not active_ob:
            debug['reject_reason'] = 'price_not_at_ob'
            return None, debug

        # FVG + Sweep (bonuses)
        fvgs = _smc.find_fvg(df1, bias, lookback=25)
        fvg_near = None
        for fvg in fvgs:
            if fvg['bottom'] < active_ob['top'] and fvg['top'] > active_ob['bottom']:
                fvg_near = fvg; break
        sweep = _smc.recent_liquidity_sweep(df1, bias, highs1, lows1, lookback=20)

        # Score
        score, reasons, failed, trigger_label, trigger_pts = score_setup(
            bias, active_ob, structure, sweep, fvg_near,
            df1, df15, df4, pd_label, hh_ll_ok
        )
        debug['score'] = score
        debug['bias']  = bias

        if score < MIN_SCORE:
            debug['reject_reason'] = f'score_{score}_below_{MIN_SCORE}'
            return None, debug

        if   score >= 92: quality = 'ELITE'
        elif score >= 85: quality = 'PREMIUM'
        else:             quality = 'HIGH'

        atr1  = df1['atr'].iloc[-1]
        entry = price

        if bias == 'LONG':
            sl  = min(active_ob['bottom'] - atr1 * 0.2, entry - atr1 * 0.6)
        else:
            sl  = max(active_ob['top'] + atr1 * 0.2, entry + atr1 * 0.6)

        risk = abs(entry - sl)
        if risk < entry * 0.001:
            debug['reject_reason'] = 'degenerate_sl'
            return None, debug

        if bias == 'LONG':
            tps = [entry + risk*1.5, entry + risk*2.5, entry + risk*4.0]
        else:
            tps = [entry - risk*1.5, entry - risk*2.5, entry - risk*4.0]

        # Extra metadata for backtest analysis
        ob_size_pct = (active_ob['top'] - active_ob['bottom']) / active_ob['bottom'] * 100
        adx_val     = df1.iloc[-1].get('adx', 0)
        rsi_val     = df1.iloc[-1].get('rsi', 50)
        vol_ratio   = df1.iloc[-1].get('vol_ratio', 1.0)

        sig = {
            'symbol':        symbol.replace('/USDT:USDT',''),
            'full_symbol':   symbol,
            'signal':        bias,
            'quality':       quality,
            'score':         score,
            'hh_ll':         hh_ll_ok,
            'entry':         entry,
            'stop_loss':     sl,
            'tp1':           tps[0],
            'tp2':           tps[1],
            'tp3':           tps[2],
            'risk_pct':      risk / entry * 100,
            'rr1':           1.5,
            'rr2':           2.5,
            'rr3':           4.0,
            'pd_zone':       pd_label,
            'pd_pos':        round(pd_pos, 3),
            'structure_kind':structure['kind'] if structure else 'none',
            'has_sweep':     sweep is not None,
            'has_fvg':       fvg_near is not None,
            'trigger':       trigger_label,
            'trigger_pts':   trigger_pts,
            'ob_size_pct':   round(ob_size_pct, 3),
            'adx':           round(adx_val, 1) if not pd.isna(adx_val) else 0,
            'rsi':           round(rsi_val, 1) if not pd.isna(rsi_val) else 50,
            'vol_ratio_1h':  round(vol_ratio, 2) if not pd.isna(vol_ratio) else 1.0,
            'reasons':       '|'.join(reasons),
            'failed':        '|'.join(failed),
            'bar_time':      df1.iloc[-1]['ts'],
            # placeholders — filled by walk-forward simulator
            'outcome':       None,   # TP1/TP2/TP3/SL/TIMEOUT
            'bars_to_exit':  None,
            'pnl_rr':        None,
            'exit_price':    None,
        }
        return sig, debug

    except Exception as e:
        debug['reject_reason'] = f'exception:{e}'
        return None, debug


# ══════════════════════════════════════════════════════════════════════════
#  DATA  DOWNLOADER  (with local cache)
# ══════════════════════════════════════════════════════════════════════════

async def download_ohlcv(exchange, symbol: str, tf: str,
                         since_ms: int, until_ms: int) -> pd.DataFrame:
    cache_key = symbol.replace('/', '_').replace(':', '_')
    cache_path = CACHE_DIR / f"{cache_key}_{tf}.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=['ts'])
        if 'ts' in df.columns and not df.empty:
            # ensure tz-aware
            if df['ts'].dt.tz is None:
                df['ts'] = df['ts'].dt.tz_localize('UTC')
            need_more = df['ts'].iloc[-1] < pd.Timestamp(until_ms, unit='ms', tz='UTC') - pd.Timedelta(hours=4)
            if not need_more:
                mask = (df['ts'] >= pd.Timestamp(since_ms, unit='ms', tz='UTC')) & \
                       (df['ts'] <= pd.Timestamp(until_ms, unit='ms', tz='UTC'))
                return df[mask].reset_index(drop=True)

    all_rows = []
    cur = since_ms
    limit = 1000
    while cur < until_ms:
        try:
            raw = await exchange.fetch_ohlcv(symbol, tf, since=cur, limit=limit)
            if not raw: break
            all_rows.extend(raw)
            cur = raw[-1][0] + 1
            await asyncio.sleep(0.12)
            if raw[-1][0] >= until_ms: break
        except Exception as e:
            print(f"  ⚠  {symbol} {tf}: {e}")
            await asyncio.sleep(2)
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df.drop_duplicates('ts', inplace=True)
    df.sort_values('ts', inplace=True)
    df.reset_index(drop=True, inplace=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


# ══════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD SIMULATOR
# ══════════════════════════════════════════════════════════════════════════

def simulate_trade_outcome(signal: dict, df1_future: pd.DataFrame) -> dict:
    """
    Given a signal and the 1H bars AFTER entry, walk forward and find
    the first of: TP1 / TP2 / TP3 / SL to be touched (high/low).
    Returns updated signal dict with outcome fields filled.
    """
    s   = signal
    sig = s['signal']

    for i, row in df1_future.iterrows():
        h, l = row['high'], row['low']
        bars = i + 1

        if sig == 'LONG':
            # SL check first (conservative — if same bar both hit, SL wins)
            if l <= s['stop_loss']:
                s['outcome']      = 'SL'
                s['bars_to_exit'] = bars
                s['exit_price']   = s['stop_loss']
                s['pnl_rr']       = -1.0
                return s
            if h >= s['tp3']:
                s['outcome'] = 'TP3'; s['exit_price'] = s['tp3']; s['pnl_rr'] = 4.0
            elif h >= s['tp2']:
                s['outcome'] = 'TP2'; s['exit_price'] = s['tp2']; s['pnl_rr'] = 2.5
            elif h >= s['tp1']:
                s['outcome'] = 'TP1'; s['exit_price'] = s['tp1']; s['pnl_rr'] = 1.5
        else:
            if h >= s['stop_loss']:
                s['outcome']      = 'SL'
                s['bars_to_exit'] = bars
                s['exit_price']   = s['stop_loss']
                s['pnl_rr']       = -1.0
                return s
            if l <= s['tp3']:
                s['outcome'] = 'TP3'; s['exit_price'] = s['tp3']; s['pnl_rr'] = 4.0
            elif l <= s['tp2']:
                s['outcome'] = 'TP2'; s['exit_price'] = s['tp2']; s['pnl_rr'] = 2.5
            elif l <= s['tp1']:
                s['outcome'] = 'TP1'; s['exit_price'] = s['tp1']; s['pnl_rr'] = 1.5

        if s['outcome'] not in (None, 'SL'):
            s['bars_to_exit'] = bars
            return s

    # timed out
    last_price        = df1_future.iloc[-1]['close'] if not df1_future.empty else s['entry']
    s['outcome']      = 'TIMEOUT'
    s['bars_to_exit'] = len(df1_future)
    s['exit_price']   = last_price
    pnl               = (last_price - s['entry']) / s['entry'] * 100
    s['pnl_rr']       = (last_price - s['entry']) / abs(s['entry'] - s['stop_loss']) \
                        if sig == 'LONG' else \
                        (s['entry'] - last_price) / abs(s['stop_loss'] - s['entry'])
    return s


def resample_to_tf(df_1h: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample 1H data to 4H or 15M approximately for slice alignment."""
    # This is used only during the walk-forward to create aligned 4H/15M slices
    # from the master 1H index.  In practice we keep separate downloaded DFs.
    return df_1h  # placeholder — actual slicing is done below


def walk_forward_pair(symbol: str, df4: pd.DataFrame,
                      df1: pd.DataFrame, df15: pd.DataFrame) -> list:
    """
    Walk forward bar by bar on 1H, apply analyse_slice(), simulate outcomes.
    Returns list of trade dicts.
    """
    trades = []
    n1     = len(df1)
    # Deduplicate bar times to avoid signal spam at same candle
    active_signal_bar = -99  # last bar index where a signal was fired

    for bar_i in range(MIN_WARMUP_BARS, n1 - MAX_TRADE_BARS - 1):
        # ── Slice data up to current bar (no lookahead) ────────────────
        df1_slice = add_indicators(df1.iloc[:bar_i+1].copy())
        cur_ts    = df1_slice.iloc[-1]['ts']

        # Align 4H slice: all 4H bars whose open time <= cur_ts
        df4_slice_raw = df4[df4['ts'] <= cur_ts]
        if len(df4_slice_raw) < 55:
            continue
        df4_slice = add_indicators(df4_slice_raw.copy())

        # Align 15M slice: last 80 bars up to cur_ts
        df15_slice_raw = df15[df15['ts'] <= cur_ts].iloc[-80:]
        if len(df15_slice_raw) < 40:
            continue
        df15_slice = add_indicators(df15_slice_raw.copy())

        data = {'4h': df4_slice, '1h': df1_slice, '15m': df15_slice}
        sig, debug = analyse_slice(data, symbol)

        if sig is None:
            continue

        # Deduplicate: don't fire on same bar twice
        if bar_i == active_signal_bar:
            continue
        active_signal_bar = bar_i

        # ── Simulate outcome on forward bars ──────────────────────────
        future_bars = df1.iloc[bar_i+1 : bar_i+1+MAX_TRADE_BARS].reset_index(drop=True)
        sig = simulate_trade_outcome(sig, future_bars)
        trades.append(sig)

    return trades


# ══════════════════════════════════════════════════════════════════════════
#  STATISTICS ENGINE
# ══════════════════════════════════════════════════════════════════════════

def compute_stats(trades: list) -> dict:
    if not trades:
        return {}

    df = pd.DataFrame(trades)

    total  = len(df)
    wins   = df['outcome'].isin(['TP1','TP2','TP3']).sum()
    losses = (df['outcome'] == 'SL').sum()
    timeouts = (df['outcome'] == 'TIMEOUT').sum()

    winrate = wins / total * 100 if total else 0
    avg_rr  = df['pnl_rr'].mean()
    avg_rr_wins   = df[df['pnl_rr'] > 0]['pnl_rr'].mean()
    avg_rr_losses = df[df['pnl_rr'] < 0]['pnl_rr'].mean()

    tp1_rate = (df['outcome']=='TP1').sum() / total * 100
    tp2_rate = (df['outcome']=='TP2').sum() / total * 100
    tp3_rate = (df['outcome']=='TP3').sum() / total * 100
    sl_rate  = (df['outcome']=='SL').sum()  / total * 100

    # expectancy (per trade, in RR units)
    expectancy = winrate/100 * avg_rr_wins + (1 - winrate/100) * (avg_rr_losses if not pd.isna(avg_rr_losses) else -1)

    # ── Break down by variable ─────────────────────────────────────────
    def breakdown(col, label):
        if col not in df.columns:
            return {}
        result = {}
        for val in df[col].unique():
            sub = df[df[col] == val]
            w   = sub['outcome'].isin(['TP1','TP2','TP3']).sum()
            result[str(val)] = {
                'count':   len(sub),
                'winrate': round(w/len(sub)*100, 1),
                'avg_rr':  round(sub['pnl_rr'].mean(), 2),
                'label':   label
            }
        return dict(sorted(result.items(), key=lambda x: -x[1]['winrate']))

    # Score buckets
    df['score_bucket'] = pd.cut(df['score'],
                                bins=[74,79,84,89,94,100],
                                labels=['75-79','80-84','85-89','90-94','95-100'])

    # ADX buckets
    df['adx_bucket'] = pd.cut(df['adx'],
                               bins=[0,20,30,40,60,200],
                               labels=['<20','20-30','30-40','40-60','>60'])

    # OB size buckets
    df['ob_bucket'] = pd.cut(df['ob_size_pct'],
                              bins=[0,0.5,0.8,1.5,2.0,10],
                              labels=['<0.5%','0.5-0.8%','0.8-1.5%','1.5-2%','>2%'])

    stats = {
        'total':      total,
        'wins':       int(wins),
        'losses':     int(losses),
        'timeouts':   int(timeouts),
        'winrate':    round(winrate, 1),
        'tp1_rate':   round(tp1_rate, 1),
        'tp2_rate':   round(tp2_rate, 1),
        'tp3_rate':   round(tp3_rate, 1),
        'sl_rate':    round(sl_rate, 1),
        'avg_rr':     round(avg_rr, 3),
        'avg_rr_wins':round(avg_rr_wins, 3)  if not pd.isna(avg_rr_wins) else 0,
        'avg_rr_loss':round(avg_rr_losses,3) if not pd.isna(avg_rr_losses) else 0,
        'expectancy': round(expectancy, 3),
        'avg_bars':   round(df['bars_to_exit'].mean(), 1),
        'by_quality': breakdown('quality', 'Quality'),
        'by_signal':  breakdown('signal',  'Direction'),
        'by_pd_zone': breakdown('pd_zone', 'PD Zone'),
        'by_structure': breakdown('structure_kind', 'Structure'),
        'by_trigger': breakdown('trigger', 'Trigger Candle'),
        'by_hh_ll':   breakdown('hh_ll',   'HH/LL'),
        'by_sweep':   breakdown('has_sweep','Liq Sweep'),
        'by_fvg':     breakdown('has_fvg',  'FVG Overlap'),
        'by_score':   breakdown('score_bucket','Score Bucket'),
        'by_adx':     breakdown('adx_bucket', 'ADX'),
        'by_ob_size': breakdown('ob_bucket',  'OB Size'),
        'by_symbol':  breakdown('symbol',     'Pair'),
    }
    return stats, df


# ══════════════════════════════════════════════════════════════════════════
#  HTML REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════

def generate_html_report(stats: dict, trades_df: pd.DataFrame,
                         date_from: str, date_to: str,
                         pairs_tested: list) -> str:

    def pct_color(val):
        if val >= 60: return '#00c853'
        if val >= 50: return '#ffd600'
        return '#ff1744'

    def rr_color(val):
        if val >= 1.0: return '#00c853'
        if val >= 0: return '#ffd600'
        return '#ff1744'

    def breakdown_table(title, d):
        if not d: return ''
        rows = ''
        for k, v in d.items():
            wr = v['winrate']
            rr = v['avg_rr']
            rows += f"""<tr>
              <td>{k}</td>
              <td>{v['count']}</td>
              <td style="color:{pct_color(wr)};font-weight:bold">{wr}%</td>
              <td style="color:{rr_color(rr)};font-weight:bold">{rr}</td>
            </tr>"""
        return f"""
        <div class="card">
          <h3>🔍 {title}</h3>
          <table>
            <tr><th>Value</th><th>Trades</th><th>Win%</th><th>Avg RR</th></tr>
            {rows}
          </table>
        </div>"""

    # Outcome chart data
    outcomes      = trades_df['outcome'].value_counts().to_dict()
    monthly       = trades_df.copy()
    monthly['month'] = pd.to_datetime(monthly['bar_time']).dt.to_period('M').astype(str)
    monthly_wins  = monthly[monthly['outcome'].isin(['TP1','TP2','TP3'])].groupby('month').size()
    monthly_total = monthly.groupby('month').size()
    monthly_wr    = (monthly_wins / monthly_total * 100).fillna(0).round(1)

    month_labels = list(monthly_wr.index)
    month_values = list(monthly_wr.values)

    # Trade list (last 50)
    recent = trades_df.sort_values('bar_time', ascending=False).head(50)
    trade_rows = ''
    for _, r in recent.iterrows():
        outcome_color = {'TP1':'#4caf50','TP2':'#00e676','TP3':'#1de9b6',
                         'SL':'#f44336','TIMEOUT':'#888'}.get(r['outcome'],'#888')
        trade_rows += f"""<tr>
          <td>{str(r['bar_time'])[:16]}</td>
          <td>{r['symbol']}</td>
          <td>{'🟢' if r['signal']=='LONG' else '🔴'} {r['signal']}</td>
          <td>{r['score']}</td>
          <td>{r['quality']}</td>
          <td>{r['pd_zone']}</td>
          <td>{r['trigger']}</td>
          <td style="color:{outcome_color};font-weight:bold">{r['outcome']}</td>
          <td style="color:{rr_color(r['pnl_rr'])}">{r['pnl_rr']:.2f}</td>
          <td>{r['bars_to_exit']}h</td>
        </tr>"""

    # What-to-adjust table
    suggestions = generate_suggestions(stats, trades_df)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SMC Pro v4.0 — Backtest Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0 }}
  body {{ background:#0d1117; color:#e6edf3; font-family:'Segoe UI',sans-serif; padding:24px }}
  h1 {{ color:#58a6ff; font-size:2rem; margin-bottom:4px }}
  h2 {{ color:#58a6ff; font-size:1.3rem; margin:32px 0 12px }}
  h3 {{ color:#8b949e; font-size:1rem; margin-bottom:12px }}
  .subtitle {{ color:#8b949e; margin-bottom:32px }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:16px; margin-bottom:32px }}
  .kpi {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:20px; text-align:center }}
  .kpi .val {{ font-size:2.2rem; font-weight:800; margin-bottom:4px }}
  .kpi .lbl {{ color:#8b949e; font-size:.8rem }}
  .card {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:20px; margin-bottom:20px }}
  .breakdowns {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:20px }}
  table {{ width:100%; border-collapse:collapse; font-size:.85rem }}
  th {{ background:#21262d; color:#58a6ff; padding:8px 12px; text-align:left }}
  td {{ padding:7px 12px; border-bottom:1px solid #21262d }}
  tr:hover td {{ background:#21262d }}
  .chart-wrap {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:20px; margin-bottom:20px }}
  .suggest {{ background:#0d2137; border-left:4px solid #58a6ff; border-radius:6px; padding:14px 18px; margin-bottom:12px }}
  .suggest .tag {{ color:#58a6ff; font-weight:bold; font-size:.8rem }}
  .suggest .msg {{ margin-top:4px; color:#e6edf3 }}
  .suggest .action {{ margin-top:6px; color:#3fb950; font-size:.85rem }}
  .green {{ color:#3fb950 }} .red {{ color:#f85149 }} .yellow {{ color:#d29922 }}
  .pill {{ display:inline-block; padding:2px 8px; border-radius:12px; font-size:.75rem }}
  footer {{ color:#444; text-align:center; margin-top:40px; font-size:.8rem }}
</style>
</head>
<body>
<h1>🏦 SMC Pro v4.0 — Backtest Report</h1>
<p class="subtitle">
  📅 {date_from} → {date_to} &nbsp;|&nbsp;
  🪙 {len(pairs_tested)} pairs &nbsp;|&nbsp;
  ⏱ Walk-forward 1H step &nbsp;|&nbsp;
  🎯 Min score {MIN_SCORE} &nbsp;|&nbsp;
  ⏳ Max trade {MAX_TRADE_BARS}H
</p>

<!-- KPIs -->
<div class="grid">
  <div class="kpi">
    <div class="val" style="color:{pct_color(stats['winrate'])}">{stats['winrate']}%</div>
    <div class="lbl">Overall Win Rate</div>
  </div>
  <div class="kpi">
    <div class="val" style="color:{rr_color(stats['avg_rr'])}">{stats['avg_rr']}</div>
    <div class="lbl">Avg RR (all trades)</div>
  </div>
  <div class="kpi">
    <div class="val" style="color:{rr_color(stats['expectancy'])}">{stats['expectancy']}</div>
    <div class="lbl">Expectancy (RR/trade)</div>
  </div>
  <div class="kpi">
    <div class="val">{stats['total']}</div>
    <div class="lbl">Total Signals</div>
  </div>
  <div class="kpi">
    <div class="val class=green">{stats['wins']}</div>
    <div class="lbl">Winners (any TP)</div>
  </div>
  <div class="kpi">
    <div class="val" style="color:#f85149">{stats['losses']}</div>
    <div class="lbl">Stop Losses Hit</div>
  </div>
  <div class="kpi">
    <div class="val">{stats['tp1_rate']}%</div>
    <div class="lbl">TP1 Hit Rate</div>
  </div>
  <div class="kpi">
    <div class="val">{stats['tp2_rate']}%</div>
    <div class="lbl">TP2 Hit Rate</div>
  </div>
  <div class="kpi">
    <div class="val">{stats['tp3_rate']}%</div>
    <div class="lbl">TP3 Hit Rate</div>
  </div>
  <div class="kpi">
    <div class="val">{stats['sl_rate']}%</div>
    <div class="lbl">SL Hit Rate</div>
  </div>
  <div class="kpi">
    <div class="val">{stats['avg_rr_wins']}</div>
    <div class="lbl">Avg RR (wins only)</div>
  </div>
  <div class="kpi">
    <div class="val">{stats['avg_bars']}h</div>
    <div class="lbl">Avg Bars to Exit</div>
  </div>
</div>

<!-- Charts -->
<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px">
  <div class="chart-wrap">
    <h3>📊 Outcome Distribution</h3>
    <canvas id="pieChart" height="220"></canvas>
  </div>
  <div class="chart-wrap">
    <h3>📈 Monthly Win Rate %</h3>
    <canvas id="monthChart" height="220"></canvas>
  </div>
</div>

<!-- Suggestions -->
<h2>🎯 What to Adjust — Actionable Findings</h2>
{"".join(suggestions)}

<!-- Breakdowns -->
<h2>📋 Performance Breakdowns</h2>
<div class="breakdowns">
  {breakdown_table('By Signal Quality', stats.get('by_quality',{}))}
  {breakdown_table('By Direction (Long/Short)', stats.get('by_signal',{}))}
  {breakdown_table('By 1H Entry Trigger', stats.get('by_trigger',{}))}
  {breakdown_table('By Score Bucket', stats.get('by_score',{}))}
  {breakdown_table('By PD Zone', stats.get('by_pd_zone',{}))}
  {breakdown_table('By Structure Type', stats.get('by_structure',{}))}
  {breakdown_table('By HH/LL Confirmed', stats.get('by_hh_ll',{}))}
  {breakdown_table('By Liquidity Sweep', stats.get('by_sweep',{}))}
  {breakdown_table('By FVG Overlap', stats.get('by_fvg',{}))}
  {breakdown_table('By OB Size', stats.get('by_ob_size',{}))}
  {breakdown_table('By ADX Strength', stats.get('by_adx',{}))}
  {breakdown_table('By Pair', stats.get('by_symbol',{}))}
</div>

<!-- Trade log -->
<h2>📜 Recent Trade Log (last 50)</h2>
<div class="card" style="overflow-x:auto">
<table>
  <tr>
    <th>Time</th><th>Pair</th><th>Dir</th><th>Score</th><th>Quality</th>
    <th>Zone</th><th>Trigger</th><th>Outcome</th><th>RR</th><th>Duration</th>
  </tr>
  {trade_rows}
</table>
</div>

<footer>SMC Pro Backtester — Generated {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</footer>

<script>
const pieCtx = document.getElementById('pieChart').getContext('2d');
new Chart(pieCtx, {{
  type: 'doughnut',
  data: {{
    labels: {list(outcomes.keys())},
    datasets: [{{ data: {list(outcomes.values())},
      backgroundColor: ['#4caf50','#00e676','#1de9b6','#f44336','#607d8b'],
      borderColor:'#0d1117', borderWidth:2 }}]
  }},
  options: {{ plugins:{{ legend:{{ labels:{{ color:'#e6edf3' }} }} }} }}
}});

const mCtx = document.getElementById('monthChart').getContext('2d');
new Chart(mCtx, {{
  type: 'bar',
  data: {{
    labels: {json.dumps(month_labels)},
    datasets: [{{
      label: 'Win Rate %',
      data: {json.dumps(month_values)},
      backgroundColor: {json.dumps([('#3fb950' if v>=55 else '#d29922' if v>=45 else '#f85149') for v in month_values])},
      borderRadius: 4
    }}]
  }},
  options: {{
    plugins:{{ legend:{{ labels:{{ color:'#e6edf3' }} }} }},
    scales:{{
      x:{{ ticks:{{ color:'#8b949e' }}, grid:{{ color:'#21262d' }} }},
      y:{{ ticks:{{ color:'#8b949e' }}, grid:{{ color:'#21262d' }}, min:0, max:100 }}
    }}
  }}
}});
</script>
</body>
</html>"""
    return html


def generate_suggestions(stats: dict, df: pd.DataFrame) -> list:
    """Produce data-driven actionable suggestions."""
    suggs = []

    def s(tag, icon, msg, action):
        return f"""<div class="suggest">
          <div class="tag">{icon} {tag}</div>
          <div class="msg">{msg}</div>
          <div class="action">→ {action}</div>
        </div>"""

    wr = stats['winrate']
    if wr < 45:
        suggs.append(s("LOW WIN RATE","🚨",
            f"Overall winrate is {wr}% — below breakeven for 1:1.5 RR.",
            "Raise MIN_SCORE to 80 or tighten OB_TOLERANCE_PCT to 0.005"))
    elif wr >= 60:
        suggs.append(s("STRONG WIN RATE","✅",
            f"Winrate {wr}% — strategy is profitable.",
            "Consider scaling: lower MIN_SCORE to 72 and test impact on signal count vs winrate"))

    # Trigger quality
    trig = stats.get('by_trigger', {})
    best_t = max(trig.items(), key=lambda x: x[1]['winrate'], default=(None,{}))
    worst_t= min(trig.items(), key=lambda x: x[1]['winrate'], default=(None,{}))
    if best_t[0]:
        suggs.append(s("BEST TRIGGER","🕯️",
            f"{best_t[0]} wins at {best_t[1]['winrate']}% vs {worst_t[0]} at {worst_t[1]['winrate']}%.",
            f"Penalise '{worst_t[0]}' harder (-15 instead of -12) in the no-trigger penalty if its winrate is <45%"))

    # OB size
    ob = stats.get('by_ob_size', {})
    if '<0.5%' in ob and ob['<0.5%']['winrate'] > 60:
        suggs.append(s("TIGHT OB EDGE","📦",
            f"OBs <0.5% size win at {ob['<0.5%']['winrate']}%.",
            "Lower OB_IMPULSE_ATR_MULT from 1.0 → 0.7 to find more tight OBs"))
    if '>2%' in ob and ob['>2%']['winrate'] < 45:
        suggs.append(s("WIDE OB DRAG","📦",
            f"OBs >2% size only win {ob['>2%']['winrate']}% — dragging results.",
            "Add hard gate: reject OBs where ob_size_pct > 1.5% (currently only loses 7pts)"))

    # Direction bias
    sig_b = stats.get('by_signal', {})
    long_wr  = sig_b.get('LONG',  {}).get('winrate', 50)
    short_wr = sig_b.get('SHORT', {}).get('winrate', 50)
    diff = abs(long_wr - short_wr)
    if diff > 10:
        weaker = 'SHORT' if long_wr > short_wr else 'LONG'
        suggs.append(s("DIRECTION BIAS","📊",
            f"LONG wins {long_wr}% vs SHORT {short_wr}% — {diff:.0f}pt gap.",
            f"Add +3pts score bonus for the stronger direction, or raise MIN_SCORE for {weaker} trades by 5pts"))

    # HH/LL impact
    hh_b = stats.get('by_hh_ll', {})
    hh_true  = hh_b.get('True',  {}).get('winrate', 50)
    hh_false = hh_b.get('False', {}).get('winrate', 50)
    if hh_true - hh_false > 10:
        suggs.append(s("HH/LL IS KEY","🏔️",
            f"Trending (HH/LL) trades win {hh_true}% vs ranging {hh_false}%.",
            f"Upgrade HH_LL_BONUS from {HH_LL_BONUS} → {HH_LL_BONUS+4} pts, or make it a soft gate (score -5 if absent)"))
    elif hh_true - hh_false < 3:
        suggs.append(s("HH/LL WEAK SIGNAL","〰️",
            f"HH/LL adds only {hh_true-hh_false:.1f}pt win difference — low alpha.",
            "Test removing HH_LL_BONUS entirely and reallocate those 8pts to trigger quality"))

    # Sweep impact
    sw_b = stats.get('by_sweep', {})
    sw_t = sw_b.get('True', {}).get('winrate', 50)
    sw_f = sw_b.get('False',{}).get('winrate', 50)
    if sw_t - sw_f > 8:
        suggs.append(s("SWEEP ADDS ALPHA","💧",
            f"Trades with liq sweep win {sw_t}% vs {sw_f}% without.",
            "Raise sweep bonus from 4pts → 7pts in score_setup extras"))
    elif sw_t < sw_f:
        suggs.append(s("SWEEP MISLEADS","💧",
            f"Trades WITH sweep actually underperform ({sw_t}% vs {sw_f}%).",
            "Investigate: sweep may be firing after strong momentum moves. Lower bonus to 2pts"))

    # Score bucket sweet spot
    sc_b = stats.get('by_score', {})
    best_bucket = max(sc_b.items(), key=lambda x: x[1]['winrate'], default=(None,{}))
    if best_bucket[0]:
        suggs.append(s("SCORE SWEET SPOT","⭐",
            f"Score bucket '{best_bucket[0]}' performs best at {best_bucket[1]['winrate']}% winrate "
            f"({best_bucket[1]['count']} trades).",
            f"Narrow signal quality gates: only send signals in the {best_bucket[0]} bucket as ELITE"))

    # ADX
    adx_b = stats.get('by_adx', {})
    strong_adx = adx_b.get('30-40', adx_b.get('40-60', {}))
    weak_adx   = adx_b.get('<20', {})
    if strong_adx.get('winrate',50) - weak_adx.get('winrate',50) > 10:
        suggs.append(s("ADX FILTER OPPORTUNITY","📐",
            f"ADX 30-40 wins {strong_adx.get('winrate','?')}% vs ADX<20 at {weak_adx.get('winrate','?')}%.",
            "Add optional ADX gate: skip signals when ADX < 20 (choppy market, no trend structure)"))

    # Timeout rate
    timeout_rate = stats['timeouts'] / stats['total'] * 100 if stats['total'] else 0
    if timeout_rate > 25:
        suggs.append(s("HIGH TIMEOUT RATE","⏰",
            f"{timeout_rate:.0f}% of trades time out at {MAX_TRADE_BARS}H without hitting TP/SL.",
            "Tighten TP1 to RR 1:1 (50% exit), or reduce MAX_TRADE_BARS to 32H. Some setups lack follow-through"))

    # Pair performance
    sym_b = stats.get('by_symbol', {})
    top_pairs    = [k for k,v in sym_b.items() if v['winrate'] >= 65 and v['count'] >= 3]
    bottom_pairs = [k for k,v in sym_b.items() if v['winrate'] < 40 and v['count'] >= 3]
    if top_pairs:
        suggs.append(s("BEST PAIRS","🏆",
            f"Top performers: {', '.join(top_pairs[:6])}",
            "Add a pair-priority list — scan these first, or weight their signals higher"))
    if bottom_pairs:
        suggs.append(s("UNDERPERFORMING PAIRS","🗑️",
            f"Poor pairs: {', '.join(bottom_pairs[:6])} — consistently below 40%",
            "Add a blocklist in get_pairs() for these symbols — they likely have low liquidity or unusual structure"))

    return suggs


# ══════════════════════════════════════════════════════════════════════════
#  CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════

def print_console_summary(stats: dict, pairs_done: list):
    print("\n" + "═"*65)
    print("  SMC PRO v4.0 — BACKTEST RESULTS")
    print("═"*65)
    print(f"  Pairs tested :  {len(pairs_done)}")
    print(f"  Total trades :  {stats['total']}")
    print(f"  Win rate     :  {stats['winrate']}%  "
          f"({'✅ Profitable' if stats['winrate']>=55 else '⚠️ Below target' if stats['winrate']>=45 else '❌ Losing'})")
    print(f"  TP1 / TP2 / TP3 : {stats['tp1_rate']}% / {stats['tp2_rate']}% / {stats['tp3_rate']}%")
    print(f"  SL rate      :  {stats['sl_rate']}%")
    print(f"  Avg RR       :  {stats['avg_rr']}  (wins: {stats['avg_rr_wins']})")
    print(f"  Expectancy   :  {stats['expectancy']} RR/trade")
    print(f"  Avg duration :  {stats['avg_bars']}h")
    print("─"*65)
    print("  BY QUALITY:")
    for k, v in stats.get('by_quality',{}).items():
        print(f"    {k:10s}  {v['count']:4d} trades  WR={v['winrate']}%  RR={v['avg_rr']}")
    print("─"*65)
    print("  BY DIRECTION:")
    for k, v in stats.get('by_signal',{}).items():
        print(f"    {k:8s}  {v['count']:4d} trades  WR={v['winrate']}%  RR={v['avg_rr']}")
    print("─"*65)
    print("  TOP 5 PAIRS:")
    top5 = sorted(stats.get('by_symbol',{}).items(), key=lambda x: -x[1]['winrate'])[:5]
    for k, v in top5:
        print(f"    {k:12s}  {v['count']:3d} trades  WR={v['winrate']}%")
    print("═"*65)
    print(f"  📁 Full HTML report saved to ./bt_results/")
    print("═"*65 + "\n")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

async def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    since_ms = int(datetime.strptime(DATE_FROM, "%Y-%m-%d").timestamp() * 1000)
    until_ms = int(datetime.strptime(DATE_TO,   "%Y-%m-%d").timestamp() * 1000)

    exchange = getattr(ccxt, EXCHANGE_ID)({
        'enableRateLimit': True,
        'options': {'defaultType': EXCHANGE_TYPE}
    })

    pairs = PAIRS
    if pairs is None:
        print("⏳ Fetching top-volume pairs...")
        await exchange.load_markets()
        tickers = await exchange.fetch_tickers()
        pairs = [
            s for s in exchange.symbols
            if s.endswith('/USDT:USDT') and 'PERP' not in s
            and tickers.get(s, {}).get('quoteVolume', 0) > 5_000_000
        ]
        pairs.sort(key=lambda x: tickers.get(x,{}).get('quoteVolume',0), reverse=True)
        pairs = pairs[:50]

    all_trades    = []
    pairs_done    = []
    pairs_skipped = []

    print(f"\n🔽  Downloading + backtesting {len(pairs)} pairs")
    print(f"    Period: {DATE_FROM} → {DATE_TO}")
    print(f"    Settings: MIN_SCORE={MIN_SCORE} | OB_TOL={OB_TOLERANCE_PCT} | WARMUP={MIN_WARMUP_BARS}bars\n")

    for idx, symbol in enumerate(pairs):
        short = symbol.replace('/USDT:USDT','')
        print(f"  [{idx+1:2d}/{len(pairs)}] {short:12s}", end=" ", flush=True)

        try:
            df4  = await download_ohlcv(exchange, symbol, '4h',  since_ms, until_ms)
            df1  = await download_ohlcv(exchange, symbol, '1h',  since_ms, until_ms)
            df15 = await download_ohlcv(exchange, symbol, '15m', since_ms, until_ms)

            if df1.empty or len(df1) < MIN_WARMUP_BARS + MAX_TRADE_BARS + 10:
                print("⚠  not enough data — skip")
                pairs_skipped.append(symbol)
                continue

            trades = walk_forward_pair(symbol, df4, df1, df15)
            print(f"→ {len(trades):3d} signals", end="  ")

            wins = sum(1 for t in trades if t['outcome'] in ('TP1','TP2','TP3'))
            wr   = wins/len(trades)*100 if trades else 0
            print(f"WR={wr:.0f}%")

            all_trades.extend(trades)
            pairs_done.append(symbol)

        except Exception as e:
            print(f"❌ {e}")
            pairs_skipped.append(symbol)
            continue

    await exchange.close()

    if not all_trades:
        print("\n❌ No trades generated. Check your DATE_FROM/DATE_TO and PAIRS config.")
        return

    # ── Compute stats ──────────────────────────────────────────────────────
    print(f"\n⚙️  Computing statistics over {len(all_trades)} trades...")
    stats, trades_df = compute_stats(all_trades)

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "trades.csv"
    trades_df.to_csv(csv_path, index=False)
    print(f"✅  Trade CSV  → {csv_path}")

    # ── Save HTML report ───────────────────────────────────────────────────
    html = generate_html_report(stats, trades_df, DATE_FROM, DATE_TO, pairs_done)
    html_path = RESULTS_DIR / "report.html"
    html_path.write_text(html, encoding='utf-8')
    print(f"✅  HTML report → {html_path}")

    # ── Save JSON stats ────────────────────────────────────────────────────
    json_path = RESULTS_DIR / "stats.json"
    # make JSON-serialisable
    stats_out = {k: (v if not isinstance(v, dict) else
                     {kk: {kkk: (bool(vvv) if isinstance(vvv,np.bool_) else
                                 float(vvv) if isinstance(vvv, (np.floating, float)) else
                                 int(vvv) if isinstance(vvv, (np.integer, int)) else str(vvv))
                           for kkk, vvv in vv.items()}
                      for kk, vv in v.items()})
                 for k, v in stats.items()}
    json_path.write_text(json.dumps(stats_out, indent=2, default=str))
    print(f"✅  JSON stats  → {json_path}")

    # ── Console summary ────────────────────────────────────────────────────
    print_console_summary(stats, pairs_done)

    if pairs_skipped:
        print(f"⚠  Skipped {len(pairs_skipped)} pairs: {', '.join(p.replace('/USDT:USDT','') for p in pairs_skipped)}")


if __name__ == "__main__":
    asyncio.run(main())
