"""
SMC PRO BACKTESTER v3.0  (fixed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROOT CAUSE FIX:
  v2 Sniper = 19 trades / 90d / 15 pairs
  Doubling to 180d+25 pairs only gives ~63 — still not enough.
  Solution: run 3 configs IN PARALLEL so we accumulate 100+
  on B_tuned while also stress-testing C_sniper.
  Walk is done ONCE per symbol, filtered per config.

NEW vs v2:
  1. 180d lookback, 25 pairs
  2. DEDUPE_HOURS = 2  (was 4)
  3. Improved TP: BE+0.25R trail after TP1, TP1-trail after TP2
  4. BTC 200 EMA regime filter on longs (per-config toggle)
  5. Diagnostic: raw score distribution logged to CSV
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
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════
#  SMC BASE SETTINGS
# ════════════════════════════════════════════════
OB_TOLERANCE_PCT    = 0.008
OB_IMPULSE_ATR_MULT = 1.0
STRUCTURE_LOOKBACK  = 20
HH_LL_LOOKBACK      = 10
HH_LL_BONUS         = 8

# ════════════════════════════════════════════════
#  BACKTEST SETTINGS
# ════════════════════════════════════════════════
LOOKBACK_DAYS   = 180
WALK_STEP       = 1
MAX_TRADE_BARS  = 48
WARM_UP_BARS_1H = 100
DEDUPE_HOURS    = 2        # loosened from 4 → catch more setups

OUTPUT_DIR  = "/mnt/user-data/outputs"
OUTPUT_CSV  = "backtest_v3_trades.csv"
SUMMARY_CSV = "backtest_v3_summary.csv"
DIAG_CSV    = "backtest_v3_signals.csv"   # raw signal log

# ════════════════════════════════════════════════
#  3 CONFIGS IN PARALLEL
# ════════════════════════════════════════════════
CONFIGS = {
    'A_baseline': {
        'label':           'v2 Baseline',
        'min_score_long':  75,
        'min_score_short': 75,
        'triple_ema_long': False,
        'btc_filter':      False,
        'description':     'MIN=75, no gates (control)',
    },
    'B_tuned': {
        'label':           'v3 Tuned',
        'min_score_long':  82,
        'min_score_short': 80,
        'triple_ema_long': True,
        'btc_filter':      True,
        'description':     'MIN_L=82+TripleEMA+BTC, MIN_S=80',
    },
    'C_sniper': {
        'label':           'v3 Sniper',
        'min_score_long':  87,
        'min_score_short': 85,
        'triple_ema_long': True,
        'btc_filter':      True,
        'description':     'MIN_L=87+TripleEMA+BTC, MIN_S=85',
    },
}

# ════════════════════════════════════════════════
#  25 PAIRS
# ════════════════════════════════════════════════
SYMBOLS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
    'BNB/USDT:USDT', 'XRP/USDT:USDT', 'DOGE/USDT:USDT',
    'AVAX/USDT:USDT','LINK/USDT:USDT','DOT/USDT:USDT',
    'ADA/USDT:USDT', 'ARB/USDT:USDT', 'OP/USDT:USDT',
    'SUI/USDT:USDT', 'TIA/USDT:USDT', 'INJ/USDT:USDT',
    'WLD/USDT:USDT', 'APT/USDT:USDT', 'SEI/USDT:USDT',
    'NEAR/USDT:USDT','FTM/USDT:USDT', 'ATOM/USDT:USDT',
    'LTC/USDT:USDT', 'MATIC/USDT:USDT','FIL/USDT:USDT',
    'STX/USDT:USDT',
]


# ════════════════════════════════════════════════
#  BTC REGIME TRACKER
# ════════════════════════════════════════════════
class BTCRegime:
    def __init__(self):
        self._df = None

    def load(self, df4_btc_with_indicators):
        self._df = df4_btc_with_indicators[['ts','close','ema_200']].dropna().reset_index(drop=True)
        above = (self._df['close'] > self._df['ema_200']).sum()
        below = len(self._df) - above
        logger.info(f"  BTC Regime: {len(self._df)} bars | above200={above} below200={below}")

    def longs_allowed(self, ts):
        if self._df is None:
            return True
        mask = self._df['ts'] <= ts
        if not mask.any():
            return True
        row = self._df[mask].iloc[-1]
        if pd.isna(row['ema_200']):
            return True
        return float(row['close']) > float(row['ema_200'])

btc_regime = BTCRegime()


# ════════════════════════════════════════════════
#  INDICATORS
# ════════════════════════════════════════════════
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
        df['atr'] = ta.volatility.AverageTrueRange(df['high'],df['low'],df['close']).average_true_range()
        bb = ta.volatility.BollingerBands(df['close'], 20, 2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_pband'] = bb.bollinger_pband()
        adx_i = ta.trend.ADXIndicator(df['high'],df['low'],df['close'])
        df['adx']    = adx_i.adx()
        df['di_pos'] = adx_i.adx_pos()
        df['di_neg'] = adx_i.adx_neg()
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'],df['low'],df['close'],df['volume']).chaikin_money_flow()
        df['mfi'] = ta.volume.MFIIndicator(df['high'],df['low'],df['close'],df['volume']).money_flow_index()
        df['vol_sma']   = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_sma'].replace(0, np.nan)
        tp_col = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (tp_col * df['volume']).cumsum() / df['volume'].cumsum()
        body = (df['close'] - df['open']).abs()
        uw   = df['high'] - df[['open','close']].max(axis=1)
        lw   = df[['open','close']].min(axis=1) - df['low']
        df['bull_engulf'] = ((df['close'].shift(1)<df['open'].shift(1))&(df['close']>df['open'])&(df['close']>df['open'].shift(1))&(df['open']<df['close'].shift(1))).astype(int)
        df['bear_engulf'] = ((df['close'].shift(1)>df['open'].shift(1))&(df['close']<df['open'])&(df['close']<df['open'].shift(1))&(df['open']>df['close'].shift(1))).astype(int)
        df['bull_pin']    = ((lw>body*2.5)&(lw>uw*2)&(df['close']>df['open'])).astype(int)
        df['bear_pin']    = ((uw>body*2.5)&(uw>lw*2)&(df['close']<df['open'])).astype(int)
        df['hammer']       = ((lw>body*2.0)&(lw>uw*1.5)).astype(int)
        df['shooting_star']= ((uw>body*2.0)&(uw>lw*1.5)).astype(int)
    except Exception as e:
        logger.error(f"Indicator error: {e}")
    return df


# ════════════════════════════════════════════════
#  SMC ENGINE
# ════════════════════════════════════════════════
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
        if n < lookback * 2: return False, "Not enough 4H data"
        recent = df_4h.iloc[-lookback:]; prior = df_4h.iloc[-(lookback*2):-lookback]
        if direction == 'LONG':
            rh, ph = recent['high'].max(), prior['high'].max()
            return (rh > ph), f"4H HH {ph:.5f}→{rh:.5f}"
        else:
            rl, pl = recent['low'].min(), prior['low'].min()
            return (rl < pl), f"4H LL {pl:.5f}→{rl:.5f}"

    def detect_structure_break(self, df, highs, lows, lookback=STRUCTURE_LOOKBACK):
        events = []; close = df['close']; n = len(df); start = max(0, n-lookback-15)
        for k in range(1, len(highs)):
            ph = highs[k-1]; ch = highs[k]
            if ch['i'] < start: continue
            level = ph['price']
            for j in range(ch['i'], min(ch['i']+10, n)):
                if close.iloc[j] > level:
                    events.append({'kind': 'BOS_BULL' if ch['price']>ph['price'] else 'MSS_BULL','level':level,'bar':j}); break
        for k in range(1, len(lows)):
            pl = lows[k-1]; cl = lows[k]
            if cl['i'] < start: continue
            level = pl['price']
            for j in range(cl['i'], min(cl['i']+10, n)):
                if close.iloc[j] < level:
                    events.append({'kind': 'BOS_BEAR' if cl['price']<pl['price'] else 'MSS_BEAR','level':level,'bar':j}); break
        if not events: return None
        latest = sorted(events, key=lambda x: x['bar'])[-1]
        return latest if latest['bar'] >= n - lookback else None

    def find_order_blocks(self, df, direction, lookback=60):
        obs = []; n = len(df); start = max(2, n-lookback)
        for i in range(start, n-3):
            c = df.iloc[i]
            atr_local = df['atr'].iloc[i] if 'atr' in df.columns and not pd.isna(df['atr'].iloc[i]) else (c['high']-c['low'])
            min_impulse = atr_local * OB_IMPULSE_ATR_MULT
            if direction == 'LONG':
                if c['close'] >= c['open']: continue
                if df['high'].iloc[i+1:min(i+5,n)].max() - c['low'] < min_impulse: continue
                ob = {'top':max(c['open'],c['close']),'bottom':c['low'],'mid':(max(c['open'],c['close'])+c['low'])/2,'bar':i}
                if (df['close'].iloc[i+1:n] < (ob['top']+ob['bottom'])/2).any(): continue
                obs.append(ob)
            else:
                if c['close'] <= c['open']: continue
                if c['high'] - df['low'].iloc[i+1:min(i+5,n)].min() < min_impulse: continue
                ob = {'top':c['high'],'bottom':min(c['open'],c['close']),'mid':(c['high']+min(c['open'],c['close']))/2,'bar':i}
                if (df['close'].iloc[i+1:n] > (ob['top']+ob['bottom'])/2).any(): continue
                obs.append(ob)
        obs.sort(key=lambda x: x['bar'], reverse=True)
        return obs

    def price_in_ob(self, price, ob, tolerance_pct=OB_TOLERANCE_PCT):
        tol = ob['top'] * tolerance_pct
        return (ob['bottom']-tol) <= price <= (ob['top']+tol)

    def find_fvg(self, df, direction, lookback=25):
        fvgs = []; n = len(df)
        for i in range(max(1, n-lookback), n-1):
            prev = df.iloc[i-1]; nxt = df.iloc[i+1]
            if direction == 'LONG' and prev['high'] < nxt['low']:
                fvgs.append({'top':nxt['low'],'bottom':prev['high'],'mid':(nxt['low']+prev['high'])/2,'bar':i})
            elif direction == 'SHORT' and prev['low'] > nxt['high']:
                fvgs.append({'top':prev['low'],'bottom':nxt['high'],'mid':(prev['low']+nxt['high'])/2,'bar':i})
        return fvgs

    def recent_liquidity_sweep(self, df, direction, highs, lows, lookback=25):
        n = len(df); start = n - lookback
        if direction == 'LONG':
            for sl in reversed(lows):
                if sl['i'] < start: continue
                level = sl['price']
                for j in range(sl['i']+1, min(sl['i']+8, n)):
                    c = df.iloc[j]
                    if c['low'] < level and c['close'] > level:
                        return {'level':level,'bar':j,'type':'SWEEP_LOW'}
        else:
            for sh in reversed(highs):
                if sh['i'] < start: continue
                level = sh['price']
                for j in range(sh['i']+1, min(sh['i']+8, n)):
                    c = df.iloc[j]
                    if c['high'] > level and c['close'] < level:
                        return {'level':level,'bar':j,'type':'SWEEP_HIGH'}
        return None

    def pd_zone(self, df_4h, price):
        hi = df_4h['high'].iloc[-50:].max(); lo = df_4h['low'].iloc[-50:].min()
        rang = hi - lo
        if rang == 0: return 'NEUTRAL', 0.5
        pos = (price - lo) / rang
        if pos < 0.40:   return 'DISCOUNT', pos
        elif pos > 0.60: return 'PREMIUM',  pos
        return 'NEUTRAL', pos

    def is_triple_ema_bull(self, df_4h):
        l = df_4h.iloc[-1]
        return l.get('ema_21',0) > l.get('ema_50',0) > l.get('ema_200',0)

smc = SMCEngine()


# ════════════════════════════════════════════════
#  SCORER
# ════════════════════════════════════════════════
def score_setup(direction, ob, structure, sweep, fvg_near,
                df_1h, df_15m, df_4h, pd_label, hh_ll_confirmed):
    score = 0; reasons = []
    l1=df_1h.iloc[-1]; p1=df_1h.iloc[-2]; l15=df_15m.iloc[-1]; l4=df_4h.iloc[-1]

    if structure:
        if 'MSS' in structure['kind']: score+=20; reasons.append(f"MSS({structure['kind']})")
        else:                           score+=14; reasons.append(f"BOS({structure['kind']})")

    if ob:
        ob_sz = (ob['top']-ob['bottom'])/ob['bottom']*100
        if ob_sz<0.8:   score+=20; reasons.append(f"TightOB({ob_sz:.2f}%)")
        elif ob_sz<2.0: score+=13; reasons.append(f"OB({ob_sz:.2f}%)")
        else:           score+=7;  reasons.append(f"WideOB({ob_sz:.2f}%)")

    e21=l4.get('ema_21',0); e50=l4.get('ema_50',0); e200=l4.get('ema_200',0)
    if direction=='LONG':
        if e21>e50>e200:          score+=15; reasons.append("4H_TripleEMA_Bull")
        elif e21>e50:             score+=10; reasons.append("4H_EMA_Bull")
        elif pd_label=='DISCOUNT':score+=6;  reasons.append("Discount")
    else:
        if e21<e50<e200:          score+=15; reasons.append("4H_TripleEMA_Bear")
        elif e21<e50:             score+=10; reasons.append("4H_EMA_Bear")
        elif pd_label=='PREMIUM': score+=6;  reasons.append("Premium")

    if hh_ll_confirmed: score+=HH_LL_BONUS; reasons.append(f"HH/LL+{HH_LL_BONUS}")

    trigger=False
    if direction=='LONG':
        if l1.get('bull_engulf',0):   score+=25;trigger=True;reasons.append("1H_BullEngulf")
        elif l1.get('bull_pin',0):    score+=22;trigger=True;reasons.append("1H_BullPin")
        elif l1.get('hammer',0):      score+=18;trigger=True;reasons.append("1H_Hammer")
        elif p1.get('bull_engulf',0): score+=14;trigger=True;reasons.append("1H_BullEngulf_prev")
        elif p1.get('bull_pin',0):    score+=11;trigger=True;reasons.append("1H_BullPin_prev")
        elif p1.get('hammer',0):      score+=9; trigger=True;reasons.append("1H_Hammer_prev")
    else:
        if l1.get('bear_engulf',0):      score+=25;trigger=True;reasons.append("1H_BearEngulf")
        elif l1.get('bear_pin',0):       score+=22;trigger=True;reasons.append("1H_BearPin")
        elif l1.get('shooting_star',0):  score+=18;trigger=True;reasons.append("1H_ShootStar")
        elif p1.get('bear_engulf',0):    score+=14;trigger=True;reasons.append("1H_BearEngulf_prev")
        elif p1.get('bear_pin',0):       score+=11;trigger=True;reasons.append("1H_BearPin_prev")
        elif p1.get('shooting_star',0):  score+=9; trigger=True;reasons.append("1H_SS_prev")
    if not trigger: score-=12

    rsi1=l1.get('rsi',50); macd1=l1.get('macd',0); ms1=l1.get('macd_signal',0)
    pm1=p1.get('macd',0); pms1=p1.get('macd_signal',0)
    sk1=l1.get('srsi_k',0.5); sd1=l1.get('srsi_d',0.5)
    if direction=='LONG':
        if 28<=rsi1<=55:            score+=4;reasons.append(f"RSI_reset({rsi1:.0f})")
        elif rsi1<28:               score+=3;reasons.append(f"RSI_OS({rsi1:.0f})")
        if macd1>ms1 and pm1<=pms1: score+=5;reasons.append("MACD_BullX")
        elif macd1>ms1:             score+=2;reasons.append("MACD_bull")
        if sk1<0.3 and sk1>sd1:     score+=3;reasons.append("Stoch_BullX")
    else:
        if 45<=rsi1<=72:            score+=4;reasons.append(f"RSI_OBzone({rsi1:.0f})")
        elif rsi1>72:               score+=3;reasons.append(f"RSI_OB({rsi1:.0f})")
        if macd1<ms1 and pm1>=pms1: score+=5;reasons.append("MACD_BearX")
        elif macd1<ms1:             score+=2;reasons.append("MACD_bear")
        if sk1>0.7 and sk1<sd1:     score+=3;reasons.append("Stoch_BearX")

    extras=0
    if sweep:    extras+=4;reasons.append("LiqSweep")
    if fvg_near: extras+=3;reasons.append("FVG+OB")
    vr15=l15.get('vol_ratio',1.0)
    if   vr15>=2.5: extras+=3;reasons.append(f"15M_vol{vr15:.1f}x")
    elif vr15>=1.5: extras+=1;reasons.append(f"15M_vol{vr15:.1f}x")
    close1=l1.get('close',0); vwap1=l1.get('vwap',0)
    if direction=='LONG'  and close1<vwap1: extras+=1;reasons.append("BelowVWAP")
    elif direction=='SHORT' and close1>vwap1: extras+=1;reasons.append("AboveVWAP")
    score+=min(extras,10)
    return max(0,min(int(score),100)), reasons


# ════════════════════════════════════════════════
#  BASE SIGNAL GENERATOR  (no config filtering)
# ════════════════════════════════════════════════
def analyse_slice(df4, df1, df15, symbol, ts_now):
    try:
        if len(df1)<80 or len(df15)<40: return None
        price=df1['close'].iloc[-1]
        l4=df4.iloc[-1]
        e21=l4.get('ema_21',0); e50=l4.get('ema_50',0)
        if e21>e50:   bias='LONG'
        elif e21<e50: bias='SHORT'
        else: return None

        triple_ema_bull = smc.is_triple_ema_bull(df4)
        btc_long_ok     = btc_regime.longs_allowed(ts_now)
        hh_ll_ok, _    = smc.check_4h_hh_ll(df4, bias, HH_LL_LOOKBACK)
        pd_label, _    = smc.pd_zone(df4, price)

        if bias=='LONG'  and pd_label=='PREMIUM':  return None
        if bias=='SHORT' and pd_label=='DISCOUNT': return None

        highs1, lows1 = smc.swing_highs_lows(df1, left=4, right=4)
        structure = smc.detect_structure_break(df1, highs1, lows1, lookback=STRUCTURE_LOOKBACK)
        if structure:
            if bias=='LONG'  and 'BEAR' in structure['kind']: return None
            if bias=='SHORT' and 'BULL' in structure['kind']: return None

        obs = smc.find_order_blocks(df1, bias, lookback=60)
        if not obs: return None
        active_ob = next((ob for ob in obs if smc.price_in_ob(price, ob, OB_TOLERANCE_PCT)), None)
        if not active_ob: return None

        fvgs    = smc.find_fvg(df1, bias, lookback=25)
        fvg_near= next((f for f in fvgs if f['bottom']<active_ob['top'] and f['top']>active_ob['bottom']), None)
        sweep   = smc.recent_liquidity_sweep(df1, bias, highs1, lows1, lookback=20)

        score, reasons = score_setup(bias, active_ob, structure, sweep, fvg_near,
                                     df1, df15, df4, pd_label, hh_ll_ok)

        atr1=df1['atr'].iloc[-1]; entry=price
        if bias=='LONG':
            sl=min(active_ob['bottom']-atr1*0.2, entry-atr1*0.6)
        else:
            sl=max(active_ob['top']+atr1*0.2, entry+atr1*0.6)

        risk=abs(entry-sl)
        if risk<entry*0.001: return None

        tps = ([entry+risk*1.5, entry+risk*2.5, entry+risk*4.0] if bias=='LONG'
               else [entry-risk*1.5, entry-risk*2.5, entry-risk*4.0])

        quality='ELITE' if score>=92 else ('PREMIUM' if score>=85 else 'HIGH')

        return {
            'symbol':      symbol,
            'bias':        bias,
            'quality':     quality,
            'score':       score,
            'entry':       entry,
            'sl':          sl,
            'tp1':tps[0], 'tp2':tps[1], 'tp3':tps[2],
            'risk_pct':    risk/entry*100,
            'pd_zone':     pd_label,
            'hh_ll':       hh_ll_ok,
            'triple_ema':  triple_ema_bull,
            'btc_long_ok': btc_long_ok,
            'structure':   structure['kind'] if structure else 'NONE',
            'reasons':     ' | '.join(reasons[:8]),
        }
    except Exception as e:
        logger.debug(f"analyse_slice: {e}")
        return None


def passes_config(sig, cfg):
    bias=sig['bias']
    if bias=='LONG':
        if sig['score']<cfg['min_score_long']: return False
        # BTC regime gate
        if cfg['btc_filter'] and not sig['btc_long_ok']: return False
        # Triple EMA gate (score>=87 bypasses)
        if cfg['triple_ema_long'] and sig['score']<87 and not sig['triple_ema']: return False
    else:
        if sig['score']<cfg['min_score_short']: return False
    return True


# ════════════════════════════════════════════════
#  TRADE RESOLVER  — v3 improved TP logic
#  TP1 hit → SL to BE+0.25R
#  TP2 hit → SL trails to TP1 price
# ════════════════════════════════════════════════
def resolve_trade(sig, future_df1h):
    entry=sig['entry']; sl_orig=sig['sl']
    tp1,tp2,tp3=sig['tp1'],sig['tp2'],sig['tp3']
    direction=sig['bias']; risk=abs(entry-sl_orig)

    be_plus      = entry + risk*0.25 if direction=='LONG' else entry - risk*0.25
    trail_tp2    = tp1   # after TP2, SL moves to TP1 level

    tp1_hit=tp2_hit=tp3_hit=sl_hit=False
    current_sl=sl_orig; exit_price=entry; exit_bar=len(future_df1h)-1; exit_reason='TIMEOUT'

    for i, row in future_df1h.iterrows():
        bar_idx=future_df1h.index.get_loc(i)
        hi=row['high']; lo=row['low']
        if direction=='LONG':
            if lo<=current_sl:
                sl_hit=True; exit_price=current_sl; exit_bar=bar_idx; exit_reason='SL'; break
            if not tp1_hit and hi>=tp1:
                tp1_hit=True; current_sl=be_plus
            if tp1_hit and not tp2_hit and hi>=tp2:
                tp2_hit=True; current_sl=trail_tp2
            if tp2_hit and not tp3_hit and hi>=tp3:
                tp3_hit=True; exit_price=tp3; exit_bar=bar_idx; exit_reason='TP3'; break
        else:
            if hi>=current_sl:
                sl_hit=True; exit_price=current_sl; exit_bar=bar_idx; exit_reason='SL'; break
            if not tp1_hit and lo<=tp1:
                tp1_hit=True; current_sl=be_plus
            if tp1_hit and not tp2_hit and lo<=tp2:
                tp2_hit=True; current_sl=trail_tp2
            if tp2_hit and not tp3_hit and lo<=tp3:
                tp3_hit=True; exit_price=tp3; exit_bar=bar_idx; exit_reason='TP3'; break

    # P&L accounting
    if exit_reason=='TP3':
        pnl_r=(1.5+2.5+4.0)/3; outcome='TP3'
    elif sl_hit and tp2_hit:
        pnl_r=(1.5+2.5)/2-0.05; outcome='TP2+trail'
    elif sl_hit and tp1_hit:
        pnl_r=(1.5+0.25)/2; outcome='TP1+BE'   # half at TP1, half stopped at BE+0.25
    elif tp2_hit:
        r_exit=(exit_price-entry)/risk*(1 if direction=='LONG' else -1)
        pnl_r=(1.5+2.5+r_exit)/3; outcome='TP2'
    elif tp1_hit:
        r_exit=(exit_price-entry)/risk*(1 if direction=='LONG' else -1)
        pnl_r=(1.5+r_exit)/2; outcome='TP1'
    elif sl_hit:
        pnl_r=-1.0; outcome='SL'
    else:
        pnl_r=(exit_price-entry)/risk*(1 if direction=='LONG' else -1); outcome='TIMEOUT'

    return {
        'outcome':outcome,'pnl_r':round(pnl_r,3),'bars_held':exit_bar+1,
        'tp1_hit':tp1_hit,'tp2_hit':tp2_hit,'tp3_hit':tp3_hit,'sl_hit':sl_hit,
        'exit_price':round(exit_price,8),
    }


# ════════════════════════════════════════════════
#  DATA FETCHER
# ════════════════════════════════════════════════
async def fetch_full_history(exchange, symbol, days=LOOKBACK_DAYS):
    since=int((datetime.utcnow()-timedelta(days=days+5)).timestamp()*1000)
    result={}
    try:
        for tf in ['4h','1h','15m']:
            all_ohlcv=[]; fetch_since=since
            while True:
                batch=await exchange.fetch_ohlcv(symbol, tf, since=fetch_since, limit=1000)
                if not batch: break
                all_ohlcv+=batch
                if len(batch)<1000: break
                fetch_since=batch[-1][0]+1
                await asyncio.sleep(0.05)
            df=pd.DataFrame(all_ohlcv,columns=['ts','open','high','low','close','volume'])
            df['ts']=pd.to_datetime(df['ts'],unit='ms')
            df=df.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
            result[tf]=df
            await asyncio.sleep(0.1)
        logger.info(f"  {symbol}: 4H={len(result['4h'])} 1H={len(result['1h'])} 15M={len(result['15m'])}")
        return result
    except Exception as e:
        logger.error(f"fetch {symbol}: {e}"); return None

def align_slice(df_full, ts_1h, n_bars):
    return df_full[df_full['ts']<=ts_1h].tail(n_bars).reset_index(drop=True)


# ════════════════════════════════════════════════
#  WALK-FORWARD (one pass, 3-config filter)
# ════════════════════════════════════════════════
async def backtest_symbol(exchange, symbol, data_override=None):
    logger.info(f"📊 {symbol} ({LOOKBACK_DAYS}d)...")
    data = data_override or await fetch_full_history(exchange, symbol)
    if not data: return {k:[] for k in CONFIGS}

    df4=data['4h']; df1=data['1h']; df15=data['15m']
    last_signal={k:{} for k in CONFIGS}
    config_trades={k:[] for k in CONFIGS}
    raw_signals=[]

    total_steps=len(df1)-WARM_UP_BARS_1H-MAX_TRADE_BARS
    if total_steps<=0: return {k:[] for k in CONFIGS}

    for step in range(0, total_steps, WALK_STEP):
        bar_idx=WARM_UP_BARS_1H+step
        ts_now=df1['ts'].iloc[bar_idx]

        slice_1h  = add_indicators(df1.iloc[:bar_idx+1].copy())
        slice_4h  = add_indicators(align_slice(df4, ts_now, 220))
        slice_15m = add_indicators(align_slice(df15, ts_now, 110))
        if len(slice_4h)<60 or len(slice_15m)<60: continue

        sig=analyse_slice(slice_4h, slice_1h, slice_15m, symbol, ts_now)
        if sig is None: continue

        # Log raw signal for diagnostics
        raw_signals.append({'ts':ts_now, 'sym':symbol.replace('/USDT:USDT',''),
                            'bias':sig['bias'],'score':sig['score'],'btc_ok':sig['btc_long_ok']})

        # Resolve trade outcome once
        future_df=df1.iloc[bar_idx+1:min(bar_idx+1+MAX_TRADE_BARS,len(df1))].reset_index(drop=True)
        if len(future_df)<3: continue
        result=resolve_trade(sig, future_df)

        trade_base={**sig,**result,
                    'entry_time':ts_now.strftime('%Y-%m-%d %H:%M'),
                    'symbol_clean':symbol.replace('/USDT:USDT','')}

        for cfg_key, cfg in CONFIGS.items():
            if not passes_config(sig, cfg): continue
            last=last_signal[cfg_key].get(symbol)
            if last and (ts_now-last).total_seconds()/3600 < DEDUPE_HOURS: continue
            last_signal[cfg_key][symbol]=ts_now
            config_trades[cfg_key].append({**trade_base,'config':cfg_key})
            logger.info(f"  [{cfg_key}] {symbol.replace('/USDT:USDT',''):<6} "
                        f"{sig['bias']:<5} sc={sig['score']} btc={sig['btc_long_ok']} "
                        f"→ {result['outcome']} {result['pnl_r']:+.2f}R")

    counts={k:len(v) for k,v in config_trades.items()}
    logger.info(f"  ✅ {symbol.replace('/USDT:USDT','')} raw_signals={len(raw_signals)} "
                f"A={counts['A_baseline']} B={counts['B_tuned']} C={counts['C_sniper']}")
    return config_trades, raw_signals


# ════════════════════════════════════════════════
#  STATS + REPORTING
# ════════════════════════════════════════════════
def compute_stats(trades):
    if not trades: return {}
    import statistics as st
    total=len(trades); pnls=[t['pnl_r'] for t in trades]
    wins=[p for p in pnls if p>0]; losses=[p for p in pnls if p<=0]
    total_r=sum(pnls); wr=len(wins)/total*100
    cum=peak=max_dd=0
    for p in pnls:
        cum+=p; peak=max(peak,cum); max_dd=max(max_dd,peak-cum)
    longs=[t for t in trades if t['bias']=='LONG']
    shorts=[t for t in trades if t['bias']=='SHORT']
    return {
        'total':total,'wr':round(wr,1),'total_r':round(total_r,2),
        'avg_r':round(total_r/total,3),
        'avg_win':round(st.mean(wins),3) if wins else 0,
        'avg_loss':round(st.mean(losses),3) if losses else 0,
        'max_dd':round(max_dd,2),
        'best':round(max(pnls),2),'worst':round(min(pnls),2),
        'long_wr':round(sum(1 for t in longs if t['pnl_r']>0)/len(longs)*100,1) if longs else 0,
        'short_wr':round(sum(1 for t in shorts if t['pnl_r']>0)/len(shorts)*100,1) if shorts else 0,
        'long_n':len(longs),'short_n':len(shorts),
    }


def print_comparison(all_config_trades, all_raw_signals):
    from collections import Counter
    sep="─"*64

    # Raw signal score distribution (diagnostic)
    print(f"\n📊 RAW SIGNAL DIAGNOSTICS ({len(all_raw_signals)} signals before config filter)")
    scores=[s['score'] for s in all_raw_signals]
    for lo,hi in [(60,69),(70,74),(75,79),(80,84),(85,89),(90,100)]:
        sub=[s for s in scores if lo<=s<=hi]
        btc_blocked=[s for s in all_raw_signals if lo<=s['score']<=hi and s['bias']=='LONG' and not s['btc_ok']]
        print(f"  {lo}-{hi}: {len(sub):>4} signals  (BTC-blocked longs: {len(btc_blocked)})")

    summaries={}
    print(f"\n{'═'*64}")
    print(f"   SMC PRO v3.0 — {LOOKBACK_DAYS}d | {len(SYMBOLS)} pairs | DEDUP={DEDUPE_HOURS}H")
    print(f"{'═'*64}\n")

    for cfg_key, cfg in CONFIGS.items():
        trades=all_config_trades[cfg_key]
        stats=compute_stats(trades)
        summaries[cfg_key]=stats
        print(f"{'━'*64}")
        print(f" {cfg['label']}  [{cfg_key}]  —  {cfg['description']}")
        print(f"{'━'*64}")
        if not stats: print("  ❌ No trades\n"); continue

        print(f"  Trades       : {stats['total']}")
        print(f"  Win Rate     : {stats['wr']}%")
        print(f"  Total R      : {stats['total_r']:+.2f}R")
        print(f"  Avg / trade  : {stats['avg_r']:+.3f}R")
        print(f"  Avg Win/Loss : {stats['avg_win']:+.3f}R  /  {stats['avg_loss']:+.3f}R")
        print(f"  Max Drawdown : -{stats['max_dd']:.2f}R")
        print(f"  LONG  ({stats['long_n']}t)  : WR={stats['long_wr']}%")
        print(f"  SHORT ({stats['short_n']}t) : WR={stats['short_wr']}%")

        print(f"\n  Score buckets:")
        for lo,hi in [(75,79),(80,84),(85,89),(90,100)]:
            sub=[t for t in trades if lo<=t['score']<=hi]
            if not sub: continue
            wr_s=sum(1 for t in sub if t['pnl_r']>0)/len(sub)*100
            avg_s=sum(t['pnl_r'] for t in sub)/len(sub)
            tag=" ✅" if wr_s>=55 else (" ⚠️" if wr_s>=42 else " 🔴")
            print(f"    {lo}-{hi}: {len(sub):>3}t  WR={wr_s:.0f}%  avg={avg_s:+.3f}R{tag}")

        print(f"\n  Outcomes:")
        oc=Counter(t['outcome'] for t in trades)
        for k,v in sorted(oc.items(),key=lambda x:-x[1]):
            print(f"    {k:<14} {v:>3} ({v/stats['total']*100:.1f}%)")
        print()

    print(f"{'═'*64}")
    print(f"  COMPARISON TABLE")
    print(f"{'═'*64}")
    print(f"  {'Config':<22} {'Trades':>7} {'WR%':>6} {'AvgR':>7} {'TotalR':>8} {'MaxDD':>7}")
    print(sep)
    best_avg=max((s['avg_r'] for s in summaries.values() if s),default=0)
    for cfg_key,cfg in CONFIGS.items():
        s=summaries[cfg_key]
        if not s: print(f"  {cfg['label']:<22} {'—':>7}"); continue
        tag=" ← 🏆 BEST avg/trade" if s['avg_r']==best_avg else ""
        stat_warn=" ⚠️ <100t" if s['total']<100 else ""
        print(f"  {cfg['label']:<22} {s['total']:>7} {s['wr']:>6} {s['avg_r']:>7.3f} {s['total_r']:>8.2f} {-s['max_dd']:>7.2f}{tag}{stat_warn}")
    print()
    return summaries


def save_outputs(all_config_trades, summaries, all_raw_signals):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_trades=[]
    for trades in all_config_trades.values(): all_trades.extend(trades)

    if all_trades:
        cols=['config','entry_time','symbol_clean','bias','quality','score',
              'triple_ema','hh_ll','pd_zone','structure','btc_long_ok',
              'entry','sl','tp1','tp2','tp3','risk_pct',
              'outcome','pnl_r','bars_held',
              'tp1_hit','tp2_hit','tp3_hit','sl_hit','reasons']
        path=os.path.join(OUTPUT_DIR,OUTPUT_CSV)
        with open(path,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore')
            w.writeheader(); w.writerows(all_trades)
        print(f"💾 Trades  → {path}  ({len(all_trades)} rows)")

    if summaries:
        rows=[{'config':k,'label':CONFIGS[k]['label'],**s}
              for k,s in summaries.items() if s]
        if rows:
            s_path=os.path.join(OUTPUT_DIR,SUMMARY_CSV)
            with open(s_path,'w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rows[0].keys(),extrasaction='ignore')
                w.writeheader(); w.writerows(rows)
            print(f"📊 Summary → {s_path}")

    if all_raw_signals:
        d_path=os.path.join(OUTPUT_DIR,DIAG_CSV)
        with open(d_path,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=all_raw_signals[0].keys())
            w.writeheader(); w.writerows(all_raw_signals)
        print(f"🔬 Signals → {d_path}  ({len(all_raw_signals)} raw)")


# ════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════
async def main():
    exchange=ccxt.binance({'enableRateLimit':True,'options':{'defaultType':'future'}})

    print(f"\n🚀 SMC PRO BACKTESTER v3.0 (fixed)")
    print(f"   {LOOKBACK_DAYS}d | {len(SYMBOLS)} pairs | DEDUP={DEDUPE_HOURS}H | 3 configs")
    print(f"   TP logic: BE+0.25R after TP1 | BTC regime filter on longs\n")
    for k,cfg in CONFIGS.items():
        print(f"   [{k}] {cfg['description']}")
    print()

    # BTC regime
    print("  Loading BTC 4H for regime filter...")
    btc_data=await fetch_full_history(exchange,'BTC/USDT:USDT')
    if btc_data:
        btc_regime.load(add_indicators(btc_data['4h'].copy()))

    all_config_trades={k:[] for k in CONFIGS}
    all_raw_signals=[]

    for symbol in SYMBOLS:
        try:
            data_in = btc_data if symbol=='BTC/USDT:USDT' else None
            result = await backtest_symbol(exchange, symbol, data_override=data_in)
            # handle both old (dict) and new (tuple) return
            if isinstance(result, tuple):
                cfg_trades, raw = result
            else:
                cfg_trades, raw = result, []
            for k in CONFIGS:
                all_config_trades[k].extend(cfg_trades[k])
            all_raw_signals.extend(raw)
            counts={k:len(cfg_trades[k]) for k in CONFIGS}
            print(f"  ✅ {symbol.replace('/USDT:USDT',''):<8} A={counts['A_baseline']} B={counts['B_tuned']} C={counts['C_sniper']}")
            await asyncio.sleep(1.0)
        except Exception as e:
            logger.error(f"  ❌ {symbol}: {e}")

    await exchange.close()

    total={k:len(all_config_trades[k]) for k in CONFIGS}
    print(f"\n📦 Totals: A={total['A_baseline']}  B={total['B_tuned']}  C={total['C_sniper']}")
    if total['B_tuned']<100:
        print(f"  ⚠️  B_tuned has {total['B_tuned']} trades — edge not yet confirmed (need 100+)")

    summaries=print_comparison(all_config_trades, all_raw_signals)
    save_outputs(all_config_trades, summaries, all_raw_signals)

if __name__=="__main__":
    asyncio.run(main())
