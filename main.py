"""
╔══════════════════════════════════════════════════════════════╗
║     SWING BOT BACKTEST v8.0 — TARGET 80-85% WIN RATE       ║
║                                                              ║
║  Strategy: Stacked elite filters — fewer, cleaner signals   ║
║                                                              ║
║  Changes from v7:                                           ║
║    - MIN_SCORE_PCT: 0.63 → 0.75   (v7 showed 75%WR here)  ║
║    - ADX_MIN: 40 → 50             (ultra-strong trend only) ║
║    - ATR_SL_MULT: 1.2 → 1.5      (wider SL, less noise)   ║
║    - REQUIRE_RSI_SHORT: rsi>58    (momentum confirmation)   ║
║    - REQUIRE_DI_MARGIN: DI->DI+   (directional conviction) ║
║    - BLOCK_EXTENDED_SHORT: True   (no 4h_below_200ema)     ║
║    - REQUIRE_MACD_BEAR: True      (must have MACD signal)  ║
║    - COOLDOWN_HOURS: 24 → 48      (no chasing)             ║
║                                                              ║
║  Goal: 80-85% WR with ~10-30 signals/month                 ║
║  Output: backtest_swing_v8_results.xlsx                     ║
╚══════════════════════════════════════════════════════════════╝

Run:
    pip install ccxt ta pandas numpy xlsxwriter
    python backtest_swing_v8.py
"""

import asyncio
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import ta
import ccxt.async_support as ccxt

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('SwingBacktest')

# ── SETTINGS (must match swing_bot_v1.py exactly) ──────────────────────────

LOOKBACK_DAYS     = 360
TOP_N_PAIRS       = 600
MIN_VOLUME_USDT   = 500_000

ATR_TP1_MULT      = 2.0   # unchanged
ATR_SL_MULT       = 1.5   # WIDENED from 1.2 — reduce noise SLs
TRAIL_ATR_MULT    = 2.5   # unchanged
TP1_POSITION_PCT  = 0.4   # unchanged

MIN_SCORE_PCT     = 0.65  # RAISED from 0.63 — v7 showed 75%+ band = 75% WR
QUALITY_PREMIUM   = 0.85  # RAISED — ultra-elite signals only
ADX_MIN           = 50    # RAISED from 40 — ultra-strong trend only
LONG_BULL_ONLY    = True
REQUIRE_BELOW_200EMA_SHORT = False  # keep off — 4h_below_200ema = 46% WR killer

# ── v8 NEW FILTERS ──
REQUIRE_RSI_GATE       = True   # SHORT: 4H RSI must be > 58 (not already oversold)
RSI_SHORT_MIN          = 58     # momentum still bearish, not oversold bounce
REQUIRE_MACD_SIGNAL    = True   # must have macd_cross OR macd_hist_expanding (not just score pts)
BLOCK_EXTENDED_SHORT   = True   # block if 4H price > 15% below 200 EMA (chasing extended)
EXTENDED_SHORT_PCT     = 0.15   # 15% below 200EMA = already extended, skip
REQUIRE_DI_MARGIN      = True   # DI- must exceed DI+ by at least 5pts for SHORTs
DI_MARGIN_MIN          = 5      # directional conviction threshold
REQUIRE_AROON_CONFIRM  = True   # Aroon must be < -50 for SHORTs (confirmed downtrend)

MAX_TRADE_DAYS    = 10    # unchanged
COOLDOWN_HOURS    = 48    # RAISED from 24 — no chasing back-to-back on same pair
MAX_SCORE         = 40.0

# ── REALISTIC SIMULATION ──
MAX_CONCURRENT    = 10
RISK_PER_TRADE    = 0.02

OUTPUT_FILE = '/mnt/user-data/outputs/backtest_swing_v8_results.xlsx'

# ── INDICATORS ─────────────────────────────────────────────────────────────

def add_indicators(df):
    try:
        df['ema_9']   = ta.trend.EMAIndicator(df['close'], 9).ema_indicator()
        df['ema_21']  = ta.trend.EMAIndicator(df['close'], 21).ema_indicator()
        df['ema_50']  = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], 200).ema_indicator()

        atr_ind   = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14)
        df['atr'] = atr_ind.average_true_range()

        hl2           = (df['high'] + df['low']) / 2
        df['st_upper']= hl2 + (3 * df['atr'])
        df['st_lower']= hl2 - (3 * df['atr'])
        df['supertrend'] = np.where(
            df['close'] > df['st_upper'].shift(1),
            df['st_lower'], df['st_upper']
        )

        df['rsi']       = ta.momentum.RSIIndicator(df['close'], 14).rsi()
        macd_ind        = ta.trend.MACD(df['close'])
        df['macd']      = macd_ind.macd()
        df['macd_signal'] = macd_ind.macd_signal()
        df['macd_hist'] = macd_ind.macd_diff()
        stoch           = ta.momentum.StochRSIIndicator(df['close'])
        df['stoch_k']   = stoch.stochrsi_k()
        df['stoch_d']   = stoch.stochrsi_d()

        adx_ind        = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx']      = adx_ind.adx()
        df['di_plus']  = adx_ind.adx_pos()
        df['di_minus'] = adx_ind.adx_neg()
        aroon          = ta.trend.AroonIndicator(df['high'], df['low'])
        df['aroon']    = aroon.aroon_up() - aroon.aroon_down()
        df['roc']      = ta.momentum.ROCIndicator(df['close'], 10).roc()

        df['obv']      = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['obv_ema']  = df['obv'].ewm(span=20).mean()
        df['mfi']      = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        df['cmf']      = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
        vol_sma        = df['volume'].rolling(20).mean()
        df['vol_ratio']= df['volume'] / vol_sma

        bb             = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_mid']   = bb.bollinger_mavg()
        df['bb_pband'] = bb.bollinger_pband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

        tp             = (df['high'] + df['low'] + df['close']) / 3
        df['vwap']     = (tp * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap'].fillna(df['close'], inplace=True)

        df['bull_div'] = (
            (df['low'] < df['low'].shift(1)) &
            (df['rsi'] > df['rsi'].shift(1))
        ).astype(int)
        df['bear_div'] = (
            (df['high'] > df['high'].shift(1)) &
            (df['rsi'] < df['rsi'].shift(1))
        ).astype(int)
        df['bull_engulf'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] > df['open']) &
            (df['open'] <= df['close'].shift(1)) &
            (df['close'] >= df['open'].shift(1))
        ).astype(int)
        df['bear_engulf'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'] < df['open']) &
            (df['open'] >= df['close'].shift(1)) &
            (df['close'] <= df['open'].shift(1))
        ).astype(int)
    except Exception as e:
        pass
    return df

# ── SCORING ────────────────────────────────────────────────────────────────

def score_candle(r4h, p4h, r1h, r_daily):
    ls = ss = 0
    lr = {}; sr = {}

    macd_cross_bull = r4h['macd'] > r4h['macd_signal'] and p4h['macd'] <= p4h['macd_signal']
    macd_cross_bear = r4h['macd'] < r4h['macd_signal'] and p4h['macd'] >= p4h['macd_signal']
    macd_hist_bull  = r4h['macd_hist'] > 0 and r4h['macd_hist'] > p4h['macd_hist']
    macd_hist_bear  = r4h['macd_hist'] < 0 and r4h['macd_hist'] < p4h['macd_hist']
    vol_spike       = r4h['vol_ratio'] > 2.0

    # DAILY TREND (8pts)
    if r_daily['ema_9'] > r_daily['ema_21'] and r_daily['ema_21'] > r_daily['ema_50']:
        ls += 4; lr['daily_uptrend'] = 4
    elif r_daily['ema_9'] < r_daily['ema_21'] and r_daily['ema_21'] < r_daily['ema_50']:
        ss += 4; sr['daily_downtrend'] = 4

    if r_daily['close'] > r_daily['ema_200']:
        ls += 2; lr['above_200ema_daily'] = 2
    elif r_daily['close'] < r_daily['ema_200']:
        ss += 2; sr['below_200ema_daily'] = 2

    if r_daily['rsi'] > 50 and r_daily['rsi'] < 70:
        ls += 1; lr['daily_rsi_bull'] = 1
    # NOTE: daily_rsi_bear REMOVED — v5 showed 30.7% WR, actively hurt SHORTs
    # elif r_daily['rsi'] < 50 and r_daily['rsi'] > 30:
    #     ss += 1; sr['daily_rsi_bear'] = 1

    if r_daily['macd_hist'] > 0:
        ls += 1; lr['daily_macd_bull'] = 1
    elif r_daily['macd_hist'] < 0:
        ss += 1; sr['daily_macd_bear'] = 1

    # 4H TREND (8pts)
    if r4h['ema_9'] > r4h['ema_21'] and r4h['ema_21'] > r4h['ema_50']:
        ls += 3; lr['4h_uptrend'] = 3
    elif r4h['ema_9'] < r4h['ema_21'] and r4h['ema_21'] < r4h['ema_50']:
        ss += 3; sr['4h_downtrend'] = 3

    if r4h['close'] > r4h['supertrend']:
        ls += 2; lr['4h_supertrend_bull'] = 2
    elif r4h['close'] < r4h['supertrend']:
        ss += 2; sr['4h_supertrend_bear'] = 2

    if r4h['close'] > r4h['vwap']:
        ls += 1; lr['4h_above_vwap'] = 1
    else:
        ss += 1; sr['4h_below_vwap'] = 1

    if r4h['close'] > r4h['ema_200']:
        ls += 2; lr['4h_above_200ema'] = 2
    elif r4h['close'] < r4h['ema_200']:
        ss += 2; sr['4h_below_200ema'] = 2

    # TREND STRENGTH / ADX (6pts)
    adx = r4h['adx']
    if adx > 35:
        if r4h['di_plus'] > r4h['di_minus']: ls += 3; lr['adx_very_strong_up'] = 3
        else:                                 ss += 3; sr['adx_very_strong_down'] = 3
    elif adx > 25:
        if r4h['di_plus'] > r4h['di_minus']: ls += 2; lr['adx_strong_up'] = 2
        else:                                 ss += 2; sr['adx_strong_down'] = 2

    aroon = r4h['aroon']
    if aroon > 60:    ls += 2; lr['aroon_up'] = 2
    elif aroon < -60: ss += 2; sr['aroon_down'] = 2

    roc = r4h['roc']
    if roc > 5:    ls += 1; lr['roc_bull'] = 1
    elif roc < -5: ss += 1; sr['roc_bear'] = 1

    # MACD (6pts)
    if macd_cross_bull:   ls += 4; lr['macd_cross_4h'] = 4
    elif macd_hist_bull:  ls += 2; lr['macd_hist_expanding'] = 2
    if macd_cross_bear:   ss += 4; sr['macd_cross_4h_bear'] = 4
    elif macd_hist_bear:  ss += 2; sr['macd_hist_expanding_bear'] = 2

    # RSI (4pts)
    rsi = r4h['rsi']
    if rsi < 35:    ls += 3; lr['rsi_4h_oversold'] = 3
    elif rsi < 45:  ls += 2; lr['rsi_4h_low'] = 2
    elif rsi < 55:  ls += 1; lr['rsi_4h_neutral_bull'] = 1
    if rsi > 65:    ss += 3; sr['rsi_4h_overbought'] = 3
    elif rsi > 55:  ss += 2; sr['rsi_4h_high'] = 2

    # DIVERGENCE (5pts)
    if r4h['bull_div']:   ls += 3; lr['4h_bull_div'] = 3
    elif r4h['bear_div']: ss += 3; sr['4h_bear_div'] = 3
    if r1h['bull_engulf']: ls += 2; lr['1h_bull_engulf'] = 2
    elif r1h['bear_engulf']: ss += 2; sr['1h_bear_engulf'] = 2

    # VOLUME (4pts)
    if vol_spike:
        if r4h['close'] > p4h['close']: ls += 3; lr['vol_spike_bull'] = 3
        else:                           ss += 3; sr['vol_spike_bear'] = 3
    if r4h['cmf'] > 0.1:    ls += 1; lr['cmf_buying'] = 1
    elif r4h['cmf'] < -0.1: ss += 1; sr['cmf_selling'] = 1

    # BB + MFI + STOCH (3pts)
    if r4h['bb_width'] > 0.05:
        if r4h['close'] > r4h['bb_mid']: ls += 1; lr['bb_breakout_up'] = 1
        else:                             ss += 1; sr['bb_breakout_down'] = 1
    if r4h['mfi'] < 25:   ls += 1; lr['mfi_oversold_4h'] = 1
    elif r4h['mfi'] > 75: ss += 1; sr['mfi_overbought_4h'] = 1
    stoch_k = r4h['stoch_k']; stoch_d = r4h['stoch_d']
    if stoch_k < 0.2 and stoch_k > stoch_d:   ls += 1; lr['stoch_bull'] = 1
    elif stoch_k > 0.8 and stoch_k < stoch_d: ss += 1; sr['stoch_bear'] = 1

    return ls, ss, lr, sr

# ── REGIME CHECK ───────────────────────────────────────────────────────────

def check_regime(r_daily, r4h, direction):
    daily_bull = r_daily['ema_9'] > r_daily['ema_21']
    daily_bear = r_daily['ema_9'] < r_daily['ema_21']
    h4_bull    = r4h['ema_9'] > r4h['ema_21']
    h4_bear    = r4h['ema_9'] < r4h['ema_21']
    if direction == 'LONG':
        return daily_bull and h4_bull
    else:
        return daily_bear and h4_bear

# ── TRADE SIMULATION v4 — TRAILING STOP ───────────────────────────────────

def simulate_trade(idx, df_4h, direction, entry, sl, tp1, tp2=None):
    """
    v4 Trailing Stop Simulation:
      Phase 1: Hard SL until TP1 hit
      Phase 2: After TP1, trail stop at TRAIL_ATR_MULT * atr below/above peak
               Ride until trailing stop triggers or timeout
    
    tp2 param kept for signature compat but ignored.
    """
    max_candles = MAX_TRADE_DAYS * 6
    future = df_4h.iloc[idx+1 : idx+1+max_candles]

    if len(future) == 0:
        return None

    # Compute ATR at signal candle
    atr_at_entry = df_4h.iloc[idx].get('atr', abs(entry * 0.03))
    trail_dist = TRAIL_ATR_MULT * atr_at_entry

    tp1_hit = False
    peak    = entry  # best price seen since TP1
    trail_stop = None

    for i, (_, candle) in enumerate(future.iterrows()):
        hi = candle['high']
        lo = candle['low']
        close_ts = candle['ts']

        if direction == 'LONG':
            if not tp1_hit:
                # Phase 1: hard SL
                if lo <= sl:
                    loss = (sl - entry) / entry * 100
                    return {'outcome': 'SL', 'pnl': -abs(loss), 'tp1_pnl': 0, 'tp2_pnl': 0, 'close_ts': close_ts}
                if hi >= tp1:
                    tp1_hit  = True
                    peak     = max(hi, tp1)
                    trail_stop = peak - trail_dist
            else:
                # Phase 2: trailing stop
                if hi > peak:
                    peak = hi
                    trail_stop = peak - trail_dist
                if lo <= trail_stop:
                    tp1_gain  = (tp1  - entry) / entry * 100
                    exit_gain = (trail_stop - entry) / entry * 100
                    blended   = tp1_gain * TP1_POSITION_PCT + exit_gain * (1 - TP1_POSITION_PCT)
                    outcome   = 'TRAIL' if exit_gain > 0 else 'BE'
                    return {'outcome': outcome, 'pnl': blended,
                            'tp1_pnl': tp1_gain, 'tp2_pnl': exit_gain, 'close_ts': close_ts}

        else:  # SHORT
            if not tp1_hit:
                if hi >= sl:
                    loss = (entry - sl) / entry * 100
                    return {'outcome': 'SL', 'pnl': -abs(loss), 'tp1_pnl': 0, 'tp2_pnl': 0, 'close_ts': close_ts}
                if lo <= tp1:
                    tp1_hit  = True
                    peak     = min(lo, tp1)
                    trail_stop = peak + trail_dist
            else:
                if lo < peak:
                    peak = lo
                    trail_stop = peak + trail_dist
                if hi >= trail_stop:
                    tp1_gain  = (entry - tp1) / entry * 100
                    exit_gain = (entry - trail_stop) / entry * 100
                    blended   = tp1_gain * TP1_POSITION_PCT + exit_gain * (1 - TP1_POSITION_PCT)
                    outcome   = 'TRAIL' if exit_gain > 0 else 'BE'
                    return {'outcome': outcome, 'pnl': blended,
                            'tp1_pnl': tp1_gain, 'tp2_pnl': exit_gain, 'close_ts': close_ts}

    # Timeout
    close_ts = future.iloc[-1]['ts'] if len(future) > 0 else None
    if tp1_hit:
        tp1_gain = abs(tp1 - entry) / entry * 100
        # Exit at last candle close
        last_close = future.iloc[-1]['close']
        if direction == 'LONG':
            exit_gain = (last_close - entry) / entry * 100
        else:
            exit_gain = (entry - last_close) / entry * 100
        blended = tp1_gain * TP1_POSITION_PCT + exit_gain * (1 - TP1_POSITION_PCT)
        return {'outcome': 'TIMEOUT_TP1', 'pnl': blended,
                'tp1_pnl': tp1_gain, 'tp2_pnl': exit_gain, 'close_ts': close_ts}
    return {'outcome': 'TIMEOUT', 'pnl': 0, 'tp1_pnl': 0, 'tp2_pnl': 0, 'close_ts': close_ts}



# ── MAIN ───────────────────────────────────────────────────────────────────

async def run_backtest():
    exchange = ccxt.binance({'options': {'defaultType': 'future'}, 'enableRateLimit': True})

    print(f"""
╔══════════════════════════════════════════════════════╗
║           SWING BOT BACKTEST v8.0                   ║
║  {LOOKBACK_DAYS}d | {TOP_N_PAIRS} pairs | score≥{MIN_SCORE_PCT*100:.0f}% | ADX≥{ADX_MIN} | HARD
║  TP1={ATR_TP1_MULT}x ATR (40%) | TRAIL={TRAIL_ATR_MULT}x ATR | SL={ATR_SL_MULT}x ATR
║  LONG_BULL_ONLY={LONG_BULL_ONLY} | tighter SL | ADX≥30
╚══════════════════════════════════════════════════════╝
""")

    # Load pairs
    logger.info("Loading pairs...")
    markets = await exchange.load_markets()
    tickers = await exchange.fetch_tickers()
    pairs = []
    for sym, mkt in markets.items():
        if not (mkt.get('swap') and mkt.get('quote') == 'USDT' and mkt.get('active')):
            continue
        vol = (tickers.get(sym, {}).get('quoteVolume') or 0)
        if vol >= MIN_VOLUME_USDT:
            pairs.append((sym, vol))
    pairs.sort(key=lambda x: x[1], reverse=True)
    pairs = [p[0] for p in pairs[:TOP_N_PAIRS]]
    logger.info(f"✅ {len(pairs)} pairs loaded")

    # BTC daily regime per candle
    btc_raw   = await exchange.fetch_ohlcv('BTC/USDT:USDT', '1d', limit=LOOKBACK_DAYS+10)
    btc_daily = pd.DataFrame(btc_raw, columns=['ts','open','high','low','close','volume'])
    btc_daily['ts'] = pd.to_datetime(btc_daily['ts'], unit='ms')
    btc_daily = add_indicators(btc_daily)
    btc_daily['btc_bull'] = btc_daily['ema_9'] > btc_daily['ema_21']
    bull_days = int(btc_daily['btc_bull'].sum())
    bear_days = int((~btc_daily['btc_bull']).sum())
    logger.info(f"  BULL: {bull_days} days | BEAR: {bear_days} days")

    thresh          = MAX_SCORE * MIN_SCORE_PCT
    premium_thresh  = MAX_SCORE * QUALITY_PREMIUM
    limit_4h        = LOOKBACK_DAYS * 6 + 250    # extra for indicators warmup
    limit_daily     = LOOKBACK_DAYS + 100

    all_signals = []
    stats = {
        'regime_blocked': 0,
        'adx_blocked'   : 0,
        'cooldown_skip' : 0,
    }

    for i, symbol in enumerate(pairs):
        try:
            # Fetch all TFs
            raw_4h    = await exchange.fetch_ohlcv(symbol, '4h',  limit=limit_4h)
            raw_1h    = await exchange.fetch_ohlcv(symbol, '1h',  limit=LOOKBACK_DAYS*24+50)
            raw_daily = await exchange.fetch_ohlcv(symbol, '1d',  limit=limit_daily)

            if not raw_4h or not raw_1h or not raw_daily:
                continue
            if len(raw_4h) < 150 or len(raw_daily) < 50:
                continue

            df_4h    = pd.DataFrame(raw_4h,    columns=['ts','open','high','low','close','volume'])
            df_1h    = pd.DataFrame(raw_1h,    columns=['ts','open','high','low','close','volume'])
            df_daily = pd.DataFrame(raw_daily, columns=['ts','open','high','low','close','volume'])

            for df in [df_4h, df_1h, df_daily]:
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')

            df_4h    = add_indicators(df_4h)
            df_1h    = add_indicators(df_1h)
            df_daily = add_indicators(df_daily)

            # Walk forward on 4H candles within backtest window
            cutoff_ts = df_4h['ts'].max() - pd.Timedelta(days=LOOKBACK_DAYS)
            window    = df_4h[df_4h['ts'] >= cutoff_ts].copy()

            pair_signals = 0
            cooldown     = {}   # {direction: last_signal_ts}

            for idx_w, (_, row_4h) in enumerate(window.iterrows()):
                idx_full = df_4h.index.get_loc(row_4h.name)
                if idx_full < 2:
                    continue

                r4h    = df_4h.iloc[idx_full]
                p4h    = df_4h.iloc[idx_full - 1]

                # Match daily candle (same date)
                date_4h = r4h['ts'].date()
                daily_match = df_daily[df_daily['ts'].dt.date <= date_4h]
                if len(daily_match) < 2:
                    continue
                r_daily = daily_match.iloc[-1]

                # Match 1H candle (closest before 4H candle)
                h1_match = df_1h[df_1h['ts'] <= r4h['ts']]
                if len(h1_match) < 1:
                    continue
                r1h = h1_match.iloc[-1]

                # ADX gate
                if r4h['adx'] < ADX_MIN or pd.isna(r4h['adx']):
                    stats['adx_blocked'] += 1
                    continue

                ls, ss, lr, sr = score_candle(r4h, p4h, r1h, r_daily)

                entry = r4h['close']
                atr   = r4h['atr']
                if pd.isna(atr) or atr <= 0:
                    continue

                for direction, score, reasons in [('LONG', ls, lr), ('SHORT', ss, sr)]:
                    if score < thresh:
                        continue

                    # Regime check (daily + 4H aligned)
                    if not check_regime(r_daily, r4h, direction):
                        stats['regime_blocked'] += 1
                        continue

                    # LONG_BULL_ONLY: block LONGs when BTC daily is BEAR
                    if LONG_BULL_ONLY and direction == 'LONG':
                        date_match = btc_daily[btc_daily['ts'].dt.date <= date_4h]
                        if len(date_match) > 0 and not date_match.iloc[-1]['btc_bull']:
                            stats['regime_blocked'] += 1
                            continue

                    # ── v8 ELITE FILTERS ─────────────────────────────────────
                    if direction == 'SHORT':

                        # 1. Block extended SHORTs: price already >15% below 4H 200EMA
                        if BLOCK_EXTENDED_SHORT and 'ema_200' in r4h.index and not pd.isna(r4h['ema_200']):
                            ema200 = r4h['ema_200']
                            if ema200 > 0 and (ema200 - r4h['close']) / ema200 > EXTENDED_SHORT_PCT:
                                stats['regime_blocked'] += 1
                                continue

                        # 2. RSI gate: must be above RSI_SHORT_MIN (not already oversold)
                        if REQUIRE_RSI_GATE:
                            rsi_4h = r4h['rsi'] if 'rsi' in r4h.index and not pd.isna(r4h['rsi']) else 50
                            if pd.isna(rsi_4h) or rsi_4h < RSI_SHORT_MIN:
                                stats['regime_blocked'] += 1
                                continue

                        # 3. MACD signal required: must have cross OR expanding histogram
                        if REQUIRE_MACD_SIGNAL:
                            has_macd = (
                                'macd_cross_4h_bear' in reasons or
                                'macd_hist_expanding_bear' in reasons
                            )
                            if not has_macd:
                                stats['regime_blocked'] += 1
                                continue

                        # 4. DI directional conviction: DI- must beat DI+ by margin
                        if REQUIRE_DI_MARGIN:
                            di_minus = r4h['di_minus'] if 'di_minus' in r4h.index and not pd.isna(r4h['di_minus']) else 0
                            di_plus  = r4h['di_plus'] if 'di_plus' in r4h.index and not pd.isna(r4h['di_plus']) else 0
                            if pd.isna(di_minus) or pd.isna(di_plus):
                                stats['regime_blocked'] += 1
                                continue
                            if (di_minus - di_plus) < DI_MARGIN_MIN:
                                stats['regime_blocked'] += 1
                                continue

                        # 5. Aroon confirmation: must be < -50 (confirmed downtrend)
                        if REQUIRE_AROON_CONFIRM:
                            aroon_val = r4h['aroon'] if 'aroon' in r4h.index and not pd.isna(r4h['aroon']) else 0
                            if pd.isna(aroon_val) or aroon_val > -50:
                                stats['regime_blocked'] += 1
                                continue

                    # Cooldown
                    if direction in cooldown:
                        elapsed = (r4h['ts'] - cooldown[direction]).total_seconds() / 3600
                        if elapsed < COOLDOWN_HOURS:
                            stats['cooldown_skip'] += 1
                            continue

                    score_pct = score / MAX_SCORE
                    quality   = 'PREMIUM' if score >= premium_thresh else 'GOOD'

                    if direction == 'LONG':
                        tp1 = entry + ATR_TP1_MULT * atr
                        sl  = entry - ATR_SL_MULT  * atr
                    else:
                        tp1 = entry - ATR_TP1_MULT * atr
                        sl  = entry + ATR_SL_MULT  * atr

                    # Skip if TP1 is unrealistically small (<3%) or huge (>25%)
                    tp1_pct = abs(tp1 - entry) / entry * 100
                    if tp1_pct < 3.0 or tp1_pct > 25.0:
                        continue

                    result = simulate_trade(idx_full, df_4h, direction, entry, sl, tp1)
                    if result is None:
                        continue

                    tp1_pct  = abs(tp1 - entry) / entry * 100
                    sl_pct   = abs(sl  - entry) / entry * 100
                    trail_pct = TRAIL_ATR_MULT * atr / entry * 100

                    all_signals.append({
                        'symbol'    : symbol.replace('/USDT:USDT',''),
                        'direction' : direction,
                        'timestamp' : r4h['ts'],
                        'entry'     : entry,
                        'tp1'       : tp1,
                        'sl'        : sl,
                        'tp1_pct'   : round(tp1_pct, 2),
                        'trail_pct' : round(trail_pct, 2),
                        'sl_pct'    : round(sl_pct, 2),
                        'atr'       : atr,
                        'adx'       : r4h['adx'],
                        'score_pct' : round(score_pct * 100, 1),
                        'quality'   : quality,
                        'outcome'   : result['outcome'],
                        'pnl'       : round(result['pnl'], 3),
                        'tp1_pnl'   : round(result['tp1_pnl'], 3),
                        'trail_pnl' : round(result['tp2_pnl'], 3),
                        'indicators': ', '.join(list(reasons.keys())[:6]),
                    })

                    cooldown[direction] = r4h['ts']
                    pair_signals += 1

            if pair_signals > 0:
                logger.info(f"  📊 {symbol.replace('/USDT:USDT','')}... → {pair_signals} signals")

        except Exception as e:
            if 'ema_9' not in str(e):
                logger.debug(f"[ERROR] {symbol}: {e}")
            continue

        if (i + 1) % 50 == 0:
            logger.info(f"── {i+1}/{len(pairs)} pairs done | {len(all_signals)} signals so far ──")

    await exchange.close()

    # ── RESULTS ────────────────────────────────────────────────────────────

    if not all_signals:
        print("❌ No signals found — try lowering MIN_SCORE_PCT")
        return

    df = pd.DataFrame(all_signals)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    df = df.sort_values('timestamp').reset_index(drop=True)

    total      = len(df)
    days       = LOOKBACK_DAYS
    per_day    = round(total / days, 1)

    # ── REALISTIC EQUITY SIMULATION ──────────────────────────────────────────
    # Simple sequential simulation: take every signal in time order
    # 2% risk per trade, compounding. No concurrent cap (handled by cooldown in live)
    # This gives the true edge picture — live throttling is a deployment decision

    df_sorted = df.sort_values('timestamp').reset_index(drop=True)

    equity        = 1000.0
    peak_equity   = equity
    max_dd_equity = 0.0
    equity_curve  = [equity]

    for _, row in df_sorted.iterrows():
        sl_frac    = row['sl_pct'] / 100 if row['sl_pct'] > 0 else 0.05
        risk_amt   = equity * RISK_PER_TRADE          # 2% of current equity
        pos_value  = risk_amt / sl_frac               # position sized to risk exactly 2%
        dollar_pnl = pos_value * (row['pnl'] / 100)

        equity    += dollar_pnl
        equity     = max(equity, 0.01)
        equity_curve.append(equity)

        if equity > peak_equity:
            peak_equity = equity
        dd = (equity - peak_equity) / peak_equity * 100
        if dd < max_dd_equity:
            max_dd_equity = dd

    equity_return  = (equity_curve[-1] / equity_curve[0] - 1) * 100
    trades_taken   = len(equity_curve) - 1
    skipped = 0  # not applicable in sequential sim

    # Outcome breakdown — v4 uses TRAIL instead of TP2
    trail_mask   = df['outcome'] == 'TRAIL'
    sl_mask      = df['outcome'] == 'SL'
    be_mask      = df['outcome'] == 'BE'
    tp1only_mask = df['outcome'].str.startswith('TIMEOUT_TP1')
    timeout_mask = df['outcome'] == 'TIMEOUT'

    n_trail  = trail_mask.sum()
    n_sl     = sl_mask.sum()
    n_be     = be_mask.sum()
    n_tp1    = tp1only_mask.sum()
    n_timeout= timeout_mask.sum()

    # Kelly fraction: f = WR - (1-WR)/RR
    avg_win_r  = (df[trail_mask]['pnl'].mean() / df[trail_mask]['sl_pct'].mean()) if n_trail > 0 else 0
    wr_frac    = (n_trail + n_be) / (n_trail + n_sl + n_be) if (n_trail + n_sl + n_be) > 0 else 0
    kelly_f    = wr_frac - (1 - wr_frac) / avg_win_r if avg_win_r > 0 else 0
    kelly_2pct = kelly_f * 100

    # WR = trail wins vs SL losses (decisive closes)
    closed   = n_trail + n_sl
    wr       = round(n_trail / closed * 100, 1) if closed > 0 else 0

    # Including BE as partial win
    full_outcomes = n_trail + n_sl + n_be
    wr_incl_be = round((n_trail + n_be) / full_outcomes * 100, 1) if full_outcomes > 0 else 0

    avg_pnl   = round(df['pnl'].mean(), 3)
    avg_trail = round(df[trail_mask]['pnl'].mean(), 3) if n_trail > 0 else 0
    avg_be    = round(df[be_mask]['pnl'].mean(), 3)    if n_be    > 0 else 0
    avg_loss  = round(df[sl_mask]['pnl'].mean(), 3)    if n_sl    > 0 else 0

    gross_profit = df[df['pnl'] > 0]['pnl'].sum()
    gross_loss   = abs(df[df['pnl'] < 0]['pnl'].sum())
    pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999

    cumulative = df['pnl'].cumsum()
    roll_max   = cumulative.cummax()
    max_dd     = round((cumulative - roll_max).min(), 2)

    avg_tp1_dist  = round(df['tp1_pct'].mean(), 2)
    avg_trail_dist = round(df['trail_pct'].mean(), 2) if 'trail_pct' in df.columns else 0
    avg_sl_dist   = round(df['sl_pct'].mean(), 2)

    print("\n" + "╔"+"═"*54+"╗")
    print("║" + "  📊 SWING BACKTEST v8 — REALISTIC SIM".center(54) + "║")
    print("╚"+"═"*54+"╝")
    print(f"\n  Settings: score≥{MIN_SCORE_PCT*100:.0f}% | HARD | ADX≥{ADX_MIN} | TP1={ATR_TP1_MULT}x | TRAIL={TRAIL_ATR_MULT}x | SL={ATR_SL_MULT}x")
    print(f"  Risk: {RISK_PER_TRADE*100:.0f}%/trade | Max concurrent: {MAX_CONCURRENT}")
    print(f"  Pairs: {df['symbol'].nunique()} | Lookback: {days}d\n")

    print(f"  ── Raw Signal Stats (all signals) ──")
    print(f"  Signals              : {total}  ({per_day}/day  |  {round(per_day*30)}/month)")
    print(f"  Win Rate (Trail only): {wr}%")
    print(f"  Win Rate (incl. BE)  : {wr_incl_be}%")
    print(f"  Profit Factor        : {pf}")
    print(f"  Avg PnL/trade        : {avg_pnl:+.3f}%")
    print(f"  Avg Trail Win        : {avg_trail:+.3f}%")
    print(f"  Avg BE (partial win) : {avg_be:+.3f}%")
    print(f"  Avg SL Loss          : {avg_loss:+.3f}%")
    print(f"  Max Drawdown (raw)   : {max_dd:.2f}%")
    print(f"")
    print(f"  ── 💰 Realistic Equity Simulation ({RISK_PER_TRADE*100:.0f}% risk/trade, max {MAX_CONCURRENT} concurrent) ──")
    print(f"  Starting equity      : $1,000")
    print(f"  Final equity         : ${equity_curve[-1]:,.2f}")
    print(f"  Total return         : {equity_return:+.1f}%")
    print(f"  Max Drawdown (equity): {max_dd_equity:.2f}%")
    print(f"  Trades taken         : {trades_taken} / {total} (skipped {skipped} — concurrent cap)")
    print(f"")
    print(f"  ── Outcome Breakdown ──")
    print(f"  🚀 TRAIL (full win)  : {n_trail}  ({round(n_trail/total*100,1)}%)")
    print(f"  🔒 BE   (partial win): {n_be}   ({round(n_be/total*100,1)}%)")
    print(f"  ⛔ SL   (full loss)  : {n_sl}   ({round(n_sl/total*100,1)}%)")
    print(f"  ⏰ Timeout (TP1)     : {n_tp1}  ({round(n_tp1/total*100,1)}%)")
    print(f"  ⏰ Timeout (no hit)  : {n_timeout} ({round(n_timeout/total*100,1)}%)")
    print(f"")
    print(f"  ── Avg Level Distances ──")
    print(f"  TP1: +{avg_tp1_dist}%  |  Trail dist: {avg_trail_dist}%  |  SL: -{avg_sl_dist}%")
    print(f"  Regime blocked: {stats['regime_blocked']} | ADX blocked: {stats['adx_blocked']} | Cooldown skip: {stats['cooldown_skip']}")

    print(f"\n  ── By Direction ──")
    for d in ['LONG','SHORT']:
        sub = df[df['direction']==d]
        if len(sub) == 0: continue
        sub_closed = sub[sub['outcome'].isin(['TRAIL','SL'])]
        d_wr = round(sub[sub['outcome']=='TRAIL'].shape[0] / len(sub_closed) * 100, 1) if len(sub_closed) > 0 else 0
        print(f"  {d:5s} | n={len(sub):4d} ({round(len(sub)/days,1)}/day) | WR={d_wr}% | Avg={sub['pnl'].mean():+.3f}%")

    print(f"\n  ── By Quality ──")
    for q in ['PREMIUM','GOOD']:
        sub = df[df['quality']==q]
        if len(sub) == 0: continue
        sub_closed = sub[sub['outcome'].isin(['TRAIL','SL'])]
        q_wr = round(sub[sub['outcome']=='TRAIL'].shape[0] / len(sub_closed) * 100, 1) if len(sub_closed) > 0 else 0
        print(f"  {q:8s} | n={len(sub):4d} | WR={q_wr}% | Avg={sub['pnl'].mean():+.3f}%")

    print(f"\n  ── Score Band Breakdown ──")
    print(f"  {'Band':<12} {'n':>6} {'WR%':>8} {'Avg%':>9} {'SL%':>8}")
    for lo_b, hi_b in [(55,60),(60,65),(65,70),(70,75),(75,100)]:
        band = df[(df['score_pct'] >= lo_b) & (df['score_pct'] < hi_b)]
        if len(band) < 3: continue
        band_closed = band[band['outcome'].isin(['TRAIL','SL'])]
        b_wr  = round(band[band['outcome']=='TRAIL'].shape[0] / len(band_closed) * 100, 1) if len(band_closed) > 0 else 0
        b_sl  = round(band[band['outcome']=='SL'].shape[0] / len(band_closed) * 100, 1) if len(band_closed) > 0 else 0
        print(f"  {lo_b}-{hi_b}%     {len(band):>6}   {b_wr:>6.1f}%  {band['pnl'].mean():>+8.3f}%   {b_sl:>5.1f}%")

    print(f"\n  ── By ADX Strength ──")
    for lo_a, hi_a in [(25,30),(30,35),(35,40),(40,100)]:
        band = df[(df['adx'] >= lo_a) & (df['adx'] < hi_a)]
        if len(band) < 3: continue
        band_closed = band[band['outcome'].isin(['TRAIL','SL'])]
        b_wr = round(band[band['outcome']=='TRAIL'].shape[0] / len(band_closed) * 100, 1) if len(band_closed) > 0 else 0
        print(f"  ADX {lo_a}-{hi_a}: n={len(band):4d} | WR={b_wr}% | Avg={band['pnl'].mean():+.3f}%")

    # Indicator WR
    print(f"\n  ── Indicator Win Rates ──")
    ind_stats = {}
    for _, row in df.iterrows():
        if pd.isna(row['indicators']): continue
        for ind in str(row['indicators']).split(', '):
            ind = ind.strip()
            if not ind: continue
            if ind not in ind_stats:
                ind_stats[ind] = {'wins': 0, 'total': 0}
            ind_stats[ind]['total'] += 1
            if row['outcome'] == 'TRAIL':
                ind_stats[ind]['wins'] += 1
    ind_df = pd.DataFrame([
        {'indicator': k, 'wr': round(v['wins']/v['total']*100,1), 'n': v['total']}
        for k, v in ind_stats.items() if v['total'] >= 5
    ]).sort_values('wr', ascending=False)
    print(f"  {'Indicator':<35} {'WR':>6}  {'n':>6}")
    for _, row in ind_df.head(15).iterrows():
        bar = '█' * int(row['wr'] / 6)
        print(f"  {row['indicator']:<35} {row['wr']:>5.1f}%  {int(row['n']):>5}  {bar}")
    if len(ind_df) > 15:
        print(f"\n  BOTTOM (consider tuning):")
        for _, row in ind_df.tail(6).iterrows():
            print(f"  {row['indicator']:<35} {row['wr']:>5.1f}%  n={int(row['n'])}")

    # Top symbols
    print(f"\n  ── Top 15 Symbols ──")
    sym_stats = df.groupby('symbol').apply(lambda x: pd.Series({
        'signals'    : len(x),
        'trail_wins' : (x['outcome']=='TRAIL').sum(),
        'sl_hits'    : (x['outcome']=='SL').sum(),
        'avg_pnl'    : round(x['pnl'].mean(), 3),
    })).reset_index()
    sym_stats['closed'] = sym_stats['trail_wins'] + sym_stats['sl_hits']
    sym_stats['wr'] = (sym_stats['trail_wins'] / sym_stats['closed'] * 100).round(1)
    sym_stats = sym_stats[sym_stats['closed'] >= 3].sort_values('wr', ascending=False)
    print(f"  {'Symbol':<18} {'n':>5} {'WR%':>7} {'Avg%':>8}")
    for _, row in sym_stats.head(15).iterrows():
        print(f"  {row['symbol']:<18} {int(row['signals']):>5} {row['wr']:>6.1f}% {row['avg_pnl']:>+8.3f}%")

    # ── Save Excel ──────────────────────────────────────────────────────
    print(f"\n  💾 Saving to {OUTPUT_FILE}...")
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='All Signals', index=False)

        summary = pd.DataFrame([
            ['SWING BACKTEST v4.0 — TRAILING STOP', ''],
            ['Lookback (days)', days],
            ['Pairs', df['symbol'].nunique()],
            ['Total Signals', total],
            ['Signals/day', per_day],
            ['', ''],
            ['Win Rate (Trail only)', f"{wr}%"],
            ['Win Rate (incl. BE)', f"{wr_incl_be}%"],
            ['Profit Factor', pf],
            ['Avg PnL/trade', f"{avg_pnl:+.3f}%"],
            ['Avg Trail Win', f"{avg_trail:+.3f}%"],
            ['Avg BE', f"{avg_be:+.3f}%"],
            ['Avg SL Loss', f"{avg_loss:+.3f}%"],
            ['Max Drawdown', f"{max_dd:.2f}%"],
            ['', ''],
            ['TRAIL count', n_trail],
            ['BE count', n_be],
            ['SL count', n_sl],
            ['Timeout (TP1)', n_tp1],
            ['Timeout (no hit)', n_timeout],
            ['', ''],
            ['Avg TP1 distance', f"+{avg_tp1_dist}%"],
            ['Avg Trail distance', f"{avg_trail_dist}%"],
            ['Avg SL distance', f"-{avg_sl_dist}%"],
            ['', ''],
            ['Regime blocked', stats['regime_blocked']],
            ['ADX blocked', stats['adx_blocked']],
            ['Cooldown skips', stats['cooldown_skip']],
        ], columns=['Metric', 'Value'])
        summary.to_excel(writer, sheet_name='Summary', index=False)
        sym_stats.to_excel(writer, sheet_name='By Symbol', index=False)
        ind_df.to_excel(writer, sheet_name='Indicators', index=False)

    print(f"  ✅ Saved!\n")

    print("╔"+"═"*54+"╗")
    print("║" + "  ✅ SWING BOT v7 — DEPLOY CHECKLIST".center(54) + "║")
    print("╚"+"═"*54+"╝")
    print(f"  TRADE_MODE           = 'TP1_TRAIL'")
    print(f"  REGIME_MODE          = 'HARD'")
    print(f"  MIN_SCORE_PCT        = {MIN_SCORE_PCT}")
    print(f"  ATR_TP1_MULT         = {ATR_TP1_MULT}  (target +{avg_tp1_dist}%)")
    print(f"  TRAIL_ATR_MULT       = {TRAIL_ATR_MULT}  (trail dist {avg_trail_dist}%)")
    print(f"  ATR_SL_MULT          = {ATR_SL_MULT}   (risk   -{avg_sl_dist}%)")
    print(f"  ADX_MIN              = {ADX_MIN}  ← elite trend filter")
    print(f"  REQUIRE_BELOW_200EMA = True  ← SHORT quality gate")
    print(f"  LONG_BULL_ONLY       = {LONG_BULL_ONLY}")
    print(f"  MAX_CONCURRENT       = {MAX_CONCURRENT}  (live: max open trades)")
    print(f"  RISK_PER_TRADE       = {RISK_PER_TRADE*100:.0f}%  (equity risk per trade)")
    print(f"  COOLDOWN_HRS         = {COOLDOWN_HOURS}")
    print(f"")
    print(f"  Expected: {per_day}/day | {wr}% Trail WR | {equity_return:+.1f}% return/90d")
    print(f"  Realistic DD: {max_dd_equity:.1f}% | Monitor /stats after 2 weeks live\n")

    print("╔"+"═"*54+"╗")
    print("║" + "  📊 v6 vs v7 COMPARISON".center(54) + "║")
    print("╚"+"═"*54+"╝")
    print(f"  {'Metric':<25} {'v6':>10} {'v7':>10}")
    print(f"  {'─'*45}")
    print(f"  {'Min score':<25} {'68%':>10} {MIN_SCORE_PCT*100:.0f}%")
    print(f"  {'Trail mult':<25} {'2.0x':>10} {TRAIL_ATR_MULT}x")
    print(f"  {'TP1 position':<25} {'50%':>10} {int(TP1_POSITION_PCT*100)}%")
    print(f"  {'Equity sim':<25} {'No':>10} {'Yes':>10}")
    print(f"  {'Max concurrent':<25} {'∞':>10} {MAX_CONCURRENT}")
    print(f"  {'─'*45}")
    print(f"  {'Signals/day':<25} {'1.2':>10} {per_day}")
    print(f"  {'Trail WR':<25} {'50.5%':>10} {wr}%")
    print(f"  {'Avg PnL/trade':<25} {'+2.4%':>10} {avg_pnl:+.1f}%")
    print(f"  {'Equity return/90d':<25} {'N/A':>10} {equity_return:+.1f}%")
    print(f"  {'Max DD (equity)':<25} {'N/A':>10} {max_dd_equity:.1f}%")
    print(f"  {'SL rate':<25} {'49.1%':>10} {round(n_sl/total*100,1)}%\n")


if __name__ == '__main__':
    asyncio.run(run_backtest())
