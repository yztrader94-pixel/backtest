import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import logging
import json
import copy
import warnings
import aiohttp

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ob_scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# ⚙️  CONFIGURATION — fill these in before running
# ============================================================

TELEGRAM_BOT_TOKEN = '8186622122:AAGtQcoh_s7QqIAVACmOYVHLqPX-p6dSNVA'   # from @BotFather
TELEGRAM_CHAT_ID   = '7500072234'     # your chat or channel ID

# How often to scan (seconds). 3600 = every hour on the hour.
# Set to 300 (5 min) during testing to verify alerts fire correctly.
SCAN_INTERVAL = 3600

# How many top pairs by volume to watch
TOP_N_PAIRS     = 300
MIN_VOLUME_USDT = 1_000_000

# ============================================================
# STRATEGY PARAMS — locked from v9 backtest (79.5% WR)
# ============================================================
OB_LENGTH       = 5
OB_MAX_AGE      = 18      # key finding: fresh OBs ≤18h only
USE_4H_TREND    = True
USE_BTC_MACRO   = True

MIN_BODY_PCT    = 0.30
MIN_VOL_RATIO   = 0.70

STRUCT_BARS     = 120
DISCOUNT_MAX    = 40
PREMIUM_MIN     = 60
SL_BUFFER_MULT  = 0.2
TP1_RR          = 1.5
TP2_RR          = 3.0
TP3_RR          = 5.0

SWEEP_LOOKBACK  = 20
SWEEP_MIN_WICK  = 0.3

# Cooldown: don't re-alert the same symbol within this many hours
SIGNAL_COOLDOWN_HOURS = 24
# ============================================================


class OBScanner:

    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.btc_4h_trend   = 'NEUTRAL'
        self.alerted: dict  = {}   # symbol → datetime of last alert
        self.scan_count     = 0

    # ----------------------------------------------------------
    # TELEGRAM
    # ----------------------------------------------------------
    async def send_telegram(self, message: str):
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id':    TELEGRAM_CHAT_ID,
            'text':       message,
            'parse_mode': 'HTML',
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"Telegram error {resp.status}: {text}")
                    else:
                        logger.info(f"✅ Telegram alert sent")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    def format_signal(self, symbol: str, direction: str, entry: float,
                      sl: float, tp1: float, tp2: float, tp3: float,
                      ob_age: int, structure: str, pd_level: str) -> str:
        emoji  = '🔴' if direction == 'SHORT' else '🟢'
        sl_pct = abs(entry - sl) / entry * 100
        return (
            f"{emoji} <b>OB SIGNAL — {direction}</b>\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"📌 <b>{symbol}</b>\n"
            f"💰 Entry:  <code>{entry:.6g}</code>\n"
            f"\n"
            f"🛑 SL:     <code>{sl:.6g}</code>  ({sl_pct:.2f}%)\n"
            f"🎯 TP1:    <code>{tp1:.6g}</code>  (1.5R)\n"
            f"🎯 TP2:    <code>{tp2:.6g}</code>  (3.0R)\n"
            f"🎯 TP3:    <code>{tp3:.6g}</code>  (5.0R)\n"
            f"\n"
            f"📊 Structure: {structure} | Zone: {pd_level}\n"
            f"⏱  OB age: {ob_age}h\n"
            f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
        )

    # ----------------------------------------------------------
    # DATA
    # ----------------------------------------------------------
    async def get_top_pairs(self):
        await self.exchange.load_markets()
        tickers = await self.exchange.fetch_tickers()
        pairs = []
        for symbol in self.exchange.symbols:
            if symbol.endswith('/USDT:USDT') and 'PERP' not in symbol:
                vol = (tickers.get(symbol) or {}).get('quoteVolume', 0) or 0
                if vol > MIN_VOLUME_USDT:
                    pairs.append((symbol, vol))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in pairs[:TOP_N_PAIRS]]

    async def fetch_ohlcv(self, symbol, timeframe, days=12):
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        all_data = []
        while True:
            try:
                batch = await self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            except Exception:
                break
            if not batch:
                break
            all_data.extend(batch)
            if len(batch) < 1000:
                break
            since = batch[-1][0] + 1
            await asyncio.sleep(0.1)
        if not all_data:
            return None
        df = pd.DataFrame(all_data, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.drop_duplicates('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def get_4h_trend(self, df_4h):
        if df_4h is None or len(df_4h) < 55:
            return 'NEUTRAL'
        try:
            ema21 = ta.trend.EMAIndicator(df_4h['close'], window=21).ema_indicator()
            ema50 = ta.trend.EMAIndicator(df_4h['close'], window=50).ema_indicator()
            v21, v50 = ema21.iloc[-1], ema50.iloc[-1]
            if pd.isna(v21) or pd.isna(v50):
                return 'NEUTRAL'
            return 'BULLISH' if v21 > v50 else 'BEARISH'
        except Exception:
            return 'NEUTRAL'

    # ----------------------------------------------------------
    # INDICATORS & DETECTION
    # ----------------------------------------------------------
    def add_indicators(self, df):
        if len(df) < 60:
            return None
        try:
            df = df.copy()
            df['atr']        = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            df['hl2']        = (df['high'] + df['low']) / 2
            df['vol_sma']    = df['volume'].rolling(20).mean()
            df['vol_ratio']  = df['volume'] / df['vol_sma'].replace(0, np.nan)
            df['body']       = abs(df['close'] - df['open'])
            df['rng']        = (df['high'] - df['low']).replace(0, np.nan)
            df['body_pct']   = df['body'] / df['rng']
            df['upper_wick'] = df['high'] - df[['open','close']].max(axis=1)
            df['lower_wick'] = df[['open','close']].min(axis=1) - df['low']
            return df
        except Exception:
            return None

    def detect_obs(self, df):
        n, L = len(df), OB_LENGTH
        obs, os = [], None
        for i in range(L, n - L):
            vol_slice = df['volume'].iloc[i-L : i+L+1]
            is_vpivot = (len(vol_slice) == 2*L+1 and
                         df['volume'].iloc[i] == vol_slice.max())
            upper = df['high'].iloc[i-L+1 : i+1].max()
            lower = df['low'].iloc[i-L+1  : i+1].min()
            if df['high'].iloc[i-L] > upper:   os = 0
            elif df['low'].iloc[i-L] < lower:  os = 1
            if is_vpivot and os is not None:
                ob_bar = i - L
                if os == 0:
                    obs.append({
                        'type':      'BEAR_OB',
                        'ob_top':    float(df['high'].iloc[ob_bar]),
                        'ob_btm':    float(df['hl2'].iloc[ob_bar]),
                        'ob_avg':    float((df['high'].iloc[ob_bar]+df['hl2'].iloc[ob_bar])/2),
                        'formed_at': ob_bar,
                        'valid':     True
                    })
        return obs

    def update_mitigation(self, obs, df, i):
        L = OB_LENGTH
        for ob in obs:
            if not ob['valid'] or i <= ob['formed_at']:
                continue
            start  = max(0, i-L+1)
            t_bear = df['high'].iloc[start:i+1].max()
            if ob['type'] == 'BEAR_OB' and t_bear > ob['ob_top']:
                ob['valid'] = False

    def find_swings(self, df, lb=5):
        n = len(df)
        sh, sl = [False]*n, [False]*n
        for i in range(lb, n-lb):
            if df['high'].iloc[i] == df['high'].iloc[i-lb:i+lb+1].max(): sh[i] = True
            if df['low'].iloc[i]  == df['low'].iloc[i-lb:i+lb+1].min():  sl[i] = True
        return sh, sl

    def get_structure(self, df, i, sh, sl):
        start   = max(0, i - STRUCT_BARS)
        sh_bars = [j for j in range(start, i) if sh[j]]
        sl_bars = [j for j in range(start, i) if sl[j]]
        if len(sh_bars) < 2 or len(sl_bars) < 2:
            return 'NEUTRAL', None, None
        sh1, sh2 = df['high'].iloc[sh_bars[-1]], df['high'].iloc[sh_bars[-2]]
        sl1, sl2 = df['low'].iloc[sl_bars[-1]],  df['low'].iloc[sl_bars[-2]]
        if sh1 > sh2 and sl1 > sl2:   return 'BULLISH', sh1, sl1
        elif sh1 < sh2 and sl1 < sl2: return 'BEARISH', sh1, sl1
        return 'NEUTRAL', sh1, sl1

    def get_pd(self, last_sh, last_sl, price):
        if last_sh is None or last_sl is None: return 'NEUTRAL', 50.0
        rng = last_sh - last_sl
        if rng <= 0: return 'NEUTRAL', 50.0
        pos = max(0.0, min(100.0, (price - last_sl) / rng * 100))
        if pos <= DISCOUNT_MAX: return 'DISCOUNT', pos
        if pos >= PREMIUM_MIN:  return 'PREMIUM',  pos
        return 'EQUILIBRIUM', pos

    def detect_sweep(self, df, i):
        if i < SWEEP_LOOKBACK + 2:
            return False
        window   = df.iloc[i - SWEEP_LOOKBACK : i]
        prev     = df.iloc[i - 1]
        prev_rng = float(prev['rng']) if not pd.isna(prev['rng']) and prev['rng'] > 0 else None
        if prev_rng is None:
            return False
        recent_high = window['high'].max()
        upper_wick  = float(prev['upper_wick']) if not pd.isna(prev['upper_wick']) else 0.0
        return (prev['high'] > recent_high and
                prev['close'] < recent_high and
                upper_wick / prev_rng >= SWEEP_MIN_WICK)

    # ----------------------------------------------------------
    # SIGNAL CHECK — only looks at the LAST closed candle
    # ----------------------------------------------------------
    async def check_pair(self, symbol: str):
        """Returns a signal dict if a valid setup exists on the latest candle, else None."""
        try:
            # Fetch enough 1H candles to compute structure + indicators
            df = await self.fetch_ohlcv(symbol, '1h', days=10)
            if df is None or len(df) < 100:
                return None

            df = self.add_indicators(df)
            if df is None:
                return None
            df.dropna(subset=['atr','hl2'], inplace=True)
            df.reset_index(drop=True, inplace=True)

            # We check the second-to-last candle (last CLOSED candle)
            i = len(df) - 2
            if i < OB_LENGTH*2 + 2:
                return None

            cur  = df.iloc[i]
            prev = df.iloc[i-1]
            if pd.isna(cur['atr']) or cur['atr'] == 0:
                return None

            obs = self.detect_obs(df)
            sh, sl = self.find_swings(df)

            structure, last_sh, last_sl = self.get_structure(df, i, sh, sl)
            if structure != 'BEARISH':
                return None

            pd_level, pd_pct = self.get_pd(last_sh, last_sl, cur['close'])
            if pd_level != 'PREMIUM':
                return None

            # 4H trend filter
            df_4h    = await self.fetch_ohlcv(symbol, '4h', days=10)
            trend_4h = self.get_4h_trend(df_4h)
            if USE_4H_TREND and trend_4h == 'BULLISH':
                return None

            # BTC macro gate
            if USE_BTC_MACRO and self.btc_4h_trend == 'BULLISH':
                return None

            vol_ratio = float(cur['vol_ratio']) if not pd.isna(cur['vol_ratio']) else 1.0
            body_pct  = float(cur['body_pct'])  if not pd.isna(cur['body_pct'])  else 1.0
            vol_ok    = vol_ratio >= MIN_VOL_RATIO
            body_ok   = body_pct  >= MIN_BODY_PCT
            swept     = self.detect_sweep(df, i)

            for ob in obs:
                if not ob['valid'] or ob['type'] != 'BEAR_OB':
                    continue
                self.update_mitigation([ob], df, i)
                if not ob['valid']:
                    continue

                age = i - ob['formed_at']
                if age <= 1 or age > OB_MAX_AGE:
                    continue

                touched = ((prev['high'] >= ob['ob_btm'] or prev['close'] >= ob['ob_btm']) and
                           prev['close'] <= ob['ob_top'] * 1.005)

                confirmed = (cur['close'] < cur['open'] and
                             cur['close'] < ob['ob_avg'] and
                             cur['close'] < prev['close'] and
                             (body_ok or vol_ok or swept))

                if touched and confirmed:
                    entry = float(cur['close'])
                    atr   = float(cur['atr'])
                    sl_p  = ob['ob_top'] + atr * SL_BUFFER_MULT
                    risk  = max(sl_p - entry, atr * 0.05)
                    return {
                        'symbol':    symbol.replace('/USDT:USDT', ''),
                        'direction': 'SHORT',
                        'entry':     entry,
                        'sl':        round(sl_p, 8),
                        'tp1':       round(entry - risk * TP1_RR, 8),
                        'tp2':       round(entry - risk * TP2_RR, 8),
                        'tp3':       round(entry - risk * TP3_RR, 8),
                        'ob_age':    age,
                        'structure': structure,
                        'pd_level':  pd_level,
                        'timestamp': cur['timestamp'].isoformat(),
                    }
        except Exception as e:
            logger.error(f"Error checking {symbol}: {e}")
        return None

    # ----------------------------------------------------------
    # COOLDOWN CHECK
    # ----------------------------------------------------------
    def is_on_cooldown(self, symbol: str) -> bool:
        if symbol not in self.alerted:
            return False
        elapsed = datetime.utcnow() - self.alerted[symbol]
        return elapsed < timedelta(hours=SIGNAL_COOLDOWN_HOURS)

    # ----------------------------------------------------------
    # MAIN SCAN LOOP
    # ----------------------------------------------------------
    async def scan_once(self):
        self.scan_count += 1
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        logger.info(f"━━━ Scan #{self.scan_count} | {now} | BTC 4H: {self.btc_4h_trend} ━━━")

        # Refresh BTC trend each scan
        btc_4h = await self.fetch_ohlcv('BTC/USDT:USDT', '4h', days=10)
        self.btc_4h_trend = self.get_4h_trend(btc_4h)
        logger.info(f"BTC 4H trend: {self.btc_4h_trend}")

        if self.btc_4h_trend == 'BULLISH':
            logger.info("BTC bullish — no shorts. Strategy idle.")
            await self.send_telegram(
                f"⚠️ <b>OB Scanner</b> — Scan #{self.scan_count}\n"
                f"BTC 4H flipped <b>BULLISH</b>. Strategy idle (shorts-only mode).\n"
                f"🕐 {now}"
            )
            return

        pairs   = await self.get_top_pairs()
        signals = []

        for idx, pair in enumerate(pairs, 1):
            if self.is_on_cooldown(pair.replace('/USDT:USDT', '')):
                continue
            signal = await self.check_pair(pair)
            if signal:
                signals.append(signal)
                logger.info(f"🔴 SIGNAL: {signal['symbol']} SHORT @ {signal['entry']}")
            await asyncio.sleep(0.2)

        logger.info(f"Scan #{self.scan_count} complete — {len(signals)} signal(s) found")

        if not signals:
            logger.info("No signals this scan.")
            return

        for sig in signals:
            msg = self.format_signal(
                symbol    = sig['symbol'],
                direction = sig['direction'],
                entry     = sig['entry'],
                sl        = sig['sl'],
                tp1       = sig['tp1'],
                tp2       = sig['tp2'],
                tp3       = sig['tp3'],
                ob_age    = sig['ob_age'],
                structure = sig['structure'],
                pd_level  = sig['pd_level'],
            )
            await self.send_telegram(msg)
            self.alerted[sig['symbol']] = datetime.utcnow()
            await asyncio.sleep(1)   # small gap between messages

        # Save signals to log file
        with open('signals_log.json', 'a') as f:
            for sig in signals:
                f.write(json.dumps(sig, default=str) + '\n')

    async def run(self):
        logger.info("="*55)
        logger.info("🚀 OB LIVE SCANNER — v9 strategy")
        logger.info(f"   Pairs: top {TOP_N_PAIRS} | OB_MAX_AGE={OB_MAX_AGE}h")
        logger.info(f"   Interval: every {SCAN_INTERVAL//60} minutes")
        logger.info(f"   Cooldown: {SIGNAL_COOLDOWN_HOURS}h per symbol")
        logger.info("="*55)

        # Startup message
        await self.send_telegram(
            f"🚀 <b>OB Scanner started</b>\n"
            f"Watching top {TOP_N_PAIRS} pairs | OB age ≤{OB_MAX_AGE}h\n"
            f"Scanning every {SCAN_INTERVAL//60} min | v9 strategy (79.5% WR backtest)"
        )

        while True:
            try:
                await self.scan_once()
            except Exception as e:
                logger.error(f"Scan error: {e}")
                await self.send_telegram(f"⚠️ Scanner error: {e}")

            logger.info(f"Sleeping {SCAN_INTERVAL//60} min until next scan...")
            await asyncio.sleep(SCAN_INTERVAL)


async def main():
    scanner = OBScanner()
    try:
        await scanner.run()
    finally:
        await scanner.exchange.close()

if __name__ == "__main__":
    asyncio.run(main())
