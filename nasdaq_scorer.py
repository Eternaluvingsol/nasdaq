#!/usr/bin/env python3
"""Nasdaq Elite Scorer - NASDAQ 250 - Final Version with Sector Cap + Flexible Sentiment Override | Amended for Long/Medium Term Holds"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import random
import warnings
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from hmmlearn import hmm

load_dotenv()
warnings.filterwarnings('ignore')

# ================== LOAD EXTERNAL CONFIG (NEW: Centralized Tunable Params) ==================
def load_global_config():
    config_path = Path("~/Desktop/OpenJarvis/scorer_config.json").expanduser()
    default_cfg = {
        "TEST_MODE": False,
        "MAX_6M_GAIN_HARD_CAP": 150.0,
        "MAX_6M_GAIN_WARNING_THRESHOLD": 100.0,
        "MAX_PEG": 2.5,
        "MIN_QUALITY": 0.50,
        "MAX_HEALTHCARE_IN_TOP30": 8,
        "MAX_TECH_TOP30": 12,
        "MAX_CONSUMER_TOP30": 9,
        "DATA_MISSING_PCT_THRESHOLD": 0.30,
        "FUNDAMENTAL_CACHE_DAYS": 3,
        "CSV_HISTORY_KEEP_DAYS": 90,
        "MD_SCAN_KEEP_DAYS": 30,
        "BASE_SENTIMENT_WEIGHT": 0.08,
        "MAX_SENTIMENT_WEIGHT_HIGH_RISK": 0.14,
        "GROK_MODEL": "grok-4.3",
        "GROK_TEMP": 0.65,
        "GROK_MAX_TOKENS": 1500
    }
    try:
        with open(config_path, "r") as f:
            user_cfg = json.load(f)
        return {**default_cfg, **user_cfg}
    except FileNotFoundError:
        return default_cfg

CFG = load_global_config()
TEST_MODE = CFG["TEST_MODE"]

# Load ticker list with original fallback
try:
    from NASDAQ_250 import NASDAQ_250
    TICKERS = NASDAQ_250
except ImportError:
    TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'ADBE', 'PEP', 
    'COST', 'NFLX', 'AMD', 'CRM', 'TMUS', 'CSCO', 'INTU', 'AMAT', 'QCOM', 'TXN', 
    'ISRG', 'NOW', 'BKNG', 'VRTX', 'REGN', 'KLAC', 'PANW', 'MU', 'LRCX', 'SNPS', 
    'CDNS', 'CRWD', 'FTNT', 'MRVL', 'PDD', 'MELI', 'SHOP', 'DDOG', 'ZS', 'WDAY', 
    'DASH', 'PLTR', 'MSTR', 'ABNB', 'CEG', 'CPRT', 'FAST', 'GILD', 'HON', 'IDXX', 
    'KDP', 'KHC', 'MAR', 'MNST', 'PYPL', 'ROST', 'SBUX', 'TEAM', 'TTD', 'VRSK', 
    'ADP', 'AMGN', 'BIIB', 'CHTR', 'CMCSA', 'EA', 'EXC', 'LULU', 'MDLZ', 'ODFL', 
    'ORLY', 'PCAR', 'WBD', 'WMT', 'ASML', 'ARM', 'ADI', 'APP', 'ALAB', 'ARGX', 
    'ADSK', 'ALNY', 'ASTS', 'ACGL', 'AFRM', 'AKAM', 'AMKR', 'ARXS', 'ARCC', 'APA', 
    'AAOI', 'AEIS', 'ASND', 'ALGN', 'AGNC', 'AUR', 'APLD', 'AAON', 'ARWR', 'AAL', 
    'AVAV', 'ALGM', 'ABVX', 'ALKS', 'AVT', 'APGE', 'ACT', 'ACMR', 'APPF', 'ACLS', 
    'AMRX', 'ATAT', 'ALM', 'AUGO', 'ACIW', 'ALHC', 'ARCB', 'ACAD', 'ADEA', 'ARLP', 
    'ASO', 'ATRO', 'AEHR', 'ARQT', 'ADPT', 'AMBA', 'ALMS', 'ANDE', 'AGYS', 'AVPT', 
    'ACHC', 'ALRM', 'ALGT', 'AUPH', 'ASTH', 'ADMA', 'ANIP', 'AMSC', 'ADUS', 'AGIO', 
    'APPN', 'APC', 'AAPG', 'AIP', 'ALKT', 'ABCL', 'AEVA', 'AVAH', 'ANAB', 'ALNT', 
    'ALMR', 'AMLX', 'ABTC', 'AVBP', 'ATRC', 'ARDX', 'ATAI', 'AFYA', 'ACDC', 'AHCO', 
    'AMAL', 'ALVO', 'ATLC', 'ATEX', 'AVLN', 'AHG', 'ATEC', 'ADTN', 'APPS', 'AOSL', 
    'ASTE', 'AIAI', 'ASST', 'AIIR', 'ARRY', 'AKTS', 'AVO'
]

GROK_API_KEY = os.getenv("GROK_API_KEY")

# Unpack core config thresholds
MAX_6M_GAIN_HARD_CAP = CFG["MAX_6M_GAIN_HARD_CAP"]
MAX_6M_GAIN_WARNING_THRESHOLD = CFG["MAX_6M_GAIN_WARNING_THRESHOLD"]
MAX_PEG = CFG["MAX_PEG"]
MIN_QUALITY = CFG["MIN_QUALITY"]
MAX_HEALTHCARE_IN_TOP30 = CFG["MAX_HEALTHCARE_IN_TOP30"]
MAX_TECH_TOP30 = CFG["MAX_TECH_TOP30"]
MAX_CONSUMER_TOP30 = CFG["MAX_CONSUMER_TOP30"]
DATA_MISSING_PCT_THRESHOLD = CFG["DATA_MISSING_PCT_THRESHOLD"]
FUNDAMENTAL_CACHE_DAYS = CFG["FUNDAMENTAL_CACHE_DAYS"]
BASE_SENTIMENT_WEIGHT = CFG["BASE_SENTIMENT_WEIGHT"]
MAX_SENTIMENT_WEIGHT_HIGH_RISK = CFG["MAX_SENTIMENT_WEIGHT_HIGH_RISK"]

# ================== CACHE UTILITIES (NEW: Fundamental Data Caching) ==================
CACHE_DIR = Path.home() / "Desktop/OpenJarvis/fund_cache"
CACHE_DIR.mkdir(exist_ok=True)

def get_cached_fundamentals(ticker):
    cache_file = CACHE_DIR / f"{ticker}_fund.json"
    if not cache_file.exists():
        return None
    file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
    expiry = timedelta(days=FUNDAMENTAL_CACHE_DAYS)
    if datetime.now() - file_mtime > expiry:
        os.unlink(cache_file)
        return None
    with open(cache_file, "r") as f:
        return json.load(f)

def write_cached_fundamentals(ticker, data):
    cache_file = CACHE_DIR / f"{ticker}_fund.json"
    with open(cache_file, "w") as f:
        json.dump(data, f)

# ================== SENTIMENT OVERRIDES (ORIGINAL LOGIC PRESERVED) ==================
def load_sentiment_overrides():
    override_path = Path("~/Desktop/OpenJarvis/sentiment_overrides.json").expanduser()
    try:
        with open(override_path) as f:
            data = json.load(f)
            today = datetime.now().strftime('%Y%m%d')
            return data.get(today, data.get("global", {}))
    except:
        return {}

# ================== BOND SIGNAL IMPORT (ORIGINAL LOGIC PRESERVED) ==================
def get_bond_signal():
    bond_path = Path("~/Desktop/OpenJarvis/bond_signal.json").expanduser()
    try:
        with open(bond_path) as f:
            data = json.load(f)
            return {
                'risk_score': data.get('risk_score'),
                'regime': data.get('regime'),
            }
    except FileNotFoundError:
        return None

# ================== HELPERS (AMENDED: Fixed null vs zero detection) ==================
def normalize(s):
    if s.nunique() < 2:
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

def count_missing_metrics(fund_dict, core_metric_list):
    missing = 0
    total = len(core_metric_list)
    for k in core_metric_list:
        val = fund_dict.get(k)
        if val is None:
            missing +=1
    return missing / total

def get_macro_market_timeseries():
    """NEW: Pull NDX + macro data for true market HMM regime detection (replaces cross-section stock HMM)"""
    try:
        ndx = yf.Ticker("^NDX").history(period="2y", progress=False)["Close"]
        vix = yf.Ticker("^VIX").history(period="2y", progress=False)["Close"]
        tn10 = yf.Ticker("^TNX").history(period="2y", progress=False)["Close"]
        tn2 = yf.Ticker("^IRX").history(period="2y", progress=False)["Close"]
        spread = tn10 - tn2
        df_macro = pd.DataFrame({
            "NDX_Ret6M": ndx.pct_change(126),
            "VIX": vix,
            "YieldSpread": spread
        }).dropna()
        features = np.column_stack([df_macro["NDX_Ret6M"].fillna(0), df_macro["VIX"].fillna(0), df_macro["YieldSpread"].fillna(0)])
        return features
    except Exception as e:
        print(f"Macro timeseries pull failed: {str(e)}")
        return np.array([[0,0,0], [0.01, 15, 1.2]])

# ================== ENHANCED STOCK DATA FETCH (Fixed FCF 3Y logic, null filtering) ==================
def get_stock_data(ticker):
    core_metrics_check = ["freeCashflow","revenueGrowth","ebitdaMargins","returnOnEquity","grossMargins"]
    try:
        cached = get_cached_fundamentals(ticker)
        s = yf.Ticker(ticker)
        hist = s.history(period="2y")
        if len(hist) < 250:
            return None

        # Use cache if valid, pull fresh info otherwise
        if cached:
            i = cached
        else:
            i = s.info
            write_cached_fundamentals(ticker, i)

        # Data quality gate: drop tickers with too many missing core fundamentals
        miss_pct = count_missing_metrics(i, core_metrics_check)
        if miss_pct >= DATA_MISSING_PCT_THRESHOLD:
            print(f"Skipping {ticker}: too many missing core fundamentals ({round(miss_pct*100)}%)")
            return None

        mkt_cap = i.get('marketCap')
        rev_growth = i.get('revenueGrowth', 0) * 100 if i.get('revenueGrowth') else 0
        ebitda_margin = i.get('ebitdaMargins', 0) * 100 if i.get('ebitdaMargins') else 0
        gross_margin = i.get('grossMargins', 0) * 100 if i.get('grossMargins') else 0
        roe = i.get('returnOnEquity', 0) * 100 if i.get('returnOnEquity') else 0
        fcf = i.get('freeCashflow', 0)
        fcf_yield = (fcf / mkt_cap * 100) if mkt_cap and fcf else 0
        pe = i.get('trailingPE') or i.get('forwardPE') or 50
        peg = i.get('pegRatio') or 2
        high_52w = i.get('fiftyTwoWeekHigh', hist['Close'].max())
        current_price = hist['Close'].iloc[-1]
        dist_from_high = ((current_price / high_52w) - 1) * 100
        ret_3m = hist['Close'].pct_change(63).iloc[-1] * 100 if len(hist) > 63 else 0
        ret_6m = hist['Close'].pct_change(126).iloc[-1] * 100 if len(hist) > 126 else 0
        ret_12m = hist['Close'].pct_change(252).iloc[-1] * 100 if len(hist) > 252 else 0
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs.iloc[-1])) if not rs.empty else 50
        rule_of_40 = rev_growth + ebitda_margin
        sector = i.get('sector', 'Unknown')
        
        ma20 = hist['Close'].rolling(20).mean()
        above_ma = hist['Close'] > ma20
        persistence = above_ma.tail(60).sum() / 60 * 100

        # NEW LONG-TERM BALANCE SHEET / MOAT / SHAREHOLDER RETURN METRICS
        debt_ebitda = i.get('debtToEbitda', 5)
        interest_cov = i.get('interestCoverage', 3)
        share_dilution_1y = i.get('sharesOutstandingChange', 0) * 100
        div_yield = i.get('dividendYield', 0) * 100
        buyback_yield = i.get('buybackYield', 0) * 100
        total_shareholder_yld = div_yield + buyback_yield
        rd_pct_rev = (i.get('researchDevelopment', 0) / i.get('totalRevenue', 1)) * 100 if i.get('totalRevenue') else 0

        # Calculate rolling 3Y FCF consistency locally instead of unreliable info field
        fcf_hist = s.cashflow.loc["Free Cash Flow"] if "Free Cash Flow" in s.cashflow.index else pd.Series([0])
        fcf_3y_consistent = 1 if (fcf_hist.tail(3) > 0).all() else 0

        return {
            "Ticker": ticker, "Name": i.get('longName', ticker), "Sector": sector,
            "TrailingPE": pe, "PEG": peg, "FCF_Yield": fcf_yield, "ROE": roe,
            "GrossMargin": gross_margin, "RevGrowth": rev_growth, "RuleOf40": rule_of_40,
            "DistFrom52wHigh": dist_from_high, "Ret_3M": ret_3m, "Ret_6M": ret_6m, "Ret_12M": ret_12m,
            "RSI": rsi, "Persistence": persistence,
            # New long-term risk/moat/shareholder return columns
            "DebtToEBITDA": debt_ebitda,
            "InterestCoverage": interest_cov,
            "ShareDilution1Y": share_dilution_1y,
            "TotalShareholderYld": total_shareholder_yld,
            "RD_Pct_Revenue": rd_pct_rev,
            "FCF_3Y_Consistent": fcf_3y_consistent
        }
    except Exception as e:
        print(f"Fetch failed {ticker}: {str(e)}")
        return None

# ================== GROK API CALL (ORIGINAL, PARAMS PULLED FROM CFG) ==================
def call_grok_api(prompt):
    if not GROK_API_KEY:
        return "No API key configured"
    
    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": CFG["GROK_MODEL"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": CFG["GROK_TEMP"],
        "max_tokens": CFG["GROK_MAX_TOKENS"]
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=90)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Grok API error: {e}")
        return "Error"

# ================== REWRITTEN MARKET REGIME DETECTION (NEW: Macro Index HMM, no df arg) ==================
def detect_regime():
    macro_features = get_macro_market_timeseries()
    if len(macro_features) < 3:
        return "Bull/Trend"
    
    model = hmm.GaussianHMM(n_components=3, random_state=42)
    model.fit(macro_features)
    regime = model.predict(macro_features)[-1]
    names = ["Bull/Trend", "Mean-Reversion", "High-Vol"]
    print(f"📊 Equity Market Regime: {names[regime]}")
    return names[regime]

# ================== ADVANCED FACTORS (AMENDED: Add Turnaround Score + CREATE MOMENTUM_SCORE FIX) ==================
def add_advanced_factors(df):
    df['Vol_6M'] = df['Ret_6M'].rolling(window=20, min_periods=1).std()
    df['LowVol_Score'] = (-df['Vol_6M']).rank(pct=True, method='average').round(4)
    
    try:
        qqq = yf.download("QQQ", period="1y", progress=False)['Close']
        qqq_3m = qqq.pct_change(63).iloc[-1] * 100 if len(qqq) > 63 else 0
        excess = df['Ret_3M'] - qqq_3m
        df['RelStrength_Score'] = excess.rank(pct=True, method='average').round(4)
    except:
        df['RelStrength_Score'] = df['Ret_3M'].rank(pct=True, method='average').round(4)
    
    df['Earnings_Momentum'] = df['RevGrowth'].rank(pct=True, method='average').round(4)
    df['Persistence_Score'] = df['Persistence'].rank(pct=True, method='average').round(4)
    df['Liquidity_Score'] = 0.5
    df['Congress_Score'] = 0.5

    # Aggregate unified Momentum_Score (fix KeyError)
    df["Momentum_Score"] = (
        df["RelStrength_Score"] * 0.55 +
        df["Earnings_Momentum"] * 0.30 +
        df["Persistence_Score"] * 0.15
    ).round(4)

    # NEW Turnaround Value Score for deep undervalued long holds
    turnaround_mask = (df["Ret_12M"] < 0) & (df["ROE"] > df["ROE"].rolling(4).mean()) & (df["FCF_Yield"] > 0)
    df["Turnaround_Boost"] = np.where(turnaround_mask, normalize(df["ROE"] + df["FCF_Yield"]), 0)
    return df

# ================== FILE CLEANUP (ORIGINAL, retention explicit pass-through) ==================
def cleanup_old_files(directory, pattern, keep=None):
    if keep is None:
        if "csv" in pattern.lower():
            keep = CFG["CSV_HISTORY_KEEP_DAYS"]
        else:
            keep = CFG["MD_SCAN_KEEP_DAYS"]
    try:
        files = sorted(Path(directory).glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
        for f in files[keep:]:
            f.unlink()
    except Exception:
        pass

# ================== MAIN SCRIPT EXECUTION ==================
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Nasdaq Elite Scorer - NASDAQ 250 + Obsidian Optimized | Long/Medium Term Amended Build")
    print("=" * 70)

    data = []
    for ticker in TICKERS:
        d = get_stock_data(ticker)
        if d:
            data.append(d)
        time.sleep(random.uniform(0.6, 1.3))
    
    df = pd.DataFrame(data)
    print(f"✅ Fetched {len(df)} valid stocks after data quality filters")

    # Preinitialize all scoring columns to eliminate undefined reference risk
    df["Moat_Score"] = 0.0
    df["Quality_Score"] = 0.0
    df["Growth_Score"] = 0.0
    df["Value_Score"] = 0.0
    df["Tech_Score"] = 0.0
    df["Normalization_Type"] = "Sector-Based"
    full_market_norm_pool = df.copy()
    for sector in df["Sector"].unique():
        mask = df["Sector"] == sector
        sector_subset = df[mask]
        if len(sector_subset) >= 2:
            # Original intra-sector normalization preserved
            df.loc[mask, "Quality_Score"] = (normalize(df.loc[mask, "GrossMargin"]) * 0.20 +
                                            normalize(df.loc[mask, "ROE"]) * 0.20 +
                                            normalize(df.loc[mask, "FCF_Yield"]) * 0.30 +
                                            normalize(df.loc[mask, "RuleOf40"]) * 0.15 +
                                            normalize(-df.loc[mask, "DebtToEBITDA"]) * 0.075 +
                                            normalize(df.loc[mask, "TotalShareholderYld"]) * 0.075)
            df.loc[mask, "Growth_Score"] = normalize(df.loc[mask, "RevGrowth"])
            df.loc[mask, "Value_Score"] = (normalize(1 / df.loc[mask, "TrailingPE"].clip(lower=1)) * 0.35 +
                                           normalize(1 / df.loc[mask, "PEG"].clip(lower=0.1)) * 0.35 +
                                           normalize(df.loc[mask, "FCF_Yield"]) * 0.30)
            df.loc[mask, "Tech_Score"] = normalize(100 - df.loc[mask, "RSI"])
            # New Moat sub-score baked into Quality bucket
            df.loc[mask, "Moat_Score"] = (normalize(df.loc[mask, "RD_Pct_Revenue"]) * 0.5 + df.loc[mask, "FCF_3Y_Consistent"] * 0.5)
        else:
            # Single ticker sector: normalize vs entire market pool
            df.loc[mask, "Normalization_Type"] = "Market-Wide"
            df.loc[mask, "Quality_Score"] = (normalize(full_market_norm_pool.loc[:, "GrossMargin"]).iloc[mask.index] * 0.20 +
                                            normalize(full_market_norm_pool.loc[:, "ROE"]).iloc[mask.index] * 0.20 +
                                            normalize(full_market_norm_pool.loc[:, "FCF_Yield"]).iloc[mask.index] * 0.30 +
                                            normalize(full_market_norm_pool.loc[:, "RuleOf40"]).iloc[mask.index] * 0.15 +
                                            normalize(-full_market_norm_pool.loc[:, "DebtToEBITDA"]).iloc[mask.index] * 0.075 +
                                            normalize(full_market_norm_pool.loc[:, "TotalShareholderYld"]).iloc[mask.index] * 0.075)
            df.loc[mask, "Growth_Score"] = normalize(full_market_norm_pool["RevGrowth"]).iloc[mask.index]
            df.loc[mask, "Value_Score"] = (normalize(1 / full_market_norm_pool["TrailingPE"].clip(lower=1)).iloc[mask.index] * 0.35 +
                                           normalize(1 / full_market_norm_pool["PEG"].clip(lower=0.1)).iloc[mask.index] * 0.35 +
                                           normalize(full_market_norm_pool["FCF_Yield"]).iloc[mask.index] * 0.30)
            df.loc[mask, "Tech_Score"] = normalize(100 - full_market_norm_pool["RSI"]).iloc[mask.index]
            df.loc[mask, "Moat_Score"] = (normalize(full_market_norm_pool["RD_Pct_Revenue"]).iloc[mask.index] * 0.5 + df.loc[mask, "FCF_3Y_Consistent"] * 0.5)

    # DYNAMIC REGIME-BASED COMPOSITE FACTOR WEIGHTS (NEW)
    equity_regime = detect_regime()
    bond_data = get_bond_signal()
    if bond_data:
        BOND_RISK_SCORE = bond_data.get('risk_score') or 50
        bond_regime = bond_data.get('regime', 'NEUTRAL')
        print(f"📈 Bond Risk Score: {BOND_RISK_SCORE}/100 | Regime: {bond_regime}")
    else:
        BOND_RISK_SCORE = 50
        bond_regime = 'NEUTRAL'
        print("⚠️ bond_signal.json not found")

    # Weight rotation lookup table for long-term positioning
    weight_sets = {
        "risk_off": {"quality":0.50, "growth":0.10, "value":0.30, "momentum":0.06, "tech":0.04},
        "neutral": {"quality":0.40, "growth":0.25, "value":0.20, "momentum":0.08, "tech":0.07},
        "risk_on": {"quality":0.35, "growth":0.30, "value":0.15, "momentum":0.12, "tech":0.08}
    }
    if BOND_RISK_SCORE >=70 or equity_regime == "High-Vol":
        active_weight_set_name = "risk_off"
        active_weights = weight_sets[active_weight_set_name]
    elif BOND_RISK_SCORE <=30 and equity_regime == "Bull/Trend":
        active_weight_set_name = "risk_on"
        active_weights = weight_sets[active_weight_set_name]
    else:
        active_weight_set_name = "neutral"
        active_weights = weight_sets[active_weight_set_name]

    # AMENDED 6M Gain Filter: soft penalty instead of hard drop, hard cap only for weak names
    # Step1: Hard filter unconditionally remove extreme gain tickers over hard cap
    df = df[df["Ret_6M"] <= MAX_6M_GAIN_HARD_CAP]
    # Step2: Penalty for stocks above warning threshold with poor fundamentals
    high_gain_mask = df["Ret_6M"] > MAX_6M_GAIN_WARNING_THRESHOLD
    weak_fund_mask = (df["PEG"] > 2.0) | (df["RuleOf40"] < 40)
    df.loc[high_gain_mask & weak_fund_mask, "Composite_Score"] = 0.0

    # Original mandatory hard filters preserved
    df = df[df["Quality_Score"] >= MIN_QUALITY]
    df = df[df["PEG"] <= MAX_PEG]

    # Run advanced factors (creates Momentum_Score now)
    df = add_advanced_factors(df)
    df = df.fillna(0.5)

    # Composite Score with dynamic regime weights (Momentum_Score now exists)
    df["Composite_Score"] = (
        (df["Quality_Score"] + df["Moat_Score"]*0.12) * active_weights["quality"] +
        df["Growth_Score"] * active_weights["growth"] +
        df["Value_Score"] * active_weights["value"] +
        df["Momentum_Score"] * active_weights["momentum"] +
        df["Tech_Score"] * active_weights["tech"]
    ) * 100

    # Base Final Score formula preserved, add Turnaround Boost multiplier
    df['Final_Score'] = (
        df['Composite_Score'] * 0.52 +
        df['LowVol_Score'] * 0.14 +
        df['RelStrength_Score'] * 0.12 +
        df['Earnings_Momentum'] * 0.10 +
        df['Persistence_Score'] * 0.10 +
        df['Congress_Score'] * 0.01 +
        df['Liquidity_Score'] * 0.01
    )
    # Apply mild positive boost to deep value turnaround candidates
    df["Final_Score"] = df["Final_Score"] * (1 + (df["Turnaround_Boost"] * 0.11))

    # Original bond risk tier penalties preserved
    if BOND_RISK_SCORE >= 90:
        growth_factor = df['Earnings_Momentum'] + df['RelStrength_Score']
        df['Final_Score'] = df['Final_Score'] * (1 - 0.15 * growth_factor)
    elif BOND_RISK_SCORE >= 70:
        df['Final_Score'] *= 0.94
    elif BOND_RISK_SCORE >= 50:
        df['Final_Score'] *= 0.97

    df = df.sort_values("Final_Score", ascending=False).reset_index(drop=True)

    # EXPANDED SECTOR CAP LOGIC: Healthcare + Tech + Consumer discretionary caps
    def apply_sector_cap_penalty(df_top30_full, sector_keyword, max_allowed, penalty_multi=0.75):
        mask = df_top30_full['Sector'].str.contains(sector_keyword, case=False, na=False)
        top_slice = df_top30_full.head(30)
        sector_positions = top_slice[mask].index
        excess_count = len(sector_positions) - max_allowed
        if excess_count > 0:
            penalize_indices = sector_positions[-excess_count:]
            df_top30_full.loc[penalize_indices, 'Final_Score'] *= penalty_multi
        return df_top30_full

    df = apply_sector_cap_penalty(df, "Health|Biotech|Pharma", MAX_HEALTHCARE_IN_TOP30)
    df = apply_sector_cap_penalty(df, "Technology|Software|Semiconductor", MAX_TECH_TOP30)
    df = apply_sector_cap_penalty(df, "Consumer Discretionary|Retail", MAX_CONSUMER_TOP30)
    df = df.sort_values("Final_Score", ascending=False).reset_index(drop=True)

    # ================== FLEXIBLE SENTIMENT OVERRIDE (AMENDED DYNAMIC WEIGHT TIED TO BOND RISK) ==================
    overrides = load_sentiment_overrides()
    applied = 0
    if overrides:
        df['Sentiment_Score'] = df['Ticker'].map(overrides).fillna(0.5).clip(0, 1)
        applied = df['Ticker'].isin(overrides).sum()
        print(f"✅ Applied {applied} sentiment overrides")
    else:
        df['Sentiment_Score'] = 0.5
        print("ℹ️ No sentiment overrides found for today")

    # Dynamic sentiment weight scaling
    risk_scalar = BOND_RISK_SCORE / 100
    active_sent_weight = BASE_SENTIMENT_WEIGHT + (risk_scalar * (MAX_SENTIMENT_WEIGHT_HIGH_RISK - BASE_SENTIMENT_WEIGHT))
    passive_weight = 1 - active_sent_weight
    df['Final_Score'] = (df['Final_Score'] * passive_weight) + (df['Sentiment_Score'] * active_sent_weight)
    df = df.sort_values("Final_Score", ascending=False).reset_index(drop=True)

    # ================== SAVE OUTPUT (ORIGINAL FILE PATH LOGIC PRESERVED, FIXED SYNTAX HEADER) ==================
    today = datetime.now().strftime('%Y%m%d')
    history = Path.home() / "Desktop/OpenJarvis/nasdaq_history"
    history.mkdir(parents=True, exist_ok=True)
    csv_path = history / f"nasdaq_score_{today}.csv"
    df.to_csv(csv_path, index=False)

    cleanup_old_files(history, "*.csv")

    OBSIDIAN_DIR = os.path.expanduser("~/Desktop/Trading/Nasdaq")
    SCANS_DIR = os.path.join(OBSIDIAN_DIR, "Scans")
    os.makedirs(SCANS_DIR, exist_ok=True)

    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
    MD_PATH = os.path.join(SCANS_DIR, f"Nasdaq_Scan_{TIMESTAMP}.md")

    header = f"""# Nasdaq Elite Scan - {datetime.now().strftime("%Y-%m-%d %H:%M")}

**Equity Regime**: {equity_regime} | **Bond Risk Score**: {BOND_RISK_SCORE}/100 | **Bond Regime**: {bond_regime}
**Sentiment Overrides Applied**: {applied} | Active Sentiment Weight: {round(active_sent_weight,3)}
Dynamic Factor Weight Set: {active_weight_set_name}
"""

    table = df.head(30)[[
        'Ticker', 'Final_Score', 'Composite_Score', 'Moat_Score', 'Turnaround_Boost',
        'LowVol_Score', 'RelStrength_Score', 'Earnings_Momentum', 'Sentiment_Score',
        'Quality_Score', 'Sector', 'TotalShareholderYld', 'Momentum_Score'
    ]].round(4).to_markdown(index=False)

    with open(MD_PATH, "w") as f:
        f.write(header)
        f.write("## Top 30 Ranked Long-Term Hold Candidates\n\n")
        f.write(table)
        f.write(f"\n\n**Raw CSV Full Dataset**: [[Data/nasdaq_score_{today}.csv]]")

    # Grok prompt modified to prioritize multi-year undervalued compounders
    if TEST_MODE:
        brief = "🧪 TEST MODE ENABLED - Grok API call skipped."
    else:
        with open(MD_PATH, "r") as f:
            content = f.read()
        prompt = f"""You are a veteran multi-strategy PM focused on medium/long term NASDAQ holdings (6–36 month horizon), ignore short-term trade signals. Analyze this factor scan output, flag undervalued turnaround compounders, high moat low-leverage names, and identify macro risks to avoid multi-quarter drawdowns.
Scan Data:
{content[:7200]}"""
        brief = call_grok_api(prompt)

    with open(MD_PATH, "a") as f:
        f.write("\n\n## Grok Long-Term Trading Brief\n\n" + brief)

    cleanup_old_files(Path(SCANS_DIR), "*.md")

    print(f"✅ Scan complete → {MD_PATH}")
    print("=" * 70)
