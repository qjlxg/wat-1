import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from joblib import Parallel, delayed

# =================================================================================
# 战法名称：极度缩量反包战法 (虚拟账本回测完整版)
# 备注：旨在寻找主力洗盘彻底、缩量到极致后的反转爆发点
# =================================================================================

STRATEGY_NAME = "backtest_reversal_strategy"
DATA_DIR = "stock_data"
NAMES_FILE = "stock_names.csv"

def get_performance(df, idx):
    """虚拟持仓账本：计算买入后不同周期的真实表现"""
    perf = {}
    for label, days in [('7天', 7), ('14天', 14), ('20天', 20), ('60天', 60)]:
        target = idx + days
        if target < len(df):
            change = (df['收盘'].iloc[target] - df['收盘'].iloc[idx]) / df['收盘'].iloc[idx]
            perf[label] = round(change * 100, 2)
        else:
            perf[label] = None
    return perf

def analyze_stock(file_path, names_dict):
    try:
        # 1. 高速读取
        df = pd.read_csv(file_path, usecols=['日期', '开盘', '收盘', '最高', '最低', '成交量', '涨跌幅', '换手率'])
        if len(df) < 120: return None
        
        code = os.path.basename(file_path).split('.')[0]
        name = names_dict.get(code, "未知")
        
        # 2. 基础过滤 (上海时区/深沪A股/价格区间)
        if "ST" in name or code.startswith("30") or not (5.0 <= df['收盘'].iloc[-1] <= 20.0):
            return None

        # 3. 核心指标预计算
        close = df['收盘'].values
        vol = df['成交量'].values
        high = df['最高'].values
        ma20_vol = df['成交量'].rolling(20).mean().values
        
        # --- 定义“缩量反包”判定函数 ---
        def is_hit(i):
            if i < 20: return False
            # 逻辑：今日阳线且收盘盖过昨日最高价
            c_reversal = (close[i] > high[i-1]) and (close[i] > df['开盘'].iloc[i])
            # 逻辑：前两天成交量萎缩 (地量)
            c_shrink = (vol[i-1] < ma20_vol[i] * 0.75) and (vol[i-2] < ma20_vol[i] * 0.75)
            # 逻辑：10日内有过异动放量 (证明有主力在)
            c_active = (vol[i-10:i] > ma20_vol[i-10:i] * 1.5).any()
            return c_reversal and c_shrink and c_active

        # 4. 全量回测 (扫描该个股过去2年的所有表现)
        all_signals = []
        lookback_limit = max(0, len(df) - 500) # 最近约2年
        for j in range(lookback_limit, len(df) - 1):
            if is_hit(j):
                perf = get_performance(df, j)
                if perf['20天'] is not None:
                    all_signals.append(perf['20天'])

        # 5. 今日信号捕捉
        today_idx = len(df) - 1
        if is_hit(today_idx):
            # 统计历史战绩
            hit_count = len(all_signals)
            win_rate = np.mean([1 if p > 0 else 0 for p in all_signals]) if hit_count > 0 else 0
            avg_return = np.mean(all_signals) if hit_count > 0 else 0
            
            # 强度评估
            strength = "⭐⭐⭐⭐⭐" if win_rate > 0.6 and avg_return > 3 else "⭐⭐⭐"
            advice = "加仓/重仓" if win_rate > 0.7 else "试错/观察"
            
            return {
                "日期": df['日期'].iloc[today_idx],
                "代码": code, "名称": name, "现价": close[today_idx],
                "涨跌幅": df['涨跌幅'].iloc[today_idx],
                "换手率": df['换手率'].iloc[today_idx],
                "战法历史触发数": hit_count,
                "历史20日胜率": f"{win_rate*100:.1f}%",
                "历史平均收益": f"{avg_return:.2f}%",
                "买入信号强度": strength,
                "操作建议": advice
            }
    except:
        return None

def main():
    names_df = pd.read_csv(NAMES_FILE)
    names_dict = dict(zip(names_df['code'].astype(str).str.zfill(6), names_df['name']))
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    print(f"[{datetime.now()}] 启动高性能复盘引擎...")
    # 限制并行，确保稳定
    results = Parallel(n_jobs=2)(delayed(analyze_stock)(f, names_dict) for f in files)
    
    final_list = [r for r in results if r is not None]
    if final_list:
        res_df = pd.DataFrame(final_list).sort_values(by="历史20日胜率", ascending=False)
        out_dir = datetime.now().strftime("%Y-%m")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{STRATEGY_NAME}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
        res_df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"复盘完毕！今日捕获 {len(res_df)} 只潜力股，结果已更新至仓库。")
    else:
        print("今日暂未发现符合条件的『极度缩量反包』信号。")

if __name__ == "__main__":
    main()
