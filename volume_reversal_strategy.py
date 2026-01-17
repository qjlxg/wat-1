import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from joblib import Parallel, delayed

# =================================================================================
# 战法名称：极度缩量反包战法 (完整回测版)
# ---------------------------------------------------------------------------------
# 战法核心逻辑 (买入信号触发点):
# 1. 价格过滤：最新收盘价在 5.0 - 20.0 元之间。
# 2. 选股过滤：只要深沪A股，排除 ST 股和创业板 (代码30开头)。
# 3. 前期活跃：5个交易日内曾出现过一根明显的放量阳线 (成交量 > 20日均量 * 1.5)。
# 4. 缩量回调：最近 2 个交易日成交量极度萎缩，均低于 20日均量的 0.7 倍 (地量洗盘)。
# 5. 反包确认：今日收盘价 > 昨日最高价，且今日成交量开始温和放大。
# 
# 虚拟账本回测逻辑:
# - 一旦今日满足信号，脚本会回溯该个股过去所有符合上述条件的日期。
# - 统计该信号出现后 7天、14天、20天、60天的持仓表现，计算胜率及期望收益。
# =================================================================================

STRATEGY_NAME = "backtest_reversal_strategy"
DATA_DIR = "stock_data"
NAMES_FILE = "stock_names.csv"

def backtest_virtual_ledger(df, signal_idx):
    """
    计算该信号点在历史上持有不同天数后的收益率
    """
    results = {}
    periods = {'7天': 7, '14天': 14, '20天': 20, '60天': 60}
    buy_price = df['收盘'].iloc[signal_idx]
    
    for label, days in periods.items():
        future_idx = signal_idx + days
        if future_idx < len(df):
            profit = (df['收盘'].iloc[future_idx] - buy_price) / buy_price
            results[label] = profit
        else:
            results[label] = None
    return results

def analyze_stock(file_path, names_dict):
    try:
        # 优化1：只读取必要列，大幅降低内存消耗，防止 GitHub Action 卡死
        cols = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '换手率', '涨跌幅']
        df = pd.read_csv(file_path, usecols=cols)
        if len(df) < 100: return None
        
        code = os.path.basename(file_path).split('.')[0]
        name = names_dict.get(code, "未知")
        
        # 基础过滤逻辑
        last_row = df.iloc[-1]
        curr_price = last_row['收盘']
        if not (5.0 <= curr_price <= 20.0) or "ST" in name or code.startswith("30"):
            return None

        # 预计算技术指标 (向量化)
        vol_arr = df['成交量'].values
        ma20_vol = df['成交量'].rolling(20).mean().values
        close_arr = df['收盘'].values
        high_arr = df['最高'].values
        
        # --- 今日信号捕获 ---
        i = len(df) - 1
        # A. 前期活跃 (5日内放量)
        cond_active = (vol_arr[i-5:i] > ma20_vol[i-5:i] * 1.5).any()
        # B. 极度缩量 (近2日地量)
        cond_shrink = (vol_arr[i-1] < ma20_vol[i] * 0.7) and (vol_arr[i-2] < ma20_vol[i] * 0.7)
        # C. 今日反包
        cond_reversal = (close_arr[i] > high_arr[i-1]) and (close_arr[i] > df['开盘'].iloc[i])
        
        if not (cond_active and cond_shrink and cond_reversal):
            return None

        # --- 如果今日出信号，则开启历史回测全量复盘 ---
        all_history_profits = []
        # 遍历历史寻找该战法的“基因”
        for j in range(20, len(df) - 21): # 预留空间计算收益
            h_active = (vol_arr[j-5:j] > ma20_vol[j-5:j] * 1.5).any()
            h_shrink = (vol_arr[j-1] < ma20_vol[j] * 0.7) and (vol_arr[j-2] < ma20_vol[j] * 0.7)
            h_reversal = (close_arr[j] > high_arr[j-1]) and (close_arr[j] > df['开盘'].iloc[j])
            
            if h_active and h_shrink and h_reversal:
                # 获取该历史点位 20 天后的表现
                ledger = backtest_virtual_ledger(df, j)
                if ledger['20天'] is not None:
                    all_history_profits.append(ledger['20天'])

        # 统计指标
        hit_count = len(all_history_profits)
        win_rate = np.mean([1 if p > 0 else 0 for p in all_history_profits]) if hit_count > 0 else 0
        avg_ret = np.mean(all_history_profits) if hit_count > 0 else 0

        # --- 自动复盘结论 ---
        strength = "⭐⭐⭐⭐⭐" if win_rate > 0.6 and avg_ret > 0.03 else "⭐⭐⭐"
        if win_rate > 0.7:
            advice = "历史高胜率标的，重点关注"
        elif win_rate > 0.5:
            advice = "形态合规，建议小仓试错"
        else:
            advice = "历史表现一般，仅作观察"

        return {
            "日期": last_row['日期'], "代码": code, "名称": name, "现价": curr_price,
            "涨跌幅": f"{last_row['涨跌幅']}%", "换手率": last_row['换手率'],
            "该股历史触发次数": hit_count,
            "历史20日胜率": f"{win_rate*100:.1f}%",
            "历史20日均益": f"{avg_ret*100:.2f}%",
            "信号强度": strength, "复盘建议": advice
        }
    except Exception:
        return None

def main():
    if not os.path.exists(NAMES_FILE):
        print(f"错误: 找不到 {NAMES_FILE}")
        return

    # 加载名称
    names_df = pd.read_csv(NAMES_FILE)
    names_dict = dict(zip(names_df['code'].astype(str).str.zfill(6), names_df['name']))
    
    # 扫描数据
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始全量并行复盘: {len(files)} 只股票")
    
    # 优化2：n_jobs=2。GitHub Action 服务器内存有限，2个并行能确保不卡死。
    results = Parallel(n_jobs=2)(delayed(analyze_stock)(f, names_dict) for f in files)
    
    valid_results = [r for r in results if r is not None]
    
    if valid_results:
        # 结果按胜率从高到低排序，优中选优
        res_df = pd.DataFrame(valid_results).sort_values(by="历史20日胜率", ascending=False)
        
        now = datetime.now()
        out_dir = now.strftime("%Y-%m")
        os.makedirs(out_dir, exist_ok=True)
        file_name = f"{STRATEGY_NAME}_{now.strftime('%Y%m%d_%H%M')}.csv"
        save_path = os.path.join(out_dir, file_name)
        
        res_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"成功！扫描出 {len(res_df)} 个战法信号，结果已存入 {save_path}")
    else:
        print("今日未发现符合战法逻辑的个股。")

if __name__ == "__main__":
    main()
