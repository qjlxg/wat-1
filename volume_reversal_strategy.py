import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from joblib import Parallel, delayed

# ==========================================
# 战法：极度缩量反包 (带虚拟持仓账本回测)
# 逻辑：寻找放量异动后，缩量回调至极点，今日反包确认的个股
# ==========================================

STRATEGY_NAME = "backtest_reversal_strategy"
DATA_DIR = "stock_data"
NAMES_FILE = "stock_names.csv"

def analyze_stock(file_path, names_dict):
    try:
        # 加载必要数据
        df = pd.read_csv(file_path, usecols=['日期', '开盘', '收盘', '最高', '最低', '成交量', '涨跌幅', '换手率'])
        if len(df) < 120: return None
        
        code = os.path.basename(file_path).split('.')[0]
        name = names_dict.get(code, "未知")
        
        # 基础过滤
        last_price = df['收盘'].iloc[-1]
        if "ST" in name or code.startswith("30") or not (5.0 <= last_price <= 20.0):
            return None

        # 向量化准备
        close = df['收盘'].values
        vol = df['成交量'].values
        high = df['最高'].values
        ma20_vol = df['成交量'].rolling(20).mean().values
        
        # 战法判定函数
        def is_strategy_hit(i):
            if i < 20: return False
            # 1. 前期活跃：10日内有过放量 (量 > 20日均量 1.5倍)
            active = (vol[i-10:i] > ma20_vol[i-10:i] * 1.5).any()
            # 2. 极度缩量：前两日成交量 < 20日均量 * 0.75
            shrink = (vol[i-1] < ma20_vol[i] * 0.75) and (vol[i-2] < ma20_vol[i] * 0.75)
            # 3. 反包确认：今日收盘 > 昨日最高 且 今日收阳
            reversal = (close[i] > high[i-1]) and (close[i] > df['开盘'].iloc[i])
            return active and shrink and reversal

        # --- 建立“虚拟持仓账本” (扫描历史所有信号点) ---
        history_ledger = []
        # 扫描过去 500 个交易日
        start_scan = max(20, len(df) - 500)
        for j in range(start_scan, len(df) - 1): # -1 是为了排除掉“今天”
            if is_strategy_hit(j):
                # 计算模拟持有收益率
                res = {}
                for days in [7, 14, 20, 60]:
                    target = j + days
                    if target < len(df):
                        res[f'p{days}'] = (close[target] - close[j]) / close[j]
                    else:
                        res[f'p{days}'] = None
                history_ledger.append(res)

        # 统计账本战绩
        hit_count = len(history_ledger)
        win_rate_20d = 0
        avg_ret_20d = 0
        if hit_count > 0:
            ledger_df = pd.DataFrame(history_ledger)
            p20_valid = ledger_df['p20'].dropna()
            if not p20_valid.empty:
                win_rate_20d = (p20_valid > 0).sum() / len(p20_valid)
                avg_ret_20d = p20_valid.mean()

        # --- 判断今日是否触发信号 ---
        today_idx = len(df) - 1
        if is_strategy_hit(today_idx):
            # 综合强度逻辑
            strength = "⭐⭐⭐⭐⭐" if win_rate_20d > 0.6 and avg_ret_20d > 0.05 else "⭐⭐⭐"
            if hit_count == 0: strength = "⭐⭐ (新股或首次触发)"
            
            # 操作建议
            if win_rate_20d > 0.7: advice = "历史强势基因，重仓买入"
            elif win_rate_20d > 0.5: advice = "概率占优，分批建仓"
            else: advice = "历史表现平平，轻仓试错"

            return {
                "日期": df['日期'].iloc[today_idx],
                "代码": code, "名称": name, "现价": last_price,
                "涨跌幅": f"{df['涨跌幅'].iloc[today_idx]}%",
                "换手率": f"{df['换手率'].iloc[today_idx]}%",
                "虚拟账本触发数": hit_count,
                "历史20日胜率": f"{win_rate_20d*100:.1f}%",
                "历史20日均益": f"{avg_ret_20d*100:.2f}%",
                "买入信号强度": strength,
                "操作建议": advice
            }
    except:
        return None

def main():
    if not os.path.exists(NAMES_FILE): return
    names_df = pd.read_csv(NAMES_FILE)
    names_dict = dict(zip(names_df['code'].astype(str).str.zfill(6), names_df['name']))
    
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"[{datetime.now()}] 启动虚拟账本全量回测，扫描 {len(files)} 只标的...")
    
    # 使用 n_jobs=2 稳定运行，防止 Actions 卡死
    results = Parallel(n_jobs=2)(delayed(analyze_stock)(f, names_dict) for f in files)
    
    final_hits = [r for r in results if r is not None]
    if final_hits:
        res_df = pd.DataFrame(final_hits).sort_values(by="历史20日胜率", ascending=False)
        
        # 存储路径
        now = datetime.now()
        out_dir = now.strftime("%Y-%m")
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, f"{STRATEGY_NAME}_{now.strftime('%Y%m%d_%H%M')}.csv")
        
        res_df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"复盘完成！今日发现 {len(res_df)} 只符合『缩量反包』战法且历史胜率较高的标的。")
    else:
        print("今日暂未发现符合条件的信号（尝试放宽缩量条件或检查数据更新）。")

if __name__ == "__main__":
    main()
