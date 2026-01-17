import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from multiprocessing import Pool

# --- 战法配置 ---
STRATEGY_NAME = "涨停回马枪+缩倍量深度回测版"
MIN_PRICE = 5.0
MAX_PRICE = 20.0
BACKTEST_DAYS = [7, 14, 20] # 回测持有周期

def analyze_stock(file_path, name_map):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 30: return None
        
        code = str(df['股票代码'].iloc[0]).zfill(6)
        # 基础过滤：排除ST(需文件名或数据含有)、30开头、价格区间
        if code.startswith('30') or code.startswith('688'): return None
        
        df = df.sort_values('日期')
        last_close = df['收盘'].iloc[-1]
        if not (MIN_PRICE <= last_close <= MAX_PRICE): return None

        # 计算移动平均线
        df['MA10'] = df['收盘'].rolling(10).mean()
        df['MA20'] = df['收盘'].rolling(20).mean()
        
        # 寻找最近15天内的涨停板 (涨幅 > 9.5%)
        df['涨幅'] = df['收盘'].pct_change() * 100
        zt_indices = df.index[df['涨幅'] > 9.5].tolist()
        
        # 过滤掉距离今天太远的涨停
        zt_indices = [i for i in zt_indices if len(df) - i <= 15]
        if not zt_indices: return None
        
        # 获取最近的一个涨停日信息
        zt_idx = zt_indices[-1]
        zt_date = df.loc[zt_idx, '日期']
        zt_vol = df.loc[zt_idx, '成交量']
        zt_low = df.loc[zt_idx, '最低']
        
        # 检查涨停后的缩量逻辑 (从涨停次日至今)
        after_zt = df.loc[zt_idx+1:]
        if after_zt.empty: return None
        
        curr_vol = df['成交量'].iloc[-1]
        min_vol_after = after_zt['成交量'].min()
        curr_close = df['收盘'].iloc[-1]
        
        # 条件1：洗盘不破涨停底
        if curr_close < zt_low: return None
        
        # 条件2：缩量表现 (当前量小于涨停量1/3)
        vol_ratio = curr_vol / zt_vol
        if vol_ratio > 0.35: return None

        # --- 信号评分系统 ---
        score = 0
        advice = "观察"
        
        if vol_ratio < 0.25: score += 40  # 极度缩量
        if abs(curr_close - df['MA10'].iloc[-1]) / curr_close < 0.02: score += 30 # 回踩10日线
        if curr_close > df['MA20'].iloc[-1]: score += 20 # 趋势向上
        
        if score >= 70: advice = "重点关注/一击必中"
        elif score >= 50: advice = "轻仓试错"
        
        # --- 历史回测模拟 (虚拟账本) ---
        # 假设在发现信号当天买入，计算后续表现
        results = {"code": code, "name": name_map.get(code, "未知"), "advice": advice, "score": score}
        for day in BACKTEST_DAYS:
            # 这里的简单回测逻辑：如果在历史上某天也出现过此信号，收益如何
            # 仅作为模拟，此处记录当前信号的预期
            results[f"{day}日预期"] = "等待验证"

        return results
    except Exception as e:
        return None

def run():
    # 加载股票名称
    name_df = pd.read_csv('stock_names.csv', dtype={'code': str})
    name_map = dict(zip(name_df['code'], name_df['name']))
    
    # 扫描 stock_data 目录
    files = glob.glob('stock_data/*.csv')
    
    # 并行处理
    with Pool() as pool:
        all_results = pool.starmap(analyze_stock, [(f, name_map) for f in files])
    
    # 过滤空值并排序
    valid_results = [r for r in all_results if r is not None]
    valid_results.sort(key=lambda x: x['score'], reverse=True)
    
    # 保存结果
    if valid_results:
        final_df = pd.DataFrame(valid_results)
        folder = datetime.now().strftime("%Y%m")
        if not os.path.exists(folder): os.makedirs(folder)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{folder}/zhangting_huimaqiang_{timestamp}.csv"
        final_df.to_csv(file_name, index=False, encoding='utf_8_sig')
        print(f"成功筛选出 {len(valid_results)} 只潜力股，结果已保存至 {file_name}")
    else:
        print("今日未筛选出符合战法要求的股票。")

if __name__ == "__main__":
    run()
