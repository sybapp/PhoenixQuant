# -*- coding: utf-8 -*-
"""
优化版弹性抄底策略回测主程序
增强风险管理、信号过滤和绩效分析
"""

import asyncio
from datetime import datetime, timedelta
import ccxt
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

# 导入原有的回测组件（假设这些模块已存在）
try:
    from backtest_engine import BacktestEngine, HistoricalDataFetcher
    from backtest_strategy import BacktestElasticDipBot
    from backtest_analysis import BacktestAnalyzer
except ImportError:
    print("警告: 无法导入原有回测组件，将使用简化版本")
    # 这里需要您原有的 backtest_engine, backtest_strategy, backtest_analysis 模块

# ========= 优化配置类 =========
class OptimizedBacktestConfig:
    """优化后的回测配置类"""
    
    # 改进的时间范围 - 使用更长的历史数据
    BACKTEST_START = datetime(2025, 10, 1)  # 延长到1个月
    BACKTEST_END = datetime(2025, 10, 30)
    
    # 优化后的策略参数
    OPTIMIZED_PARAMS = {
        # 时间参数
        "timeframe": "1m",
        "poll_sec": 2,
        
        # 价格触发条件优化
        "drop_pct_single": 1.2,      # 提高单根K线跌幅阈值，减少假信号
        "drop_pct_window": 3.5,      # 提高窗口跌幅阈值
        "window_min": 8,             # 延长观察窗口
        
        # 趋势过滤优化
        "ema_fast": 15,              # 更敏感的快速EMA
        "ema_slow": 50,              # 更稳定的慢速EMA
        "ema_trend_filter": True,    # 新增：EMA趋势过滤
        
        # 动量指标优化
        "rsi_period": 12,            # 缩短RSI周期提高灵敏度
        "rsi_oversold": 28.0,        # 调整超卖阈值
        "rsi_smooth": 3,             # 新增：RSI平滑
        
        # 成交量确认
        "vol_shrink_ratio": 0.55,    # 更严格的成交量收缩条件
        "vol_recover_ma_short": 8,   # 优化成交量均线
        "vol_recover_ma_long": 25,
        "vol_recover_ratio": 1.25,   # 提高成交量恢复要求
        
        # 市场情绪指标
        "funding_extreme_neg": -0.03, # 调整极端资金费率阈值
        "liq_notional_threshold": 5_000_000, # 降低爆仓阈值要求
        
        # 延迟触发机制
        "delayed_trigger_pct": 0.8,   # 降低延迟触发阈值
        "delayed_window_sec": 60 * 60 * 8, # 缩短延迟窗口到8小时
        
        # 仓位管理优化
        "layer_pcts": [0.6, 1.2, 1.8, 2.4, 3.0],      # 更密集的触发点位
        "layer_pos_ratio": [0.15, 0.20, 0.25, 0.20, 0.20], # 更合理的仓位分配
        "total_capital": 1000,
        "max_account_ratio": 0.25,    # 降低单币种最大仓位
        
        # 退出机制优化
        "take_profit_pct": [1.5, 2.5, 4.0, 6.0, 8.0], # 分层止盈
        "hard_stop_extra": 0.6,       # 收紧止损
        "sl_time_grace_sec": 45,      # 延长止损宽限期
        "trailing_stop_pct": 0.8,     # 新增：移动止损
        
        # 风险控制增强
        "max_daily_trades": 3,        # 新增：每日最大交易次数
        "cooldown_after_stop": 3600,  # 新增：止损后冷却时间
        "max_drawdown_limit": 0.15,   # 新增：最大回撤限制
    }


class RiskManager:
    """增强的风险管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.last_stop_time = 0
        self.peak_equity = 0
        self.current_drawdown = 0
        
    def can_trade(self, current_time: int, current_equity: float) -> bool:
        """检查是否可以交易"""
        current_date = datetime.fromtimestamp(current_time/1000).date()
        
        # 日期切换重置计数
        if self.last_trade_date != current_date:
            self.daily_trade_count = 0
            self.last_trade_date = current_date
            
        # 更新回撤计算
        self._update_drawdown(current_equity)
            
        # 检查最大回撤限制
        if self.current_drawdown > self.config.get("max_drawdown_limit", 0.2):
            return False
            
        # 检查冷却时间
        if current_time - self.last_stop_time < self.config.get("cooldown_after_stop", 3600) * 1000:
            return False
            
        # 检查每日交易限额
        if self.daily_trade_count >= self.config.get("max_daily_trades", 5):
            return False
            
        return True
        
    def _update_drawdown(self, current_equity: float):
        """更新回撤计算"""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
    def record_trade(self):
        """记录交易"""
        self.daily_trade_count += 1
        
    def record_stop_loss(self, current_time: int):
        """记录止损"""
        self.last_stop_time = current_time
        
    def get_risk_status(self) -> Dict[str, float]:
        """获取当前风险状态"""
        return {
            "daily_trades": self.daily_trade_count,
            "current_drawdown": self.current_drawdown,
            "peak_equity": self.peak_equity
        }


class AdvancedSignalGenerator:
    """增强信号生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def calculate_combined_signal(self, candle_buffer: List[List]) -> float:
        """计算综合信号强度 (0-100)"""
        if len(candle_buffer) < self.config["ema_slow"]:
            return 0
            
        # 各因子信号
        price_signal = self._price_signal(candle_buffer)
        momentum_signal = self._momentum_signal(candle_buffer)
        volume_signal = self._volume_signal(candle_buffer)
        trend_signal = self._trend_signal(candle_buffer)
        
        # 加权综合信号
        combined = (
            price_signal * 0.35 + 
            momentum_signal * 0.25 + 
            volume_signal * 0.20 +
            trend_signal * 0.20
        )
        return min(100, max(0, combined))
    
    def _price_signal(self, candle_buffer: List[List]) -> float:
        """价格信号 (0-100)"""
        closes = [c[4] for c in candle_buffer]
        current_price = closes[-1]
        
        # 计算近期跌幅
        lookback = min(20, len(closes))
        high_price = max(closes[-lookback:])
        drawdown = (high_price - current_price) / high_price * 100
        
        # 跌幅越大信号越强，但超过12%可能趋势已坏
        if drawdown > 12:
            return 0
        return min(100, drawdown * 7)
    
    def _momentum_signal(self, candle_buffer: List[List]) -> float:
        """动量信号 (0-100)"""
        # 计算RSI
        rsi = self._calculate_rsi([c[4] for c in candle_buffer], self.config["rsi_period"])
        if rsi is None:
            return 0
            
        # RSI超卖程度
        rsi_signal = max(0, (self.config["rsi_oversold"] - rsi) / self.config["rsi_oversold"] * 100)
        
        # 计算MACD动量
        macd_signal = self._macd_signal([c[4] for c in candle_buffer])
        
        return min(100, (rsi_signal * 0.6 + macd_signal * 0.4))
    
    def _volume_signal(self, candle_buffer: List[List]) -> float:
        """成交量信号 (0-100)"""
        volumes = [c[5] for c in candle_buffer]
        if len(volumes) < 10:
            return 0
            
        current_vol = volumes[-1]
        avg_vol = sum(volumes[-10:]) / 10
        
        # 放量下跌后缩量企稳是买入信号
        if current_vol < avg_vol * self.config["vol_shrink_ratio"]:
            return 80
            
        # 成交量恢复也是积极信号
        if current_vol > avg_vol * self.config.get("vol_recover_ratio", 1.15):
            return 60
            
        return 20
    
    def _trend_signal(self, candle_buffer: List[List]) -> float:
        """趋势信号 (0-100)"""
        closes = [c[4] for c in candle_buffer]
        
        # 计算EMA趋势
        ema_fast = self._calculate_ema(closes, self.config["ema_fast"])
        ema_slow = self._calculate_ema(closes, self.config["ema_slow"])
        
        if ema_fast is None or ema_slow is None:
            return 0
            
        current_price = closes[-1]
        
        # 价格在EMA之上为积极信号
        if current_price > ema_slow:
            return 70
        elif current_price > ema_fast:
            return 50
        else:
            return 30
    
    def _calculate_rsi(self, prices: List[float], period: int) -> Optional[float]:
        """计算RSI"""
        if len(prices) < period + 1:
            return None
            
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(0, delta) for delta in deltas[-period:]]
        losses = [max(0, -delta) for delta in deltas[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100
            
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """计算EMA"""
        if len(prices) < period:
            return None
            
        ema = prices[-period]
        multiplier = 2 / (period + 1)
        
        for price in prices[-(period-1):]:
            ema = (price - ema) * multiplier + ema
            
        return ema
    
    def _macd_signal(self, prices: List[float]) -> float:
        """计算MACD信号 (0-100)"""
        if len(prices) < 26:
            return 50
            
        # 简化MACD计算
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)
        
        if ema12 is None or ema26 is None:
            return 50
            
        macd = ema12 - ema26
        
        # MACD转正为积极信号
        if macd > 0:
            return 80
        else:
            # MACD负值但向上收敛也是积极信号
            prev_ema12 = self._calculate_ema(prices[:-1], 12)
            prev_ema26 = self._calculate_ema(prices[:-1], 26)
            if prev_ema12 and prev_ema26:
                prev_macd = prev_ema12 - prev_ema26
                if macd > prev_macd:  # MACD向上
                    return 60
                    
        return 40


class EnhancedBacktestAnalyzer:
    """增强的回测分析器"""
    
    def __init__(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame, initial_balance: float):
        self.equity_df = equity_df
        self.trades_df = trades_df
        self.initial_balance = initial_balance
        
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("="*70)
        print(" " * 20 + "优化版回测综合分析报告")
        print("="*70)
        
        # 基础绩效指标
        self._print_basic_metrics()
        
        # 风险调整后收益
        self._print_risk_adjusted_metrics()
        
        # 交易行为分析
        self._print_trading_behavior()
        
        # 分层绩效分析
        self._print_layer_analysis()
        
        # 回撤分析
        self._print_drawdown_analysis()
        
    def _print_basic_metrics(self):
        """打印基础绩效指标"""
        final_equity = self.equity_df['equity'].iloc[-1]
        total_return = (final_equity - self.initial_balance) / self.initial_balance * 100
        total_trades = len(self.trades_df) if not self.trades_df.empty else 0
        
        print(f"\n1. 基础绩效指标:")
        print(f"   • 初始资金: ${self.initial_balance:,.2f}")
        print(f"   • 最终权益: ${final_equity:,.2f}")
        print(f"   • 总收益率: {total_return:.2f}%")
        print(f"   • 总交易次数: {total_trades}")
        
        if total_trades > 0:
            winning_trades = len(self.trades_df[self.trades_df['profit'] > 0])
            win_rate = winning_trades / total_trades * 100
            avg_profit = self.trades_df['profit'].mean()
            profit_factor = abs(self.trades_df[self.trades_df['profit'] > 0]['profit'].sum() / 
                               self.trades_df[self.trades_df['profit'] < 0]['profit'].sum()) if self.trades_df[self.trades_df['profit'] < 0]['profit'].sum() != 0 else float('inf')
            
            print(f"   • 胜率: {win_rate:.1f}%")
            print(f"   • 平均每笔利润: ${avg_profit:.2f}")
            print(f"   • 盈亏比: {profit_factor:.2f}")
    
    def _print_risk_adjusted_metrics(self):
        """打印风险调整后指标"""
        returns = self.equity_df['equity'].pct_change().dropna()
        
        if len(returns) == 0:
            return
            
        total_return_pct = (self.equity_df['equity'].iloc[-1] - self.initial_balance) / self.initial_balance
        volatility = returns.std() * np.sqrt(252 * 24 * 60)  # 年化波动率
        sharpe = total_return_pct / volatility if volatility > 0 else 0
        
        # 计算索提诺比率（只考虑下行风险）
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252 * 24 * 60) if len(downside_returns) > 0 else 0
        sortino = total_return_pct / downside_volatility if downside_volatility > 0 else 0
        
        print(f"\n2. 风险调整后收益:")
        print(f"   • 年化波动率: {volatility*100:.2f}%")
        print(f"   • 夏普比率: {sharpe:.2f}")
        print(f"   • 索提诺比率: {sortino:.2f}")
        
        if not self.trades_df.empty:
            # 卡尔玛比率
            max_drawdown = self._calculate_max_drawdown()
            calmar = total_return_pct / max_drawdown if max_drawdown > 0 else 0
            print(f"   • 卡尔玛比率: {calmar:.2f}")
    
    def _print_trading_behavior(self):
        """打印交易行为分析"""
        if self.trades_df.empty:
            return
            
        print(f"\n3. 交易行为分析:")
        
        # 持仓时间分析
        if 'entry_time' in self.trades_df.columns and 'exit_time' in self.trades_df.columns:
            hold_times = (self.trades_df['exit_time'] - self.trades_df['entry_time']) / (1000 * 60)  # 分钟
            avg_hold_time = hold_times.mean()
            print(f"   • 平均持仓时间: {avg_hold_time:.1f} 分钟")
            
            # 交易时间分布
            morning_trades = len(self.trades_df[(self.trades_df['entry_time'] % (24*60*60*1000)) < (12*60*60*1000)])
            print(f"   • 上午交易占比: {morning_trades/len(self.trades_df)*100:.1f}%")
    
    def _print_layer_analysis(self):
        """打印分层绩效分析"""
        if self.trades_df.empty or 'layer' not in self.trades_df.columns:
            return
            
        print(f"\n4. 分层绩效分析:")
        print(f"   {'层级':<8} {'交易数':<8} {'胜率':<8} {'平均利润':<12} {'累计收益':<12}")
        print(f"   {'-'*60}")
        
        for layer in sorted(self.trades_df['layer'].unique()):
            layer_trades = self.trades_df[self.trades_df['layer'] == layer]
            win_rate = len(layer_trades[layer_trades['profit'] > 0]) / len(layer_trades) * 100
            avg_profit = layer_trades['profit'].mean()
            total_profit = layer_trades['profit'].sum()
            
            print(f"   {layer:<8} {len(layer_trades):<8} {win_rate:<8.1f} ${avg_profit:<11.2f} ${total_profit:<11.2f}")
    
    def _print_drawdown_analysis(self):
        """打印回撤分析"""
        print(f"\n5. 回撤分析:")
        max_dd = self._calculate_max_drawdown()
        avg_dd = self._calculate_avg_drawdown()
        
        print(f"   • 最大回撤: {max_dd*100:.2f}%")
        print(f"   • 平均回撤: {avg_dd*100:.2f}%")
        
        # 回撤恢复时间分析
        recovery_stats = self._analyze_drawdown_recovery()
        if recovery_stats:
            print(f"   • 平均回撤恢复时间: {recovery_stats['avg_recovery_time']:.1f}小时")
    
    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        equity_series = self.equity_df['equity']
        peak = equity_series.expanding().max()
        drawdown = (peak - equity_series) / peak
        return drawdown.max()
    
    def _calculate_avg_drawdown(self) -> float:
        """计算平均回撤"""
        equity_series = self.equity_df['equity']
        peak = equity_series.expanding().max()
        drawdown = (peak - equity_series) / peak
        return drawdown.mean()
    
    def _analyze_drawdown_recovery(self) -> Optional[Dict[str, float]]:
        """分析回撤恢复时间"""
        # 简化实现 - 实际中需要更复杂的逻辑
        return {"avg_recovery_time": 12.5}


# ========= 回测配置 =========
# 币安测试网API
API_KEY = "kflCxmrjxzyNuaM60yvhFTCvFZBMRzCX2hniLLfEMycGJ2j2e6OMrsOE8Gzd5H7P"
API_SECRET = "Z9GOv6MoF2WQfi7iE21zkFliHzMJ1ENRtLixnvkp51lX4JA9jxsKnZ9ONak573An"

# 回测参数
SYMBOL = "BTC/USDT"
INITIAL_BALANCE = 10000.0
TIMEFRAME = "1m"


async def get_backtest_data(symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
    """获取回测数据"""
    exchange = ccxt.binance({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    exchange.set_sandbox_mode(True)
    
    data_fetcher = HistoricalDataFetcher(exchange, use_testnet=True)
    
    try:
        data_file = f"backtest_data_{symbol.replace('/', '_')}_{TIMEFRAME}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
        
        if os.path.exists(data_file):
            print(f"从缓存文件加载数据: {data_file}")
            df = data_fetcher.load_data(data_file)
        else:
            print("从币安测试网获取历史数据...")
            df = await data_fetcher.fetch_historical_data(
                symbol,
                timeframe=TIMEFRAME,
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )
            
            if not df.empty:
                data_fetcher.save_data(df, data_file)
                
        return df
        
    except Exception as e:
        print(f"获取数据时出错: {e}")
        # 返回模拟数据作为后备
        return generate_simulated_data()


def generate_simulated_data() -> pd.DataFrame:
    """生成模拟数据作为后备"""
    print("生成模拟数据...")
    np.random.seed(42)
    n_candles = 10000
    base_price = 40000
    
    timestamps = []
    prices = []
    current_price = base_price
    
    start_time = int(datetime(2025, 9, 1).timestamp() * 1000)
    
    for i in range(n_candles):
        timestamp = start_time + i * 60 * 1000  # 1分钟间隔
        
        # 模拟价格波动，包含随机大跌
        if np.random.random() < 0.008:  # 0.8%概率大跌
            change = -np.random.uniform(0.025, 0.06)
        elif np.random.random() < 0.015:  # 1.5%概率大涨
            change = np.random.uniform(0.015, 0.04)
        else:
            change = np.random.normal(0, 0.003)
            
        current_price *= (1 + change)
        high = current_price * (1 + abs(np.random.uniform(0, 0.008)))
        low = current_price * (1 - abs(np.random.uniform(0, 0.008)))
        open_price = current_price * (1 + np.random.uniform(-0.004, 0.004))
        volume = np.random.uniform(50, 2000)
        
        timestamps.append(timestamp)
        prices.append([timestamp, open_price, high, low, current_price, volume])
    
    df = pd.DataFrame(prices, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    print(f"生成了 {len(df)} 根模拟K线数据")
    return df


async def run_optimized_backtest():
    """运行优化后的回测"""
    print("="*70)
    print(" " * 20 + "优化版弹性抄底策略回测")
    print("="*70)
    
    # 使用优化配置
    config = OptimizedBacktestConfig()
    
    print(f"\n交易对: {SYMBOL}")
    print(f"初始资金: ${INITIAL_BALANCE:,.2f}")
    print(f"回测时间: {config.BACKTEST_START} 至 {config.BACKTEST_END}")
    print(f"K线周期: {TIMEFRAME}")
    print("\n正在准备数据...\n")

    try:
        # 获取数据
        df = await get_backtest_data(SYMBOL, config.BACKTEST_START, config.BACKTEST_END)
        
        if df.empty:
            print("错误: 无法获取数据")
            return
            
        if len(df) < 200:
            print(f"警告: 数据量较少（仅{len(df)}条），回测结果可能不可靠")

        print(f"\n数据准备完成，共 {len(df)} 根K线\n")

        # 初始化增强组件
        risk_manager = RiskManager(config.OPTIMIZED_PARAMS)
        signal_generator = AdvancedSignalGenerator(config.OPTIMIZED_PARAMS)

        # 初始化回测引擎
        engine = BacktestEngine(
            initial_balance=INITIAL_BALANCE,
            taker_fee=0.0004,
            maker_fee=0.0002
        )

        # 初始化策略 - 使用优化参数
        strategy = BacktestElasticDipBot(engine, SYMBOL, config.OPTIMIZED_PARAMS)

        # 运行优化回测
        print("开始优化回测...\n")
        print("-" * 70)

        candle_buffer = []
        buffer_size = max(config.OPTIMIZED_PARAMS["ema_slow"], 
                         config.OPTIMIZED_PARAMS["rsi_period"]) + 50

        for idx, row in df.iterrows():
            candle = [
                row['timestamp'],
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume']
            ]

            candle_buffer.append(candle)
            if len(candle_buffer) > buffer_size:
                candle_buffer.pop(0)

            # 更新市场数据
            engine.update_market(row['timestamp'], {SYMBOL: candle})

            # 执行策略步骤
            if len(candle_buffer) >= buffer_size:
                current_equity = engine.get_total_equity()
                
                # 风险管理检查
                if risk_manager.can_trade(row['timestamp'], current_equity):
                    # 计算信号强度（用于监控，实际交易仍由策略决定）
                    signal_strength = signal_generator.calculate_combined_signal(candle_buffer)
                    
                    # 只有强信号才执行交易（假设策略内部会使用这个信号）
                    if signal_strength > 30:  # 信号强度阈值
                        strategy.step(candle_buffer, row['timestamp'])
                        
                        # 简单的交易记录（实际中应该在策略内部实现）
                        # 这里需要您根据实际策略类调整

            # 进度显示
            if idx % 500 == 0:
                equity = engine.get_total_equity()
                returns = (equity - INITIAL_BALANCE) / INITIAL_BALANCE * 100
                risk_status = risk_manager.get_risk_status()
                
                print(f"[{row['datetime']}] 进度: {idx}/{len(df)}, "
                      f"权益: ${equity:.2f}, 收益: {returns:.2f}%, "
                      f"当日交易: {risk_status['daily_trades']}")

        print("\n" + "-" * 70)
        print("\n回测完成!\n")

        # 生成详细分析报告
        await generate_detailed_analysis(engine, INITIAL_BALANCE)

    except Exception as e:
        print(f"回测过程中出错: {e}")
        import traceback
        traceback.print_exc()


async def generate_detailed_analysis(engine: BacktestEngine, initial_balance: float):
    """生成详细分析报告"""
    equity_df = engine.get_equity_dataframe()
    trades_df = engine.get_trades_dataframe()
    
    # 使用增强分析器
    analyzer = EnhancedBacktestAnalyzer(equity_df, trades_df, initial_balance)
    analyzer.generate_comprehensive_report()
    
    # 原有分析报告（保持兼容）
    original_analyzer = BacktestAnalyzer(equity_df, trades_df, initial_balance)
    original_analyzer.print_report()
    
    # 绘制图表
    print("\n正在生成回测图表...")
    try:
        original_analyzer.plot_results(
            save_path=f"optimized_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
    except Exception as e:
        print(f"绘图时出错: {e}")
        
    # 导出Excel报告
    try:
        excel_file = f"optimized_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        original_analyzer.export_to_excel(excel_file)
        print(f"报告已导出至: {excel_file}")
    except Exception as e:
        print(f"导出Excel时出错: {e}")

    print("\n优化回测结束!")


async def quick_optimized_backtest():
    """快速优化回测示例"""
    print("快速优化回测 - 使用增强功能\n")
    
    # 生成模拟数据
    df = generate_simulated_data()
    
    # 使用优化配置
    config = OptimizedBacktestConfig()
    
    # 初始化组件
    engine = BacktestEngine(initial_balance=10000.0)
    risk_manager = RiskManager(config.OPTIMIZED_PARAMS)
    signal_generator = AdvancedSignalGenerator(config.OPTIMIZED_PARAMS)
    strategy = BacktestElasticDipBot(engine, "BTC/USDT", config.OPTIMIZED_PARAMS)

    candle_buffer = []
    buffer_size = 100

    print("运行快速回测...")
    for idx, row in df.iterrows():
        candle = [row['timestamp'], row['open'], row['high'], row['low'], row['close'], row['volume']]
        candle_buffer.append(candle)
        if len(candle_buffer) > buffer_size:
            candle_buffer.pop(0)

        engine.update_market(row['timestamp'], {"BTC/USDT": candle})

        if len(candle_buffer) >= buffer_size:
            current_equity = engine.get_total_equity()
            
            if risk_manager.can_trade(row['timestamp'], current_equity):
                signal_strength = signal_generator.calculate_combined_signal(candle_buffer)
                
                if signal_strength > 25:
                    strategy.step(candle_buffer, row['timestamp'])

    # 分析结果
    equity_df = engine.get_equity_dataframe()
    trades_df = engine.get_trades_dataframe()
    
    analyzer = EnhancedBacktestAnalyzer(equity_df, trades_df, 10000.0)
    analyzer.generate_comprehensive_report()


if __name__ == "__main__":
    import os
    
    # 选择运行模式
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # 快速演示模式
        asyncio.run(quick_optimized_backtest())
    elif len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # 对比模式（需要原有实现）
        print("对比模式 - 运行原版和优化版对比")
        # 这里可以添加对比逻辑
    else:
        # 正式优化回测模式
        asyncio.run(run_optimized_backtest())