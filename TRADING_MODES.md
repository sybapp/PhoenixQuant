# 交易模式完整指南

## 📌 重要更新

1. **Binance 测试网已废弃**（2024年） → 改用 **Demo 模式**
2. **支持现货和合约** → 通过 `market_type` 配置

## 🎯 支持的交易模式

| 模式 | exchange_id | market_type | leverage | 杠杆支持 | 适用场景 |
|------|-------------|-------------|----------|---------|---------|
| 现货（真实） | binance | spot | 1 | ❌ | 低风险，适合新手 |
| 合约（真实） | binanceusdm | future | 1-125 | ✅ | 高风险，支持杠杆 |
| 现货（Demo） | binance | spot | 1 | ❌ | 模拟交易测试 |
| 合约（Demo） | binanceusdm | future | 1-20 | ✅ | 模拟杠杆交易测试 |

## 📁 配置文件参考

### 1. 现货 Demo 模式 (`spot_demo.yaml`)
```yaml
exchange:
  exchange_id: "binance"      # 现货交易所
  market_type: "spot"         # 现货类型
  options:
    demo: true                # Demo 模式

engine:
  leverage: 1                 # 现货固定为 1

strategy:
  risk:
    max_account_ratio: 0.2    # 无杠杆，可提高使用比例
```

**运行：**
```bash
python run_live_trading.py --config configs/spot_demo.yaml --dry-run
```

---

### 2. 合约 Demo 模式 (`future_demo.yaml`)
```yaml
exchange:
  exchange_id: "binanceusdm"  # 合约交易所（USDT 本位）
  market_type: "future"       # 合约类型
  options:
    demo: true                # Demo 模式

engine:
  leverage: 3                 # 可设置杠杆（Demo 建议 1-3）

strategy:
  direction: "both"           # 合约支持双向
  risk:
    max_account_ratio: 0.1    # 使用杠杆时降低比例
```

**运行：**
```bash
python run_live_trading.py --config configs/future_demo.yaml --dry-run
```

---

### 3. 旧测试网配置迁移

**旧配置（已废弃）：**
```yaml
data:
  use_testnet: true           # ❌ 不再支持

exchange:
  exchange_id: "binanceusdm"
  api_key: "TESTNET_KEY"      # ❌ 测试网 Key
```

**新配置（Demo 模式）：**
```yaml
data:
  use_testnet: false          # ✅ 改为 false

exchange:
  exchange_id: "binanceusdm"
  market_type: "future"       # ✅ 指定市场类型
  api_key: "REAL_API_KEY"     # ✅ 真实网 Key
  options:
    demo: true                # ✅ 启用 Demo
```

---

## 🔧 配置详解

### market_type 参数

```yaml
exchange:
  market_type: "spot"   # 或 "future"
```

- **spot（现货）**：
  - 使用 `exchange_id: "binance"`
  - 不支持杠杆（leverage 固定为 1）
  - 风险较低
  - 手续费较高（0.1%）

- **future（合约）**：
  - 使用 `exchange_id: "binanceusdm"`（USDT 本位）
  - 或 `exchange_id: "binancecoinm"`（币本位）
  - 支持杠杆（1-125 倍）
  - 可做多做空
  - 手续费较低（0.02%-0.05%）

### Demo 模式设置

```yaml
data:
  use_testnet: false        # ⚠️ 必须为 false

exchange:
  api_key: "REAL_KEY"       # ⚠️ 使用真实网 API Key
  options:
    demo: true              # ⚠️ 启用 Demo 模式
```

**重要：**
- Demo 模式使用真实网 API Key
- 但所有订单都是模拟的，不会下真单
- 使用真实市场数据

---

## 🚀 快速开始

### 步骤 1：获取 API Key

访问 Binance 官网：
https://www.binance.com/zh-CN/my/settings/api-management

权限设置：
- ✅ 读取权限（必需）
- ❌ 交易权限（Demo 模式不需要）
- ❌ 提现权限（绝对不要开启）

### 步骤 2：选择配置文件

| 你的需求 | 使用配置 |
|---------|---------|
| 测试现货策略 | `spot_demo.yaml` |
| 测试合约策略（无杠杆） | `future_demo.yaml` + leverage: 1 |
| 测试合约策略（有杠杆） | `future_demo.yaml` + leverage: 3 |
| 真实现货交易 | 复制 `spot_demo.yaml`，关闭 demo |
| 真实合约交易 | 复制 `future_demo.yaml`，关闭 demo |

### 步骤 3：修改配置

```yaml
exchange:
  api_key: "YOUR_REAL_API_KEY"      # 填入你的 Key
  secret: "YOUR_REAL_SECRET"         # 填入你的 Secret
  market_type: "spot"                # 或 "future"
  options:
    demo: true                       # Demo 模式
```

### 步骤 4：运行

```bash
# Dry-run（双重保护）
python run_live_trading.py --config configs/spot_demo.yaml --dry-run

# Demo 模式（模拟订单）
python run_live_trading.py --config configs/spot_demo.yaml

# 查看详细日志
python run_live_trading.py --config configs/spot_demo.yaml --log DEBUG
```

---

## 📊 日志验证

### 成功启动现货

```
市场类型: SPOT | 杠杆: 1.0x
✅ Demo 模式已启用（模拟交易，不会下真实订单）
杠杆设置为 1.0（无杠杆模式）
```

### 成功启动合约

```
市场类型: FUTURE | 杠杆: 3.0x
✅ Demo 模式已启用（模拟交易，不会下真实订单）
成功设置杠杆: 3x for ETH/USDT
```

### 错误：现货使用杠杆

```
现货交易不支持杠杆，leverage 参数将被忽略
```

---

## ⚠️ 常见问题

### Q1: 测试网 API Key 还能用吗？
❌ 不能。Binance 已废弃期货测试网，请使用真实网 API Key + Demo 模式。

### Q2: Demo 模式会下真实订单吗？
❌ 不会。Demo 模式所有订单都是模拟的，不会使用真实资金。

### Q3: 现货可以用杠杆吗？
❌ 不可以。现货交易不支持杠杆，leverage 必须为 1。

### Q4: 如何切换现货/合约？
修改 `market_type` 和 `exchange_id`：
```yaml
# 现货
exchange_id: "binance"
market_type: "spot"

# 合约
exchange_id: "binanceusdm"
market_type: "future"
```

### Q5: Demo 模式和 Dry-run 的区别？
- **Demo 模式**：使用真实 API，但模拟订单（由 Binance 提供）
- **Dry-run**：本地模拟，不调用 API（由 PhoenixQuant 提供）
- **推荐**：两者同时开启（双重保护）

---

## 🔗 相关链接

- Binance API 文档: https://binance-docs.github.io/apidocs/
- CCXT 文档: https://docs.ccxt.com/
- Demo 模式说明: [BINANCE_DEMO_MODE.md](./BINANCE_DEMO_MODE.md)
