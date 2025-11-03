# Binance 测试网完整指南

## 📊 测试网现状总览

| 测试网类型 | 状态 | 地址 | 说明 |
|----------|------|------|------|
| **现货测试网** | ✅ 正常运行 | https://testnet.binance.vision/ | 完全免费，推荐使用 |
| **期货测试网** | ❌ 已废弃 | ~~testnet.binancefuture.com~~ | 改用真实网 Demo 模式 |

## 🎯 现货测试网（推荐）

### 特点
- ✅ **完全免费**，无需真实资金
- ✅ **真实市场数据**
- ✅ **独立的测试账户**
- ✅ **支持所有现货交易对**
- ✅ **可获取测试代币**
- ❌ 无杠杆（现货特性）

### 获取测试网 API Key

**步骤 1：访问测试网**
```
https://testnet.binance.vision/
```

**步骤 2：登录**
- 使用 GitHub 账号登录
- 无需 Binance 账号

**步骤 3：生成 API Key**
1. 点击右上角头像 → API Management
2. 创建新的 API Key
3. 保存 API Key 和 Secret

**步骤 4：获取测试资金**
1. 进入 Wallet
2. 点击 "Get Test Funds"
3. 选择要充值的币种（如 USDT, ETH）
4. 获取免费测试代币

### 配置文件

**使用 `spot_testnet.yaml`：**

```yaml
exchange:
  exchange_id: "binance"      # 现货交易所
  market_type: "spot"         # 现货模式
  api_key: "YOUR_SPOT_TESTNET_API_KEY"    # 测试网 Key
  secret: "YOUR_SPOT_TESTNET_SECRET"       # 测试网 Secret

data:
  use_testnet: true           # ✅ 现货测试网仍支持

engine:
  leverage: 1                 # 现货固定为 1
```

### 运行

```bash
# Dry-run 模式
python run_live_trading.py --config configs/spot_testnet.yaml --dry-run

# 实际测试网交易（使用测试资金）
python run_live_trading.py --config configs/spot_testnet.yaml
```

---

## ⚙️ 期货模拟交易（Demo 模式）

由于期货测试网已废弃，现在使用 **Demo 模式**。

### 特点
- ⚠️ **需要真实网 API Key**（但不会下真单）
- ✅ **真实市场数据**
- ✅ **模拟订单执行**
- ✅ **支持杠杆**
- ✅ **可做多做空**

### 配置文件

**使用 `future_demo.yaml`：**

```yaml
exchange:
  exchange_id: "binanceusdm"  # 合约交易所
  market_type: "future"       # 合约模式
  api_key: "YOUR_REAL_API_KEY"        # ⚠️ 真实网 Key
  secret: "YOUR_REAL_SECRET"           # ⚠️ 真实网 Secret
  options:
    demo: true                # ✅ Demo 模式

data:
  use_testnet: false          # ⚠️ 必须为 false

engine:
  leverage: 3                 # 支持杠杆
```

### 运行

```bash
# Dry-run 模式
python run_live_trading.py --config configs/future_demo.yaml --dry-run

# Demo 模式交易（模拟订单）
python run_live_trading.py --config configs/future_demo.yaml
```

---

## 📋 配置对比

### 现货测试网 vs 期货 Demo

| 特性 | 现货测试网 | 期货 Demo |
|------|-----------|----------|
| **API Key 来源** | testnet.binance.vision | www.binance.com |
| **资金** | 免费测试币 | 虚拟余额 |
| **市场数据** | 真实 | 真实 |
| **订单执行** | 测试网撮合 | 模拟执行 |
| **杠杆** | ❌ 不支持 | ✅ 支持 |
| **做空** | ❌ 不支持 | ✅ 支持 |
| **use_testnet** | true | false |
| **options.demo** | 不需要 | true |

### 配置示例对比

**现货测试网：**
```yaml
exchange:
  exchange_id: "binance"
  api_key: "SPOT_TESTNET_KEY"
  market_type: "spot"

data:
  use_testnet: true           # ✅
```

**期货 Demo：**
```yaml
exchange:
  exchange_id: "binanceusdm"
  api_key: "REAL_API_KEY"     # ⚠️ 真实网
  market_type: "future"
  options:
    demo: true                # ✅

data:
  use_testnet: false          # ⚠️
```

---

## 🚀 快速选择指南

### 我应该用哪个？

| 你的需求 | 推荐配置 | 配置文件 |
|---------|---------|---------|
| **测试现货策略** | 现货测试网 | `spot_testnet.yaml` |
| **测试无杠杆合约** | 期货 Demo | `future_demo.yaml` + leverage: 1 |
| **测试杠杆交易** | 期货 Demo | `future_demo.yaml` + leverage: 3 |
| **测试做空** | 期货 Demo | `future_demo.yaml` + direction: short |
| **测试双向交易** | 期货 Demo | `future_demo.yaml` + direction: both |

### 推荐学习路径

1. **第一步**：现货测试网（`spot_testnet.yaml`）
   - 最安全，完全免费
   - 熟悉策略基本运行

2. **第二步**：期货 Demo 无杠杆（`future_demo.yaml` + leverage: 1）
   - 了解合约交易机制
   - 测试做空功能

3. **第三步**：期货 Demo 低杠杆（leverage: 2-3）
   - 理解杠杆的作用
   - 体验风险放大效果

4. **第四步**：双向自适应（direction: both）
   - 测试完整策略
   - 准备真实交易

---

## ⚠️ 常见问题

### Q1: 现货测试网需要充值吗？
❌ 不需要。访问 testnet.binance.vision 可以免费获取测试代币。

### Q2: 现货测试网的币可以提现吗？
❌ 不可以。测试网的币只能在测试网内使用，没有真实价值。

### Q3: 为什么期货没有测试网？
Binance 已于 2024 年废弃期货测试网，改用 Demo 模式替代。

### Q4: Demo 模式安全吗？
✅ 安全。虽然使用真实 API Key，但所有订单都是模拟的，不会使用真实资金。

### Q5: 现货测试网支持杠杆吗？
❌ 不支持。现货交易本身就不支持杠杆，leverage 必须为 1。

---

## 📁 配置文件清单

| 配置文件 | 模式 | use_testnet | options.demo | 说明 |
|---------|------|-------------|--------------|------|
| `spot_testnet.yaml` | 现货测试网 | true | - | ⭐ 推荐入门 |
| `spot_demo.yaml` | 现货 Demo | false | true | 真实网模拟 |
| `future_demo.yaml` | 期货 Demo | false | true | 合约模拟 |
| `live_eth.yaml` | 合约真实 | false | false | 真实交易 |

---

## 🔗 相关链接

- 现货测试网：https://testnet.binance.vision/
- Binance API 文档：https://binance-docs.github.io/apidocs/spot/cn/
- 期货废弃公告：https://t.me/ccxt_announcements/92
- Demo 模式详解：[BINANCE_DEMO_MODE.md](./BINANCE_DEMO_MODE.md)
