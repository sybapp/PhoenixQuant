# Binance Demo 模式使用说明

## ⚠️ 重要变更

**Binance 已于 2024 年废弃期货测试网（testnet.binancefuture.com）！**

现在需要使用 **Demo 模式** 进行模拟交易。

## 🆚 测试网 vs Demo 模式对比

| 特性 | 旧测试网（已废弃） | 新 Demo 模式 |
|------|-------------------|------------|
| API Key | 测试网专用 | **真实网 API Key** |
| 数据源 | 测试网数据 | **真实市场数据** |
| 订单执行 | 测试网撮合 | **模拟执行（不下真单）** |
| 余额 | 测试资金 | 虚拟余额 |
| 杠杆 | ❌ 不支持 | ✅ 支持（但建议用1） |
| 状态 | 🔴 已废弃 | ✅ 正常使用 |

## 📋 配置步骤

### 1. 获取真实网 API Key

访问 Binance 官网获取 API Key：
https://www.binance.com/zh-CN/my/settings/api-management

**重要：**
- ✅ 使用真实网 API Key（不是测试网）
- ✅ Demo 模式不会下真实订单
- ✅ 只需要读取权限即可

### 2. 配置文件设置

```yaml
data:
  use_testnet: false        # ⚠️ 必须为 false

exchange:
  exchange_id: "binanceusdm"
  api_key: "YOUR_REAL_API_KEY"      # 真实网 API Key
  secret: "YOUR_REAL_SECRET"         # 真实网 Secret
  options:
    defaultType: "future"
    demo: true              # ⚠️ 启用 Demo 模式
```

### 3. 运行

```bash
# Dry-run 模式（双重保护）
python run_live_trading.py --config configs/testnet_example.yaml --dry-run

# Demo 模式交易（模拟订单）
python run_live_trading.py --config configs/testnet_example.yaml
```

## 🔍 日志验证

成功启用 Demo 模式会看到：
```
✅ Demo 模式已启用（模拟交易，不会下真实订单）
```

如果看到警告：
```
Binance 期货已废弃测试网模式！请使用 demo 模式
```

说明配置有误，请检查：
1. `data.use_testnet` 是否为 `false`
2. `exchange.options.demo` 是否为 `true`

## 💡 推荐配置

```yaml
engine:
  leverage: 1               # Demo 模式建议无杠杆

live:
  dry_run: true             # 开启双重保护
  enable_trading: false     # 初次测试建议关闭

strategy:
  max_account_ratio: 0.15   # 无杠杆时可提高到 15%
```

## ⚙️ 从旧测试网迁移

如果你之前使用测试网配置：

```yaml
# 旧配置（已废弃）
data:
  use_testnet: true
exchange:
  api_key: "TESTNET_KEY"

# 新配置（Demo 模式）
data:
  use_testnet: false        # 改为 false
exchange:
  api_key: "REAL_API_KEY"   # 换成真实网 key
  options:
    demo: true              # 加上这一行
```

## 🛡️ 安全说明

**Demo 模式安全吗？**

✅ 是的！Demo 模式有多重保护：
1. 所有订单都是模拟的，不会在真实市场成交
2. 不会动用你的真实资金
3. 使用真实市场数据，回测更准确

**额外建议：**
- 首次使用同时开启 `dry_run: true`（双重保护）
- API Key 只给读取权限
- 定期检查 Binance 账户，确保没有真实订单

## 🔗 相关链接

- Binance API 文档: https://binance-docs.github.io/apidocs/futures/cn/
- CCXT Demo 模式说明: https://docs.ccxt.com/#/README?id=sandbox-mode
- 废弃公告: https://t.me/ccxt_announcements/92
