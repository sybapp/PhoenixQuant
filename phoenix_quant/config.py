"""配置加载与数据类定义"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - 按需降级解析
    import yaml  # type: ignore
except ModuleNotFoundError:  # 运行环境可能缺少PyYAML
    yaml = None  # type: ignore


@dataclass
class DataSourceConfig:
    """历史数据源配置"""

    source: str = "binance"
    use_testnet: bool = True
    cache: Optional[Path] = None
    limit: int = 1000


@dataclass
class EngineConfig:
    """回测引擎配置"""

    initial_balance: float = 10_000.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004
    leverage: float = 1.0


@dataclass
class ExchangeConfig:
    """实盘交易所连接配置"""

    exchange_id: str = "binance"
    api_key: str = ""
    secret: str = ""
    password: Optional[str] = None
    enable_rate_limit: bool = True
    options: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    leverage: float = 1.0


@dataclass
class LayerConfig:
    """分层挂单配置"""

    offset_pct: float
    size_ratio: float


@dataclass
class RiskConfig:
    """风险控制配置"""

    max_account_ratio: float = 0.3
    take_profit_pct: float = 1.0
    hard_stop_pct: float = 2.0
    trailing_pct: float = 0.5
    trailing_activation_pct: float = 0.6
    cooldown_minutes: int = 30
    max_concurrent_trades: int = 1
    max_hold_minutes: int = 360


@dataclass
class StrategyConfig:
    """弹性抄底策略配置"""

    timeframe: str
    drop_single_pct: float
    drop_window_pct: float
    drop_window: int
    delayed_trigger_pct: float
    delayed_window_minutes: int
    ema_fast: int
    ema_slow: int
    rsi_period: int
    rsi_oversold: float
    volume_short: int
    volume_long: int
    volume_recover_ratio: float
    volume_tick_ratio: float
    min_signal: float
    volatility_window: int = 30
    layers: List[LayerConfig] = field(default_factory=list)
    risk: RiskConfig = field(default_factory=RiskConfig)
    scale_multiplier: float = 1.0
    trail_step_pct: float = 0.3


@dataclass
class LiveSettings:
    """实盘运行参数"""

    poll_interval: float = 30.0
    warmup_bars: int = 500
    backfill_limit: int = 1000
    enable_trading: bool = True
    dry_run: bool = False
    heartbeat_interval: float = 120.0


@dataclass
class BacktestWindow:
    """回测时间窗口"""

    start: Optional[datetime]
    end: Optional[datetime]


@dataclass
class BacktestConfig:
    """完整的回测配置"""

    symbol: str
    timeframe: str
    engine: EngineConfig
    strategy: StrategyConfig
    data: DataSourceConfig
    window: BacktestWindow


@dataclass
class LiveTradingConfig:
    """实盘交易配置"""

    symbol: str
    timeframe: str
    engine: EngineConfig
    strategy: StrategyConfig
    data: DataSourceConfig
    exchange: ExchangeConfig
    settings: LiveSettings


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """解析ISO时间字符串"""

    if not value:
        return None
    return datetime.fromisoformat(value)


def _ensure_layers(raw_layers: List[dict]) -> List[LayerConfig]:
    """转换层级配置"""

    layers = []
    for item in raw_layers:
        layers.append(
            LayerConfig(
                offset_pct=float(item["offset_pct"]),
                size_ratio=float(item["size_ratio"]),
            )
        )
    return layers


def _parse_scalar(value: str):
    """解析基础标量类型"""

    value = value.strip()
    if not value:
        return ""

    lower = value.lower()
    if lower in {"true", "yes"}:
        return True
    if lower in {"false", "no"}:
        return False
    if lower in {"null", "none"}:
        return None

    if (value.startswith("\"") and value.endswith("\"")) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    try:
        if value.startswith("0") and value != "0" and not value.startswith("0."):
            # 避免将类似交易对中的前导0截断
            raise ValueError
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def _simple_yaml_load(text: str):
    """在PyYAML缺失时的简易YAML解析"""

    lines = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))

    result: dict = {}
    stack: list[tuple[int, object]] = [(-1, result)]

    def current_parent(indent: int, is_list_item: bool):
        while len(stack) > 1:
            top_indent, container = stack[-1]
            if indent > top_indent:
                break
            if indent == top_indent and isinstance(container, list) and is_list_item:
                break
            stack.pop()
        return stack[-1][1]

    i = 0
    while i < len(lines):
        indent, stripped = lines[i]
        is_list_item = stripped.startswith("- ")
        parent = current_parent(indent, is_list_item)

        if is_list_item:
            if not isinstance(parent, list):
                raise ValueError("列表项缺少父级列表定义")

            item_str = stripped[2:].strip()
            if not item_str:
                item: object = {}
                parent.append(item)
                stack.append((indent + 1, item))
            elif ":" in item_str:
                key, value_part = item_str.split(":", 1)
                key = key.strip()
                value_part = value_part.strip()
                item_dict: dict
                if value_part:
                    item_dict = {key: _parse_scalar(value_part)}
                    parent.append(item_dict)
                    stack.append((indent + 1, item_dict))
                else:
                    nested: dict = {}
                    item_dict = {key: nested}
                    parent.append(item_dict)
                    stack.append((indent + 1, nested))
            else:
                parent.append(_parse_scalar(item_str))
        else:
            key, _, value_part = stripped.partition(":")
            key = key.strip()
            value_part = value_part.strip()

            if not isinstance(parent, dict):
                raise ValueError("键值对缺少父级字典定义")

            if value_part == "":
                next_container: object = {}
                if i + 1 < len(lines):
                    next_indent, next_stripped = lines[i + 1]
                    if next_indent >= indent and next_stripped.startswith("- "):
                        next_container = []
                parent[key] = next_container
                stack.append((indent if isinstance(next_container, list) else indent + 1, next_container))
            else:
                parent[key] = _parse_scalar(value_part)

        i += 1

    return result


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        text = f.read()

    if yaml is not None:  # type: ignore[truthy-bool]
        return yaml.safe_load(text)

    return _simple_yaml_load(text)


def load_config(path: str | Path) -> BacktestConfig:
    """从YAML文件加载配置"""

    path = Path(path)
    raw = _load_yaml(path)

    engine = EngineConfig(**raw["engine"])
    risk = RiskConfig(**raw["strategy"].get("risk", {}))
    strategy_dict = raw["strategy"].copy()
    strategy_dict["risk"] = risk
    strategy_dict["layers"] = _ensure_layers(strategy_dict.get("layers", []))
    strategy = StrategyConfig(**strategy_dict)

    data = raw.get("data", {})
    if data.get("cache"):
        data["cache"] = Path(data["cache"])
    data_config = DataSourceConfig(**data)

    window_cfg = raw.get("window", {})
    window = BacktestWindow(
        start=_parse_datetime(window_cfg.get("start")),
        end=_parse_datetime(window_cfg.get("end")),
    )

    return BacktestConfig(
        symbol=raw["symbol"],
        timeframe=raw.get("timeframe", strategy.timeframe),
        engine=engine,
        strategy=strategy,
        data=data_config,
        window=window,
    )


def _dict_or_empty(value: Optional[dict]) -> Dict[str, Any]:
    if value is None:
        return {}
    return dict(value)


def load_live_config(path: str | Path) -> LiveTradingConfig:
    """加载实盘交易配置"""

    path = Path(path)
    raw = _load_yaml(path)

    engine = EngineConfig(**raw["engine"])

    strategy_dict = raw["strategy"].copy()
    risk_cfg = strategy_dict.get("risk", {})
    strategy_dict["risk"] = RiskConfig(**risk_cfg)
    strategy_dict["layers"] = _ensure_layers(strategy_dict.get("layers", []))
    strategy = StrategyConfig(**strategy_dict)

    data_cfg = raw.get("data", {})
    if data_cfg.get("cache"):
        data_cfg["cache"] = Path(data_cfg["cache"])
    data = DataSourceConfig(**data_cfg)

    exchange_cfg = raw.get("exchange", {})
    exchange_cfg = exchange_cfg.copy()
    exchange_id = exchange_cfg.pop("id", None)
    if exchange_id:
        exchange_cfg["exchange_id"] = exchange_id
    exchange_cfg["options"] = _dict_or_empty(exchange_cfg.get("options"))
    exchange_cfg["params"] = _dict_or_empty(exchange_cfg.get("params"))
    exchange = ExchangeConfig(**exchange_cfg)

    settings_cfg = raw.get("live", raw.get("settings", {})) or {}
    settings = LiveSettings(**settings_cfg)

    return LiveTradingConfig(
        symbol=raw["symbol"],
        timeframe=raw.get("timeframe", strategy.timeframe),
        engine=engine,
        strategy=strategy,
        data=data,
        exchange=exchange,
        settings=settings,
    )
