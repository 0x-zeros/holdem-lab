# Equity Calculator - 功能规格说明

## 概述

基于 holdem-lab 的 Web 端德州扑克 Equity 计算器，参考 PokerStove 设计。

**目标用户**: 德州扑克学习者和研究者
**平台**: Web 应用 (WASM) / 桌面应用 (Tauri)
**计算引擎**: holdem-core Rust 库

---

## 功能模块

### 1. 手牌输入 (Hand Input)

#### 1.1 具体手牌模式
- 输入格式: `AhKh`, `QsQd` 等
- 支持键盘快捷输入
- 点击卡牌选择器可视化选择

#### 1.2 范围模式 (Range Mode)
- 13×13 手牌矩阵
- 支持点击选择单个手牌组合
- 支持拖拽选择多个手牌
- 范围滑块: 按百分比选择顶部手牌

#### 1.3 快捷范围按钮
| 按钮 | 选择范围 |
|------|----------|
| All | 所有 169 种手牌 |
| Any Suited | 所有同花组合 (78 种) |
| Any Broadway | T+的高牌组合 |
| Any Pair | 所有对子 (13 种) |
| Clear | 清空选择 |

#### 1.4 范围语法 (高级)
支持解析常见范围表示法:
- `AA` - 对子
- `AKs` - 同花
- `AKo` - 杂花
- `AK` - 同花+杂花
- `QQ+` - QQ 及以上对子
- `ATs+` - ATs, AJs, AQs, AKs
- `77-TT` - 77 到 TT 的对子

---

### 2. 公共牌输入 (Board)

- 最多 5 张牌 (Flop 3张, Turn 1张, River 1张)
- 支持手动输入: `Ah Kc Qd`
- 支持点击卡牌选择器
- 自动排除已使用的牌

---

### 3. 死牌输入 (Dead Cards)

- 输入已知不在牌堆中的牌
- 用于更精确的计算
- 支持手动输入或点击选择

---

### 4. 玩家管理

- 支持 2-10 个玩家
- 每个玩家可以是:
  - 具体手牌 (1 combo)
  - 范围 (多个 combos)
- 动态添加/删除玩家
- 显示每个玩家的 combo 数量

---

### 5. 计算模式

#### 5.1 Enumerate All (枚举)
- 遍历所有可能的发牌组合
- 精确计算
- 适合 2 玩家 + 短范围
- 显示计算的游戏数量

#### 5.2 Monte Carlo (蒙特卡洛)
- 随机采样
- 快速近似
- 适合多玩家或大范围
- 可设置采样次数 (默认 100,000)

---

### 6. 结果展示

#### 6.1 Equity 表格
| 列 | 说明 |
|----|------|
| Player | 玩家标识 |
| Hand/Range | 手牌或范围描述 |
| Equity | 总体胜率 (含平局加权) |
| Win % | 纯赢的概率 |
| Tie % | 平局概率 |
| Combos | 范围内的组合数 |

#### 6.2 Equity 条形图
- 水平条形图显示各玩家 equity
- 颜色区分赢/平

#### 6.3 计算信息
- 计算耗时
- 计算的游戏/组合数量

---

### 7. 听牌分析 (Draw Analysis)

当输入具体手牌 + 公共牌时，显示听牌信息:

#### 7.1 同花听牌 (Flush Draw)
- 4张同花 = Flush Draw (9 outs)
- 3张同花 (flop) = Backdoor Flush
- 标注是否为 Nut Flush Draw

#### 7.2 顺子听牌 (Straight Draw)
- Open-Ended (OESD): 8 outs
- Gutshot: 4 outs
- Double Gutshot: 8 outs
- Backdoor Straight (flop only)

#### 7.3 总 Outs
- 去重后的总 outs 数量
- 标注 Combo Draw (同时有 flush + straight draw)

---

## UI 组件清单

### 基础组件
| 组件 | 说明 |
|------|------|
| Card/Empty | 空牌位 (虚线边框) |
| Card/Face | 显示具体牌面 |
| Button/Primary | 主要操作按钮 |
| Button/Secondary | 次要操作按钮 |
| Input/Text | 文本输入框 |
| Radio | 单选按钮组 |

### 业务组件
| 组件 | 说明 |
|------|------|
| PlayerRow | 玩家行 (按钮 + 手牌显示 + equity 条) |
| EquityBar | 胜率条形图 |
| MatrixCell | 13×13 矩阵单元格 |
| HandMatrix | 完整手牌矩阵 (169 cells) |
| RangeSlider | 范围百分比滑块 |
| ResultTable | 结果数据表格 |
| DrawPanel | 听牌分析面板 |

### 复合组件
| 组件 | 说明 |
|------|------|
| BoardInput | 公共牌输入区 |
| DeadCardsInput | 死牌输入区 |
| PlayerList | 玩家列表 |
| CalculatePanel | 计算模式选择 + 按钮 |
| ResultPanel | 完整结果区域 |

---

## 交互流程

### 基本流程
1. 点击 [Player 1] 按钮 → 打开范围选择弹窗
2. 在矩阵中选择手牌范围 → 点击 [OK]
3. 点击 [Player 2] 按钮 → 选择对手范围
4. (可选) 输入 Board 和 Dead Cards
5. 选择计算模式 (Enumerate All / Monte Carlo)
6. 点击 [Evaluate] 按钮
7. 查看结果表格和 equity 条形图

### 范围选择弹窗交互
- 单击: 切换单个手牌选中状态
- Shift+单击: 选择从上次点击到当前的范围
- Ctrl+单击: 添加到选择
- 拖拽: 框选多个手牌
- 快捷按钮: 预设范围
- 滑块: 按百分比选择

---

## API 对应

| UI 操作 | holdem-lab API |
|---------|----------------|
| 解析手牌文本 | `parse_cards("AhKh")` |
| 获取 169 种规范手牌 | `get_all_canonical_hands()` |
| 规范手牌转字符串 | `str(canonical)` → "AKs" |
| 展开为具体 combos | `get_all_combos(canonical)` |
| 排除死牌 | `get_combos_excluding(canonical, dead)` |
| 计算 equity | `calculate_equity(EquityRequest(...))` |
| 分析听牌 | `analyze_draws(hole, board)` |
| 获取 flush outs | `count_flush_outs(hole, board)` |
| 获取 straight outs | `count_straight_outs(hole, board)` |

---

## 设计参考

- [PokerStove](https://github.com/andrewprock/pokerstove) - 经典 equity 计算器
- [Equilab](https://www.pokerstrategy.com/poker-software-tools/equilab/) - 现代化界面
- [PokerCruncher](https://www.pokercruncher.com/) - 移动端设计

---

## 附录: 169 种规范手牌

手牌矩阵布局 (行=第一张牌, 列=第二张牌):

```
     A    K    Q    J    T    9    8    7    6    5    4    3    2
A   AA   AKs  AQs  AJs  ATs  A9s  A8s  A7s  A6s  A5s  A4s  A3s  A2s
K   AKo  KK   KQs  KJs  KTs  K9s  K8s  K7s  K6s  K5s  K4s  K3s  K2s
Q   AQo  KQo  QQ   QJs  QTs  Q9s  Q8s  Q7s  Q6s  Q5s  Q4s  Q3s  Q2s
J   AJo  KJo  QJo  JJ   JTs  J9s  J8s  J7s  J6s  J5s  J4s  J3s  J2s
T   ATo  KTo  QTo  JTo  TT   T9s  T8s  T7s  T6s  T5s  T4s  T3s  T2s
9   A9o  K9o  Q9o  J9o  T9o  99   98s  97s  96s  95s  94s  93s  92s
8   A8o  K8o  Q8o  J8o  T8o  98o  88   87s  86s  85s  84s  83s  82s
7   A7o  K7o  Q7o  J7o  T7o  97o  87o  77   76s  75s  74s  73s  72s
6   A6o  K6o  Q6o  J6o  T6o  96o  86o  76o  66   65s  64s  63s  62s
5   A5o  K5o  Q5o  J5o  T5o  95o  85o  75o  65o  55   54s  53s  52s
4   A4o  K4o  Q4o  J4o  T4o  94o  84o  74o  64o  54o  44   43s  42s
3   A3o  K3o  Q3o  J3o  T3o  93o  83o  73o  63o  53o  43o  33   32s
2   A2o  K2o  Q2o  J2o  T2o  92o  82o  72o  62o  52o  42o  32o  22
```

- 对角线 (13): 对子
- 右上三角 (78): 同花 (suited)
- 左下三角 (78): 杂花 (offsuit)
- 总计: 169 种
