# Poker AI 上强度路线（CFR / 求解器参考）

> 目标：**在 Steam（Poker Legends, appid 758980）上真正赢过其他玩家。** 当前的
> `HeuristicPolicy` 只是 bootstrap baseline，用来驱动本地游戏 / bot 管线和当 sanity 对手；
> 真实强度必须切到下面这条业界已验证的 **CFR 求解器 / 自博弈** 主线。本文件是该路线的
> canonical 调研与分阶段计划；roadmap 总纲见 `docs/plan.md`，规则见 `AGENTS.md`。
>
> 纪律：**不长期手调启发式**（天然过拟合、不可传递）；尽早切 CFR，并用
> **exploitability + 固定参照对手 + 外部基准**量化强度，而不是只看同族自博弈。

## 1. 为什么不从零造算法

德扑是不完美信息博弈 AI 最经典的 benchmark，被学界啃了近 20 年，核心算法、开源实现、
评估工具都成熟。我们**只写"薄 facade + 共用 `ai.decide()` 接口"**：规则用 PokerKit，
求解器/自博弈用 OpenSpiel / RLCard / PokerRL，评估用 OpenSpiel best-response。
我们的分阶段路线（启发式 → preflop GTO → CFR → 自博弈，exploitability 评估）与业界主线一致。

## 2. 里程碑（系统 + 论文）

| 局型 | 系统 | 出处 | 方法要点 |
|------|------|------|---------|
| 限注 HU（基本"解出"） | **Cepheus** | Bowling, Burch, Johanson, Tammelin, *Science* 2015 | CFR+ 全博弈树求解 |
| 无限注 HU（超人） | **DeepStack** | Moravčík 等, *Science* 2017（U. Alberta） | continual re-solving + 深度反事实价值网络 |
| 无限注 HU（胜顶级职业） | **Libratus** | Brown & Sandholm, *Science* 2018（CMU；Brains-vs-AI 2017.1） | 抽象 + CFR+ blueprint + nested subgame solving + self-improver |
| 6-max 无限注（胜职业） | **Pluribus** | Brown & Sandholm, *Science* 2019 | MCCFR blueprint + 深度受限搜索；**训练仅约 $150 算力** |

> Pluribus 的低算力是关键信号：**多人 NLHE 的超人水平对个人项目可行**，不需要大集群。

## 3. 算法主线（演进链）

- **CFR** — Zinkevich, Johanson, Bowling, Piccione, *Regret Minimization in Games with
  Incomplete Information*, NeurIPS 2007。反事实遗憾最小化，自博弈收敛到纳什均衡（双人零和）。
- **MCCFR** — Lanctot, Waugh, Zinkevich, Bowling, NeurIPS 2009。蒙特卡洛采样版（external /
  outcome sampling），省内存、可扩展，Pluribus 用的就是它。
- **CFR+** — Tammelin 2014（arXiv）。收敛更快、实践默认，Cepheus 用它。
- **Deep CFR** — Brown, Lerer, Gross, Sandholm, ICML 2019。用神经网络逼近遗憾，**免去手工
  card abstraction**。**Single Deep CFR (SD-CFR)** — Steinberger 2019，简化且更稳。
- **NFSP** — Heinrich & Silver 2016。深度 RL + 自博弈逼近均衡（RLCard 内置）。
- **ReBeL** — Brown, Bakhtin, Lerer, Gong, NeurIPS 2020。统一深度 RL + 搜索的通用框架。

> 我们**到 CFR+/MCCFR 在抽象 HUNL 上就已远强于任何手调启发式**；Deep CFR / ReBeL 是后续可选增强。

## 4. 可直接使用的开源（仅官方来源，遵守 `AGENTS.md`）

- **OpenSpiel**（DeepMind，官方 repo `github.com/google-deepmind/open_spiel`，PyPI `open_spiel`）
  —— CFR / CFR+ / MCCFR / Deep CFR 实现 + `universal_poker`（基于 ACPC）+ **exploitability /
  best_response 现成工具**。我们 engine 已有 `adapters/openspiel.py`。**最重要的一个。**
- **RLCard**（Texas A&M DATA Lab，官方 `github.com/datamllab/rlcard`，PyPI `rlcard`）——
  NLHE / Limit 环境 + CFR / Deep CFR / NFSP + 示例 agent。我们 engine 已有 `adapters/rlcard.py`。
- **PokerRL**（Eric Steinberger，官方 `github.com/EricSteinberger/PokerRL`）—— Deep CFR / SD-CFR
  的 poker 专用参考实现。
- **PokerKit**（UofT CPRG，官方 `github.com/uoftcprg/pokerkit`，PyPI `pokerkit`）—— 规则/发牌/
  边池/摊牌权威，已是我们 engine 的底层。
- 手牌评估加速（可选）：`eval7` / `treys` / `phevaluator`（均官方 PyPI），或日后用 PyO3 包
  `equity-calculator/` 的 Rust 评估器。

## 5. 评估 / 基准（强度怎么量化）

- **Exploitability / best-response**（OpenSpiel 现成）：在抽象博弈上算"离纳什均衡多远"，
  是 CFR 阶段的**首要指标**。
- **固定参照对手**（已落地，`holdem_ai.baselines`）：random / call_station / rock / maniac，
  给绝对 bb/100 标尺；每次改动卡"对参照对手不回退"。
- **外部基准**：**Slumbot**（强 HUNL 开放 bot，有公开 API 可直接对战）；历年 **ACPC** agent。
- **preflop 对照**：公开 Nash push/fold 表 + GTO 开池 / 3bet 范围（基本已解，免费）。

## 6. 我们的上强度阶段（结合本仓库）

接口不变：本地 game 与 Steam bot 始终调用同一个 `ai.decide(state) -> Action`；新增的求解器
策略以同样接口替换 `HeuristicPolicy`。abstraction / 查表 / 网络都放在 `ai/`。

- **S0（进行中）启发式 baseline + 绝对评估**：修明显漏洞、打磨到"像正常人"。仅 bootstrap，
  不追 GTO。产物：`holdem_ai.heuristic` + `holdem_ai.baselines` + `holdem-ai-evaluate-heads-up`。
- **S1 preflop GTO 查表**：引入 HU（及 6-max）push/fold + 开池/3bet 范围（公开图表或小 CFR 解），
  翻前直接查表，翻后仍走启发式。**性价比最高的单点强度提升**，正好补当前翻前最大漏洞。
- **S2 CFR/CFR+/MCCFR（抽象 HUNL）**：用 OpenSpiel `universal_poker` + card bucketing + 有限
  bet sizing 抽象，自博弈求 blueprint；用 best-response/exploitability 评估；落成可被 `decide()`
  调用的查表策略。
- **S3 Deep CFR / SD-CFR**：用 PokerRL / OpenSpiel 去掉手工抽象，提升精度；或 ReBeL 式
  depth-limited search 做实时再求解。
- **S4 面向 Poker Legends 的实战形态**：Poker Legends 是 **6-max 多人 play-money**，终局更贴近
  **Pluribus 路线（MCCFR blueprint + 深度受限搜索）**，而非纯 HUNL。

## 7. 关键纪律与风险

- **GTO 是不可被剥削的下限，不是终点**：面对 Steam 上的**弱玩家池**，**剥削式打法（GTO floor +
  对手建模 / 群体偏差利用）通常比纯 GTO 赢更多**。所以 S2+ 的 blueprint 之上要留**对手建模 /
  exploit 层**的位置。
- **不长期手调启发式**：S0 只为"像正常人"和当 sanity 对手；强度靠 S2+。
- **抽象 soundness**：card/bet 抽象会引入"abstraction pathology"，必须用 best-response 验证抽象
  策略在真实博弈里的可被剥削度。
- **算力预算**：tabular CFR+/MCCFR 用 CPU 即可（Pluribus 量级 ~$150）；Deep CFR / RL 再上 GPU。
- **合规**：Poker Legends 为 play-money 社交扑克（非真钱），属个人自动化/学习用途；其 ToS 可能
  禁止自动化，bot 仅供本地学习，保留安全闸门与停手口径（见 `docs/plan.md` 阶段 4）。

## 8. 下一步

S0 收尾（修 sizing / 翻前偷盲 / 多街收手，对参照对手不回退）后，进入 **S1 preflop 查表**。
每个阶段以 exploitability（S2+）或参照对手 bb/100（S0/S1）量化，不靠主观感觉。
