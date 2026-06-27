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
- **S1 preflop 真值表 + 查表**：
  - **S1a（已落地）169 手牌类 + all-in 真值表**：`holdem_ai.preflop` 用本项目评估器（确定性蒙特卡洛、
    12000 样本/类、无外部下载）算出 169 个等价类对随机手的 heads-up all-in 胜率
    `PREFLOP_ALLIN_EQUITY`，并提供 `hand_class()` / `preflop_equity()` / `all_in_equity_vs_random()`
    （同 seed 可精确复算）。这是 push/fold 与 **S2 CFR 抽象（card bucketing）所需的基础数据**。
    实测发现：把它接进“开池范围选择”相对现有公式阈值并无可测增益（两者都开约 88%，参照池不惩罚边际
    选牌差异，还引入 vs-random 回退风险），故**按“不回退”纪律暂不改动已验证的启发式开池**，先把真值表
    作为基础设施沉淀。
  - **S1b（下一步）push/fold + 开池/3bet 查表**：在该真值表上做短筹码 jam-or-fold 与开池/3bet 范围；
    push/fold 是经典已解子博弈，在 ≤~12BB 短筹码下用真值表替代“min-open 再对 shove 弃牌”是明确的
    正确性提升，且可在短筹码评估里量化（vs maniac/random 的 all-in 决策）。
- **S2 CFR/CFR+/MCCFR（抽象 HUNL）**：
  - **S2a（已落地）CFR + exploitability 管线 + 小型 NLHE 抽象**：`holdem_ai.cfr` 用 OpenSpiel 的
    CFR/CFR+ 求解器与 exploitability 评估，验证 `solve -> average policy -> exploitability` 闭环：
    kuhn CFR+ 200 iters → ~3e-4，leduc CFR+ → ~0.01-0.03，CFR+ 明显快于 vanilla。并已从玩具游戏扩到
    **真正的无限注德扑抽象**：`nolimit_holdem_abstraction()` 生成 `universal_poker` 的 `fcpa`
    （fold/call/pot/allin）缩减牌堆游戏，`--game nlhe-small` 80 iters → exploitability ~0.006。
    入口 `uv run holdem-ai-train-cfr --game <kuhn_poker|leduc_poker|nlhe-small|...> --variant cfr_plus`。
    实测边界：全树 tabular CFR 可解 1 手牌缩减 NLHE（秒级），2 手牌即超时——**扩到真实 HUNL 需上
    MCCFR + card/bet 抽象**（S2b）。
  - **S2b（已落地）自有抽象游戏 → CFR blueprint → `decide()` 桥接**：不逆向 `universal_poker` 的
    infostate，而是写**自己掌控的 pyspiel 游戏** `holdem_ai.preflop_game`（heads-up push/fold、手牌
    分 8 个 equity 桶、叶子用 `preflop.BUCKET_EQUITY` 桶间胜率矩阵），infostate 由我定义因而能从
    `GameState` 精确复现。OpenSpiel CFR+ 把它解到 exploitability ~2e-5，得到阈值型 push/fold 策略
    （按钮 jam 范围 ⊇ 大盲跟注范围、筹码越短 jam 越宽）。`holdem_ai.blueprint.PushFoldPolicy` 把它
    包成同一个 `explain()/decide()` 接口：短筹码翻前查 blueprint、其余回退启发式——**第一个真正由
    CFR 求解、game/bot 共用的 `decide()` 策略**。
    诚实结果（10BB、400 手）：pushfold **赢所有参照对手**（random +45 / call_station +35 / rock +27 /
    maniac +75），但**输给启发式 current（-31.6）**、且赢参照对手赢得比 current 少——因为它只 jam/fold、
    不做小注价值剥削。这正实证了 §7 的判断：**GTO 是不可剥削下限、但对弱玩家池剥削式打法赢更多**。
    唯一反例 vs maniac（pushfold +75 / current -25），jam 抓疯狂诈唬更稳。
  - **S2c（下一步）扩抽象**：更细的桶 + 限注 bet sizing（limp/raise）+ 翻后街道（board bucketing），
    或换 MCCFR；目标是 blueprint 在弱场也能逼近或超过剥削式启发式，并保留 exploitability 下限。
    考虑“启发式为主 + 短筹码/关键节点查 blueprint”的混合，兼得剥削收益与不可剥削性。
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

S0 已修 sizing / 翻前偷盲（多街收手暂缓，见 S2 注）；S1a 真值表已落地。下一步 **S1b：在真值表上做
短筹码 push/fold + 开池/3bet 查表**，用短筹码评估量化。每个阶段以 exploitability（S2+）或参照对手
bb/100（S0/S1）量化，不靠主观感觉。
