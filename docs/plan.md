# holdem-lab 路线图：德扑游戏 + AI + Steam CV Bot

> 项目规范化路线图（canonical roadmap），从初期规划固化进仓库，供容器内任意 agent
> （Claude / Codex）读取。规则类约束见 `AGENTS.md`。

## 当前进度（截至 2026-06-26 阶段 4 Poker Legends ScreenState + timeline v0）

**已完成**
- Dev container（tier ① 无头核心）已 build 并在容器内验证通过：`ubuntu:24.04` + Ubuntu apt
  Python 3.12 + uv + Node 24（全部官方源 + 校验）+ Claude/Codex CLI。
- `scripts/dev/`（`venv-dir` / `check-venv` / `setup-dev-env` / `verify-dev-env` / `py` /
  `claude-shell`）已迁移并适配为 Python-at-root；`postCreate` 跑 `scripts/dev/setup-dev-env.sh`。
- `pyproject.toml`（uv，非包；dev: ruff/pytest/mypy，已排除 legacy）、`AGENTS.md`
  （官方下载源规则）已就位。
- 阶段 0 完成：`rust/` + `web/` 已整体 `git mv` 到 `equity-calculator/`；Pages workflow、
  `.gitignore`、验证脚本路径已跟随更新；根 `pyproject.toml` 已转 uv workspace，成员为
  `common/ engine/ ai/ game/ bot/`。
- `common/` 已落地共享 `GameState` / `Action` / `Card` / `Street` / `PlayerState` / `Pot`
  dataclass，作为 engine / game / ai / bot 的通用接口。
- 阶段 1 engine 基础闭环：`engine/` 已引入 PokerKit（官方 PyPI，锁文件记录哈希），提供
  `PokerKitFacade`、`HoldemEnv`、PokerKit → `GameState` 转换；支持固定 deck 测试入口、
  单座位隐藏观测、no-limit bet/raise-to 金额语义。
- 阶段 1 adapter 边界：已引入 RLCard 与 OpenSpiel（均来自官方 PyPI，锁文件记录哈希），
  `engine/holdem_engine/adapters/{rlcard.py,openspiel.py}` 已提供 `GameState` → 训练/求解观测、
  合法动作 id、动作 id → `Action` 的映射；OpenSpiel 后端验证到 `universal_poker` 可用。
- 阶段 3 最小 AI 入口：`ai/` 已提供 `holdem_ai.decide(state) -> Action` 与确定性启发式
  `HeuristicPolicy`；当前不新增评估器/训练依赖，先让 game/bot 共用同一个决策入口。
- 阶段 2 最小 pygame 牌桌：`game/` 已引入 pygame（官方 PyPI，锁文件记录哈希），提供
  `uv run holdem-game` / `python -m holdem_game` 启动入口；可渲染座位、手牌/暗牌、公共牌、
  筹码、底池与动作按钮，人类座位直连 `HoldemEnv.step()`，AI 座位调用 `holdem_ai.decide()`。
- 阶段 4 bot 骨架：`bot/` 已提供 `Capture` / `Recognizer` / `Automator` / `BotOrchestrator`
  接口；已实现无截图依赖的 in-process `GameState` 适配器，用于先验证
  capture → recognize → `ai.decide()` → automate 的控制闭环。
- 阶段 4 in-process bot 闭环：`game/` 已接入 bot pipeline，`uv run holdem-game --bot-seat 0`
  可让 bot 控制原本人类座位；该路径仍然不加截图/CV 依赖，先验证同一个
  `GameState` / `ai.decide()` / `Action` 控制面。
- 阶段 4 视觉 fixture：`bot/vision/annotations.py` 已定义截图标注 JSON schema；`game/` 已提供
  `uv run holdem-game-capture-fixture --output-dir artifacts/vision-fixtures`，可生成自家 pygame
  牌桌 PNG + JSON 标注，用于后续 CV/OCR 识别准确率评估。
- 阶段 4 视觉评估口径：`bot/vision/recognition.py` 已定义 `RecognizedTable` 中间结果、
  annotation oracle recognizer 与 `evaluate_recognition()`；后续真实 CV/OCR 只要输出同一结构，
  即可按 cards/buttons/chips/seats/table 分组统计准确率。
- 阶段 4 最小 CV/OCR 识别器：已引入 `opencv-python` / `pytesseract`（官方 PyPI，锁文件记录哈希）
  与 `tesseract-ocr` / `libgl1`（Ubuntu 官方 apt 源，已写入 devcontainer Dockerfile）；
  `bot/vision/roi_ocr.py` 可基于 fixture ROI 做 OpenCV 裁剪预处理、Tesseract OCR，并用
  `evaluate_recognition()` 统计识别准确率。
- 阶段 4 pygame CV/OCR 基线：自家 pygame fixture 的 ROI/OCR 识别已覆盖牌面、按钮、筹码、
  座位状态、当前行动座位；`uv run holdem-bot-evaluate-fixture <png> <json> --min-accuracy 1.0`
  在当前 fixture 上达到 overall/cards/buttons/chips/seats/table 全部 1.0。
- 阶段 4 Poker Legends 视频 ingest：已提供 `uv run holdem-bot-ingest-video <video> --out <dir>`
  工具，使用 OpenCV 顺序解码、按采样 FPS + 画面差分去重、输出 keyframe PNG、draft annotation
  JSON、`manifest.json`、`process_report.md`、分页 contact sheet 与首屏 `contact_sheet.jpg`；已对
  `artifacts/poker-legends-videos/raw/session_001.mov` 跑通，保留 177 个关键帧，产物在
  `artifacts/poker-legends-videos/session_001_ingest/`。
- 阶段 4 Poker Legends 代表帧选择：已提供
  `uv run holdem-bot-select-keyframes <manifest.json> --out <dir> --select ...` 工具，可从 ingest
  manifest 中复制代表 PNG/草稿 JSON、生成 `selected_manifest.json`、`selection_report.md` 与
  选择联系表；当前已在 `artifacts/poker-legends-videos/session_001_selection/` 生成 20 张代表帧，
  覆盖核心牌局街道、动作按钮、摊牌/all-in 与弹窗/菜单类 no-action 负样本。
- 阶段 4 Poker Legends ROI layout bootstrap：已提供
  `uv run holdem-bot-apply-poker-legends-layout <annotations...>` 工具，可把第一版可缩放 ROI 模板
  写入代表帧草稿标注，并输出 overlay PNG 与 `layout_report.md`；当前
  `artifacts/poker-legends-videos/session_001_selection/annotations/` 的 20 个草稿已应用
  `poker_legends_1600w_v1`，overlay 在同目录 `layout_overlays/`。
- 阶段 4 LLM 标注工厂：已引入 OpenAI Python SDK 与 Google Gemini SDK（均来自官方 PyPI，锁文件
  记录解析版本），提供
  `uv run holdem-bot-llm-annotate <annotations...> --image-root <dir> --out <dir>` 工具；支持
  `--provider gemini|openai`，默认 Gemini `gemini-3.1-flash-lite`，默认只生成离线请求包，
  `--execute` 才会调用对应 provider API。当前已为 20 个代表帧生成
  省流版 `artifacts/poker-legends-videos/session_001_selection/llm_annotation_slim/`：20 张 1280 宽 JPEG
  整图、360 个必要 ROI crop（board/cards/buttons/texts）、`requests.jsonl`、`manifest.json` 和
  `package_report.md`，总包约 6.3MB。执行器默认跳过已有候选结果，支持中断后 resume，避免重复计费。
- 阶段 4 LLM 候选标注执行：Gemini key 已通过本地 `.env` 提供；当前 `llm_annotation_slim/` 已成功
  生成 20/20 个 `candidate_annotations` 与 provider 原始响应，可中断 resume，避免重复计费。
- 阶段 4 Poker Legends LLM/CV 对比：已提供
  `uv run holdem-bot-compare-poker-legends-recognition <annotations...> --image-root ... --candidate-dir ... --out ...`
  工具，把现有 Tesseract ROI/OCR 输出与 LLM 候选逐字段比较，并写出
  `comparison_report.md` / `comparison.json` / `roi_ocr_results/`。当前 20 帧对比结果：
  cards 0/84、buttons 17/43、text_numbers 5/75、overall 22/202（约 10.9%）与 LLM 候选一致；
  这说明旧 OCR 不能直接作为 Poker Legends 主识别器。LLM 候选整体明显更强；对
  `keyframe_000145` 的人工复核确认第三张公共牌是 `9S`，LLM 该字段正确。候选标注仍需二次校验后
  才能作为真值。
- 阶段 4 bot 安全闸门：已新增 `ScreenState` / `ScreenKind` / `evaluate_safety()`，把截图先分类为
  `actionable_table`、`table_observe`、`blocked_overlay`、`non_table_ui`、`unknown_or_transition`；
  `BotOrchestrator` 现在只有在 screen 为可行动牌桌、识别置信度达标、有 `GameState`、且轮到受控座位
  时才会调用 `ai.decide()` 和 automator，其余状态一律返回停手原因。人工复核确认
  `keyframe_000124` 是“底部 hero 等待选择”的可行动帧，后续标注需要显式保留 bottom hero/current。
- 阶段 4 Poker Legends truth overlay：已提供
  `uv run holdem-bot-build-poker-legends-truth <candidate_annotations/*.json> --review-decisions ... --out ...`
  工具，把 20 个 LLM candidate 与 `human_review_decisions.json` 合成为第一版可审阅真值层；输出
  `truth_overlays/*.json`、`truth_overlay_summary.json`、`truth_overlay_report.md`。当前产物在
  `artifacts/poker-legends-videos/session_001_selection/llm_annotation_slim/truth_overlay_v1/`：
  20 帧、9 帧带人工复核覆盖、`actionable_table` 8 / `blocked_overlay` 6 / `table_observe` 6、
  结构警告 0。关键人工结论已落地：`keyframe_000124` 为 bottom hero/current 可行动帧；
  `keyframe_000132` 为 observe/showdown，普通底部手牌 ROI 被忽略；`keyframe_000145` 第三张公共牌为
  `9S`。
- 阶段 4 Poker Legends ScreenState recognizer v0：已提供图像版
  `uv run holdem-bot-evaluate-poker-legends-screen-state <truth_overlays/*.json> --annotation-dir ... --image-root ... --out ...`
  工具，基于当前 ROI layout 的按钮亮度/方差与 overlay 区域亮度信号，把真实截图先分类为
  `actionable_table` / `table_observe` / `blocked_overlay`。当前在 20 张代表帧上相对
  `truth_overlay_v1` 达到 20/20（1.000）screen-kind 匹配，报告在
  `truth_overlay_v1/screen_state_eval/screen_state_report.md`。运行时
  `PokerLegendsScreenStateRecognizer` 可消费 truth/candidate JSON 或截图路径；v0 只输出
  `ScreenState`，`GameState` 仍为 `None`，因此现有安全闸门会在 `no_game_state` 停手，不会误点。
- 阶段 4 Poker Legends card template v0：已提供
  `uv run holdem-bot-build-poker-legends-card-templates <truth_overlays/*.json> --annotation-dir ... --image-root ... --out ...`
  工具，从 `truth_overlay_v1` 的正常桌面帧（`actionable_table` / `table_observe`）抽取 hero 手牌与公共牌
  crop，生成标准化模板库、manifest 与 self / leave-frame 评估报告。当前产物在
  `truth_overlay_v1/card_templates_v1/`：73 个模板、覆盖 31/52 张牌；self sanity 可见牌准确率 1.000、
  hidden false-positive 0；保守 leave-frame precision 1.000、coverage 0.288。结论：模板链路可用且
  fail-closed，但当前 20 帧数据不足以支撑完整实时牌面识别，后续需要更多覆盖样本或更强的 rank/suit
  分类器。
- 阶段 4 Poker Legends primary button recognizer v0：已提供
  `uv run holdem-bot-build-poker-legends-button-templates <truth_overlays/*.json> --annotation-dir ... --image-root ... --out ...`
  工具，识别三个主动作按钮：`primary_left` 用模板区分 `check/call`（OCR 对 `Call` 不可靠）；
  `primary_middle` / `primary_right` 按固定位置映射为 `raise/fold`；raise shortcut 暂不作为动作输出，
  避免和非动作 blind 控件混淆。当前产物在 `truth_overlay_v1/button_templates_v1/`：8 个
  primary-left 模板（call=2、check=6），8 个 actionable 帧共 24 个主按钮 self 与 leave-frame
  accuracy/precision/coverage 均为 1.000。
- 阶段 4 Poker Legends session_002 分析：已对
  `artifacts/poker-legends-videos/raw/session_002.mov` 完成 ingest（22 分钟，413 个关键帧，18 页
  contact sheet），并生成 26 帧复核集与 ROI overlay。Gemini `gemini-3.1-flash-lite` 已执行
  26/26 个 LLM candidate（中途一次 503 后 resume 成功，未重复已成功帧），合成未人工复核版
  `truth_overlay_v1`：`actionable_table` 15 / `table_observe` 7 / `blocked_overlay` 4，uncertain 11。
  已人工复核左侧活动/信息栏口径：`keyframe_000091`-`keyframe_000096` 按自动化安全口径标为
  `blocked_overlay`；`keyframe_000045` 已确认是底部 hero 筹码不足/买入提示 + 左下 winner 展示，也按
  `blocked_overlay` 处理。当前 `truth_overlay_v5` 为 `actionable_table` 15 / `blocked_overlay` 10 /
  `table_observe` 1，7 帧带人工复核覆盖，ScreenState v0 达到 25/26（0.962）；唯一 mismatch 仍是
  `keyframe_000045`，因为 v0 尚无右下买入提示信号，但该帧没有主动作按钮簇，暂不构成误点击风险。
  ROI/OCR 对 LLM candidate overall agreement 仅
  0.074，继续确认旧 OCR 不能作为 Poker Legends 主识别器。session_002 未复核牌面模板覆盖
  40/52，和 session_001 合并可达 47/52（仍缺 `2H` / `5S` / `7D` / `8S` / `QD`）。已把 truth 合成里的
  主按钮 action_type 规则化：可操作桌面中 `primary_left` 只接受 `check/call`，`primary_middle`
  映射 `raise`，`primary_right` 映射 `fold`，非可操作桌面清除扑克动作；session_002 未复核按钮模板
  提升到 self accuracy 1.000、leave-frame precision 0.975 / coverage 1.000。
- 阶段 4 Poker Legends session timeline v0：已新增
  `uv run holdem-bot-build-poker-legends-session-timeline ...`，可把 truth overlay 帧转换为带时间戳的
  `PokerLegendsFrameObservation`，并输出 `session_timeline.json` / `session_timeline.md`。当前在
  session_002 的 26 张 v5 复核帧上生成 79 个时间线事件与 13 个 hand segment；前两段能保留跨买入提示
  / 左侧活动栏的上下文，后续单帧 segment 主要来自 22 分钟视频的稀疏抽样，不代表完整牌谱。该 tracker
  是后续“连续视频状态机”的第一版骨架，下一步要接更密的 ScreenState / card / button recognizer 输出。
- 阶段 4 Poker Legends rank/suit 牌面原型：已新增
  `uv run holdem-bot-build-poker-legends-card-part-templates ...`，把每张牌拆成 rank 局部模板与 suit
  局部模板，并额外输出 `leave_card` 评估（排除同一张具体牌，粗略衡量未见 rank+suit 组合的泛化）。
  synthetic 测试覆盖 `AS/AH/KS/KH`，证明同一 rank 与 suit 在其他组合中出现时可以组合识别。真实
  session_002 未复核 truth 上，保守阈值 `0.04/0.04` 得到 self 1.000、hidden false-positive 0、
  leave-frame precision 0.925 / coverage 0.626、leave-card precision 0.676 / coverage 0.346。结论：
  该原型可作为 fail-closed 辅助信号，但尚不能替代 LLM truth 或整牌模板来可靠识别完全未见组合。
- 根目录 `.env.example` 已提供 LLM provider/API key 样例；真实 `.env` 已被 `.gitignore` 忽略。
- 规则测试已覆盖：盲注、下注轮推进、全员弃牌终局、heads-up all-in 自动 runout、边池数学、
  摊牌平分、边池派奖、筹码守恒。
- 当前验证：`scripts/dev/verify-dev-env.sh` 通过（CV/OCR runtime、ruff format/check、mypy、pytest，
  97 tests）。
- 历史提交：`ea3dace`（dev container + AGENTS.md）、`086682e`（scripts/dev）、`7eb9f48`、
  `0a79c5c`、`c93b1bd`、`d90410a`、`f6090e8`、`560386d`、`d903102`、`380f477`、
  `d20669b`、`cabe333`、`b40eef9`、`ba3befd`、`718cf1e`。尚未 push。

**下一步：Poker Legends truth 复核与牌面/按钮/筹码识别到 GameState**
- 保持 ScreenState v0 作为最外层安全闸门；继续用 `truth_overlay_v1` 评估，不让可疑帧进入
  `ai.decide()`。
- 后续补 `keyframe_000045` 这类右下买入提示检测；当前没有主动作按钮簇，不会触发点击。
- 把 session timeline tracker 接到更密的关键帧识别输出上，用连续上下文稳定 hand boundary、
  overlay pause/resume 与 showdown/winner 展示，而不是只依赖单帧判断。
- 按钮 truth 已规则化：主按钮中间/右侧优先按固定位置映射，不直接吸收 LLM 的 `other` /
  `all_in` / `cancel` action_type；左侧继续只区分 `check` / `call`，不确定则 needs_review。
- 扩展牌面识别：当前 card template v0 是 fail-closed 基线；要进入可用 GameState，需要补更多视频样本
  覆盖剩余未见牌（session_001 + session_002 未复核候选仍缺 `2H` / `5S` / `7D` / `8S` / `QD`），
  或继续把 rank/suit 局部分类器升级为更强的分类器来泛化到未见牌；当前朴素 rank/suit 模板的
  `leave_card` precision 仍不足以直接上线。
- 按钮识别 v0 已覆盖 `check/call/raise/fold` 三主按钮；后续只有在需要快捷下注额时再处理
  raise shortcut，不把弹窗 confirm/cancel 映射为扑克动作。
- 做筹码/底池专用数字 OCR：只在 ScreenState 为可行动牌桌或观察牌桌时读 pot/stack/commit，并把
  overlay 后面的数字字段继续标为 ignored。
- 在 actionable truth 帧上实现 Poker Legends `RecognizedTable` → `GameState` 原型；只有当
  screen、牌面、按钮、筹码都超过阈值时才允许把 `state` 交给安全闸门，否则继续停在 `no_game_state` /
  `low_confidence`。

**环境备注（容器内）**
- 你在 Linux devcontainer 里，用户 `node`，`UV_PROJECT_ENVIRONMENT=/workspace/.venv-docker`。
- 跑 Python 用 `scripts/dev/py …` 或 `uv run …`；验证用 `scripts/dev/verify-dev-env.sh`。
- 只有 `game`（手玩）和真·Poker Legends bot 在 macOS host 跑，其余都在容器内。

---

## Context（背景）

现有 `holdem-lab` 只是一个德扑 equity 计算器（Rust core + WASM + Tauri + React），
在整体目标里占比很小。本次要在**同一仓库**（保留名字）里，基本**从零**构建三块大功能：

1. 一个可玩的德州扑克游戏；
2. 用 AI 玩这个本地游戏；
3. 用 AI（画面识别 + 自动化）玩 Steam 上的扑克游戏。

现有计算器先**归档到一边**，不约束新设计；日后或可作为加速器（用 PyO3 把 Rust 手牌评估
包给 Python 提速），但当前不依赖它。核心洞察：**引擎产出的 `GameState` 是通用接口**——本地
游戏和 Steam bot 都把状态喂给同一个 `ai.decide(state) -> action`，bot 只是“用 CV 把屏幕翻成
同样的状态”。AI 只写一次，两处复用。

## 已确定的技术选型（来自与用户确认）

- 归档：`rust/` + `web/` → `equity-calculator/` 子目录（不叫 legacy）。
- 主语言：Python；`uv` 管理的单仓工作区。
- 引擎：自研**薄 facade**（掌控我们自己的 `GameState`），底层包 PokerKit（全规则），
  并提供适配器接 OpenSpiel / RLCard。
- 游戏 UI：pygame（在 host 运行）。
- AI：借力 OpenSpiel / RLCard，启发式 → CFR → 自博弈，分阶段。
- Bot：Python（host 运行）；**先拿自家 pygame 游戏练管线**（Xvfb 容器），真目标
  **Poker Legends（Steam 758980）在 macOS 原生**跑。
- 开发环境：hybrid dev container（见下）。

## 目标结构

```
holdem-lab/
├── equity-calculator/      # 归档：原 rust/ + web/（整体搬入）
├── common/                 # 共享类型：GameState / Action / Card / Street
├── engine/                 # 规则权威引擎 + 多智能体环境（Python）
├── ai/                     # 决策引擎（Python）
├── game/                   # pygame 牌桌（host）
├── bot/                    # Steam CV bot（Python, host）
├── .devcontainer/          # hybrid 容器（engine/ai 用）
├── pyproject.toml          # uv workspace
└── README.md
```

## 分阶段实施

### 阶段 0 — 归档 + 脚手架
- `git mv rust/ web/` → `equity-calculator/`（整体搬：Tauri 的 `../../holdem-core` 相对路径
  不变）。更新 `.github/workflows/deploy.yml` 里 `rust/holdem-*` → `equity-calculator/rust/holdem-*`，
  或暂停该 workflow。
- 建 uv workspace：根 `pyproject.toml` + 成员 `common/ engine/ ai/ game/ bot/`。
- 工具：ruff（lint+format）、pytest、mypy（可选）。
- `common/`：`GameState`、`Action`、`Card`、`Street` 等 dataclass——全项目的“通用接口”。
- `.devcontainer/`（见“开发环境”）。

### 阶段 1 — engine/（先有一局能跑的德扑）
- 包 **PokerKit** 实现全规则：盲注、下注轮、合法动作、边池、摊牌、定胜负；对外只暴露
  我们自己的 `GameState` 与 `step(action)`。手牌评估用 PokerKit 自带（或 `eval7`/`treys`/
  `phevaluator` 备选）。
- 环境 API：Gym/PettingZoo 风格 `reset / step / legal_actions / observe`，供 AI 自博弈。
- 适配器：`engine/holdem_engine/adapters/{openspiel.py,rlcard.py}`（GameState ↔ 各库格式）。
- 测试：边池数学、摊牌平分、筹码守恒（property test）。
- 代表文件：`engine/holdem_engine/{state.py,facade.py,env.py}`。

### 阶段 2 — game/（pygame，host 运行）
- 牌桌：座位、手牌/公共牌、筹码/底池、动作按钮、简单动画。
- 座位可设 人/AI；AI 座位调用 `ai.decide()`，同进程直连 engine。
- 兼作 AI 自博弈“观战”可视化。
- 代表文件：`game/holdem_game/{app.py,table_view.py,widgets.py,assets/}`。

### 阶段 3 — ai/（决策引擎，可与阶段 2 并行）
- API：`decide(state: GameState) -> Action`。
- A 启发式：Monte-Carlo equity + 底池赔率 + 位置 + 听牌 outs → 动作 + 下注量
  （快速上线，驱动游戏 AI 座位与首个 bot）。
- B preflop GTO：push/fold 与开池范围（查表或 CFR 解）。
- C CFR/CFR+：经 OpenSpiel 在抽象博弈上求解；或经 RLCard 做 deep-CFR/自博弈。
- 评估：AI vs AI 的 bb/100；OpenSpiel best-response 算 exploitability。
- 代表文件：`ai/holdem_ai/{api.py,heuristic.py,preflop.py,cfr.py,eval.py}`。

### 阶段 4 — bot/（Steam CV bot，host 运行）
- 抽象层：`Capture`、`Recognizer`（像素→GameState）、`Automator`——平台/游戏/后端皆可替换。
  `Capture`/`Automator` 两套后端：**xvfb 容器**（打自家游戏、可 CI、确定性）/ **host 原生**
  （mss + pyautogui 打真实屏幕）；`Recognizer` 与 `ai.decide()` 两边共用同一份代码。
- 识别默认：卡牌模板匹配 + 数字 OCR（Tesseract/PaddleOCR）+ 按钮检测；VLM（复用现有
  Qwen-VL/豆包）做兜底与冷启动。
- 编排：轮到我方 → 读屏 → `ai.decide()` → 操作；安全：急停热键、仅在我方回合操作、
  拟人随机延时。
- 策略：**先对自家 pygame 游戏**跑通全链路（Xvfb 容器内、可控/可 CI），再写
  `bot/holdem_bot/adapters/poker_legends.py` 对接真目标（**Poker Legends 758980**，macOS 原生）。
- 合规：Poker Legends 是 play-money 社交扑克（非真钱），属个人自动化/学习；注意其 ToS 可能
  禁止自动化，仅供本地学习用途。
- 代表文件：`bot/holdem_bot/{capture.py,recognize/,automate.py,orchestrator.py}`。

## 开发环境（三层 hybrid，融合两个参考项目）

参考 quant（无头核心）+ web3-tycoon（Xvfb + noVNC 的 GUI/测试容器），分三层：

- **① 无头核心容器（engine + ai）** —— **骨架两项目通用**（同作者：Claude/Codex 挂载、
  firewall、runArgs、zsh 都一致）。**base 取 web3-tycoon 的 `ubuntu:24.04`**（比 quant 的
  `node:20` 更适合 Python+图形项目，且与 ② 共用 base、apt 包好叠加），**Python 工具链照搬
  quant**：uv + 固定 Python 3.12 + 双 venv；Node 仅为装 Claude/Codex CLI 而保留。负责
  训练/求解/测试/CI，并把 **OpenSpiel 的 C++ 依赖在镜像里一次构建好**。Rust 精简（仅日后做
  PyO3 加速器时再装），加 `cmake build-essential` 等。
- **② GUI / “bot 打自家游戏”测试容器（game + 阶段 4 练管线）** —— 借用 web3-tycoon 的
  **Xvfb + x11vnc + websockify + noVNC + fluxbox** 方案：pygame 渲染到 `DISPLAY=:0`，浏览器开
  `localhost:7900` 即可观看。关键收益：**阶段 4“先拿自家游戏练管线”可整套塞进同一个 Xvfb 容器**
  —— pygame 游戏 + 截图（mss 抓 Xvfb 帧）+ OCR + 注入式点击（Xlib/pyautogui 打同一帧）在容器内
  闭环，确定性、可 CI、免 host 权限。镜像 `FROM` ① 再叠加图形层，加
  `xvfb x11vnc websockify novnc fluxbox libgl1 libsdl2* tesseract-ocr`。（人要“手玩”想更跟手时，
  game 也可直接在 host 原生跑。）
- **③ 真·Steam bot（host 原生，macOS 先行）** —— 目标游戏 **Poker Legends（Steam 758980）
  有 Mac/Win 版、无 Linux 版**，所以**直接在 macOS 开发机原生跑整套**（截图+CV+自动化+游戏同机），
  无需 Windows 机/VM；进不了容器（游戏无 Linux 版，Xvfb 也帮不上）。macOS 需授予**屏幕录制 +
  辅助功能(Accessibility)** 权限；capture 用 Quartz `CGWindowListCreateImage` 或 mss 抓游戏窗口，
  automate 用 pyautogui/pynput。Windows 版日后可作第二目标。`Capture`/`Automator` 的
  **xvfb（练自家游戏）/ macOS-host（打真游戏）双后端**，识别与决策代码两边复用。
- **venv**：沿用 quant 的双 venv（`.venv-docker` / `.venv-mac`），同一 uv 工作区两处各用。
- **GPU**：启发式 + tabular CFR 用 CPU 足够；deep-CFR/RL（C 阶段）再定（Linux+NVIDIA host
  `--gpus all`；macOS MPS 只能 host 侧）。

## 验证

- engine：`pytest`（规则/边池/摊牌/筹码守恒 property test）。
- ai：评估脚本跑 AI vs random / vs 启发式 的 bb/100；CFR 用 OpenSpiel 算 exploitability。
- game：host 上手玩一局 人 vs AI。
- bot：对自家 pygame 游戏跑 读屏 → 决策 → 点击 闭环，统计识别准确率，再迁 Steam。

## 已确认 / 待后续

- **目标游戏已定**：Poker Legends（Steam 758980），有 Mac/Win 版、无 Linux 版 → 阶段 4 在
  **macOS 开发机原生**跑，首个 `adapters/poker_legends.py`。
- 待办（不阻塞阶段 0–3）：进入阶段 4 前采集游戏窗口的牌/按钮/数字布局（模板 + OCR 区域）；
  Windows 版作为可选第二目标。
