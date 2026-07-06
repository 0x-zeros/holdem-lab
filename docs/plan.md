# holdem-lab 路线图：德扑游戏 + AI + Steam CV Bot

> 项目规范化路线图（canonical roadmap），从初期规划固化进仓库，供容器内任意 agent
> （Claude / Codex）读取。规则类约束见 `AGENTS.md`。

## 当前进度（截至 2026-07-01 阶段 2/3 local game + AI evaluation v2）

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
  避免和非动作 blind 控件混淆。`primary_left` 模板特征已改为 RGB 归一化图像，避免灰度 equalize
  弱化 `Call` 蓝色 C 与 `Check` 绿色对勾差异。当前产物在 `truth_overlay_v1/button_templates_v1/`：
  8 个 primary-left 模板（call=2、check=6），8 个 actionable 帧共 24 个主按钮 self 与
  leave-frame accuracy/precision/coverage 均为 1.000。
- 阶段 4 Poker Legends session_002 分析：已对
  `artifacts/poker-legends-videos/raw/session_002.mov` 完成 ingest（22 分钟，413 个关键帧，18 页
  contact sheet），并生成 26 帧复核集与 ROI overlay。Gemini `gemini-3.1-flash-lite` 已执行
  26/26 个 LLM candidate（中途一次 503 后 resume 成功，未重复已成功帧），合成未人工复核版
  `truth_overlay_v1`：`actionable_table` 15 / `table_observe` 7 / `blocked_overlay` 4，uncertain 11。
  已人工复核左侧活动/信息栏口径：`keyframe_000091`-`keyframe_000096` 按自动化安全口径标为
  `blocked_overlay`；`keyframe_000045` 已确认是底部 hero 筹码不足/买入提示 + 左下 winner 展示，也按
  `blocked_overlay` 处理。当前 `truth_overlay_v5` 为 `actionable_table` 15 / `blocked_overlay` 10 /
  `table_observe` 1，7 帧带人工复核覆盖；ScreenState v0 在补入右下买入提示 magenta 信号后达到
  26/26（1.000），session_001 也保持 20/20（1.000）无回归。
  ROI/OCR 对 LLM candidate overall agreement 仅
  0.074，继续确认旧 OCR 不能作为 Poker Legends 主识别器。已把 truth 合成里的
  主按钮 action_type 规则化：可操作桌面中 `primary_left` 只接受 `check/call`，`primary_middle`
  映射 `raise`，`primary_right` 映射 `fold`，非可操作桌面清除扑克动作；session_002 未复核按钮模板
  提升到 self accuracy 1.000、leave-frame precision 0.975 / coverage 1.000。
- 阶段 4 Poker Legends session timeline v0：已新增
  `uv run holdem-bot-build-poker-legends-session-timeline ...`，可把 truth overlay 帧转换为带时间戳的
  `PokerLegendsFrameObservation`，并输出 `session_timeline.json` / `session_timeline.md`。当前在
  session_002 的 26 张 v5 复核帧上生成 79 个时间线事件与 13 个 hand segment；前两段能保留跨买入提示
  / 左侧活动栏的上下文，后续单帧 segment 主要来自 22 分钟视频的稀疏抽样，不代表完整牌谱。该 tracker
  是后续“连续视频状态机”的第一版骨架，下一步要接更密的 ScreenState / card / button recognizer 输出。
- 阶段 4 Poker Legends dense ScreenState scan + LLM 候选选择：已新增
  `uv run holdem-bot-scan-poker-legends-screen-state <manifest.json> --out ... --selection-out ...`，
  可对 ingest 的全量 keyframes 跑 ScreenState v0，输出 screen runs、状态计数、overlay/button 特征与
  高价值 LLM/人工复核候选帧。当前 session_002 全量 413 帧扫描为 `actionable_table` 87 /
  `table_observe` 316 / `blocked_overlay` 10，生成 124 个 screen run；候选策略按 blocked overlay、
  2-button actionable 边界、按时间间隔的 actionable/observe 样本优先，得到覆盖 0s-1302s 的 51 张候选帧。
  已生成 `auto_review_selection_v2/` 并套用 ROI layout；离线 LLM 省流包
  `auto_review_selection_v2/llm_annotation_slim/` 为 51 帧、918 个 ROI crop、约 16MB。Gemini
  `gemini-3.1-flash-lite` 已执行 51/51（中途一次 503 后 resume，已成功帧被跳过），合成未复核
  `truth_overlay_v1`：`actionable_table` 32 / `table_observe` 17 / `blocked_overlay` 2，uncertain 21、
  warnings 8。ScreenState v0 对该未复核 truth 为 37/51（0.725）：8 张是安全阻塞口径
  （买入提示/左侧活动栏）应覆盖为 blocked，6 张是 LLM 把“在局中/底部预选条”当作 actionable 但
  ScreenState 未见三主按钮簇。人工确认：底部圆形 action strip 是等待别人时的预选/快捷操作，
  用于减少网络延迟，不作为安全自动点击信号；另外 4 张为轮到其他玩家操作。当前 reviewed
  `truth_overlay_v2` 为 `actionable_table` 26 / `table_observe` 15 / `blocked_overlay` 10，
  14 帧人工复核、warnings 0，ScreenState v0 达到 51/51（1.000）。
- 阶段 4 Poker Legends rank/suit 牌面原型：已新增
  `uv run holdem-bot-build-poker-legends-card-part-templates ...`，把每张牌拆成 rank 局部模板与 suit
  局部模板，并额外输出 `leave_card` 评估（排除同一张具体牌，粗略衡量未见 rank+suit 组合的泛化）。
  synthetic 测试覆盖 `AS/AH/KS/KH`，证明同一 rank 与 suit 在其他组合中出现时可以组合识别。真实
  session_002 未复核 truth 上，保守阈值 `0.04/0.04` 得到 self 1.000、hidden false-positive 0、
  leave-frame precision 0.925 / coverage 0.626、leave-card precision 0.676 / coverage 0.346。结论：
  该原型可作为 fail-closed 辅助信号，但尚不能替代 LLM truth 或整牌模板来可靠识别完全未见组合。
- 阶段 4 Poker Legends multi-source template 评估：已新增
  `uv run holdem-bot-build-poker-legends-multi-templates --source <name> <truth_dir> <annotation_dir> <image_root> ... --out ...`，
  可把多个 reviewed truth source 合并成带 source 前缀的 frame_id，解决不同视频 `keyframe_000023`
  这类重名问题，并一次输出整牌模板、rank/suit 拆分模板和按钮模板评估。v1 用 session_001
  `truth_overlay_v1` + session_002 `auto_review_selection_v2/truth_overlay_v2` 生成 71 帧，覆盖 48/52。
- 阶段 4 Poker Legends card-review 候选选择：已新增
  `uv run holdem-bot-select-poker-legends-card-review-candidates --source <name> <ingest_manifest> ... --target-cards ...`，
  用当前整牌模板和 rank/suit 模板扫描全量 ingest，排除已有 truth，只把目标缺牌与模板 gap 高价值帧
  复制出来并套 ROI layout，供小 LLM 包复核。第一轮扫描 session_001 + session_002 共 590 帧，排除
  已有 truth 73 帧，从 485 个可用桌面帧中选 48 帧；LLM 包 7.9MB（48 帧、336 个 card/board crop），
  Gemini `gemini-3.1-flash-lite` 中途一次 503 后 resume 成功，补到 `5S` / `7D` / `QD`。加入
  `card_review_selection_v1/truth_overlay_v1` 后生成
  `artifacts/poker-legends-videos/multi_source_templates_v2/`：119 帧；整牌模板 529 个、覆盖 51/52，
  leave-frame precision 0.956 / coverage 0.558；rank/suit 模板 1058 个、覆盖 51/52，
  leave-card precision 0.627 / coverage 0.675；按钮仍为 21 个且 leave-frame precision/coverage 1.000。
  第二轮只针对 `8S`，排除新增 truth 后从 437 个可用桌面帧中选 24 帧，LLM 包 3.1MB；未发现 `8S`。
  现有两段视频的 A 路线已把缺牌收敛到只剩 `8S`。
- 阶段 4 Poker Legends card classifier / consensus v1：已新增
  `uv run holdem-bot-build-poker-legends-card-classifier ...`，训练本地可见性门控 + rank/suit 加权
  KNN 分类器；另新增 `uv run holdem-bot-evaluate-poker-legends-card-consensus ...`，运行策略为
  “整牌模板先命中；否则 rank/suit part 模板与 classifier 必须对同一 slot 给出同一张牌才输出”。
  B 阶段发现全量 `table_observe` / showdown truth 不能直接作为固定 board ROI 训练数据：Poker Legends
  会把 winner/hero 展示牌摆到桌面附近，LLM truth 可能把展示牌错并进 board slot。当前 B 的可用口径先
  收敛到 `actionable_table`：classifier 单体 leave-frame precision/coverage 0.899/0.958、leave-card
  0.510/0.903、hidden false-positive 0；旧 part actionable baseline leave-card 0.680/0.606；整牌模板
  leave-frame 0.945/0.571；consensus actionable v1 leave-frame 0.911/0.889、leave-card 0.745/0.502、
  hidden false-positive 0。结论：classifier 单体不直接上线；`actionable_table` 上用 full-card +
  part/classifier consensus 作为 fail-closed 牌面信号，observe/showdown 继续走 timeline/暂停口径。
  诊断产物在 `artifacts/poker-legends-videos/multi_source_templates_v2/card_truth_audit_v1/`。
- 阶段 4 Poker Legends table recognizer prototype：已新增 `PokerLegendsTableRecognizer`，把
  ScreenState、card consensus、button recognizer 与已复核 truth 元数据合成为
  `RecognizedTable`，并且只在 `actionable_table`、pot/seat/hero 手牌/board/按钮等字段齐全时构造
  prototype `GameState`；纯截图或元数据不足仍保持 `state=None` fail-closed。当前在
  `multi_source_templates_v2` 的 57 个 actionable truth 帧上扫描：15 帧生成 prototype state；其余按
  `missing_pot` 22、`missing_hero_seat` 7、`not_enough_players` 6、`missing_call_amount` 3、
  `board_count_mismatch` 3、`hero_not_current` 1 停手。产物在
  `artifacts/poker-legends-videos/multi_source_templates_v2/table_recognizer_actionable_v1/`。
- 阶段 4 Poker Legends numeric OCR fallback v1：已新增 `PokerLegendsNumberRecognizer` 与
  `uv run holdem-bot-evaluate-poker-legends-numbers ...`，专门识别 pot / hero stack / primary-left call
  amount；table recognizer 仅在 reviewed truth 缺字段时才调用 OCR，并且只接受置信度达到阈值的数字，
  低置信结果保留在 metadata/artifact 中但不进入 `GameState`。当前在同一批 57 个 actionable truth 帧上
  扫描：OCR 尝试 23 帧、接受 20 帧；`missing_pot` 从 22 降到 3，prototype state 从 15 增到 16。
  新暴露的后续 blocker 为 `missing_hero_seat` 13、`missing_legal_actions` 11、`not_enough_players` 7、
  `missing_call_amount` 3、`board_count_mismatch` 3、`hero_not_current` 1。产物在
  `artifacts/poker-legends-videos/multi_source_templates_v2/table_recognizer_numeric_v1/`。
- 阶段 4 Poker Legends seat/action fallback v1：在不放宽安全闸门的前提下，继续收紧
  `RecognizedTable -> GameState`。新增保守规则：hero 手牌可见且有 reviewed `hero_stack` 时补受控
  hero seat；truth 只有 hero 且有 reviewed `right_top_stack` / `opponent_stack` 时补一个最小对手 seat；
  图像按钮识别为空时，只吸收 reviewed truth 中明确的直接动作按钮，继续排除 `Call Any`、
  `Check/Fold` 与快捷下注；当 reviewed street 与可见公共牌数量冲突时，用公共牌数量推断真实街道。
  当前同一批 57 个 actionable truth 帧：prototype state 从 numeric v1 的 16 提升到 47；
  `missing_hero_seat` 13 -> 0，`missing_legal_actions` 11 -> 3，`not_enough_players` 7 -> 3，
  `missing_call_amount` 3 -> 1，`missing_pot` 3 -> 2，`board_count_mismatch` 3 -> 0，
  `hero_not_current` 仍为 1。产物在
  `artifacts/poker-legends-videos/multi_source_templates_v2/table_recognizer_seat_action_v1/`。
- 阶段 4 Poker Legends host dry-run v0：已新增 macOS host 侧捕获/点击规划骨架，不引入第三方下载。
  `MacOSScreenCapture` 只调用 macOS 自带 `screencapture -x`（可选 `-l <window_id>`），输出
  `CapturedFrame` 与 image 坐标系 metadata；`PokerLegendsLayoutClickPlanner` 把通用 `Action`
  保守映射到三主按钮中心：`check/call -> primary_left`、`bet/raise/all_in -> primary_middle`、
  `fold -> primary_right`；`PokerLegendsDryRunAutomator` 只写 JSONL 审计日志，`executed=false`，
  不做真实点击。命令行入口为 `uv run holdem-bot-capture-macos-screen --out-dir ...` 与
  `uv run holdem-bot-plan-poker-legends-click --layout-annotation ... --action ...`。该阶段仍停在
  dry-run，不进入真 Poker Legends 点击测试。
- 阶段 4 Poker Legends perception redesign Slice 0-3：按
  `docs/poker-legends-perception-redesign.md` 落地 observation-first / contract-first 骨架。Slice 0/1
  已把 `RecognitionMode`、frame/ROI evidence、`VisualObservation`、`GameStateAssemblyResult`、
  structured issues 与 contract level 变成 `RecognitionResult` 的一等字段，旧 metadata 只作兼容镜像；
  image-only 模式会拒绝 reviewed-truth critical field。Slice 2 已新增 replay safety summary /
  Markdown report，显式统计 authorization、unsafe authorization、truth-assisted authorization、
  expected non-actionable 与 false actionable。Slice 3 新增 `PokerLegendsTemporalTracker` 并接入
  replay dry-run 与 live HUD 连续帧路径：默认需要 2 帧当前窗口稳定后才把 `single_frame_valid` 升级为
  `temporally_stable_valid`，overlay clear / 新手牌边界会重新稳定，旧 stable state 不会授权当前行动。
  仍然只做 dry-run / overlay，不做真实点击。
- 阶段 3/4 AI heuristic v1：`holdem_ai.decide(state) -> Action` 入口保持不变，新增
  `explain_decision(state) -> PolicyDecision`，供 bot/dry-run 审计策略理由。策略评分从原先只看私牌
  扩展为可解释手牌评估：成牌类型、flush/straight/combo draw、估算 outs、pot odds、位置 bonus、
  多人底池 penalty、价值下注与强听牌 semi-bluff 下注尺度。`BotOrchestrator` 现在调用
  `explain_decision()`，`BotStepResult` 带 `policy_decision`。新增
  `uv run holdem-bot-run-poker-legends-dry-run ...` 单步 dry-run 入口，可把已保存图片或 macOS 截图、
  layout/truth 标注、牌面/按钮 manifests 串到 ScreenState/Table recognizer、AI 决策与 dry-run JSONL；
  仍然没有真实点击入口。
- 阶段 2/3 本地游戏与 AI polish v1：当前方向先回到自家 pygame 游戏和共用 AI。AI 在强价值下注与
  半诈唬之外新增中等成牌保护下注（如 flop/turn 顶对按半池保护），`decide()` 接口保持兼容；
  pygame 游戏新增多档 bet/raise 按钮（min / half-pot / pot 去重后展示）、键盘动作入口、
  行动日志面板、AI 策略理由展示，以及 CLI 桌面配置参数（玩家数、人类座位、起始筹码、大小盲、
  bot seat/delay）。这批改动不引入新依赖，继续通过同一个 `GameState` 和 `explain_decision()` 让
  本地游戏与 bot 共用策略。
- 阶段 2/3 本地游戏与 AI polish v2：新增无外部依赖的轻量 hand evaluator / rollout equity 模块，
  暴露 `evaluate_best_hand()` 与 `estimate_showdown_equity()`；postflop `HeuristicPolicy` 会用固定
  seed 抽样 equity 混入 strength，并把 `showdown_equity` 写入 `PolicyDecision.metadata`，方便本地游戏
  和 bot 日志审计。pygame 游戏新增多手牌 button/SB/BB 自动轮转，终局消息改为可读的 winners/payoff
  摘要，行动日志也保留 hand complete 结果。
- 阶段 2/3 本地游戏与 AI polish v3：新增
  `uv run holdem-ai-evaluate-heads-up`，可让 `current` / `no_equity` / `tight` / `loose` 等本地
  heuristic profile 做 heads-up 对战评估，输出 JSON 里的 chips、bb/100、胜负、动作频率和策略理由
  频率；该脚本只依赖 workspace 内 `holdem-engine`，不下载外部包。pygame 游戏新增键盘数字下注输入：
  有 bet/raise 时可直接输入总下注额，Backspace/Delete 编辑，Up/Down 按 big blind 调整，Enter 提交；
  合法自定义金额会作为额外 `Bet/Raise X` 按钮展示。终局如果真正摊牌，会在行动日志里列出各亮牌玩家
  的牌型分类，弃牌结束则不伪造摊牌信息。
- 阶段 2/3 本地游戏与 AI polish v4：本地 pygame 会话从单手 demo 变成可连续试玩的 cash-session
  模式：每手结束后累计 session profit，下一手沿用当前 session stacks；座位破产时自动按初始筹码
  rebuy 并记录到行动日志；桌面左上角显示 session 盈亏与当前手前筹码。`P` 键可暂停/恢复 AI 自动
  推进，暂停后 human 行动会停在下一个 AI seat，便于观察局面；恢复后继续自动推进。CV fixture 生成
  时会关闭 session/action overlay，保持自家 pygame OCR 基线稳定。
- 阶段 2/3 AI evaluation v1：`uv run holdem-ai-evaluate-heads-up` 进入系统评估阶段。默认可跑两两
  heads-up；新增 `--matrix current no_equity tight loose` 模式，可自动跑所有 profile pairings，
  输出 pairings 明细与 leaderboard（chips、bb/100、胜负统计）。当前 profile 仍是可解释 heuristic
  族，作为后续手调 range 或接 CFR/RL 前的基线。
- 阶段 2/3 本地试玩 UX v2：AI profile 定义已从评估脚本抽成 `holdem_ai.profiles` 公共入口，
  `game` 与 `evaluate` 共用 `current` / `no_equity` / `tight` / `loose`；pygame 新增
  `uv run holdem-game-heads-up` 与 `uv run holdem-game --heads-up` 单人对 AI 入口，并支持
  `--ai-profile` 选择自动座位策略。`--bot-seat` 保持为 bot pipeline 接管座位的测试入口，不作为普通
  AI 对手开关。试玩说明见 `docs/local-game.md`。
- 阶段 2 本地 no-limit stake 对齐：本地 pygame 默认级别改为 `5/10`，默认买入按 `100BB`
  计算为 `1000`；新增 `--stake-level`，从 `5/10` 开始按层翻倍（如 `10/20`、`20/40`），仍可用
  `--small-blind` / `--big-blind` / `--starting-stack` 手动覆盖。动作金额继续使用 no-limit
  `bet/raise-to` 总投入语义。
- 阶段 2 本地结算 UX：一手结束后停在中心结算面板，明确显示赢家、hero 本手输赢、各座位 payoff /
  stack 与摊牌牌型；只有按 `N` 或点击 `Next hand` 才进入下一手。
- 阶段 2 本地试玩 UX v3：本地 AI 自动行动从同步循环改为按帧调度，默认每次行动前等待 `700ms`，
  并新增 `--ai-delay-ms` / `--turn-timeout-sec`；牌桌顶部显示当前行动座位、最大等待倒计时和进度条。
  pygame 人类 UI 会保留 Steam 风格的 Fold 快捷按钮；核心 `GameState.legal_actions` 仍只在有跟注压力时
  暴露 Fold，避免 AI/训练接口学到免费弃牌。
- 阶段 2 本地试玩 UX v4：修复 PokerKit 内部座位顺序与 holdem-lab public seat 的映射，确保
  button/SB/BB、实际盲注投入、当前行动座位一致；AI/bot 动作导致终局时先显示最后动作，短暂延迟后再
  弹出结算面板，避免 `New hand` 后看起来直接跳回结算。
- 根目录 `.env.example` 已提供 LLM provider/API key 样例；真实 `.env` 已被 `.gitignore` 忽略。
- 规则测试已覆盖：盲注、下注轮推进、全员弃牌终局、heads-up all-in 自动 runout、边池数学、
  摊牌平分、边池派奖、筹码守恒。
- 阶段 2/3 AI evaluation v2（绝对基线参照对手）：新增 `holdem_ai.baselines`，提供四个确定性参照
  对手 `random` / `call_station` / `rock` / `maniac`（与 `HeuristicPolicy` 同样实现
  `explain()->PolicyDecision` / `decide()` 接口），并入 `profiles`（`REFERENCE_PROFILE_NAMES`）与
  `holdem-ai-evaluate-heads-up --matrix`，给出脱离同族自博弈的绝对 bb/100 标尺。250 手对战发现：
  `current` 暴打 `random`（约 +830 bb/100）、`maniac`（约 +1090）、`call_station`（约 +310），但对
  **只弃牌/从不下注的 `rock` 仅 +4.6 bb/100**——暴露翻前按钮位只 limp 不加注、几乎不偷盲/不持续
  下注的被动漏洞；`no_equity` 与 `current` 对参照对手几乎同分，说明当前 160 样本 equity 采样对绝对
  收益贡献很小、却是主要算力开销。
- 阶段 2/3 AI heuristic v2（按参照对手实测修漏洞）：基于绝对基线小步修两处明显漏洞，每步以“对
  参照对手 bb/100 不回退”为准绳。① 价值下注 sizing：去掉“目标触顶时回落 min-bet”的塌缩（低 SPR
  下一组 A 在 200 底池只剩 40 筹码竟 min-bet），改为正常 clamp / all-in。② 翻前改为 raise-or-fold
  （取消 limp）：未加注底池按位置放宽开池范围（in-position base 0.30、OOP +0.06、每多一名对手
  +0.05、开池 2.5x），实测扫描 + 换 seed 验证后定档。效果（250–300 手）：对**只弃牌的 rock 从
  +4.6 跃升到约 +87 bb/100**、对 call_station 从约 +312 升到约 +837，同时仍碾压 random/maniac；
  同族 self-play 的 bb/100 离散度从约 ±90 收敛到约 ±25、排序变正常（current 最优、no_equity 最差），
  equity 混合在 self-play 中开始正向（current 反超 no_equity）。多街“收手 / 弃中等成牌”逻辑暂缓——
  参照池里唯一多街进攻者是 maniac（疯狂诈唬），对它收手反而回退，需等 CFR/GTO 级理性对手才能正向
  验证（见 `docs/ai-strength.md` S2）。
- 阶段 3 AI S1a（preflop 真值表基础设施）：新增 `holdem_ai.preflop`，把 169 个等价类（13 对子 /
  78 同花 / 78 不同花）规范化，并用本项目评估器（确定性蒙特卡洛、12000 样本/类、无外部下载）算出
  `PREFLOP_ALLIN_EQUITY`——每类对随机手的 heads-up all-in 胜率（AA 0.855…32o 0.324，与公认值吻合）；
  `hand_class()` / `preflop_equity()` / `all_in_equity_vs_random()`（同 seed 可精确复算）使其可审计、
  可再生。这是 push/fold 与 S2 CFR card bucketing 的基础数据。**刻意没有接进启发式开池决策**：实测
  显示用真实 equity 选 ~88% 开池范围相对现有公式阈值无可测增益、且有 vs-random 回退风险，按“对参照
  对手不回退”纪律暂不改动已验证的开池；真值表留待短筹码 all-in 决策（S1b）等明确正确性场景再消费。
- 阶段 3 AI S2a（CFR + exploitability 管线）：新增 `holdem_ai.cfr`，用 OpenSpiel（官方 PyPI
  `open-spiel>=1.6.15`，已是 engine 依赖）的 CFR/CFR+ 求解器与 exploitability 评估，在小博弈上验证
  `solve -> average policy -> exploitability` 闭环：kuhn CFR+ 200 iters → exploitability ~3e-4、
  leduc CFR+ → ~0.01-0.03、CFR+ 明显快于 vanilla。已从玩具游戏扩到**真正的无限注德扑抽象**：
  `nolimit_holdem_abstraction()` 生成 `universal_poker` 的 `fcpa`（fold/call/pot/allin）缩减牌堆游戏，
  CLI `uv run holdem-ai-train-cfr --game nlhe-small --variant cfr_plus` 80 iters → exploitability
  ~0.006。实测：全树 tabular CFR 可解 1 手牌缩减 NLHE（秒级），2 手牌即超时——扩真实 HUNL 需上
  MCCFR + card/bet 抽象。这是“先把求解+评估打通、再扩抽象 HUNL（S2b）”的标准路径；blueprint 落成后
  经 `engine/adapters/openspiel.py` 的离散动作映射回 `Action`，由同一个 `decide()` 消费。详见
  `docs/ai-strength.md` S2。
- 阶段 3 AI S2b（自有抽象游戏 → CFR blueprint → `decide()` 桥接）：为了把 CFR 解出的策略可靠地接进
  `decide()`，不逆向 `universal_poker` 的 infostate，而是写**自己掌控的 pyspiel 游戏**
  `holdem_ai.preflop_game`（heads-up push/fold、手牌分 8 个 equity 桶、叶子用 `preflop.BUCKET_EQUITY`
  桶间胜率矩阵）。infostate 由我定义，能从 `GameState` 精确复现。OpenSpiel CFR+ 解到 exploitability
  ~2e-5、得到阈值型 push/fold 策略；`holdem_ai.blueprint.PushFoldPolicy` 把它包成同一个
  `explain()/decide()`（短筹码翻前查 blueprint、其余回退启发式），并加入 `pushfold` profile。**这是
  第一个真正由 CFR 求解、game/bot 共用的 `decide()` 策略。** 诚实结果（10BB、400 手）：pushfold 赢
  所有参照对手（random +45 / call_station +35 / rock +27 / maniac +75），但输给启发式 current
  （-31.6）、且赢参照对手赢得比 current 少——它只 jam/fold、不做小注价值剥削，实证了「GTO 是不可剥削
  下限、对弱场剥削式打法赢更多」（`docs/ai-strength.md` §7）；唯一反例 vs maniac（pushfold +75 /
  current -25）。
- 阶段 3 AI S2c-1（外部 review 驱动的桥接/抽象硬化）：ChatGPT PRO 评审确认架构 seam 与「自有 infostate」
  方向正确，但指出当前 blueprint 是管线验证而非可信牌谱，并发现真 bug。已修：① **HU 硬闸**——
  `PushFoldPolicy` 仅在恰好 2 人局面生效，绝不把 HU push/fold 套到 6-max（release-blocker）；② **覆盖式
  jam 检测**——对手 all-in 即便我方更深也识别、用有效筹码（旧逻辑只在「跟注=自己全下」时触发，实测中
  BB 跟注表几乎从未被检验）；③ 默认按信息态种子**采样混合策略**（保留 exploitability 性质），
  `mode="pure"` 才取 argmax（标注剥削式投影）；④ `BUCKET_EQUITY` 对角线强制 0.5；⑤ **精确联合发桶**——
  `preflop_game` 改两段 chance：先按边际发 button 桶、再按去牌**条件分布**（`preflop.bucket_deal_conditional`）
  发 BB 桶，乘积等于全枚举 1326×1225 不相交组合对得到的精确联合 `BUCKET_PAIR_WEIGHTS`（`compute_bucket_pair_weights`
  可 0.06s 精确复算），消除「独立发桶」与去牌胜率矩阵的不自洽——hero 拿最强桶时 villain 同样最强桶概率
  0.1252→0.114；⑥ **矩阵去噪**——`BUCKET_EQUITY` 用 150k 样本（与发桶同一去牌采样 + 强制对称）重算，每格
  std 由 ±0.0065 降到 ±0.0013。重解后 6/10/16bb exploitability ~1e-5 量级（默认 400 iters：约 3e-5/7e-6/1e-5）（现为「一致、低噪抽象内」有意义的
  近纳什值），且短筹码越浅 jam 越宽（jam 5.75/4.0/3.0、BB call 4.26/3.0/2.0）。10bb 对照（1500 手）：pushfold
  胜 maniac **+40.5**、rock **+27.0**、启发式 **+5.0**（去噪前略负，现约打平偏赢）；一处有用观察：10bb 下启发式
  自身负于 maniac（−45.6）而 pushfold 胜 maniac，正是 S2c-3 护栏要补的短筹码漏洞。完整 S2c 计划见 `docs/ai-strength.md`。
- 阶段 3 AI S2c-4（评估硬化）：让"谁更强"的结论可信，再谈翻后。① **CRN 配对评估**
  `holdem_ai.evaluate.evaluate_match`：同一副牌换座各打一次（duplicate poker）抵消发牌方差，bb/100 配
  **bootstrap 95% 置信区间**与 button/BB 位置切片，CLI `--match FOCAL OPP --pairs N`；抽出共享
  `_play_hand`，`evaluate_heads_up` 行为不变。实证 CRN 很紧：current vs rock @10bb +69.0、CI[+66,+72.5]
  仅 300 手。② **会惩罚的对手**：加 equity 接地的 `TAGPolicy`（紧凶 reg）与 `ThreeBetJammerPolicy`（短筹码
  极化 3bet-jam），接进 `REFERENCE_PROFILE_NAMES`；CRN 证明它们有强度非送钱（10bb 下 `current` vs `tag`
  +33.2 是真赢、不是白送）。③ **多人 smoke + HU 闸收紧**：闸由 `active_players!=2` 改
  `players!=2`（按入座人数），修掉"3 人弃到 2 人剩者仍套 HU blueprint（死钱失真）"的潜在误用，加 `test_multiway`
  3 人桌端到端（每手零和、pushfold 确有行动但绝不出 blueprint）。④ 可复现网格脚本 `ai/scripts/reference_eval.py`
  → `docs/ai-reference-eval.md`（10/25/100bb，1200/800/300 对）。**头条**：10bb 启发式负 maniac（−18.8）而
  pushfold 胜 maniac（+40.8）——GTO 短筹码下限；对被动/会防守对手 `current` 反赢更多（GTO≠榨取上限）。
  后续 BB-defender / minraise / checkraise-bluffer 对手按需再加。
- 阶段 3 AI S2c-2（短筹码 preflop game v2）：`preflop_game_v2.ShortStackPreflopGame`——在 push/fold 上加
  button 的 limp/min-raise(2bb)/2.5x/jam 与 BB 的 check/fold/call/jam 应对、BB-jam-over-open 后 button 再
  fold/call；沿用 S2c-1 去牌联合发桶 + 去噪矩阵，CFR+ 默认 600 iters 解到 exploitability ~1e-4 量级（迭代更多可降到 ~1e-5），`solve_short_stack_preflop`
  抽出 per-bucket 蓝图（button_open / bb_vs_{limp,minraise,raise25,jam} / button_vs_jam）。**关键发现**：纯无翻后
  摊牌让 min-raise/2.5x 被 limp-or-jam 严格支配；加单参数 `oop_realization`(R=0.85，位置/主动权粗代理)后尺度才被
  使用（8bb 多 jam、12bb 宽 limp、16bb 出 2.5x）。`test_preflop_game_v2` 覆盖可遍历/零和、近纳什、分布归一、
  越浅越 jam、R=1 退化 vs R<1 尺度涌现。诚实边界：R 有 artifact、非真翻后(→S3)，蓝图尚未接出牌(→S2c-3)。
- 阶段 3 AI S2c-3（CFR 作启发式护栏）：用 CRN+CI harness **先证伪再落地**。给 `PushFoldPolicy` 加 `defend_only`，
  落成 `hybrid` profile（启发式出牌 + blueprint 只供不可剥削的跟注-vs-jam 下限，开池交还启发式）。**关键发现**：
  原假设"防守叠加同时拿 vs maniac/tag"被证伪——maniac 打 pot-size 而非全下，`_is_facing_jam` 少触发，pushfold
  赢 maniac 靠主动 open-jam；而 open-jam 在 5/6/8/10bb **全大胜 maniac**（+41/+37/+63/+67）却**全输 tag**
  （−12/−14/−14/−26，5bb 也不消失）=对手相关剥削、非普适增益。故 `hybrid`=只上不可剥削下限（≈current、对真 jam
  对手 + 且零成本），完整 `pushfold` 留给已知激进桌；二者并存，并**界定 S4 对手自适应的必要性**。
  `test_blueprint` 加 defend_only 行为 + CFR profile 解析。
- 当前验证：`scripts/dev/verify-dev-env.sh` 通过（CV/OCR runtime、ruff format/check、mypy、pytest，
  243 tests）。新 CLI：`uv run holdem-ai-evaluate-heads-up --match hybrid maniac --pairs 20`（CRN+CI）。
- **多 agent 对抗式 review（4 路并行，逐条跑代码验证）**：核心逻辑（v2 树/payoff/信息态隐藏/精确联合发桶/R 模型/
  CRN/竞争对手合法性/HU 闸）经独立枚举与 fuzz 验证均正确；查出并已修：① **CRN `hand_id` 含 `focal_seat`**——
  会把镜像两半喂进不同 equity/decision RNG，破坏 antithetic 不变性（对 equity 型策略），去掉后镜像精确相消
  （加 tag/maniac 不变性测试）；② **`position or name` 回退**——名叫 "dealer"/"d" 的座位被误判为庄位、覆盖
  OOP 安全默认，去掉 name 回退；③ v2 蓝图抽取在 stack≤2.5bb 对非法 min-raise/2.5x 上下文 KeyError——按合法性
  门控、非法上下文填惰性 fold 默认；④ underwater 角落（stack<to_call+raise 钮）会出低于跟注的非法加注——无合法
  加注时丢弃 RAISE；⑤ v2 `max_game_length` 5→3（OpenSpiel 不计 chance）；⑥ 对手未知筹码会 crash 而非 fail-closed
  ——加 `missing_seat_stack` 闸；⑦ exploitability 文案夸大已据实修正（push/fold ~1e-5 量级、v2 默认 ~1e-4）。
- 阶段 3 AI S4-lite（对手自适应，离线最高杠杆）：S2c-3 界定的"open-jam 是对手相关剥削、需对手建模"在此
  交付第一块。`holdem_ai.adaptive.AdaptivePolicy`（profile `adaptive`）——纯路由器：跨手累积"面对激进度"频率
  （只从自己行动时的 `GameState` 重建——翻前 `villain.committed>bb` 算加注、翻后任意 `to_call>0` 即对手下注；
  `last_aggressor` 全仓库未赋值故弃用），暖机 20 次决策后频率≥0.55 判 maniac → 切完整 `pushfold`（open-jam），
  否则恒走安全 `hybrid`。**CRN 指纹干净**（maniac 0.68 / random 0.45 / current 0.28 / tag 0.21 / rock·station 0），
  0.55 阈值落 random 与 maniac 间。**fork 实测两头通吃**（seed 20260627，300 对，5/6/8/10bb）：对 tag 与 `current`
  **逐位完全相同**（+33/+35/+35/+36，频率永不过阈→恒 hybrid，零成本继承启发式价值）；对 maniac 把启发式短筹码
  大漏洞（−56/−46/−50/−28，`hybrid`=`current` 证实 call-floor 无济于事）收回到 ≈`pushfold`（−20/−18/+13/+18）。
  静态二选一做不到此双赢；只新增"对手读数"一处状态、`reset()` 换对手清零，可直接挂 bot 同一 `ai.decide`。诚实修正：
  S2c-3 旧记"open-jam vs tag −12..−26"是更早/pure 模式更噪估计，本次更紧的 CRN-300 mixed 下成本其实很小（近乎打平），
  但 adaptive 双赢不依赖该值（靠 tag 侧按构造等同 current + maniac 侧漏洞大而符号稳健）；网格详见 `docs/ai-strength.md`
  §6 S4-lite。验证：`verify-dev-env.sh` 全绿（ruff/mypy/**256 tests**，新增 `ai/tests/test_adaptive.py` 13 例）；
  新 CLI 路径 `uv run holdem-ai-evaluate-heads-up --match adaptive maniac --pairs 300`。下一步：把同一对手读数接进
  bot 侧（`bot/` 的 6-max 适配里按席位累积激进度），并在 host dry-run 验证多人桌的读数稳定性。
- 阶段 3 AI S4 弱场剥削（6-max，Phase 1–2：对站薄价值 + 对 nit 多弃）：真正能赢 Steam 弱场的杠杆。三件套：① **6-max
  CRN 评估器** `evaluate_field`（每副牌焦点轮转坐遍 6 座位，抵消发牌运 + bootstrap CI；CLI `--field FOCAL OPP --decks N`）；
  ② **per-seat `OpponentModel`**（`holdem_ai.opponents`）——只从自己行动快照重建每座位 VPIP/PFR，**只取翻前**（避开
  `committed` 整手累计在翻后把跟注伪装成加注），PFR 用"唯一最高投入且 >bb"干净排除跟注污染；③ **`FieldExploitPolicy`**
  （`holdem_ai.field`，profile `field_exploit`）——按"当前相关对手+上下文"切两套：读到**跟注站在场**→薄价值/不诈唬/大尺寸；
  面对**被判 nit 的下注**→多弃（紧手下注=价值；**nit 门用低 VPIP 判定，天然排除高 VPIP 的 maniac → 绝不对诈唬者弃牌**）。
  诚实大背景：`current` 对纯弱场本就海赢（vs 5×call_station +875、rock +76、maniac +1120）却输 5×tag（−228），故剥削是
  增量、以同副牌"剥削−current"差度量。**实测（CRN，seed 20260627，200 副牌）**：field_exploit 对 call_station +691
  [CI 不重叠]、**把 tag 漏洞 −228 翻成 +62（delta +290）**、对真实鱼 `loose_passive` +165；对 rock/maniac **逐字节相同
  （delta 0，零误伤）**。**maniac 无专用配置**——实测"更轻跟到底" delta −448（100bb 深码 current 已 value-own maniac，
  再轻跟反流血）**测试后明确砍掉**。新增 `loose_passive` 参照（松被动鱼，会下注成手牌，被判 STATION）。诚实标注：observed
  VPIP 被观察时序低估（站理论~0.85 实测~0.50）但相对分离干净；**端到端 nit/鱼增益高方差**（需 100+ 副牌才稳定，故门**逻辑**
  由瞬时单元测试钉死、**增益**由文档大样本背书，不进快测；站剥削效应巨大稳健保留为集成测试）。验证：`verify-dev-env.sh`
  全绿（ruff/mypy/**277 tests**）；网格详见 `docs/ai-strength.md` §6 S4。
- 阶段 3 AI public API 接入 S4 弱场剥削（2026-06-29）：`holdem_ai.decide()` / `explain_decision()` 默认
  从纯 `HeuristicPolicy` 升级为 **seat-scoped `FieldExploitPolicy`**（每个受控 seat 一份 `OpponentModel`，避免本地
  多 AI 对局互相污染读数；新增 `reset_decision_policy()` 供新 session/测试清零）。`profile_from_name("current")`
  仍保留纯启发式，方便做旧基线评估。验收对打（CRN，seed 20260627，6-max，200 副牌=1200 手，100bb，对
  5×`call_station`）：`current` +875.3 bb/100 [597.5, 1207.2]；`field_exploit` **+1566.2** [1154.7, 2044.5]；
  delta **+690.8 bb/100**，且 value-bet 由 46 次增至 140 次，符合“跟注站薄价值”的预期。
- 阶段 4 bot 接入对手读数（S4 ②）：把 `FieldExploitPolicy`（持久 `OpponentModel`）作为 Poker Legends host dry-run
  的 `policy_explainer` 注入 orchestrator——orchestrator 跨帧持有策略对象，故连续会话里每座位读数会累积；单帧 dry-run
  时读数恒 UNKNOWN→回退 base，决策与裸启发式**逐字节相同**（安全）。dry-run 输出新增 `opponent_reads`（每座位
  profile/VPIP/PFR/hands）+ `policy_decision.metadata.exploit`/`opponent_profiles`。新增 **`replay_dry_run_main`**
  （CLI `holdem-bot-replay-poker-legends-dry-run`）：用**同一持久 policy** 顺序重放一串保存帧 → 读数累积，离线验证
  "感知→读数"管道（`--use-truth` 隔离 CV 质量）。验证：bot 19 测试通过（含
  `test_orchestrator_with_field_exploit_accumulates_opponent_read` 钉死跨帧累积+分类）。**离线发现**：现有
  `session_001_selection` 20 帧即便 `--use-truth` 也全 `non_table_ui`（本就是菜单/弹窗负样本，无 actionable state），
  故读数在这批帧上不填充——需 host 实拍 actionable 帧才会累积（指南 `docs/bot-host-dryrun.md` Step 3）。
- 阶段 4 捕获/迭代策略调研（官方源核实，3 路并行）：手动「截屏→存→跑→贴 JSON」迭代太慢，调研加速工具。
  结论：**① Airtest/Poco 跳过**——Airtest 桌面端只有 Windows 无 macOS 后端；Poco 必须把 `poco-sdk` 编进游戏
  （第三方 Steam 改不了），其无 SDK 的 Windows UIA 驱动也看不见 Unity/DirectX 画进 GPU 的牌桌；AirtestIDE 闭源停更、
  无实时识别叠加。**② 抓屏**：GDI 系（mss/Pillow/pywin32）对 GPU/全屏出黑帧、但 2D 窗口扑克 OK；Windows 最佳
  `windows-capture`(WGC,MIT，单窗口/跟随/抓遮挡)、`dxcam`(2026.3 复活) 兜底；macOS 用 `mss`(MIT 跨平台) 起步、
  Quartz 已被 SDK15 废弃。**③ OBS 虚拟摄像头跳过**（输出缩放+黑边 canvas，与像素标定打架）。**建议接入点击时把
  host 切 Windows**（无每月录屏授权 + Per-Monitor-v2 真像素 + 库现役）；只读 HUD 现在先在 Mac 跑（mss 跨平台）。
- 阶段 4 感知 HUD（加速迭代的工具，自研薄壳不引 Airtest）：新增 `holdem-bot-watch-poker-legends`（`watch_main`）+
  纯渲染 `bot/vision/perception_overlay.py`（numpy/cv2，无 GUI/无游戏依赖，可无头测）。把「机器人看到了什么」实时
  叠加回画面：布局 ROI（按 base→帧缩放，misalign 一眼可见）+ 文字面板（screen kind、安全门判定、**`state_block_reason`**、
  pot/to_call/legal、策略意图动作、每座位读数）。两路：`--image` 单帧无头渲染叠加 PNG（测试+寄证据用）；live `mss`
  循环 `cv2.imshow` ~4fps（`s` 存 frame+overlay+json，`q` 退）。**只读、无点击路径**。忠实复用 `evaluate_safety`
  安全门（与 orchestrator 一致）但保留完整 `RecognitionResult.metadata` 以显示卡因。新增 `mss>=10.2` 依赖（官方 PyPI，
  带 py.typed）；host 脚本加 `watch`/`watch-once` 子命令。验证：ruff/mypy/**287 tests**（新增
  `test_perception_overlay.py` 6 例 + `test_poker_legends_watch.py` 3 例）。**离线发现**：`watch-once 000080` 是**真牌桌**
  （hero 5h8s、Call/Raise/Fold 可见）却仍 `no_game_state` + `state_block_reason: missing_table_metadata`——selection 集
  标注是 `roi_applied_values_pending`（只放 ROI 未填值），即真牌桌也得布局带 table-metadata 块才能拼出 `GameState`。
- 阶段 4 本地 VLM 评测（LM Studio）：新增/扩展 `holdem-bot-evaluate-local-vlm`
  （`bot/src/holdem_bot/eval/local_vlm.py`），可用 LM Studio/OpenAI-compatible 后端跑同一运行时 prompt，并与
  Gemini/CV/ reviewed truth 对齐评估；新增 `--reference-dir` 直接用 reviewed truth overlay 作字段基准（不把
  Gemini 当真值），新增 `--timeout` 防止本地模型啰嗦卡死，并修复在 reviewed-truth 模式下 Gemini 也参与字段评分。
  当前 `docs/local-vlm-eval.md` 结论：13 张完整 hero 决策帧上 Gemini `gemini-3.1-flash-lite` 解析 13/13、
  box 约 11px、均延迟 7.8s；Qwen3-VL-30B 解析 12/13、box 约 69px、字段相对 Gemini 约 69%、均延迟 13.1s。
  追加 reviewed-truth 小样本（session_002 3 张 actionable）：Gemini 字段一致率 62%、均 6.5s；Qwen3-VL-30B
  38%、均 17.6s；Qwen3-VL-8B 19%、均 15.1s。结论：**Gemini 仍是整屏主读者；30B 是本地高准确率备选；
  8B 只适合 smoke/fallback 或后续 ROI crop 小任务；点击坐标继续以 CV 精修为准**。
- 阶段 4 Poker Legends 数字 OCR hardening v2（字段语义先保守）：`PokerLegendsNumberPrediction`
  现在同时输出 `base_number` / `overlay_number` / `total_number`，把 `$334+10` 这类叠加筹码文本拆成可审计
  组件；`normalized_number` 保持兼容语义（总额），但 evaluator 额外统计 component-level truth comparison。
  为避免把“stack 展示 + 已下注/叠加标记”误当可用 stack，image-only accepted critical path 暂时拒绝
  未经规则验证的 stack overlay；`stack overlay validator v1` 已接入 accepted-number gate：目前只放行
  hero `base+overlay`，且必须满足 `base+overlay == total == normalized_number`、reviewed hero seat 的
  `committed == overlay`、若 seat stack 已知则必须等于 total。opponent overlay 因缺少稳定 seat-to-ROI 映射仍
  fail-closed。`hero_current_bet` ROI v1 已加入 canonical layout（手牌上方当前投入区域），旧 layout JSON
  缺该 region 时 number recognizer 会按图像尺寸补齐 canonical ROI；image-only validator 可用同帧
  `hero_current_bet == overlay` 验证 hero stack overlay。
  最新 image-only replay（119 帧含 non-actionable，输出
  `/tmp/poker-legends-table-eval-image-only-hero-current-bet-roi-v1`）：authorization 0、unsafe 0、stale 0、
  accepted-critical-wrong 0；accepted 数字从上一版的 hero_stack 9 / right_top_stack 2 收紧到
  hero_stack 6 / right_top_stack 1（pot 8，hero_current_bet 15），其中右上 stack 仍有 1 个 mismatch，
  继续 fail-closed。
  number readiness 现在直接使用 recognizer 的 rejection reason：多数 overlay blocker 其实是低置信
  （hero 32 / opponent 31），真正高置信但缺 committed/current-bet 证据的是 hero 3 / opponent 1，另有 opponent OCR
  missing 8。
  truth-assisted 对照（`/tmp/poker-legends-table-eval-truth-assisted-hero-current-bet-roi-v1`）：authorization 39、
  unsafe/stale/accepted-critical-wrong 均为 0。结论：字段语义路线正确，但下一步不能单纯提 coverage，
  image-only 要继续推进应先处理剩余低置信 stack OCR/ROI，opponent overlay 要等稳定 seat-to-ROI 映射。
- 阶段 4 Poker Legends stack OCR hardening v3（只保留无错收益）：数字 confidence 现在把干净的
  `base+overlay` token 扩展到千分位/后缀形式，`$1,005 +10`、`$1,146+80` 这类 stack OCR 不再因
  `,` 被降到 0.65，而是按 clean split stack 给 0.82；`+110 4`、多 `+`、`$` 前混入数字/字母等碎片
  仍保持低置信。当前 image-only replay（119 帧含 non-actionable，
  `/tmp/poker-legends-table-eval-image-only-stack-clean-plus-final-v1`）：authorization 0、unsafe 0、stale 0、
  screen false actionable 0、source-policy violation 0、accepted-critical-wrong 0；accepted hero_stack
  6 -> 7 且 7/7 exact match（pot 8、hero_current_bet 15、right_top_stack 1，右上 stack 的 1 个 mismatch
  仍不进入授权链路）。对比实验结论：放宽 `1$990+10` / `28 $990+10` 这类 noisy-prefix stack 会把
  `session_002__keyframe_000131` 的 `$890` 错收为 `900`（相邻 current-bet/筹码区域污染），因此已放弃；
  简单收紧全局 ROI/pad 也在抽样上出现 accepted mismatch，不作为安全修复。剩余 blocker 主要是
  低置信/污染 stack ROI：hero low-confidence 29、hero unverified overlay 5、opponent low-confidence 29、
  opponent missing OCR 8、opponent unverified overlay 3。下一步应做字段专用 ROI/字符级 OCR 或更稳定的
  seat-to-ROI 映射，而不是降低阈值或信任 noisy-prefix。
- 阶段 4 Poker Legends stack OCR hardening v4（字段语义 + ROI 证据 + safety report 前置）：`*_stack`
  OCR 的 accepted `normalized_number` 现在按可用 stack 的 `base_number` 语义输出，`overlay_number` /
  `total_number` 只作为组件证据；evaluator 对历史 truth 中 base/total 混标的 stack overlay 做
  stack-aware comparison，同时继续保留 component-level base/total mismatch 诊断。`hero_stack` 增加
  `crop_variant` / `roi_rect` 证据字段，并只在默认 OCR 低置信且 raw 显示边缘污染时懒触发
  `hero_stack_no_pad` 或 `hero_stack_trim_right_16`，避免每帧无条件多跑 Tesseract。`right_top_stack`
  到最小 opponent seat 的合成改为显式 seat-to-ROI 映射（`right_top_stack -> seat 1 / ui_slot=right_top`，
  controlled seat 冲突时才回退下一个空 seat），opponent overlay 仍 fail-closed。Evaluator 新增
  `negative_safety_tag_counts` / `negative_safety_by_tag` 与 `temporal_tracker_*_counts`，为 hard negative
  与后续 tracker 接入提供固定报告口径。当前 image-only replay（119 帧含 non-actionable，
  `/tmp/poker-legends-table-eval-image-only-stack-field-roi-final-v1`）：authorization 0、unsafe 0、stale 0、
  screen false actionable 0、source-policy violation 0、accepted-critical-wrong 0；accepted hero_stack 7 -> 10
  且 stack-aware accepted match 10/10，raw hero_stack match 36/41；负样本统计覆盖 62 个 non-actionable
  frames（blocked_overlay 17、table_observe 45，含 preselect/shortcut 2、modal/menu 5）。剩余 blocker：
  hero low-confidence 6、hero unverified overlay 25、opponent low-confidence 29、opponent missing OCR 8、
  opponent unverified overlay 3；下一步应继续做真正字符级 OCR/stack ROI 数据集，而不是扩大 opponent
  accepted path。
- 阶段 4 Poker Legends number crop dataset v1（字符级 OCR 前置数据层）：新增离线
  `build_poker_legends_number_crop_dataset` / `holdem-bot-build-poker-legends-number-crops`，从 layout
  annotations + reviewed truth 导出数字 ROI crops、truth canonical text / tokens、chip number labels、
  screen kind、blocking reason、ROI rect、pad 与 crop variant；不进入 live accepted path。为 stack 字段
  固定导出多变体：`hero_stack` 的 default / no-pad / trim-right-16，`right_top_stack` 的 default /
  no-pad / trim-left-16；同时修正 number OCR report/dataset 的 frame_id 解析，merged dataset 里优先用
  annotation 文件 stem，避免 source-prefixed truth 对不齐。`hero_current_bet` crop label 可从 reviewed hero
  seat `committed` 派生，`hero_stack` 可从 reviewed hero seat `stack` 派生。当前真实数据集输出
  `/tmp/poker-legends-number-crop-dataset-v3`：119 帧、1071 crops、584 labeled crops；标签分布为
  `hero_stack` 336、`right_top_stack` 144、`pot` 62、`hero_current_bet` 30、`primary_left` 12。下一步可以
  基于该 manifest 做字符/CTC OCR baseline 或人工 review queue，不应直接把 unlabeled / opponent overlay
  crop 放进授权链路。
- 阶段 4 Poker Legends number crop OCR baseline v1（离线评估，不接 live）：新增
  `evaluate_poker_legends_number_crop_dataset` / `holdem-bot-evaluate-poker-legends-number-crops`，读取
  `number_crop_dataset_manifest.json` 后按 field / crop variant 输出 raw accuracy、accepted precision 与
  accepted-wrong 明细；支持 `--max-crops`，避免 1071 crops 全量 Tesseract 慢跑被误用作日常检查。同时收紧
  crop label 语义：`*_stack` direct text 的 `base+overlay` 统一标为 base/available stack；缺 direct text
  的 hero stack 只有 reviewed `committed == 0` 时才从 seat.stack 派生，否则保持 unlabeled，避免旧
  total/base 混标污染训练。当前真实数据集 v5 为 119 帧、1071 crops、575 labeled crops（`hero_stack`
  327、`right_top_stack` 144、`pot` 62、`hero_current_bet` 30、`primary_left` 12）。120-crop smoke
  baseline（`/tmp/poker-legends-number-crop-ocr-baseline-sample-v3`）显示现有 Tesseract crop OCR 仍不可接入：
  labeled 43、raw accuracy 0.558、accepted precision 0.710、accepted wrong 9；后续应先做 accepted-wrong
  review queue / 字符模板或小模型 OCR，而不是放宽 runtime gate。
- 阶段 4 Poker Legends number OCR review queue v1：crop OCR evaluator 现在额外输出
  `number_crop_ocr_review_queue.json` / `.md`，按 `accepted_wrong`、`missing_labeled`、
  `mismatch_labeled`、`accepted_unlabeled` 分类并携带 crop path / raw OCR / confidence / expected /
  observed，供人工复核和主动学习选样；stdout 同步报告 queue 数量。当前 120-crop smoke
  `/tmp/poker-legends-number-crop-ocr-review-queue-sample-v1` 生成 41 条 review rows：accepted_wrong 9、
  missing_labeled 2、mismatch_labeled 8、accepted_unlabeled 22。结论：短期最有价值的是复核 accepted_wrong
  和 accepted_unlabeled 中的 right_top/按钮误读，继续避免把 Tesseract accepted 直接接进授权链路。
- 阶段 4 Poker Legends number crop accepted policy v1：crop OCR evaluator 的 accepted 口径改为字段感知，
  stack crop 必须通过 `_is_safe_stack_variant` 才计入 accepted；同时把无 `$` 的 `5900+100m` /
  `6780+1559` 等 plus OCR 明确降置信，避免把左缘/右缘污染当高置信 stack。120-crop smoke
  `/tmp/poker-legends-number-crop-ocr-field-aware-sample-v1`：raw accuracy 仍为 0.558，但 accepted
  precision 从 0.710 提升到 0.786，accepted wrong 从 9 降到 6，review queue 40。剩余 high-priority
  blocker 集中在 `hero_current_bet` 小数字误读和 `$0+30` 这类真值/ROI歧义；结论仍是不接 runtime，
  但 review queue 已更接近真实安全风险。
- 阶段 4 Poker Legends hero current-bet confidence hardening v1：数字 OCR confidence 现在对
  `hero_current_bet` 做字段感知降置信：无 `$` 的单个小数字或 `.7` 这类 punctuation-leading OCR 不再按
  0.90 accepted，而是压到 0.65；带 `$` 的小额（如 `$5`）不受影响。120-crop smoke
  `/tmp/poker-legends-number-crop-ocr-current-bet-sample-v1`：raw accuracy 仍为 0.558，但 accepted precision
  从 0.786 提升到 0.880，accepted wrong 从 6 降到 3；剩余 accepted wrong 全集中在同一帧
  `hero_stack=$0+30` 的三个 crop variant，适合人工复核 ROI/truth，而不是继续靠 Tesseract 规则硬猜。
- 阶段 4 Poker Legends number crop source triage v1：人工复核确认旧 review queue 中
  `accepted_wrong` 的前三行实际都是 `$0+30`，不是 truth 给出的 `1229`；`hero_current_bet` 两个
  missing crop 实际落在手牌局部、看不到数字；`buttons.primary_left` accepted_unlabeled 多为 check
  图标/文字或背景，不应作为 numeric OCR 默认样本。因此 number crop dataset 默认只导出
  `pot` / `hero_stack` / `right_top_stack` 文本 ROI，`hero_current_bet` 和 button amount 必须显式
  `--text-name` / `--button-name` 才会导出；未 human-reviewed 且 hero 有 committed、但
  `texts.hero_stack.value` 没有 `+` overlay 的 truth 不再作为 crop 硬标签。新真实数据集
  `/tmp/poker-legends-number-crop-dataset-v6`：119 帧、833 crops、479 labeled crops，字段分布为
  `hero_stack` 357、`right_top_stack` 357、`pot` 119。120-crop smoke
  `/tmp/poker-legends-number-crop-ocr-triaged-sample-v1`：labeled 30、raw accuracy 0.700、
  accepted labeled 19、accepted precision 1.000、accepted wrong 0、review queue 37
  （accepted_unlabeled 28、mismatch_labeled 9）。结论：当前 Tesseract 仍只适合作离线 baseline；
  下一步应针对 stack overlay 样本补 human-reviewed 标签/专门字符 OCR，而不是把 accepted_unlabeled
  接入 runtime。
- 阶段 4 Poker Legends number crop review refinement v1：第二轮人工复核确认 `mismatch_labeled`
  的 ROI 本身可用，错误主要来自 Tesseract raw OCR（例如 `$1000` 被读成带额外前后缀/`m` 后缀）；
  `accepted_unlabeled` 中一批 right-top stack 变体左侧裁掉 `$`，不应继续作为 accepted 样本。
  因此 dataset 不再导出 `right_top_stack_trim_left_16`，stack overlay OCR 若缺 `$` 也不再通过
  `_is_safe_stack_variant`。新真实数据集 `/tmp/poker-legends-number-crop-dataset-v7`：119 帧、
  714 crops、431 labeled crops，字段分布为 `hero_stack` 357、`right_top_stack` 238、`pot` 119。
  120-crop smoke `/tmp/poker-legends-number-crop-ocr-review-refined-sample-v1`：labeled 30、
  raw accuracy 0.700、accepted labeled 19、accepted precision 1.000、accepted wrong 0、review queue
  34（accepted_unlabeled 25、mismatch_labeled 9）。结论不变：ROI 层已能筛掉明显不安全变体；
  剩余 `mismatch_labeled` 是 OCR 能力问题，应转向专门字符/template OCR，而不是继续放宽 Tesseract。
- 阶段 4 Poker Legends number char recognizer prototype v1：新增离线
  `holdem-bot-evaluate-poker-legends-number-chars`，从 `number_crop_dataset_manifest.json` 构建
  `hero_stack` 字符分割样本，并在同一批 test crops 上比较 template、OpenCV MLP 与 Tesseract；
  当前环境没有 PyTorch/TensorFlow，因此真 CNN 暂未运行，报告中明确标为 `not_run`。在
  `/tmp/poker-legends-number-char-prototype-v2`（真实 v7 manifest，273 行 hero_stack、216 train /
  57 test、1034 glyph samples）上：template raw exact 24/57，保守 accepted 24/57、accepted
  precision 1.000、accepted wrong 0、accepted coverage 0.421；Tesseract exact 22/57，但若全部接受则
  accepted wrong 35、accepted precision 0.386；OpenCV MLP exact 4/57 且 0 accepted，不值得继续。
  结论：字符 template 路线已经证明比 Tesseract 更适合作 fail-closed 高精度数字信号；下一步应
  改善分割覆盖，并决定是否引入官方 PyPI 的 PyTorch 做真正 CNN，而不是继续 OpenCV MLP。
- 阶段 4 Poker Legends number char recognizer prototype v2：按人工复核结论改为识别
  `hero_stack` 的白色 base/available stack 文本，忽略 cyan overlay/current-bet，避免 `$1000`
  被 `$0+30` / `$840+160` 这类叠加区域污染；同时引入 PyTorch `torch==2.12.1`（官方 PyPI，
  lockfile 固定）实现离线小型 CNN，并新增 `template_cnn` 共识指标。真实 v7 manifest 仍为
  273 行 hero_stack、216 train / 57 test：字符分割 match 从 v1 的 153/273（0.560）提升到
  242/273（0.886），测试集 match 54/57。最终报告
  `artifacts/poker-legends-videos/number_char_cnn_v2/number_char_recognizer_report.md`：
  Tesseract exact 5/57、accepted wrong 52；template exact 48/57、accepted coverage 54/57，
  但 accepted wrong 6（不能单独作为安全信号）；CNN exact 51/57、accepted coverage 51/57、
  accepted precision 1.000、accepted wrong 0；template+CNN 共识 accepted 48/57、precision 1.000、
  accepted wrong 0。结论：Tesseract 继续只作弱 baseline，OpenCV MLP 放弃；后续若要接 runtime，
  应优先让 CNN 或 template+CNN 共识进入 observation-only/contract 评估，且必须继续以
  accepted-critical-wrong=0 和更大的 hard negative set 为门槛，不因模板 coverage 较高而单独放行。
- 阶段 4 Poker Legends number char recognizer prototype v3（stack component split）：根据人工复核
  反馈，v2 的“只识别白色 base”会系统性漏掉 `+105` 这类 cyan overlay，因此离线评估改为同一 crop
  同时输出三个目标：`base`（白色可用筹码）、`overlay`（cyan 加号/current-bet）、`display`（完整显示）。
  HTML 也改为每行一张 crop，分 base/overlay/display 三块展示 CNN、template+CNN、template、Tesseract
  与 segmentation，便于继续人工 review。当前报告在
  `artifacts/poker-legends-videos/number_char_components_v2/number_char_recognizer_report.md`：
  base 273 行/57 test，seg 0.886，CNN exact 54/57、accepted 51/57、wrong 0；
  overlay 153 行/24 test，seg 0.895，CNN/template/template+CNN exact 22/24、accepted 22/24、wrong 0；
  display 273 行/57 test，seg 0.601，CNN/template/template+CNN exact 27/57、accepted 27/57、wrong 0。
  结论：component split 解决了“加号后数字完全没评估”的问题；完整 display 的主要瓶颈是分割覆盖，
  runtime 前仍应优先使用结构化 base+overlay observation，再由规则/seat committed 校验组合语义。
- 阶段 4 Poker Legends number char recognizer prototype v4（CRNN/Transformer CTC 技术选型）：按
  “绕过字符分割”的思路，新增两个显式 opt-in 离线序列 OCR baseline：`crnn_ctc` 与
  `transformer_ctc`，都吃整段 target mask（32x160）并用 CTC 输出可变长字符串；默认不跑，避免日常
  报告被序列训练拖慢。快速试验命令为 `--enable-ctc --disable-tesseract --crnn-epochs 12
  --transformer-epochs 12`，报告在
  `artifacts/poker-legends-videos/number_char_sequence_v1/number_char_recognizer_report.md`。结果不佳：
  CRNN 在 base/display/overlay 全部 blank decode（0 accepted）；Transformer base/display raw exact
  3/57、overlay 0/24，且 0 accepted。结论：CRNN/Transformer CTC 技术路线可以继续作为后续研究项，
  但当前轻量实现和小数据规模下明显不如 segmentation+CNN/template；近期不应替代现有 component
  CNN/template pipeline，除非先完成数据清洗、mask/box 可视化和更系统的序列模型调参。
- 历史提交：`ea3dace`（dev container + AGENTS.md）、`086682e`（scripts/dev）、`7eb9f48`、
  `0a79c5c`、`c93b1bd`、`d90410a`、`f6090e8`、`560386d`、`d903102`、`380f477`、
  `d20669b`、`cabe333`、`b40eef9`、`ba3befd`、`718cf1e`。尚未 push。

**下一步：以绝对基线为准绳，按业界既有研究路线推进 AI（启发式只作 bootstrap）**
- 评估方法已升级：除同族 self-play matrix 外，新增 random/call_station/rock/maniac 绝对参照对手；
  之后所有启发式改动都以“对参照对手的 bb/100 是否回退”为准绳，避免同族调参出现的非传递 RPS 假象。
- 先小步修启发式明显漏洞（可测、对参照对手不回退）：翻前按钮位只 limp 不加注、几乎不偷盲；价值
  下注金额触顶时回落 min-bet 的 sizing bug；缺乏对多街进攻的收手。目标只是把 baseline 打磨到“像
  正常玩家”，不追求 GTO。
- 真实强度走业界既有路线、不从零造核心算法（见下“参考实现 / 论文”）：preflop Nash push/fold 与
  开池范围（已基本解出、有公开图表）→ 基于 OpenSpiel 在抽象 HUNL 上跑 CFR/CFR+/MCCFR，用 OpenSpiel
  best-response/exploitability 评估 → 视需要再上 Deep CFR。engine 已有 OpenSpiel/RLCard adapter，
  PokerKit 管规则，求解器复用，不自研核心求解算法。
- 本地游戏基础 UX 已闭合（连续试玩 / 自定义下注 / 暂停 AI / 行动日志 / 摊牌摘要 / session 统计 /
  一人对 AI 入口 + profile 选择）。Poker Legends host dry-run 继续暂停，不进入真实点击测试。

**参考实现 / 论文（避免走偏，直接对标）**
- **完整的 CFR / 求解器“上强度”路线 + 论文 + 官方开源 + 分阶段计划见 `docs/ai-strength.md`**
  （Steam 上要真正赢过其他玩家，后续这些算法必须上；此处只列摘要）。
- 里程碑：HU Limit 已被“解”（Cepheus，Bowling 等，Science 2015，CFR+）；HUNL 被 DeepStack（Alberta,
  2017）与 Libratus（CMU，Brown & Sandholm, 2017）攻克；6-max 由 Pluribus（2019）攻克。
- 算法主线：CFR（Zinkevich 2007）→ CFR+（Tammelin 2014）→ MCCFR（Lanctot 2009）→ Deep CFR
  （Brown 2019）→ ReBeL（Brown 2020，统一 RL+search）。
- 开源直接可用：OpenSpiel（DeepMind，含 CFR 全家桶 + `universal_poker` + exploitability/
  best-response）、RLCard（含 NLHE / CFR / Deep CFR / NFSP）、PokerRL（Deep CFR / SD-CFR 参考实现）。
- 基准对手：Slumbot（公开 API 可直接对战）、历年 ACPC agent；preflop 用公开 GTO/Nash 图表对照 range。
- 结论：我们的分阶段路线（启发式 → preflop GTO → CFR → 自博弈，exploitability 评估）与业界主线一致，
  方向没走偏。关键纪律是“不长期手调启发式”，尽早切到 CFR 并以 exploitability / 参照对手量化强度。

**Poker Legends host dry-run 后续暂停项**
- 感知重构完整设计已单独文档化：`docs/poker-legends-perception-redesign.md`。后续实现先按
  observation-first + contract-first 路线切分：先做 recognition mode / source policy / evidence refs，
  再做 `VisualObservation` / `GameStateAssemblyResult` / safety contract / 分层 evaluator，之后接
  temporal tracker；不要继续把整屏 VLM JSON 当主感知链路。
- 保持 ScreenState v0 作为最外层安全闸门；继续用 reviewed truth overlay（session_001 v1 /
  session_002 v5 / session_002 auto review v2）评估，不让可疑帧进入 `ai.decide()`。
- 继续在更多样本上验证右下买入提示、左侧活动栏、中心弹窗等 blocked overlay 信号；安全口径仍是
  可疑界面先停手。
- 把 session timeline tracker 接到更密的关键帧识别输出上，用连续上下文稳定 hand boundary、
  overlay pause/resume 与 showdown/winner 展示，而不是只依赖单帧判断。
- 后续扩大 truth 时优先使用 dense scan 选出的 `auto_review_selection_v2` 这类候选集，不把整段视频
  直接送 LLM；底部圆形 action strip 已确认为预选/快捷操作，不进入安全可点击判定。
- 按钮 truth 已规则化：主按钮中间/右侧优先按固定位置映射，不直接吸收 LLM 的 `other` /
  `all_in` / `cancel` action_type；左侧继续只区分 `check` / `call`，不确定则 needs_review。
- 扩展牌面识别：当前可用策略是 `actionable_table` 上 full-card template 优先，失败后要求
  rank/suit part 与 classifier consensus；该路径已接入 Poker Legends `RecognizedTable` /
  prototype `GameState`。seat/action fallback v1 已把 57 个 actionable truth 帧中的 47 个转成
  prototype state；剩余 10 帧继续 fail-closed。下一步只针对残余 blocker 做小步治理：
  `missing_legal_actions` 3、`not_enough_players` 3、`missing_pot` 2、`preselect_ambiguous` 1、
  `hero_not_current` 1。
- **6-max 位置/盲注保真度（已修，AI 质量缺口）**：原型 `GameState` 旧逻辑把 `button_seat` 硬编码成 hero、
  盲注写死 config——等于让启发式**永远以为自己在按钮位**，在真实 6-max 会**乱开一堆烂牌**。已：① `RecognizedSeat`
  加 `position`，从 truth 标注 thread 进来；② `_resolve_button_seat` 识别庄位，识别不到则**默认 hero 处于 OOP**
  （宁紧勿松，而非旧的"永远按钮"），来源记入 metadata；③ 盲注从 `table_state.small_blind/big_blind` 读取、否则
  config，并驱动 legal raise 下限与 `min_raise`；④ **修 raise 下限 bug**——`_legal_actions` 改两遍扫描：旧逻辑
  `min_amount=committed+big_blind` **忽略 to_call**（committed=0/to_call=20/bb=10 时算出 10，**低于跟注额=非法
  加注**），且依赖按钮顺序；现在先定 to_call、再按 `committed+to_call+big_blind` 定合法最小 raise-to。实测影响真实：
  **18 手垃圾牌（94o/72s/54o/32s…）从 IP 开池翻转为 OOP 弃牌**。`test_poker_legends_table_recognizer` 加位置识别/
  OOP 默认/盲注 thread/raise 下限 四测。
- 按钮识别 v0 已覆盖 `check/call/raise/fold` 三主按钮；后续只有在需要快捷下注额时再处理
  raise shortcut，不把弹窗 confirm/cancel 映射为扑克动作。
- 继续校准筹码/底池专用数字 OCR：只在 ScreenState 为可行动牌桌或观察牌桌时读 pot/stack/commit，
  低置信数字只记录不使用；overlay 后面的数字字段继续标为 ignored。
- 继续收紧 Poker Legends `RecognizedTable` → `GameState` 原型：对当前 fail-closed actionable
  帧逐一归类；能用 reviewed truth 明确修正的继续收敛，无法确认的保留停手。只有当 screen、牌面、
  按钮、筹码都超过阈值时才允许把 `state` 交给安全闸门，否则继续停在 `no_game_state` /
  `low_confidence`。
- 阶段 4 Poker Legends action-row hardening v1：残余 blocker 中的
  `session_002__keyframe_000334` 原先表现为 `missing_call_amount`，但其主按钮 label 是 `Call Any`，
  属于预选/快捷语义，不能当作当前行动 call。`PokerLegendsTableRecognizer` 现在把这类
  `Call Any` / `Check/Fold` / `Fold to ...` label 显式拦为 `preselect_ambiguous`
  (`PRESELECT_AMBIGUOUS` issue)，继续 fail-closed，不再把问题误归因成 OCR 缺金额。局部复现：
  该帧 `state=False`、`screen=actionable_table`、block=`preselect_ambiguous`。
- 阶段 4 Poker Legends table recognizer evaluator v1：新增
  `uv run holdem-bot-evaluate-poker-legends-table-recognizer --dataset-manifest ... --card-part-manifest ...`
  作为正式复跑入口，输出 `table_recognizer_summary.json` / `table_recognizer_report.md`，默认只扫描
  truth 中 `screen.kind=actionable_table` 的帧。初次用 `multi_source_templates_v2` 复跑 57 帧：
  `state` 47、`missing_legal_actions` 3、`not_enough_players` 3、`missing_pot` 2、
  `preselect_ambiguous` 1、`hero_not_current` 1；报告已输出 issue count 与 blocker 明细表
  （frame/result/issues/street/pot/buttons/seats/accepted numbers），不再依赖临时脚本解释 blocker。
- 阶段 4 Poker Legends committed-pot derivation v1：在不猜未知底池的前提下，若 pot OCR/truth 缺失，
  但至少两个 active seat 都有显式 `committed` 数值且总额 >0，则把 pot 派生为 committed 总和，source
  记为 `rule_inferred_committed`；`committed=null`、单人 seat、总额 0 仍 fail-closed。当前 57 帧复跑：
  `state` 48、`missing_legal_actions` 3、`not_enough_players` 3、`missing_pot` 1、
  `preselect_ambiguous` 1、`hero_not_current` 1。
- 阶段 4 Poker Legends action-panel evidence v1：`VisualObservation.action_panels` 现在显式标记
  actionable 但缺当前动作按钮行的 `missing_current_action_row`，并在遇到 `Call Any` / `Check/Fold`
  等预选/快捷标签时追加 `preselect_strip` negative panel；table recognizer evaluator 的 blocker 表同步展示
  action panel flags。当前 57 帧复跑 state 覆盖不变（48/57），但 3 个 `missing_legal_actions` 均可解释为
  `missing_current_action_row`，`session_002__keyframe_000334` 明确为 `preselect_strip`。
- 阶段 4 Poker Legends action-row contract hardening v2：table recognizer 现在要求免费行动局面必须看到
  `check`/`bet` 这类 passive/start action，面对跟注压力必须看到 `call`，否则继续 fail-closed 为
  `missing_legal_actions`；action-panel flags 同步标出 `missing_passive_action` 与
  `button_label_action_mismatch`。同时只在高置信 OCR 读到 `right_top_stack` 时才为单 hero seat 补最小对手
  seat。当前 57 帧复跑：`state` 39、`missing_legal_actions` 13、`not_enough_players` 2、
  `missing_pot` 1、`preselect_ambiguous` 1、`hero_not_current` 1；覆盖下降来自旧 valid prototype 中 9 帧按钮行
  不完整/错位，按安全设计视为 discovered previous false valid，而不是追覆盖。
- 阶段 4 Poker Legends number parser hardening v2：数字 parser 对无 K/M 后缀的 `31.0` / `1,250`
  这类 OCR 插点/千分位按整数筹码处理（去标点），但保留 `1.23K` 的小数倍率语义。`card_review_v1__session_001__keyframe_000112`
  的 pot 从低置信 `31` 修成可接受的 `310`；当前 57 帧复跑：`state` 40、`missing_legal_actions` 13、
  `not_enough_players` 2、`preselect_ambiguous` 1、`hero_not_current` 1，`missing_pot` 清零。
- 阶段 4 Poker Legends table evaluator diagnostics v2：table recognizer evaluator 新增
  `action_panel_flag_counts` 与 `blocking_action_panel_flag_counts`，把 state 行的诊断 flags 和真正阻塞原因
  分开统计。当前 57 帧中 blocking flags 为 `missing_passive_action` 10、
  `missing_current_action_row` 6、`button_label_action_mismatch` 4、`preselect_shortcut_label` 2。
- 阶段 4 Poker Legends reviewed-label action derivation v1：truth-assisted replay 中，若当前按钮槽位有明确
  reviewed label（如 `Call $100` / `Check` / `Raise`）且不是 `Call Any` / `Check/Fold` 等预选语义，
  `RecognizedButton.action_type` 由 label 派生而不是继续信任槽位默认值；image-only/live 仍不会因此绕过
  source policy。当前 57 帧复跑：`state` 41、`missing_legal_actions` 11、`missing_call_amount` 1、
  `not_enough_players` 2、`preselect_ambiguous` 1、`hero_not_current` 1；`button_label_action_mismatch`
  从 action-panel flags 中清零，旧 `session_002__keyframe_000200` 更准确地暴露为 label 缺金额的
  `missing_call_amount`。
- 阶段 4 Poker Legends committed-call derivation v1：当 call 按钮 label/按钮 OCR 都没有金额时，只在 active
  seats 的 committed 数值完整且 `max_committed - hero_committed > 0` 时把 call amount 标为
  `rule_inferred_committed`；否则继续 fail-closed。当前 57 帧复跑：`state` 42、`missing_legal_actions` 11、
  `not_enough_players` 2、`preselect_ambiguous` 1、`hero_not_current` 1，`missing_call_amount` 清零。
- 阶段 4 Poker Legends hidden-button filter v1：truth-assisted replay 中，reviewed truth 明确标记不可见的
  中/右 primary 按钮不再因为静态槽位存在而进入 `RecognizedTable`；左键仍保留模板识别路径以兼容已覆盖的
  shifted-label 场景。当前 57 帧 coverage 不变（`state` 42），但 `session_002__keyframe_000041/000089/000228/000229`
  的 false `primary_middle:raise` 已从 blocker 表消失，只剩 reviewed 可见的 fold/all-in 按钮。
- 阶段 4 Poker Legends direct truth action buttons v1：truth-assisted replay 现在优先使用 reviewed direct action
  buttons（包括 `call`/`check`/`raise` 语义命名），且 action type 同样由 label 派生；`primary_left=Call Any`
  这类主按钮槽位预选语义仍独立标为 `preselect_ambiguous`。当前 57 帧复跑：`state` 44、
  `missing_legal_actions` 9、`not_enough_players` 2、`preselect_ambiguous` 1、`hero_not_current` 1；
  blocking `missing_passive_action` 从 8 降到 6。
- 阶段 4 Poker Legends accepted action-panel evidence v1：`VisualObservation.action_panels` 现在优先展示
  已进入 `RecognizedTable.buttons` 的 accepted buttons，只有没有 accepted buttons 时才回退到 raw image
  button predictions；这让 truth-assisted state 与 HUD/evaluator 证据一致。当前 57 帧 coverage 不变（`state` 44），
  但 `missing_current_action_row` 总数从 16 降到 4，且 4 个全部为真正 blocking 行，不再污染 state 行诊断。
- 阶段 4 Poker Legends action-block reason split v1：`_legal_actions` 不再把所有动作行失败都归为
  `missing_legal_actions`，而是拆为 `missing_current_action_row`（无当前按钮行）、
  `missing_passive_action`（只有 fold/all-in/raise，缺 check/call/bet）和后续可用的 `missing_call_action`。
  当前 57 帧 coverage 不变（`state` 44），blocker 为 `missing_passive_action` 6、
  `missing_current_action_row` 3、`not_enough_players` 2、`preselect_ambiguous` 1、`hero_not_current` 1。
- 阶段 4 Poker Legends table evaluator safety counts v1：table recognizer evaluator 新增
  `screen_kind_counts`、`authorization_events`、`non_actionable_frames`、`false_actionable_count` 与
  `false_actionable_examples`，让 `--include-non-actionable` 可直接衡量 hard negatives。当前 manifest
  全量 119 帧复跑：`actionable_table` 57、`table_observe` 45、`blocked_overlay` 17；authorization events 44，
  non-actionable frames 62，false actionable 0。
- 阶段 4 Poker Legends table evaluator truth summaries v1：每个 evaluator row 现在包含 reviewed truth 的
  buttons/seats/texts 摘要，blocker report 增加 Truth Buttons / Truth Seats / Truth Texts 与 recognized
  Buttons / Seats 对照列。当前全量 119 帧复跑结果计数不变，但剩余 blocker 可以直接判断是 truth 缺 seat、
  当前动作行缺失，还是 recognizer 未接受按钮/数字，不再依赖临时脚本追查。
- 阶段 4 Poker Legends table evaluator review tags v1：evaluator 为每个 non-state row 生成 `review_tags`，
  并汇总 `review_tag_counts`，把剩余 blocker 变成可执行清单。当前全量 119 帧复跑：
  `negative_screen_state` 62、`truth_missing_passive_action` 6、`truth_missing_current_action_row` 3、
  `truth_missing_opponent_seat` 2、`primary_preselect_shortcut` 1、`hero_turn_not_confirmed` 1；false actionable
  仍为 0。
- 阶段 4 Poker Legends table evaluator review queue v1：evaluator 额外输出
  `table_recognizer_review_queue.json`，只包含需要补标注/ROI/审核的非 state 行，排除普通
  `negative_screen_state` hard negatives。当前全量 119 帧复跑 review queue 为 13 行：
  `truth_missing_passive_action` 6、`truth_missing_current_action_row` 3、`truth_missing_opponent_seat` 2、
  `primary_preselect_shortcut` 1、`hero_turn_not_confirmed` 1。
- 阶段 4 Poker Legends table evaluator review queue evidence v1：每个 evaluator row 现在带
  `truth_path` 与 `layout_annotation_path`，并额外输出
  `table_recognizer_review_queue_by_tag.json` / report 的 `Review Queue By Tag`。当前全量 119 帧复跑核心
  计数不变：authorization events 44、non-actionable frames 62、false actionable 0；review queue 仍为
  13 行，并可直接按 tag 跳转补标注或查 ROI。
- 阶段 4 Poker Legends table evaluator critical safety summary v1：evaluator 现在输出
  `recognition_mode_counts`、`contract_counts`、`assembly_status_counts`、`unsafe_authorization_events`、
  `stale_authorization_events`、`truth_assisted_authorization_events`、`source_policy_violation_count`、
  `accepted_critical_wrong_count`，每个 row 同步导出 `accepted_critical_fields` 与 mismatch/source-policy
  examples。当前全量 119 帧复跑：`truth_assisted_replay` 119，authorization events 44，其中
  truth-assisted authorization 44、unsafe authorization 0、stale authorization 0、accepted critical wrong 0。
- 阶段 4 Poker Legends table evaluator image-only replay v1：新增 `--image-only-replay`，同一 reviewed manifest
  可只把 image + layout annotation 传入 recognizer，truth 只作为 evaluator expected data，不进入主链路。当前全量
  119 帧 image-only 复跑：`image_only_replay` 119，authorization events 0、truth-assisted authorization 0、
  unsafe authorization 0、accepted critical wrong 0；其中 41 帧到达 table recognizer 后因
  `missing_table_metadata` fail-closed，78 帧被 screen gate 归为 `screen_not_actionable`。
- 阶段 4 Poker Legends table evaluator screen confusion v1：evaluator 新增 truth-vs-recognized screen 混淆矩阵、
  `screen_false_actionable_count` 与 `screen_missed_actionable_count`，区分 screen gate 风险和最终授权风险。当前
  全量 119 帧 image-only 复跑：screen false actionable 0、screen missed actionable 16；混淆为
  truth `actionable_table` -> recognized `actionable_table` 41 / `table_observe` 16，truth `blocked_overlay` ->
  `blocked_overlay` 16 / `table_observe` 1，truth `table_observe` -> `table_observe` 45。
- 阶段 4 Poker Legends table evaluator screen review tags v1：`screen_not_actionable` 不再一律当普通
  `negative_screen_state`；truth actionable 但 image-only screen gate 漏判时标记为
  `screen_missed_actionable`，recognized actionable 但 truth non-actionable 时标记为
  `screen_false_actionable`。当前 image-only review queue 为 57 行：`missing_table_metadata` 41、
  `screen_missed_actionable` 16；普通 non-actionable hard negatives 仍排除在 queue 外。
- 阶段 4 Poker Legends table evaluator image-only readiness v1：对 image-only 且 screen 判为
  `actionable_table` 的 row 输出 `table_readiness_flags`，并汇总 `table_readiness_flag_counts`，只量化 primitive
  缺口，不启用 image-only GameState 组装。当前 119 帧 image-only 复跑 readiness blocker：
  `readiness_not_enough_players` 30、`readiness_missing_hero_seat` 8、`readiness_missing_passive_action` 8；
  `readiness_missing_hero_hole_cards` 误报已修正（metadata 内存里是 tuple，evaluator 现在按 list/tuple 都能读）。
  authorization 仍为 0。
- 阶段 4 Poker Legends table evaluator number readiness v1：image-only row 现在同时导出原始
  `number_predictions` 与 `accepted_number_predictions`，并把部分 table readiness blocker 拆成数字证据缺口，
  额外输出 `table_recognizer_number_readiness_by_flag.json` 与
  `table_recognizer_number_readiness_rows.json` 供按 frame 复核，仍不放宽数字接受阈值、不启用
  image-only GameState 组装。当前 119 帧 image-only 复跑，number readiness row 30：
  `readiness_low_confidence_opponent_stack` 19、`readiness_missing_opponent_stack_ocr` 8、
  `readiness_low_confidence_hero_stack` 8；数字覆盖分布显示 41 个 image-only actionable-table row 均有
  raw `pot`/`hero_stack`/`right_top_stack`/`primary_left` 预测，accepted 分别为 pot 41、hero_stack 33、
  right_top_stack 14、primary_left 38；authorization events 0、unsafe authorization 0。
- 阶段 4 Poker Legends table evaluator number truth comparison v1：evaluator 现在把 replay 中的 raw/accepted
  text 数字预测与 reviewed truth text 做离线 exact-match 对照，并输出 mismatch examples；该指标只用于评估，
  不进入 recognizer/safety gate。当前 119 帧 image-only 复跑显示数字 OCR 仍不可作为安全 critical source：
  raw hero_stack 19 match / 22 mismatch、raw pot 22 / 12、raw right_top_stack 4 / 19；accepted hero_stack
  16 / 17、accepted pot 22 / 12、accepted right_top_stack 2 / 7。因此下一步应修 parser/ROI/规则校验，
  不能直接通过降低阈值来解锁 image-only GameState。
- 阶段 4 Poker Legends number parser hardening v3：数字 parser 不再把跨空白的孤立 `K/M` 粘成单位后缀，
  避免 `1,052+50\n\nM` 被误归一成 50,001,052；正常紧邻后缀如 `$1.25K` 仍解析为 1250。当前
  119 帧 image-only 复跑 raw right_top_stack 从 4 match / 19 mismatch 改善到 7 / 16；accepted 数字分布不变，
  仍不放宽阈值、不启用 image-only GameState。
- 阶段 4 Poker Legends number confidence hardening v1：数字 OCR confidence 现在会把明显碎片化的结果降到
  0.65，包括多行孤立数字、跨空白孤立 K/M、`+110 4`、多 `+`、`$` 前混入数字/字母、无货币符号的异常
  加号组合等。当前 119 帧 image-only 复跑 accepted exact-match 明显收紧：pot 7 match / 0 mismatch、
  hero_stack 8 / 1、right_top_stack 1 / 1；accepted 覆盖同步降到 pot 8、hero_stack 9、right_top_stack 2。
  image-only authorization events 仍为 0；truth-assisted 复跑因不再信任碎片 pot OCR 从 44 个 state 收紧到
  39 个 state，unsafe/stale 仍为 0。剩余 `$1054`、`$334+10` 类错例需要字段语义/ROI 校验，不能由通用 parser
  猜。
- 三大块已闭合：离线 `RecognizedTable -> GameState` 稳定性、macOS 捕获/自动化 dry-run 安全链路、
  共用 AI heuristic v1。下一步可以做 macOS host dry-run 实测，但仍不做真实点击。
- **host dry-run 操作指南见 `docs/bot-host-dryrun.md`**（已验证的精确命令 + manifest/layout 路径 + 输出字段
  解读 + 回报清单）。Step 1（`--image` 跑 bundled 帧，无需 macOS）已在容器内验证：keyframe_000042 正确
  fail-closed 为 `blocked_overlay`（buy_in_prompt），证明 capture→recognize→安全闸→输出 全链路接通。
- 第一轮 host 测试建议先用 `uv run holdem-bot-run-poker-legends-dry-run --image ...` 跑已保存截图和
  reviewed annotation，确认输出里的 `screen`、`state`、`policy_decision`、`dry_run_record` 一致。
- 第二轮再用 `--capture-out-dir ... --window-id ...` 对 Poker Legends 当前窗口截图；没有 reviewed
  annotation 的 live 截图很可能继续 fail-closed 在 `no_game_state` / `missing_table_metadata`，这是预期
  安全行为。真实点击仍需单独实现并显式开闸。

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
