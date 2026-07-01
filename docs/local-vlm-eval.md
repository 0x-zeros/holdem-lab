# 本地多模态 vs Gemini vs CV — Poker Legends 识别评测

_生成于 2026-06-29；共 13 帧，全部为真·hero 决策帧(Gemini `is_actionable=true`，含 call+raise+fold 完整按钮行)。box 误差与字段一致率**仅在这些决策帧上**统计(box 真值 = CV `detect_action_buttons` 中心；字段基线 = Gemini)。延迟/解析率统计全部帧。复跑见文末。_

候选本地模型由 LM Studio 提供 OpenAI 兼容服务(`host.docker.internal:1234`)，评测脚本后端无关：`bot/src/holdem_bot/eval/local_vlm.py`。

## 汇总(仅 hero 决策帧)

| 后端 | 解析成功 | box 平均误差 | 漏检按钮 | 字段一致率(vs Gemini) | 平均延迟 |
|---|---|---|---|---|---|
| `gemini`(gemini-3.1-flash-lite) | 13/13 | **11px** | 0 | — | **7.8s** |
| `qwen/qwen3-vl-30b`(A3B Instruct) | 12/13 | 69px | 5 | 69% (n=84) | 13.1s |
| `cv`(detect_action_buttons) | 13/13 | 0px(真值) | 0 | — | 0.0s |

## 逐字段一致率(vs Gemini，仅 hero 决策帧)

| 后端 | hero_cards | board | pot | seat_count | stacks | street | actionable |
|---|---|---|---|---|---|---|---|
| `qwen/qwen3-vl-30b` | 83% | 83% | 58% | 83% | 75% | 0%¹ | 100% |

¹ Qwen 在 `text` 模式下从不填 `table_state.street`；street 可由公共牌数直接推导(0=preflop, 3=flop, 4=turn, 5=river)，不必问模型，故非实际阻碍。

## 结论与推荐

**Qwen3-VL-30B-A3B-Instruct 是同机实测中唯一可用的本地候选，字段读取够用、边际成本≈0；但当前 Gemini 在速度/坐标精度/稳定性上仍更优。** 推荐**混合**架构、Gemini 暂为主，把本地作为已验证的降本备选：

| 维度 | 现状最优 | 说明 |
|---|---|---|
| 速度 | **Gemini**(7.8s) | 本地中位 ~13s、尾部达 18.6s，**可能超出 ~15s 行动钟** |
| 点击坐标 box_2d | **Gemini**(11px) / **CV**(0px) | 本地 69px 系统偏移，够"认按钮"但点不准 → 用 CV 精修 |
| 字段(牌/座位/actionable) | 接近持平 | actionable 100%、hero_cards/seat 83%、board 83% |
| 稳定性 | **Gemini**(13/13) | 本地 12/13，1 帧啰嗦循环到 120s+ 截断 |
| 长期成本 | **本地**(≈电费) | Gemini ≈ $1.5/千次；本地在自有 Mac 上跑边际≈0 |

## 2026-06-29 追加：Qwen3-VL 8B vs 30B（reviewed truth 小样本）

用户本机 LM Studio 已同时暴露 `qwen/qwen3-vl-8b` 与 `qwen/qwen3-vl-30b`。新增
`--reference-dir` 后，可直接拿 reviewed truth overlay 当字段基准，不再依赖 Gemini。两组小样本只用于
快速判断 8B 是否值得替代 30B；样本小且部分 truth 只标最小座位集，seat/stacks 指标偏严。

| 样本 | 后端 | 解析成功 | box 平均误差 | 漏检按钮 | 字段一致率(vs reviewed truth) | 平均延迟 |
|---|---|---:|---:|---:|---:|---:|
| session_002 actionable 3 帧 | `qwen/qwen3-vl-8b` | 3/3 | 146px | 4 | 19% | 15.0s |
| session_002 actionable 3 帧 | `qwen/qwen3-vl-30b` | 3/3 | 250px | 2 | 38% | 18.1s |
| session_001 actionable 2 帧 | `qwen/qwen3-vl-8b` | 2/2 | 185px | 2 | 21% | 12.4s |
| session_001 actionable 2 帧 | `qwen/qwen3-vl-30b` | 2/2 | 133px | 0 | 36% | 13.2s |

同一 session_002 三帧把 Gemini 也加入 reviewed-truth 对比后：

| 后端 | 解析成功 | 字段一致率(vs reviewed truth) | 平均延迟 | actionable | hero_cards | street |
|---|---:|---:|---:|---:|---:|---:|
| `gemini` | 3/3 | **62%** | **6.5s** | 100% | 100% | 100% |
| `qwen/qwen3-vl-30b` | 3/3 | 38% | 17.6s | 100% | 33% | 0% |
| `qwen/qwen3-vl-8b` | 3/3 | 19% | 15.1s | 33% | 33% | 0% |

关键差异：

- **Gemini 仍明显领先**：在 reviewed-truth 小样本里，整屏结构化字段、hero cards、street、actionable 与延迟
  都优于两个本地 Qwen。
- **8B 能解析，但不能替代 30B**：两组 reviewed-truth 小样本里字段一致率约 19–21%，明显低于 30B 的
  36–38%。
- **actionable 稳定性差异很大**：30B 在 5/5 帧都输出 actionable；8B 只有 1/5 帧匹配 truth。
- **牌面花色 8B 更容易错**：例如 `JS` 被读成 `JH`、`8D` 被读成 `8H`；30B 在 session_001 的两帧
  hero cards 为 2/2。
- **box 仍不能直接点击**：8B/30B 都有百像素级偏差；点击继续以 CV `detect_action_buttons` 为准。
- **延迟不是 8B 的决定性优势**：本次 8B 平均 12–15s，30B 13–18s；8B 更轻，但在当前 LM Studio/图像尺寸下
  没快到可以抵消准确率差距。

当前建议：**8B 作为低成本 smoke / fallback，30B 仍作为本地高准确率候选；线上点击与金额继续走 CV/规则兜底。**
8B 更适合先看单个 ROI crop（牌面、按钮、数字），不适合作为整屏 `GameState` 主读者。

复跑命令：

```bash
scripts/dev/py -m holdem_bot.eval.local_vlm \
  --frames <comma-separated-actionable-frame-pngs> \
  --reference-dir <truth_overlay_dir> \
  --backends qwen/qwen3-vl-8b,qwen/qwen3-vl-30b,cv \
  --timeout 60 \
  --out artifacts/poker-legends-videos/qwen3vl_8b_vs_30b_truth_eval.md \
  --stamp "$(date +%F)"
```

**落地（与现有 HUD 混合点击设计一致）：**

| 用途 | 主用 | 兜底/校验 |
|---|---|---|
| 牌 / 座位 / actionable | Gemini 主、本地备(可配置切换) | 互为参照 |
| street | **由公共牌数推导** | 不问模型 |
| 点击坐标 box_2d | **CV 像素精修** | LLM box 粗定位兜底 |
| 鲁棒性 | 硬超时 + 失败回退 | 本地超时/失败 → 回退 CV/Gemini |

**何时切本地为主**：当调用量大到 Gemini 成本可观、且完成下列 hardening——① 降延迟(更小输入图 / 更低输出上限)；② 输出硬上限 + 超时 + 失败回退(治 1/13 啰嗦截断)；③ prompt 补强 board 读取。鉴于长期刷分目标，推进本地 hardening 是划算的。

## 成本（长期）

| | 每次调用边际成本 | 来源 |
|---|---|---|
| Gemini 3.1 Flash-Lite | ~$0.0015（≈$1.5/千次） | $0.25/1M 输入 + $1.50/1M 输出；单帧约 1.8k 输入 + 0.7k 输出 token |
| 本地 Qwen3-VL（你的 Mac） | ≈ 电费（~0） | 硬件已自有(沉没成本)；权重一次性下载 |

只要是**持续/高频自我对弈**，本地边际成本几乎立刻更划算；只有**偶发、低频**且不想占用 Mac 时，Gemini 的免运维更省心。

## 诚实解读各维度

- **box**：Gemini 11px(优) / Qwen 69px(系统性偏移，CV 精修可压到个位数；偶发 156px) / CV 0px(真值)。
- **hero_cards 83%**：原始仅 33%，主因是**格式差异**——Qwen 输出花色符号 `♠♥♦♣` 和 `10`(而非 `S/H/D/C`、`T`)，已归一化。剩余不一致是**真实花色误读**(JD↔JS、JC↔JS，各 1 帧)。
- **board 83%**：2 帧(翻牌/河牌)Qwen 漏读公共牌返回空 → 偶发漏读。
- **pot 58%**：真实分歧——底池数字最难读，双方都可能错；bot 侧可用各家 committed 之和交叉校验。
- **seat_count 83% / stacks 75%**：过滤空座(stack=0)后，剩余差异为边缘座位(刚 fold / 边界筹码)。
- **延迟**：本地 9–18.6s(中位~13s，方差大、尾部风险)，Gemini 7.8s。
- **鲁棒性**：13 帧中 1 帧(keyframe_000332)本地进入啰嗦循环(120s+、输出到 6000 token 上限被截断)→ 必须配硬超时 + 回退。

## 被淘汰的候选（同机实测，win_capture）

| 模型 | 不可用原因 |
|---|---|
| `qwen3.5-9b` | 思考型，关不掉(`/no_think`、`enable_thinking=false` 均无效)→ 146s、输出被截断 |
| `gemma-4-26b-a4b` | 同为思考型 → 80s、截断、甚至误判"无 hero" |
| `gemma-3-4b` | 4B 太小：非法 JSON + box_2d 离谱(grounding 弱) |
| `gemma-3-27b` | 唯一干净，但 box 偏 75–220px、延迟~40s，grounding 弱 |

## 实现要点（已写进 harness，可复跑）

- **box 轴序按后端分**：Qwen-VL 用原生 `[x1,y1,x2,y2]`(无视 prompt 要求的 yxyx)；Gemini 用 `[ymin,xmin,ymax,xmax]`。
- **键名大小写归一**：本地模型把 prompt 小标题 `Buttons`/`GAME_REGION` 直接当 JSON 键。
- **牌面归一**：`♠♥♦♣ → SHDC`、`10 → T`。
- **响应格式**：LM Studio 不支持 `json_object`；复杂 `json_schema` 会语法卡死出空 → 本地走 `text` + 容错解析(`raw_decode` 容忍 ``` 围栏与尾部杂讯)。
- **思考型兜底**：抓 `reasoning_content`，content 为空时回退解析。

## 复跑

```bash
# 本地后端需 LM Studio 开 "Serve on Local Network"(默认 :1234)并能加载 qwen3-vl-30b
# 决策帧可用 CV 完整三色按钮行筛选(见 detect_action_buttons)
scripts/dev/py -m holdem_bot.eval.local_vlm \
  --frames "<逗号分隔的帧路径>" \
  --backends gemini,qwen/qwen3-vl-30b,cv \
  --out docs/local-vlm-eval.md --stamp "$(date +%F)"
```

> 合规：本地权重经 LM Studio 从 HuggingFace 拉取；正式部署应选官方 Qwen org / lmstudio-community 构建并校验完整性。
