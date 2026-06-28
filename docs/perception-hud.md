# 感知 HUD（perception HUD）—— 原理 / 库 / 安全 / 通用性 / 操作指南

> 工具：`holdem-bot-watch-poker-legends`（host 脚本子命令 `watch` / `watch-once`）。
> **只读截屏 + 本地 CV + 叠加显示，永不点击。** 轮到 hero 行动时还会定位动作按钮(**LLM `box_2d`
> 当主坐标 + CV 校验**)、算出「机器人**打算**点哪个像素」并把靶标画在 HUD 上——**仍然只算不点**
> (不合成任何鼠标输入)。配套的完整 host 流程见
> `docs/bot-host-dryrun.md`；这份文档是 HUD 本身的参考（它是什么、怎么做到的、有什么风险、
> 能不能用到别处、怎么跑）。

---

## 1. 它是什么 / 原理

HUD 把慢的手动循环（截屏 → 存 → 跑 → 贴 JSON → 等人看）换成「开游戏 → 一条命令 → 实时看到机器人
看到了什么」。每一帧做 7 步，**全在本机内存里完成**：

1. **抓屏**：`mss` 抓一块屏幕区域 → 一张像素图（numpy 数组）。等价于连续截图，**不碰游戏进程、
   不读游戏内存、不注入、不 hook**。
2. **写临时 PNG**：把这张图落一个临时文件（只是为了复用我们现成的「按文件路径识别」的识别器）。
3. **识别**：现成的 `PokerLegendsTableRecognizer` 用 OpenCV 模板匹配 + 训练好的发牌分类器，在布局
   ROI 上认牌/底池/座位/按钮 → 一个 `GameState`（或失败原因 `state_block_reason`）。
4. **安全门**：同一个 `evaluate_safety`，判断这帧能不能行动（与真 bot 一致）。
5. **决策**：若可行动，对手感知策略算出「打算怎么打」—— **只算不点**。
6. **算点击靶标(混合)**：若有决策,**主坐标 = LLM 在识别调用里顺带返回的每个按钮 `box_2d`**
   (`[ymin,xmin,ymax,xmax]` 归一化 0-1000,反归一化映射回整帧像素中心;随识别免费、对取景/主题/
   缩放鲁棒、可跨游戏)。同时用 CV（`detect_action_buttons`,按颜色/形状找红=fold·蓝=call·金=raise
   的圆钮）在整帧里定位**同一**按钮做**校验**:两者吻合→取 CV 像素精确中心(标 `llm+cv`);只有一方→
   取那方(`llm`/`cv`);相距过远→标 `conflict`;都没有→**不点**(fail-closed)。窗口模式加窗口原点→屏幕坐标。
7. **叠加 + 显示**：纯 numpy/cv2 把 ROI 框 + 文字面板 + 点击靶标（选中按钮高亮、标来源 + `read-only`）
   画到画面**副本**上，`cv2.imshow` 开本地窗口；`s` 存证据，`q` 退。

一句话：**它是只读的屏幕阅读器 + 可视化器**，会算出但**不执行**点击，没有任何「对游戏施加动作」的代码路径。
按钮坐标走「LLM box_2d 主 + CV 校验」混合(`conflict`/缺失则 fail-closed);Gemini 用官方 `box_2d`
归一化 0-1000 约定能给出 ~10px 内的可点坐标。独立验证 CV 按钮定位:
`holdem-bot-detect-poker-legends-buttons --image X.png`（纯 CV，无需 API）。

---

## 2. 叠加显示在哪、怎么做到的

**在旁边的一个独立窗口里，不是盖在真游戏画面上。**

- HUD 先把屏幕**截一张副本**（内存里的像素数组）；
- 在**这张副本**上用 `cv2.rectangle` / `cv2.putText` 画 ROI 框和文字面板（代码里就是
  `render_overlay()` 先 `frame.copy()` 再画）；
- 用 `cv2.imshow` 把这张「画过的副本」显示在**我们自己的窗口**里，每秒刷几次。

真 Poker Legends 窗口**完全没被碰**：没有透明悬浮层、没有改游戏像素、没有注入游戏渲染。屏幕上会
同时有两样东西——真游戏窗口，和一个像「带标注的延迟镜像」的 HUD 窗口。

> 对比我们**没有**采用的方案：真正悬浮在游戏上的透明 overlay（类似 FPS 外挂/OBS 那种），需要置顶
> 透明窗口或注入游戏渲染，更复杂也更脆弱，macOS 上还要特殊窗口 API。我们故意用「截图 → 在副本上画
> → 显示副本」这条最简单、最安全的路。

---

## 3. 用了什么库（都官方源、本地运行）

| 库 | 许可 | 干什么 | 联网？ |
|---|---|---|---|
| `mss` | MIT | 截屏。macOS 底层走 **CoreGraphics/Quartz**（`CGWindowListCreateImage`），Windows 走 BitBlt/DXGI | 否（源码无 socket/http） |
| `opencv-python`（cv2） | Apache-2.0 | 图像处理 + 显示窗口 | 否 |
| `numpy` | BSD | 数组 | 否 |
| `pytesseract` + `tesseract` | Apache-2.0 | 本地 OCR 读数字 | 否 |
| 我们自己的代码（holdem_ai/holdem_bot/holdem_common） | — | 识别 / 策略 / 叠加，纯 Python | 否 |

`watch-once`（离线单帧）只需要 cv2 即可，**不需要 mss、不需要录屏权限**；`mss` 只在 live `watch`
时才用。

---

## 4. 在 Mac 上的安全风险——结论：低，但有一处要讲准

**实测事实：HUD 这条路径会把 `openai` 和 `google-genai` 两个联网客户端 import 进内存。** 原因是
`bot/vision/__init__.py` 是个「一次性全导出」的桶，顺带加载了那个**单独的 LLM 标注工具**
（`llm_annotation.py`）。

但关键区别：**「被 import」≠「会联网」。**

- 这两个客户端是**惰性**的：import 不发任何包；只有**显式调用** `.generate()` / `.create()` **且
  提供 API key** 时才联网。
- **HUD 全程不调用它们**（没 key、没调用点）；识别器和叠加模块本身**零网络调用**。
- 所以**你的屏幕内容不会经 HUD 外发**。唯一会把画面发给 OpenAI/Google 的，是你**单独主动**去跑
  `llm-annotate` 并填自己的 key —— 那是另一条命令、另一回事。

**「被黑」层面：**

- HUD **不开任何监听端口、不接收远程输入、不提权、不注入游戏、不读游戏内存** → **没有新增对外
  攻击面**。
- 依赖都来自官方 PyPI、`uv.lock` 锁哈希；mss/opencv/numpy 是被广泛审计的常用库。供应链风险 = 任何
  pip 项目的基线，低。

**真正该留心的（都不是「被黑」）：**

1. **录屏权限**：live `watch` 首次运行时，macOS 会要求给你的终端（Terminal/iTerm）授「屏幕录制」
   权限；授予期间该进程能看到你**整个屏幕**（和所有截图/直播软件一样）。授给具体那个程序，用完可
   在「系统设置 → 隐私与安全性 → 屏幕录制」撤销。
2. **dump 文件就是截图**：`~/pl-dumps` 里是牌桌截图（玩具币、低敏感，但可能带你的用户名/头像）。
   发我 = 分享截图；抓帧前把别的敏感窗口关掉。
3. **Steam ToS**：只读截图是良性的；**将来「点击」**那一步才是 ToS 敏感点。

> 可选强化：如需「构造上就离线」，可以把那两个网络客户端从 HUD 的加载链里解耦（让识别器不再经
> `vision` 那个桶顺带加载 `llm_annotation`），改完 HUD 连 import 都不会碰 openai/google。目前未做，
> 需要时再说。

---

## 5. 通用性：**方法通用，这套工具是为「扑克」适配的**

分层看最清楚：

| 层 | 通用吗 | 说明 |
|---|---|---|
| **抓屏循环**（mss→numpy） | ✅ 完全通用 | 任何屏幕任务都能用 |
| **叠加/显示/dump 框架**（`perception_overlay`） | ✅ 通用 | 它**不懂扑克**，只是「给我一份 ROI 布局我就画框 + 面板」，换任何 layout JSON 都行 |
| **capture→recognize→门→叠加 编排** | ✅ 通用模式 | 这就是「截屏 + CV 玩游戏」的通用范式（Airtest 之类也是这套思路） |
| **识别器**（认牌/池/座位/盲注） | ❌ 扑克专用 | 用了 Poker Legends 卡面训练的分类器 + 它 UI 的布局 |
| **AI/策略**（决策） | ⚠️ **德州扑克通用、非 PL 专用** | `holdem_ai` 是通用德扑——它现在就同时驱动本地 pygame 那个游戏 |

所以：

- **换另一个扑克客户端**（别的扑克 App/网站）≈ 只需**重做识别器 + 布局**，AI 和这套外壳照用。
- **换非扑克的 Steam 游戏** = 既要**新识别器**，又要**新「大脑」**——基本是新项目，**只复用抓屏/叠加
  这套脚手架**。

本质：**「截屏 + CV + 叠加」是放之四海的技术**；贵且专用的是**「教它这些像素是什么意思」（识别）和
「该怎么打」（AI）**——这两块才是每个游戏的真正工作量。

**关于 Windows**：HUD 代码本身**跨平台**（mss 两边都跑）。切 Windows 主要是为了**抓屏更稳 + 将来
点击省事**，设计上没有任何 Mac 专属的东西，迁过去零返工。

---

## 6. 操作指南：开游戏 → 跑循环 → 发我什么

所有命令从仓库根目录跑；脚本默认用 `uv run`（首次会自动 `uv sync` 装好 `mss`）。**全程不点击。**

### Step 0 — 先离线验工具（不碰真屏、不需要权限）
```bash
git pull                                              # 取最新（含 HUD）
scripts/host/poker-legends-dryrun.sh watch-once 000080
# → 渲出 /tmp/poker-legends-watch-000080.overlay.png；打开看看：一张牌桌截图，上面画了 ROI 框
#   + 左上文字面板（screen kind / 安全门 / state_block_reason / pot / 决策 / 读数）。
```

### Step 1 — 开游戏
- 打开 Poker Legends，**用窗口/无边框模式**（不要独占全屏——窗口模式抓屏最稳），坐到一张牌桌，
  轮到你行动。

### Step 2 — 抓一帧真屏发我做标定（最快通路）
live `watch` 在我们**为你的分辨率标定布局之前**，ROI 会是错位的（bundled 布局按 1600×982 定）。所以
最快的路是先给我一帧：
```bash
scripts/host/poker-legends-dryrun.sh capture ~/pl-frames   # 存一张当前屏 PNG，打印路径
sips -g pixelWidth -g pixelHeight ~/pl-frames/<那张>.png    # 告诉我它的像素尺寸
```
把**路径 + 像素尺寸**发我 → 我给你一条按你分辨率调好的 `apply-poker-legends-layout` 命令，生成
`~/pl-layout.json`。

### Step 3 — 跑实时循环
有了你的布局后：
```bash
scripts/host/poker-legends-dryrun.sh watch ~/pl-layout.json --region <L,T,W,H> --dump-dir ~/pl-dumps
# 弹出一个 HUD 窗口（带标注的游戏镜像，~4fps）。轮到你的真帧上按 's' 存证据，'q' 退。
```
> 首次会触发 macOS「屏幕录制」授权：去「系统设置 → 隐私与安全性 → 屏幕录制」给你的终端打勾，
> 重启终端再跑。`watch-once`（Step 0）不需要这个权限。
>
> 想先粗看一眼也行：直接 `... watch --dump-dir ~/pl-dumps`（用默认 bundled 布局、抓整屏），ROI 会
> 错位，但能确认「截屏→识别→叠加」链路通。

### Step 4 — 发我什么
在 `~/pl-dumps` 里挑**轮到你**那帧的 `*.overlay.png` + `*.json`（连同 Step 2 的像素尺寸）发我。我据
此：① 看 ROI 偏多少 → 调准布局；② 看 `state_block_reason` → 逐个解掉
`no_game_state` / `missing_table_metadata` 等卡点。这一步通了，读数和决策才跑得起来。
