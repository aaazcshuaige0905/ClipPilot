# ClipPilot

`ClipPilot` 是一个面向 AI 短视频高光剪辑场景的 Agentic Workflow MVP。  
当前版本已经能够完成一条完整的本地执行链路：

1. 上传 3 到 10 分钟视频
2. 提取视频元信息
3. 执行 ASR 转写
4. 生成高光候选片段
5. 生成真实可执行的剪辑时间轴 `editing_plan.json`
6. 按时间轴裁剪视频片段并合并成 `final_video.mp4`
7. 生成 `subtitles.srt`
8. 在需要时烧录字幕生成 `final_video_burned.mp4`
9. 保存 `execution_report.json`、`review_report.json`、`artifact_manifest.json`

这个项目当前的重点不是“模型一定最强”，而是把**可运行、可调试、可扩展**的 Agentic 视频工作流骨架搭稳。

---

## 1. 当前已实现功能

### 1.1 视频上传与任务创建

- 支持上传 `.mp4`、`.mov`、`.mkv`
- 支持接收以下表单字段：
  - `target_platform`
  - `target_duration`
  - `edit_style`
  - `language`
  - `need_burn_subtitle`
- 每次上传都会生成唯一 `task_id`
- 每个任务都会生成独立输出目录

### 1.2 视频元信息检测

使用 `moviepy` 提取视频元信息，当前返回：

- `duration_seconds`
- `width`
- `height`
- `fps`
- `has_audio`
- `file_size_mb`
- `source_path`

同时会做基础校验：

- 只允许 `.mp4` / `.mov` / `.mkv`
- 视频时长必须在 `180` 到 `600` 秒之间

### 1.3 ASR 转写

当前项目已经具备统一 ASR 抽象层：

- 默认 provider：`mock`
- 预留真实 `whisper`
- 预留真实 `faster-whisper`

即使本地没有安装 Whisper，也可以通过 `mock` 模式跑通完整流程。

转写结果会输出：

- `video_id`
- `segments`
- `full_text`
- `provider`

并保存为 `transcript.json`。

### 1.4 高光候选片段识别

当前使用规则法生成高光候选，不依赖外部 LLM。

规则主要包括：

- 关键词加分
- hook 句式加分
- 文本过短或过长降分
- 语义不完整降分
- 将连续 transcript 片段合并为约 8 到 20 秒候选
- 最终按 `score` 排序

结果保存为 `candidates.json`。

### 1.5 真实剪辑计划生成

`planner_agent` 已经不是占位版 top-N 选择器，而是会生成真实可执行时间轴：

- 优先选高分候选片段
- 第一个 clip 尽量选 hook
- 总时长尽量贴近 `target_duration`
- 严格不超过 `target_duration + 5`
- 每个 clip 限制在 `3` 到 `20` 秒之间
- 自动去重重复文本片段
- 如果候选不足，允许从 transcript 中补片段

结果保存为 `editing_plan.json`。

### 1.6 真实视频执行

`executor_agent` 当前已经能执行真实视频输出：

- 按 `editing_plan.clips` 逐段裁剪
- 将片段保存到 `outputs/tasks/{task_id}/clips/`
- 合并为 `outputs/tasks/{task_id}/final/final_video.mp4`
- 生成 `outputs/tasks/{task_id}/final/subtitles.srt`
- 当 `need_burn_subtitle=true` 时，尝试生成：
  - `outputs/tasks/{task_id}/final/final_video_burned.mp4`
- 生成完整 `execution_report.json`

### 1.7 字幕生成与烧录

已经支持：

- 根据**新视频时间轴**生成 `.srt`
- 第一段字幕从 `00:00:00,000` 开始
- 每段字幕按 `editing_plan.clips` 连续递增
- 使用 `ffmpeg` 可选烧录字幕

### 1.8 任务查询

当前 API 支持：

- 查询单个任务状态与结果
- 查询最近任务列表

### 1.9 中间产物持久化

当前系统会保存每个阶段的重要产物，方便：

- 调试
- 回放任务过程
- 未来前端展示
- 多 Agent 协作

---

## 2. 项目结构

```text
ClipPilot/
|-- main.py
|-- config.yaml
|-- requirements.txt
|-- README.md
|-- ARCHITECTURE.md
|-- pytest.ini
|-- .gitignore
|-- clippilot/
|   |-- api/
|   |-- core/
|   |-- schemas/
|   |-- tools/
|   |-- storage/
|   |-- harness/
|   |-- agents/
|   `-- rag/
|-- data/
|   `-- raw_videos/
|-- outputs/
|   `-- tasks/
`-- tests/
```

说明：

- `clippilot/` 是主代码目录
- `outputs/tasks/` 是任务主产物目录
- `tests/` 是自动化测试目录

更详细的模块边界说明请看 [ARCHITECTURE.md](./ARCHITECTURE.md)。

---

## 3. 核心模块说明

### `clippilot/api/`

负责 FastAPI 接口层：

- 接收上传请求
- 接收任务查询请求
- 组装 `UserRequest`
- 调用 `workflow`

### `clippilot/core/`

负责工作流编排：

- 创建任务上下文
- 管理任务状态和阶段
- 串联元信息提取、ASR、候选生成、计划生成、执行、审核

### `clippilot/schemas/`

负责定义结构化数据契约：

- 请求参数
- 视频信息
- Transcript
- 高光候选
- EditingPlan
- ExecutionReport
- TaskResult

### `clippilot/tools/`

负责原子能力实现：

- 视频信息提取
- ASR 转写
- 高光候选识别
- 视频裁剪
- 视频合并
- 字幕生成
- 字幕烧录

### `clippilot/storage/`

负责路径和持久化：

- 创建任务目录
- 保存原视频
- 保存 JSON 产物
- 读取任务结果
- 列出历史任务

### `clippilot/harness/`

负责：

- validators
- trace 日志
- 后续评估指标预留

### `clippilot/agents/`

当前已接入工作流的 Agent 模块：

- `planner_agent`
- `executor_agent`
- `review_agent`

其中：

- `planner_agent` 已升级为真实时间轴规划器
- `executor_agent` 已升级为真实视频执行器
- `review_agent` 目前仍是占位版

### `clippilot/rag/`

用于未来接入：

- 平台规则检索
- 剪辑模板检索
- 字幕规范检索
- 标题模板检索

---

## 4. 环境准备

### 4.1 使用 Conda

```powershell
conda activate clippilot
pip install -r requirements.txt
```

### 4.2 使用 venv

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 4.3 关于 ffmpeg

当前这些功能依赖系统可用的 `ffmpeg`：

- 视频裁剪
- 视频合并
- 字幕烧录

如果本机没有安装 `ffmpeg`：

- 裁剪和合并不会成功
- 字幕烧录不会成功
- 但单元测试会自动跳过依赖 ffmpeg 的集成测试，不会让整个 pytest 崩溃

---

## 5. 配置说明

项目配置来自 [config.yaml](./config.yaml) 和环境变量。

当前主要配置包括：

- 应用名称、版本、描述
- 任务主目录
- 原始测试视频目录
- 允许上传的视频后缀
- 视频最短和最长时长
- ASR provider
- Whisper 模型名
- 高光候选时长范围

示例：

```yaml
video:
  allowed_extensions:
    - .mp4
    - .mov
    - .mkv
  min_duration_seconds: 180
  max_duration_seconds: 600

asr:
  provider: mock
  whisper_model: base

highlight:
  min_candidate_duration: 8.0
  max_candidate_duration: 20.0
```

可以用环境变量覆盖部分配置，例如：

```powershell
$env:CLIP_PILOT_ASR_PROVIDER="mock"
```

---

## 6. 启动方式

在项目根目录运行：

```powershell
uvicorn main:app --reload
```

默认访问地址：

```text
http://127.0.0.1:8000
```

Swagger 文档：

```text
http://127.0.0.1:8000/docs
```

---

## 7. API 接口

### 7.1 健康检查

```powershell
curl.exe "http://127.0.0.1:8000/health"
```

### 7.2 上传视频并创建任务

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/tasks/upload" `
  -F "file=@sample.mp4" `
  -F "target_platform=bilibili" `
  -F "target_duration=30" `
  -F "edit_style=powerful" `
  -F "language=zh" `
  -F "need_burn_subtitle=true"
```

### 7.3 查询单个任务

```powershell
curl.exe "http://127.0.0.1:8000/api/v1/tasks/<task_id>"
```

### 7.4 查询最近任务列表

```powershell
curl.exe "http://127.0.0.1:8000/api/v1/tasks?limit=20"
```

---

## 8. 上传接口返回结果

当前上传成功后，返回结果会包含：

- `task_id`
- `status`
- `stage`
- `upload_file_path`
- `transcript_path`
- `candidates_path`
- `editing_plan_path`
- `execution_report_path`
- `final_video_path`
- `subtitle_path`
- `burned_video_path`

示例：

```json
{
  "task_id": "9d7d42ea07d34c0e9368ca9b2e32cf1e",
  "status": "completed",
  "stage": "completed",
  "upload_file_path": "outputs/tasks/9d7d42ea07d34c0e9368ca9b2e32cf1e/input/source.mp4",
  "user_params": {
    "target_platform": "bilibili",
    "target_duration": 30,
    "edit_style": "powerful",
    "language": "zh",
    "need_burn_subtitle": true
  },
  "metadata": {
    "duration_seconds": 362.15,
    "width": 1906,
    "height": 1062,
    "fps": 30.0,
    "has_audio": true,
    "file_size_mb": 375.58,
    "source_path": "E:/Github/agent_test/ClipPilot/outputs/tasks/.../input/source.mp4"
  },
  "transcript_path": "outputs/tasks/9d7d42ea07d34c0e9368ca9b2e32cf1e/transcript/transcript.json",
  "candidates_path": "outputs/tasks/9d7d42ea07d34c0e9368ca9b2e32cf1e/highlights/candidates.json",
  "editing_plan_path": "outputs/tasks/9d7d42ea07d34c0e9368ca9b2e32cf1e/plan/editing_plan.json",
  "execution_report_path": "outputs/tasks/9d7d42ea07d34c0e9368ca9b2e32cf1e/plan/execution_report.json",
  "final_video_path": "outputs/tasks/9d7d42ea07d34c0e9368ca9b2e32cf1e/final/final_video.mp4",
  "subtitle_path": "outputs/tasks/9d7d42ea07d34c0e9368ca9b2e32cf1e/final/subtitles.srt",
  "burned_video_path": "outputs/tasks/9d7d42ea07d34c0e9368ca9b2e32cf1e/final/final_video_burned.mp4"
}
```

说明：

- `burned_video_path` 只有在启用烧录并成功时才会有值
- 如果执行阶段失败，`status` 可能为 `failed`

---

## 9. 任务目录结构

每个任务的主目录如下：

```text
outputs/tasks/{task_id}/
|-- input/
|   `-- source.mp4
|-- metadata/
|   `-- video_info.json
|-- transcript/
|   `-- transcript.json
|-- highlights/
|   `-- candidates.json
|-- clips/
|   |-- clip_01.mp4
|   |-- clip_02.mp4
|   `-- ...
|-- final/
|   |-- final_video.mp4
|   |-- subtitles.srt
|   `-- final_video_burned.mp4
|-- plan/
|   |-- editing_plan.json
|   `-- execution_report.json
|-- review/
|   `-- review_report.json
|-- trace/
|   `-- workflow_trace.jsonl
|-- artifact_manifest.json
`-- task_result.json
```

### 主要文件说明

- `source.mp4`
  - 原始上传视频
- `video_info.json`
  - 视频元信息
- `transcript.json`
  - ASR 转写结果
- `candidates.json`
  - 高光候选片段
- `editing_plan.json`
  - 真实可执行时间轴
- `execution_report.json`
  - 真实执行结果，记录每个 clip 的裁剪情况和最终产物路径
- `final_video.mp4`
  - 合并后的短视频
- `subtitles.srt`
  - 根据新视频时间轴生成的字幕文件
- `final_video_burned.mp4`
  - 烧录字幕后的视频
- `review_report.json`
  - 当前阶段仍为占位审核报告
- `artifact_manifest.json`
  - 任务产物清单
- `task_result.json`
  - 对外返回结果的落盘版本

---

## 10. 当前工作流

当前主工作流由 [clippilot/core/workflow.py](./clippilot/core/workflow.py) 串联，阶段如下：

1. 创建任务上下文
2. 保存上传视频
3. 提取视频元信息
4. 校验格式和时长
5. 执行 ASR 转写
6. 生成高光候选
7. 生成真实剪辑时间轴
8. 执行真实视频裁剪、合并、字幕生成、可选烧录
9. 生成审核报告
10. 保存任务结果、产物清单和 trace

当前状态与阶段信息会同步写入：

- `task_result.json`
- `artifact_manifest.json`
- `workflow_trace.jsonl`

---

## 11. 如何测试当前功能

### 11.1 Swagger 页面

启动服务后打开：

```text
http://127.0.0.1:8000/docs
```

可以直接测试：

- `GET /health`
- `GET /api/v1/tasks`
- `GET /api/v1/tasks/{task_id}`
- `POST /api/v1/tasks/upload`

### 11.2 推荐验收顺序

1. 先测 `/health`
2. 上传一个 3 到 10 分钟视频
3. 记录返回的 `task_id`
4. 查看 `outputs/tasks/{task_id}/`
5. 确认以下文件存在：
   - `metadata/video_info.json`
   - `transcript/transcript.json`
   - `highlights/candidates.json`
   - `plan/editing_plan.json`
   - `plan/execution_report.json`
   - `final/final_video.mp4`
   - `final/subtitles.srt`
6. 如果开启字幕烧录，再检查：
   - `final/final_video_burned.mp4`

### 11.3 推荐异常测试

- 上传非视频文件
- 上传时长小于 180 秒的视频
- 上传时长大于 600 秒的视频
- 在没有 ffmpeg 的环境下执行真实剪辑

---

## 12. 自动化测试

运行全部测试：

```powershell
python -m pytest tests -q
```

当前测试已经覆盖这些核心能力：

- 视频信息提取
- ASR 转写
- 高光候选生成
- 剪辑计划生成
- 视频裁剪
- 视频合并
- 字幕生成与烧录
- 执行器
- TaskStorage
- 任务查询
- 任务列表
- validators

说明：

- 如果测试环境没有 `ffmpeg`，依赖 ffmpeg 的集成测试会自动跳过
- 不会因为系统缺少 ffmpeg 而导致整个 pytest 崩溃

---

## 13. 当前限制

虽然系统已经打通了“上传 -> 转写 -> 规划 -> 执行 -> 产物保存”的闭环，但当前仍属于 MVP。

当前主要限制包括：

- 默认 ASR 仍然是 `mock`
- 高光候选仍是规则法，不是语义理解模型
- `review_agent` 仍是占位实现
- `revision_agent` 尚未接入工作流
- `rag` 层尚未真正接入 Planner / Review
- 标题文案生成尚未实现
- 更复杂的镜头理解和质量检查尚未实现

---

## 14. 下一阶段建议

建议后续优先按以下顺序推进：

1. 接入真实 Whisper 或 faster-whisper
2. 将高光识别从规则法升级为语义理解或 LLM/Agent 辅助规划
3. 增加更细粒度的 review / quality check
4. 接入 RAG 的平台规则与剪辑模板
5. 接入 revision agent，形成自动修订闭环
6. 增加标题、封面文案和平台适配输出

---

## 15. 建议阅读顺序

如果你要继续开发，推荐按这个顺序看代码：

1. [ARCHITECTURE.md](./ARCHITECTURE.md)
2. [clippilot/core/workflow.py](./clippilot/core/workflow.py)
3. [clippilot/storage/task_storage.py](./clippilot/storage/task_storage.py)
4. [clippilot/agents/planner_agent.py](./clippilot/agents/planner_agent.py)
5. [clippilot/agents/executor_agent.py](./clippilot/agents/executor_agent.py)
6. `schemas/` 和 `tools/` 下对应模块
