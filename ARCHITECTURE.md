# ClipPilot 架构说明

## 1. 文档目标

本文档用于说明 `ClipPilot` 当前版本的整体架构、模块边界、数据流、任务目录结构和扩展方向。

当前项目已经从“只有上传接口的 MVP 骨架”升级为一条可真实执行的视频工作流，能够完成：

1. 视频上传与任务创建
2. 视频元信息提取
3. ASR 转写
4. 高光候选识别
5. 可执行剪辑时间轴规划
6. 视频裁剪与合并
7. 字幕文件生成
8. 可选字幕烧录
9. 执行报告与任务产物持久化

项目当前仍然是 Stage 1，但已经具备了后续扩展为 Agentic 多阶段系统的基础边界。

---

## 2. 架构设计原则

### 2.1 单一职责

每一层只做一类事情：

- `api` 只负责 HTTP 请求入口
- `core` 只负责工作流编排
- `schemas` 只负责数据契约
- `tools` 只负责原子能力
- `storage` 只负责路径和持久化
- `harness` 只负责校验和追踪
- `agents` 负责基于结构化产物做规划、执行、审核
- `rag` 预留给未来检索增强

### 2.2 数据契约优先

跨模块数据优先通过 `Pydantic` Schema 传递，避免业务主链路里到处传散乱 `dict`。

### 2.3 每个任务独立目录

每次上传都会创建唯一 `task_id`，所有产物都落到：

```text
outputs/tasks/{task_id}/
```

这样做的价值是：

- 调试更方便
- 中间结果可回放
- 便于前端展示
- 便于未来多个 Agent 通过文件产物协作

### 2.4 渐进扩展而不是推倒重来

当前版本虽然还没接入真正的 Planner/Review/Revision/RAG 闭环，但架构上已经预留接口和目录位置，后续扩展时不需要重写整套工程。

---

## 3. 当前能力边界

### 已实现能力

- 上传 `.mp4` / `.mov` / `.mkv`
- 接收剪辑请求参数
- 使用 `moviepy` 读取视频元信息
- 使用 `mock` ASR 跑通完整转写流程
- 基于规则法生成高光候选
- 生成真实可执行的 `EditingPlan`
- 按时间轴裁剪并合并视频
- 生成 `.srt` 字幕
- 可选烧录字幕视频
- 生成 `ExecutionReport`
- 生成 `TaskResult`
- 提供任务查询和任务列表接口

### 尚未完全实现的能力

- 真实 Whisper / faster-whisper 转写
- 更高级的视频理解和镜头分析
- Review Agent 的真实质量检查
- Revision Agent 自动修订
- RAG 对平台规则和剪辑模板的动态注入
- 标题文案、封面文案、平台适配导出

---

## 4. 项目目录结构

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
|   |-- __init__.py
|   |-- api/
|   |   |-- __init__.py
|   |   `-- app.py
|   |-- core/
|   |   |-- __init__.py
|   |   |-- exceptions.py
|   |   |-- states.py
|   |   |-- task_context.py
|   |   `-- workflow.py
|   |-- schemas/
|   |   |-- __init__.py
|   |   |-- agent_trace.py
|   |   |-- editing_plan.py
|   |   |-- execution_report.py
|   |   |-- review_report.py
|   |   |-- task_result.py
|   |   |-- transcript.py
|   |   |-- user_request.py
|   |   `-- video_info.py
|   |-- tools/
|   |   |-- __init__.py
|   |   |-- audio_extract.py
|   |   |-- export.py
|   |   |-- highlight.py
|   |   |-- scene_detect.py
|   |   |-- subtitle.py
|   |   |-- transcribe.py
|   |   |-- video_cut.py
|   |   |-- video_info.py
|   |   `-- video_merge.py
|   |-- storage/
|   |   |-- __init__.py
|   |   |-- json_io.py
|   |   |-- path_manager.py
|   |   `-- task_storage.py
|   |-- harness/
|   |   |-- __init__.py
|   |   |-- eval_metrics.py
|   |   |-- trace_logger.py
|   |   `-- validators.py
|   |-- agents/
|   |   |-- __init__.py
|   |   |-- executor_agent.py
|   |   |-- planner_agent.py
|   |   |-- review_agent.py
|   |   |-- revision_agent.py
|   |   `-- video_understanding_agent.py
|   `-- rag/
|       |-- __init__.py
|       |-- build_index.py
|       |-- retrieve.py
|       `-- knowledge/
|           |-- editing_templates.md
|           |-- platform_rules.md
|           |-- subtitle_rules.md
|           `-- title_templates.md
|-- data/
|   `-- raw_videos/
|-- outputs/
|   `-- tasks/
`-- tests/
```

---

## 5. 模块边界说明

## 5.1 `clippilot/api/`

### 职责

`api` 层是整个系统的 HTTP 边界。

当前核心文件：

- `app.py`

当前提供的接口：

- `GET /health`
- `GET /api/v1/tasks`
- `GET /api/v1/tasks/{task_id}`
- `POST /api/v1/tasks/upload`

### 应该做什么

- 接收上传文件和表单字段
- 将输入组装为 `UserRequest`
- 调用 `run_stage1_workflow()`
- 将领域异常映射为 HTTP 错误

### 不应该做什么

- 不直接做视频处理
- 不直接实现 ASR
- 不直接拼装复杂文件路径
- 不在路由里写长业务流程

---

## 5.2 `clippilot/core/`

`core` 是业务编排层。

### `workflow.py`

负责串联主流程：

1. 创建任务上下文
2. 保存源视频
3. 提取视频元信息
4. 执行 ASR
5. 生成高光候选
6. 生成剪辑时间轴
7. 执行真实视频裁剪、合并、字幕生成、烧录
8. 生成审核报告
9. 保存任务结果和产物清单

### `task_context.py`

负责保存运行中的任务上下文，典型字段包括：

- `task_id`
- `request`
- `paths`
- `status`
- `stage`
- `artifacts`
- `traces`

### `states.py`

统一维护任务状态和阶段名，例如：

- `pending`
- `processing`
- `completed`
- `failed`

以及：

- `uploaded`
- `transcribed`
- `editing_plan_generated`
- `execution_report_generated`

### `exceptions.py`

统一定义领域异常：

- `ClipPilotValidationError`
- `ClipPilotProcessingError`
- `ClipPilotStorageError`

### 核心边界

应该做什么：

- 编排阶段顺序
- 控制状态推进
- 调度 tools 和 agents

不应该做什么：

- 不直接实现媒体底层处理逻辑
- 不直接依赖 FastAPI

---

## 5.3 `clippilot/schemas/`

`schemas` 负责定义结构化数据契约。

### 关键模型

- `UserRequest`
- `VideoInfo`
- `TranscriptResult`
- `HighlightCandidate`
- `EditingClip`
- `EditingPlan`
- `ExecutionClipResult`
- `ExecutionReport`
- `TaskResult`

### 当前重点 Schema

#### `EditingPlan`

当前已经是“可执行时间轴”而不是占位结构，至少包含：

- `task_id`
- `target_duration`
- `total_duration`
- `clips`
- `editing_notes`
- `warnings`

每个 `clip` 至少包含：

- `clip_id`
- `source_start`
- `source_end`
- `duration`
- `purpose`
- `text`
- `subtitle`
- `score`
- `reason`

#### `ExecutionReport`

当前记录真实执行结果，至少包含：

- `task_id`
- `status`
- `clip_results`
- `final_video_path`
- `subtitle_path`
- `burned_video_path`
- `warnings`
- `errors`

### 边界要求

应该做什么：

- 定义字段和类型
- 做结构级校验

不应该做什么：

- 不实现业务流程
- 不做文件读写
- 不做媒体处理

---

## 5.4 `clippilot/tools/`

`tools` 是原子能力层，强调“可复用、可单测、输入输出清晰”。

### 已实现的关键工具

#### `video_info.py`

负责：

- 校验视频扩展名
- 提取视频元信息
- 校验时长区间

#### `transcribe.py`

负责：

- 选择 ASR provider
- 输出 `TranscriptResult`

#### `highlight.py`

负责：

- 合并 transcript 片段
- 规则法打分
- 输出候选高光

#### `video_cut.py`

负责：

- 使用 `ffmpeg subprocess` 裁剪单个 clip
- 校验 `start_time` / `end_time`
- 返回结构化 `VideoCutResult`

#### `video_merge.py`

负责：

- 使用 `ffmpeg concat demuxer` 合并多个 clip
- 自动生成临时 `concat_list.txt`
- 返回结构化 `VideoMergeResult`

#### `subtitle.py`

负责：

- 根据 `EditingPlan` 生成新视频时间轴下的 `.srt`
- 提供 `seconds_to_srt_time()`
- 使用 `ffmpeg` 烧录字幕

### `tools` 边界要求

应该做什么：

- 实现明确的媒体处理步骤
- 接收确定输入并给出确定输出

不应该做什么：

- 不依赖 FastAPI
- 不写死任务目录
- 不管理整条任务生命周期

---

## 5.5 `clippilot/storage/`

`storage` 负责持久化与路径规范。

### `path_manager.py`

统一负责：

- 加载 `config.yaml`
- 构建任务路径

当前每个任务除了基础目录外，还会生成：

- `clips_dir`
- `final_dir`
- `final_video_path`
- `subtitle_path`
- `burned_video_path`

### `task_storage.py`

负责：

- 生成 `task_id`
- 创建任务目录
- 保存源视频
- 保存 JSON 产物
- 读取任务结果和产物清单

### `json_io.py`

负责最底层 JSON 读写。

### 边界要求

应该做什么：

- 统一路径规则
- 统一 JSON 落盘
- 统一任务目录创建

不应该做什么：

- 不写业务规则
- 不直接控制 workflow 顺序

---

## 5.6 `clippilot/harness/`

`harness` 负责校验和可观测性。

### `validators.py`

当前用于：

- transcript 非空校验
- candidates 非空校验
- 输出文件存在校验

### `trace_logger.py`

负责记录结构化流程日志，当前每个任务会保存：

- `workflow_trace.jsonl`

### `eval_metrics.py`

当前为预留模块，用于后续引入质量评估指标。

---

## 5.7 `clippilot/agents/`

这是当前最关键的智能工作流层。

### `planner_agent.py`

当前已升级为真实剪辑时间轴规划器，负责：

- 按 score 选候选
- 优先选 hook 开头
- 控制总时长不超过目标 + 5 秒
- 去重重复文本
- 候选不足时从 transcript 补片段

### `executor_agent.py`

当前已升级为真实视频执行器，负责：

1. 根据 `editing_plan.clips` 逐段裁剪
2. 保存 clip 到 `outputs/tasks/{task_id}/clips/`
3. 合并为 `final/final_video.mp4`
4. 生成 `final/subtitles.srt`
5. 可选生成 `final/final_video_burned.mp4`
6. 输出 `ExecutionReport`

### `review_agent.py`

当前仍是占位审核器，后续可升级为：

- 输出文件存在性检查
- 时长和目标时长偏差检查
- 字幕一致性检查
- 平台规则检查

### `revision_agent.py`

当前仅预留模块，后续可基于 `ReviewReport` 自动修订 `EditingPlan`。

---

## 5.8 `clippilot/rag/`

当前尚未接入主流程，但已预留目录和知识文件，用于未来给 Planner / Review 提供：

- 平台规则
- 剪辑模板
- 字幕规则
- 标题模板

---

## 6. 当前主数据流

当前系统的核心数据流如下：

1. 用户提交上传请求
2. `api` 组装 `UserRequest`
3. `workflow` 创建 `TaskContext`
4. `storage` 保存源视频
5. `tools.video_info` 输出 `VideoInfo`
6. `tools.transcribe` 输出 `TranscriptResult`
7. `tools.highlight` 输出 `HighlightCandidatesResult`
8. `agents.planner_agent` 输出 `EditingPlan`
9. `agents.executor_agent` 输出 `ExecutionReport`
10. `agents.review_agent` 输出 `ReviewReport`
11. `storage` 保存全部产物
12. `api` 返回 `TaskResult`

可以概括为：

`API 接请求 -> Workflow 编排 -> Tools / Agents 执行 -> Storage 落盘 -> API 返回结果`

---

## 7. 当前任务目录与产物

每个任务的主产物目录是：

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

### 关键产物说明

- `editing_plan.json`
  - 真实剪辑时间轴
- `execution_report.json`
  - 真实执行报告，包含每段裁剪结果和最终视频产物路径
- `subtitles.srt`
  - 按新视频时间轴生成的字幕文件
- `final_video.mp4`
  - 最终合并视频
- `final_video_burned.mp4`
  - 烧录字幕后的视频

---

## 8. 当前状态机

当前 workflow 阶段包括：

1. `created`
2. `uploaded`
3. `video_info_extracted`
4. `transcribed`
5. `highlight_candidates_generated`
6. `editing_plan_generated`
7. `execution_report_generated`
8. `review_report_generated`
9. `completed`

任务状态包括：

- `pending`
- `processing`
- `completed`
- `failed`

说明：

- 如果执行阶段真正失败，任务最终状态可能是 `failed`
- 即使失败，也会尽量保留可用的中间产物和错误信息

---

## 9. 配置策略

当前配置来源于：

- `config.yaml`
- 环境变量

当前关键配置包括：

- 应用基本信息
- 任务主目录
- 原始测试视频目录
- 视频格式限制
- 视频时长限制
- ASR provider
- Whisper 模型名
- 高光候选时长范围

配置原则：

- 路径不要硬编码进业务逻辑
- 模型名不要散落在多个工具函数中
- 将来 API Key 也应该从环境变量或配置文件读取

---

## 10. 错误处理策略

当前错误类型包括：

- `ClipPilotValidationError`
  - 输入非法、时长不符合要求、候选为空等
- `ClipPilotProcessingError`
  - 视频裁剪、合并、执行过程中的处理错误
- `ClipPilotStorageError`
  - 产物保存或读取失败

处理原则：

- tools / agents / workflow 抛领域异常或结构化错误
- API 层统一转换为 HTTP 响应
- 执行器尽量将失败写入 `ExecutionReport.errors`，而不是直接让整个流程崩掉

---

## 11. 当前已稳定的能力边界

当前已经相对稳定的边界包括：

- 上传与任务创建
- 视频元信息提取
- ASR 抽象
- 高光候选生成
- 可执行剪辑计划生成
- 真实视频执行
- 字幕生成与烧录
- 任务结果查询
- 任务目录与产物管理

---

## 12. 后续建议扩展方向

下一阶段建议优先推进：

1. 接入真实 Whisper / faster-whisper
2. 将高光候选与剪辑计划升级为更强的语义规划
3. 升级 `review_agent` 为真实质量检查器
4. 接入 `rag/retrieve.py` 给 Planner / Review 提供规则知识
5. 接入 `revision_agent` 做自动修订闭环
6. 增加标题、封面文案、平台适配导出

---

## 13. 一句话总结

当前 `ClipPilot` 的架构可以概括为：

- `api` 负责接请求
- `core` 负责编排
- `schemas` 负责定义结构
- `tools` 负责原子媒体处理
- `agents` 负责规划、执行、审核
- `storage` 负责任务目录和产物持久化
- `harness` 负责校验和追踪
- `rag` 为下一阶段智能增强预留入口

这套架构的核心价值是：

**当前功能已经可以真实跑通，而且未来继续增强时不需要推倒重来。**
