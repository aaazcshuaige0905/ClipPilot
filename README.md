# ClipPilot

`ClipPilot` 是一个面向短视频高光剪辑场景的 Agentic Workflow 项目。

它当前的核心价值不是“已经接好了最强模型”，而是先把一条 **可执行、可观测、可修订、可继续演进** 的视频工作流搭稳。你可以把它理解为：

- 一个能真实跑完上传、转写、规划、裁剪、合并、字幕、审核的本地工作流
- 一个正在从规则驱动 MVP 向多 Agent 视频编排系统演进的代码基座
- 一个强调中间产物落盘、任务可回放、后续可接真实模型的工程骨架

---

## 1. 先看结论

截至 `2026-07-04`，这个项目处于 **晚期 MVP / 内测 Alpha** 阶段。

它已经完成了这些关键事情：

- 能通过 FastAPI 接口上传视频并创建任务
- 能做视频元信息提取、音频抽取、ASR 转写
- 能生成高光候选、时间线式剪辑计划、最终视频与字幕
- 能输出 `execution_report`、`review_report`、`project_state`、`artifact_manifest`
- 已经接入了 `video_understanding`、`planner_memory`、`RAG`、`revision` 这些二阶段能力的骨架

它还没有完全完成这些事情：

- 默认 ASR 仍是 `mock`
- `video_understanding` 目前是 transcript-backed stub，而不是真实多模态推理
- `planner` 已准备好面向 Qwen 的请求契约，但当前仍是本地规则 + stub candidate 组合
- `revise` 只重生成 plan，不会自动重新执行 render 和 review
- 部分测试仍停留在旧的 `EditingClip` 结构，需要迁移到新的 `TimelineItem`

如果你准备接手项目，最重要的判断是：

**这不是从零开始的 demo，而是一套已经跑通主链路、正在向更完整 Agent 系统升级的工程。**

---

## 2. 项目想解决什么问题

给定一段 `3` 到 `10` 分钟的视频，系统希望根据用户目标：

- 平台：如 `bilibili`、`douyin`
- 时长：如 `30s`
- 风格：如 `powerful`
- 语言：如 `zh`
- 是否烧录字幕

自动完成一轮短视频高光剪辑，产出：

- 可执行的剪辑时间线
- 合并后的视频
- 字幕文件
- 可选烧录字幕版本
- 结构化审核结果
- 完整中间产物，便于回放和二次修订

---

## 3. 你应该先读什么

推荐新同学按这个顺序进入项目：

1. 本文档：先建立整体认知
2. [ARCHITECTURE.md](./ARCHITECTURE.md)：看清模块边界和数据流
3. [clippilot/core/workflow.py](./clippilot/core/workflow.py)：理解主流程实际怎么串起来
4. [clippilot/agents/planner_agent.py](./clippilot/agents/planner_agent.py)：理解规划逻辑
5. [clippilot/agents/executor_agent.py](./clippilot/agents/executor_agent.py)：理解执行逻辑
6. `schemas/`：理解数据契约
7. `storage/`：理解任务目录和产物落盘

如果你只打算花 10 分钟快速上手，只看：

- `README.md`
- `ARCHITECTURE.md`
- `clippilot/core/workflow.py`

---

## 4. 当前端到端流程

一次任务当前大致会经历下面这条链路：

1. API 接收上传文件与表单参数
2. `workflow` 创建 `task_id`、任务目录、`TaskContext`
3. 保存原视频到 `outputs/tasks/{task_id}/input/`
4. 提取视频元信息并做时长/格式校验
5. 如有音轨则抽取音频
6. 执行 ASR，生成 `transcript.json`
7. 检索 RAG 上下文，注入平台规则/字幕规则/剪辑模板
8. 基于 transcript 构建 fine-grained units
9. 执行 `video_understanding`，生成 timeline 与 LLM-style highlight candidates
10. 生成规则式高光候选，作为稳定 baseline / fallback
11. `planner_agent` 基于候选、timeline、RAG、planner memory 生成 `editing_plan.json`
12. `executor_agent` 按 `timeline_items` 裁剪、拼接、生成字幕、可选烧录
13. `review_agent` 对输出文件、时长、字幕长度、执行错误等做结构化检查
14. 保存 `task_result`、`project_state`、`artifact_manifest`、`trace`

这个流程最重要的设计特点有两个：

- 每个阶段都尽量落盘，而不是只在内存里传
- 每个阶段都尽量输出结构化产物，而不是依赖隐式状态

---

## 5. 当前真实能力边界

### 已经稳定存在的能力

- 上传视频并创建独立任务目录
- 读取视频元信息并做基础校验
- 音频抽取
- `mock` / `whisper` / `faster-whisper` 三种 ASR 路径的统一接口
- 规则式高光候选生成
- 基于 `beats + timeline_items` 的新式剪辑计划
- 视频裁剪、合并、字幕生成、字幕烧录
- 任务查询、任务列表
- review 报告、trace 日志、artifact manifest
- planner short-term memory
- revision API
- RAG 检索入口和知识库目录

### 还属于“骨架先行”的能力

- 真实视频理解模型调用
- 真实 LLM 规划调用
- 自动 revision 闭环
- 更强的质量评估指标
- 标题/封面/发布文案生成

---

## 6. 项目结构

```text
ClipPilot/
|-- main.py
|-- config.yaml
|-- requirements.txt
|-- README.md
|-- ARCHITECTURE.md
|-- clippilot/
|   |-- api/            # FastAPI 接口
|   |-- core/           # 工作流编排、上下文、状态、memory builder
|   |-- schemas/        # Pydantic 数据契约
|   |-- tools/          # 原子视频/字幕/转写能力
|   |-- storage/        # 路径管理、任务目录、JSON 落盘
|   |-- harness/        # validators、trace、eval 预留
|   |-- agents/         # planner / executor / review / revision / video understanding
|   `-- rag/            # 检索、rerank、fusion、provider、knowledge
|-- data/
|   |-- raw_videos/
|   `-- rag/
|-- outputs/
|   `-- tasks/
`-- tests/
```

你可以用一句话记住：

- `api` 进请求
- `core` 串流程
- `schemas` 定契约
- `tools` 干原子活
- `agents` 做高层决策
- `storage` 管路径和产物
- `rag` 提供外部规则上下文

---

## 7. 一次任务会产出什么

每个任务都有自己的目录：

```text
outputs/tasks/{task_id}/
|-- input/
|-- audio/
|-- metadata/
|-- transcript/
|-- understanding/
|-- highlights/
|-- clips/
|-- final/
|-- plan/
|   `-- versions/
|-- review/
|-- trace/
|-- project_state.json
|-- task_result.json
`-- artifact_manifest.json
```

最常用的产物包括：

- `metadata/video_info.json`
- `transcript/transcript.json`
- `understanding/timeline.json`
- `understanding/retrieved_context.json`
- `highlights/candidates.json`
- `plan/editing_plan.json`
- `plan/planner_memory.json`
- `plan/execution_report.json`
- `review/review_report.json`
- `final/final_video.mp4`
- `final/subtitles.srt`
- `final/final_video_burned.mp4`
- `project_state.json`

对于开发者来说，这个目录布局非常关键，因为它意味着：

- 可以直接回放一个任务的所有阶段
- 可以脱离 API 排查 planner / executor / review
- 可以给未来前端、运营台、评测系统直接复用

---

## 8. 快速启动

### 8.1 安装依赖

```powershell
pip install -r requirements.txt
```

### 8.2 安装 `ffmpeg`

以下能力依赖系统可用的 `ffmpeg`：

- 音频抽取
- 视频裁剪
- 视频合并
- 字幕烧录

如果没有 `ffmpeg`，主链路中的媒体执行阶段会失败或退化。

### 8.3 启动服务

```powershell
uvicorn main:app --reload
```

默认地址：

- `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`

---

## 9. API 一览

当前接口包括：

- `GET /health`
- `GET /api/v1/tasks`
- `GET /api/v1/tasks/{task_id}`
- `POST /api/v1/tasks/upload`
- `POST /api/v1/tasks/{task_id}/revise`

### 上传任务示例

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/tasks/upload" `
  -F "file=@sample.mp4" `
  -F "target_platform=bilibili" `
  -F "target_duration=30" `
  -F "edit_style=powerful" `
  -F "language=zh" `
  -F "need_burn_subtitle=true"
```

### revision 示例

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/tasks/<task_id>/revise" `
  -H "Content-Type: application/json" `
  -d "{\"feedback_text\": \"把开头更有 hook 一点，控制在 25 秒\"}"
```

注意：当前 `revise` 只会重生成 `editing_plan` 和更新任务状态，不会自动重新执行渲染。

---

## 10. 配置说明

配置来自 `config.yaml` 与环境变量。

当前比较关键的配置包括：

- 任务输出目录
- 原始视频目录
- 视频格式和时长限制
- ASR provider
- Whisper 模型名
- RAG 开关与 top-k
- Qwen embedding 相关配置

常见环境变量：

```powershell
$env:CLIP_PILOT_ASR_PROVIDER="mock"
$env:CLIP_PILOT_WHISPER_MODEL="base"
$env:CLIP_PILOT_QWEN_API_KEY="..."
```

---

## 11. 测试与当前状态说明

运行测试：

```powershell
python -m pytest tests -q
```

当前仓库里已经有较完整的测试目录，但需要注意一件事：

- 部分测试仍然引用旧的 `EditingClip` schema
- 当前代码主干已经切换到 `TimelineItem`

这意味着项目已经进入一次明显的架构升级期，测试需要继续追平代码结构。接手项目时，建议把“测试迁移”列为优先事项之一。

---

## 12. 这套架构为什么值得继续做

这个项目目前最好的地方，不是模型效果，而是工程方向已经比较明确：

- 主流程是完整的
- 数据契约是结构化的
- 每个任务是可追踪的
- 重要产物是可回放的
- Agent、RAG、memory、revision 都已经预留并部分接通

这会让后续几件事都比较自然：

- 接真实 ASR
- 接真实视频理解
- 接真实 LLM planner
- 做自动 revision 闭环
- 做质量评测与人审台

---

## 13. 接下来最建议怎么推进

如果你是这个项目的下一位主要开发者，我建议按下面顺序推进：

1. 先把 README、ARCHITECTURE、测试与当前代码结构对齐
2. 把旧 `EditingClip` 测试迁移到 `TimelineItem`
3. 打通 `revise -> execute -> review` 闭环
4. 接真实 `faster-whisper`
5. 接通真实 `video_understanding` 和 planner LLM 调用
6. 建立质量评估集和可回归指标
7. 再扩展标题、封面、平台文案等产物

简化成一句话：

**先稳住工程，再增强智能。**

---

## 14. 相关阅读

- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [clippilot/core/workflow.py](./clippilot/core/workflow.py)
- [clippilot/agents/planner_agent.py](./clippilot/agents/planner_agent.py)
- [clippilot/agents/executor_agent.py](./clippilot/agents/executor_agent.py)
- [clippilot/agents/review_agent.py](./clippilot/agents/review_agent.py)
