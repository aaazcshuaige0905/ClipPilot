# ClipPilot 架构说明

## 1. 这份文档是给谁看的

这份文档主要给三类人：

- 新接手项目的工程师
- 需要扩展某个模块的开发者
- 需要判断“下一步该往哪里演进”的负责人

阅读目标不是记住每个函数，而是快速回答这几个问题：

- 项目现在处于什么阶段
- 一次任务到底怎么流转
- 每个目录应该放什么
- 哪些能力已经真实实现，哪些还只是骨架
- 如果我要新增一个 provider / agent / 评测指标，应该接在哪里

---

## 2. 当前架构定位

`ClipPilot` 当前不是一个“只有 API 外壳的 demo”，也不是一个已经产品化的剪辑平台。

它更准确的定位是：

**一套已经打通主链路的 Agentic 视频工作流骨架，正在从规则驱动 MVP 向多 Agent、可修订、可检索增强的系统升级。**

当前主链路已经包括：

1. 上传与任务创建
2. 视频元信息提取
3. 音频抽取
4. ASR 转写
5. RAG 上下文检索
6. transcript-based 视频理解
7. 高光候选生成
8. 时间线规划
9. 媒体执行
10. 输出审核
11. 任务产物持久化

---

## 3. 架构设计原则

### 3.1 工作流先于模型

项目优先保证：

- 流程能跑
- 任务可回放
- 中间产物可调试
- 数据契约清晰

模型能力是后续增强点，而不是第一优先级。

### 3.2 数据契约优先

跨层传递尽量使用 `Pydantic` schema，而不是散乱的 `dict`。这样做的好处是：

- 模块边界清晰
- 序列化和落盘简单
- 测试更容易做
- 后续前端和评测系统更容易接

### 3.3 每个任务都是一个独立工作区

每次上传都会生成一个 `task_id`，并在：

```text
outputs/tasks/{task_id}/
```

下保存完整产物。这样天然支持：

- 调试
- 断点排查
- 回放任务过程
- revision
- 人审和评测

### 3.4 渐进式升级

很多能力当前虽然还是 stub，但已经不是“以后重写”的占位逻辑，而是“接口与上下文已经先接入”的可升级结构。典型例子：

- `video_understanding_agent`
- `planner_memory`
- `retrieved_context`
- `revision_agent`

---

## 4. 系统总览

可以把当前系统看成 7 层：

1. `api`
2. `core`
3. `agents`
4. `tools`
5. `schemas`
6. `storage`
7. `rag`

它们的关系是：

```text
HTTP Request
   -> api
   -> core.workflow
   -> agents + tools
   -> storage
   -> HTTP Response
```

更细一点可以理解为：

```text
API 负责进入系统
Core 负责组织阶段
Agents 负责高层决策
Tools 负责原子媒体处理
Schemas 负责数据形状
Storage 负责目录与落盘
RAG 负责规则上下文
```

---

## 5. 一次任务如何流转

当前主流程的真实入口是 `clippilot/core/workflow.py`。

### 5.1 阶段顺序

一次上传任务当前按下面顺序执行：

1. 创建 `TaskContext`
2. 保存原视频
3. 提取视频元信息并校验
4. 抽取音频
5. 执行转写
6. 检索规则上下文
7. 执行视频理解
8. 生成规则式高光候选
9. 生成 `EditingPlan`
10. 执行 `EditingPlan`
11. 生成 `ReviewReport`
12. 写入任务最终状态与产物清单

### 5.2 阶段之间靠什么传数据

主要靠三类东西：

- `TaskContext`
- `ProjectState`
- 各种结构化 artifact

这三者的分工是：

- `TaskContext`：运行态上下文，负责状态、trace、artifact 注册
- `ProjectState`：任务级共享状态，负责让后续 agent 看到前序分析结果
- Artifact 文件：负责调试、回放、跨阶段持久化

### 5.3 为什么同时要有内存态和落盘态

因为这个项目不是只求“这次跑完”，而是要支持后续这些需求：

- 任务复查
- revision 重规划
- 人工审核
- 评测回放
- 前端展示

所以当前架构有意识地保留了很多中间产物。

---

## 6. 关键数据对象

### 6.1 `UserRequest`

代表用户意图，包括：

- `target_platform`
- `target_duration`
- `edit_style`
- `language`
- `need_burn_subtitle`

这是整个任务的最初约束来源。

### 6.2 `VideoInfo`

代表源视频元信息，包括：

- 时长
- 分辨率
- fps
- 是否有音轨
- 文件大小

它既用于校验，也用于 review。

### 6.3 `TranscriptResult`

代表 ASR 输出，包括：

- 分段 transcript
- 完整文本
- provider

它是后续几乎所有规划逻辑的底座。

### 6.4 `FineGrainedUnit`

这是当前架构升级里很重要的一层。

它不是原始 transcript segment，而是更适合编辑的细粒度单元，带有：

- 前后停顿信息
- 关键词
- emphasis score
- 可作为 cut boundary 的语义信息

它的作用是把“ASR 文本”进一步转成“可剪辑单位”。

### 6.5 `VideoTimeline`

代表 `video_understanding_agent` 产出的粗粒度语义时间线。

当前它还是 transcript-backed stub，但接口形状已经是为未来真实多模态理解准备的。

### 6.6 `HighlightCandidatesResult`

这是规则法生成的 baseline 候选池。它仍然重要，因为：

- 可作为 planner fallback
- 可对比未来 LLM candidate 的收益
- 在模型不可用时可保证链路继续跑

### 6.7 `EditingPlan`

当前的 `EditingPlan` 已不是旧版的简单 clip list，而是更接近“可执行时间线”：

- `beats`
- `timeline_items`
- `editing_notes`
- `warnings`
- `strategy`
- `generation_mode`
- `plan_version`

这是当前系统最核心的计划产物。

### 6.8 `ExecutionReport`

代表 executor 的真实执行结果，包括：

- 每个 timeline item 的执行情况
- 最终视频路径
- 字幕路径
- 烧录版本路径
- warnings
- errors

### 6.9 `ReviewReport`

代表输出质量检查结果。它现在已经不是空壳，会检查：

- 最终视频是否存在
- 文件是否非空
- 时长是否接近目标
- 字幕文件是否存在
- timeline item 时间是否合法
- 字幕长度是否合理
- 执行是否报错

---

## 7. 模块边界

## 7.1 `clippilot/api/`

### 职责

- 接受 HTTP 请求
- 解析上传文件和表单字段
- 组装 `UserRequest`
- 调用工作流或 revision
- 把领域错误转成 HTTP 错误

### 不负责

- 不直接处理视频
- 不自己拼接任务目录
- 不写业务编排逻辑

### 当前接口

- `GET /health`
- `GET /api/v1/tasks`
- `GET /api/v1/tasks/{task_id}`
- `POST /api/v1/tasks/upload`
- `POST /api/v1/tasks/{task_id}/revise`

---

## 7.2 `clippilot/core/`

这是整个系统的编排层。

### 核心职责

- 决定阶段顺序
- 维护状态机
- 连接 agents 和 tools
- 维护任务级上下文
- 负责最终状态落盘

### 关键文件

- `workflow.py`
- `task_context.py`
- `states.py`
- `exceptions.py`
- `memory_manager.py`
- `context_builder.py`

### 设计判断

如果你想改“任务整体怎么跑”，优先看 `core`。
如果你想改“某一步具体怎么做”，优先看 `agents` 或 `tools`。

---

## 7.3 `clippilot/agents/`

这里放“高层决策逻辑”，不是纯媒体原子能力。

### `video_understanding_agent.py`

职责：

- 生成 timeline
- 生成 LLM-style highlight candidates
- 准备未来真实视频理解模型的请求契约

当前状态：

- 已接入主流程
- 当前输出仍是 stub

### `planner_agent.py`

职责：

- 整合 timeline、候选、RAG、planner memory
- 生成 `EditingPlan`
- 保持目标时长、hook、ending、去重、candidate 选择等规则

当前状态：

- 已接入主流程
- 已从旧 clip list 升级为 `beats + timeline_items`
- 已准备未来真实 Qwen planner 请求结构

### `executor_agent.py`

职责：

- 把 `EditingPlan` 变成真实媒体产物
- 支持单段 cut 与 montage 组装
- 合并视频
- 生成字幕
- 可选烧录字幕

当前状态：

- 已真实执行
- 是最接近“生产动作”的模块

### `review_agent.py`

职责：

- 对最终产物进行结构化质量检查
- 输出 `passed / score / checks / suggestions`

当前状态：

- 已有真实检查逻辑
- 但还不算最终版质量体系

### `revision_agent.py`

职责：

- 根据用户反馈修改 planner memory
- 生成新的 `EditingPlan`
- 保留 plan version 历史

当前状态：

- 已有 API 与核心逻辑
- 目前只做到“重规划”，还没有自动重新执行 executor 和 review

---

## 7.4 `clippilot/tools/`

这里放“明确输入、明确输出”的原子能力。

典型模块：

- `video_info.py`
- `audio_extract.py`
- `transcribe.py`
- `highlight.py`
- `video_cut.py`
- `video_merge.py`
- `subtitle.py`

判断原则：

- 如果逻辑偏媒体处理、易复用、可单测，放 `tools`
- 如果逻辑偏策略决策、依赖多个上下文，放 `agents`

---

## 7.5 `clippilot/storage/`

这是任务工作区和 artifact 的管理层。

### 职责

- 统一路径生成
- 创建任务目录
- JSON 落盘
- 读取任务结果
- 列出历史任务

### 为什么它重要

这个项目很多“工程感”其实都来自 `storage`：

- 任务目录结构稳定
- artifact 路径统一
- revision 可以回到旧任务继续工作

---

## 7.6 `clippilot/schemas/`

这是系统的数据契约层。

它的目标是：

- 让跨模块交互更明确
- 让落盘与回读更稳定
- 让接口升级更可控

当前一次显著架构升级，就是从旧的 `EditingClip` 思路迁移到新的 `TimelineItem` 思路。也正因为 schema 变了，部分老测试需要一起迁移。

---

## 7.7 `clippilot/rag/`

RAG 当前的角色不是“主模型问答”，而是给 planner/review 提供策略上下文。

### 当前知识来源

- `platform_rules.md`
- `subtitle_rules.md`
- `editing_templates.md`
- `title_templates.md`

### 当前处理链

1. 读取 markdown 知识文件
2. chunking
3. BM25 检索
4. 可选 dense retrieval
5. metadata rerank
6. weighted fusion
7. 输出 `RetrievedContext`

### 当前状态

- 检索入口已接入 workflow
- 没有 dense index 也可以退化运行
- 更像策略增强层，而不是内容主生成层

---

## 8. 任务目录结构

当前单任务目录大致如下：

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

各目录职责：

- `input/`：原始上传视频
- `audio/`：抽取后的音频
- `metadata/`：视频元信息
- `transcript/`：ASR 结果
- `understanding/`：timeline、retrieved context、retrieval trace
- `highlights/`：规则法候选
- `clips/`：timeline item 渲染出的中间片段
- `final/`：最终视频与字幕
- `plan/`：editing plan、planner memory、execution report、versioned plan
- `review/`：review report
- `trace/`：工作流 trace

---

## 9. 状态、追踪与可观测性

当前系统的可观测性主要来自三部分：

### 9.1 `task_result.json`

给外部接口和任务查询使用，体现用户视角的任务结果。

### 9.2 `artifact_manifest.json`

记录当前任务有哪些产物、它们属于哪个阶段。

### 9.3 `workflow_trace.jsonl`

按阶段记录结构化 trace，用于回放和排查。

这三者结合起来，能比较清楚地回答：

- 当前任务做到哪一步了
- 哪一步失败了
- 失败前已经产出了哪些文件

---

## 10. 错误处理策略

当前错误主要分三类：

- `ClipPilotValidationError`
- `ClipPilotProcessingError`
- `ClipPilotStorageError`

整体原则是：

- 在内部抛领域错误
- 在 API 层统一转成 HTTP 错误
- 在 executor / review 尽量保留 warnings 和 errors，而不是一出错就让整个上下文丢失

这让失败任务也仍然具备排查价值。

---

## 11. 当前技术债和未完成闭环

以下是现在最真实的架构缺口：

### 11.1 测试体系落后于 schema 升级

当前代码已经切换到 `TimelineItem`，但部分测试仍然依赖旧的 `EditingClip`。

### 11.2 revision 不是完整闭环

现在只重生成 plan，没有自动触发后续 executor / review。

### 11.3 视频理解与 planner 仍是“接口先行”

当前 `video_understanding` 和 planner 的真实模型调用尚未接通，现阶段更像：

- 上下文结构已经准备好
- 真实调用只差 provider 落地

### 11.4 服务执行仍是同步链路

上传接口当前直接同步跑 workflow。对于更长任务、更高并发、更稳定的任务管理，这最终会需要后台队列或异步 job 模型。

---

## 12. 如何扩展这个项目

### 如果你要加新的 ASR provider

- 放到 `clippilot/tools/transcribe.py` 或其 provider 子模块
- 保持 `TranscriptResult` 输出不变
- 不要把 provider 细节散落到 workflow

### 如果你要加新的 planner 能力

- 优先改 `planner_agent.py`
- 需要的新上下文尽量先进入 `ProjectState`
- 不要让 executor 反向承担 planner 逻辑

### 如果你要加新的审核规则

- 优先改 `review_agent.py`
- 让每条规则都形成结构化 check
- 尽量输出 suggestion，方便未来 revision 自动消费

### 如果你要加新的知识库规则

- 放到 `clippilot/rag/knowledge/`
- 保持 chunking 与 metadata 可被检索策略理解

### 如果你要加新的任务产物

- 先在 `TaskPaths` 中定义路径
- 再在 `TaskStorage` 中补保存/读取入口
- 最后在 workflow 中注册 artifact

---

## 13. 下一阶段最合理的演进顺序

从架构角度看，最建议的推进顺序是：

1. 先把测试和文档追平当前 schema
2. 打通 `revise -> execute -> review`
3. 接入真实 `faster-whisper`
4. 接入真实视频理解 provider
5. 接入真实 planner LLM
6. 建立评测与质量回归
7. 再扩展标题、封面、发布文案等外围产物

原因很简单：

- 现在工程骨架已经有了
- 最缺的是稳定性和闭环
- 不是继续叠功能名词

---

## 14. 一句话总结

`ClipPilot` 当前的架构核心，不是“模型已经最强”，而是：

**工作流已经成型，任务边界清晰，数据契约稳定，下一阶段可以在不推倒重来的前提下持续增强智能能力。**
