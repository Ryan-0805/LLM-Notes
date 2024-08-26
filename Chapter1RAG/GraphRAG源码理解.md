# 微软GraphRAG源码理解

## 1.引言

2024年四月份，微软发布一篇文章《From Local to Global: A Graph RAG Approach to Query-Focused Summarization》，提出了graphRAG的方法，该方法很好地解决了典型RAG系统在聚焦于查询的总结性 (QFS) 任务上存在的问题，并且在生成答案的全面性和多样性方面，Graph RAG 比朴素的 RAG 基线有了显着的改进。

2024年六月份，微软开源了graphRAG的项目 https://github.com/microsoft/graphrag 

为RAG带来新的发展，因此本人想从论文的原理以及源码部分去剖析和解读，希望进一步理解其核心结构，以求能做出一些速度上的优化

## 2.探究GraphRAG原理

- 为什么要在传统的RAG上发展GraphRAG？什么是GraphRAG？

传统RAG在针对整个文本语料库的全局问题上回答不佳，比如：这个数据集的主要主题是什么？这种问题不是简单的检索任务，是聚焦于查询的总结性任务。

GraphRAG核心思想：首先从源文档中派生一个实体知识图谱（通过大模型提取实体、关系，社区检测提取社区），然后为所有密切相关的实体组预先生成社区摘要。给定一个问题，每个社区摘要都用于生成部分响应，然后再次将所有部分响应汇总到对用户的最终响应中

![entity](images/entity.png)

- graphRAG原理

  ![image-20240801160103956](images/image-20240801160103956.png)

graphRAG中的一些名词：

**文档（Documents)**：输入的文档，源码中要求是txt格式。

**文本单元（Text Unit)**:要分析的文本块。这些块的大小、它们的重叠以及它们是否遵循任何数据边界可以自行配置。一个常见的用例是将`CHUNK_BY_COLUMNS`设置为`id` ，以便文档和 TextUnit 之间存在一对多关系，而不是多对多关系

**实体（Entity）**：从 TextUnit 中提取的实体。它们代表人物、地点、事件或您提供的其他一些实体模型

**关系（Relationship）**：两个实体之间的关系。这些是从协变量生成的。

**协变量(Covariate)** ：提取的声明信息，其中包含有关实体的陈述，这些陈述可能是时间相关的。

**社区报告（Community Report）**：生成实体后，我们会对它们执行分层社区检测，并为该层次结构中的每个社区生成报告

**节点（Node）**：此表包含已嵌入和集群的实体和文档的渲染图形视图的布局信息

![flow](images/flow.png)

- **流程**

（1）   将文本切割为文本块

（2）   LLM识别文本中的实体。大模型首先识别文本中的所有实体，包括它们的名称、类型和描述，然后再识别明确相关的实体之间的所有关系，包括源实体和目标实体及其关系的描述；进行了多轮实体收集

（3）   为实体创建摘要 LLM来独立地创建可能隐含但未由文本本身陈述的概念的有意义的摘要，

（4）   实体摘要->同质无向加权图 社区检测算法来将图划分为彼此之间的连接比与图中的其他节点之间的连接更强的节点社区

（5）   生成社区摘要 

（6）  社区摘要 → 社区答案 → 全局答案；

- 准备社区摘要，社区摘要被随机打乱并分成预先指定的token大小的块。这确保了相关信息分布在各个块中，而不是集中（并可能丢失）在单个上下文窗口中；
- 映射社区答案。并行生成中间答案，每个块一个。这LLM还要求生成 0-100 之间的分数，表明生成的答案对于回答目标问题有多大帮助。得分为 0 的答案将被过滤掉。
- 减少到全局答案。中间社区答案按有用性分数的降序排序，并迭代添加到新的上下文窗口中，直到达到令牌限制。最终上下文用于生成返回给用户的全局答案。

最终提供两种搜索的方法：本地搜索和全局搜索

**本地搜索：**本地搜索方法通过将人工智能提取的知识图谱中的相关数据与原始文档的文本块相结合来生成答案。此方法适用于需要了解文档中提到的特定实体的问题（例如洋甘菊的治疗特性是什么？）

**全局搜索：**全局搜索方法通过以地图缩减方式搜索所有人工智能生成的社区报告来生成答案。这是一种资源密集型方法，但通常可以很好地回答需要理解整个数据集的问题（例如，本笔记本中提到的草药最重要的价值是什么？）

## 3.源码结构

源码的结构如下

```
├─config
│  ├─input_models
│  └─models
├─index
│  ├─cache
│  ├─config
│  ├─emit
│  ├─graph
│  │  ├─embedding
│  │  ├─extractors
│  │  │  ├─claims
│  │  │  ├─community_reports
│  │  │  ├─graph
│  │  │  └─summarize
│  │  ├─utils
│  │  └─visualization
│  ├─input
│  ├─llm
│  ├─progress
│  ├─reporting
│  ├─storage
│  ├─text_splitting
│  ├─utils
│  ├─verbs
│  │  ├─covariates
│  │  │  └─extract_covariates
│  │  │      └─strategies
│  │  │          └─graph_intelligence
│  │  ├─entities
│  │  │  ├─extraction
│  │  │  │  └─strategies
│  │  │  │      └─graph_intelligence
│  │  │  └─summarize
│  │  │      └─strategies
│  │  │          └─graph_intelligence
│  │  ├─graph
│  │  │  ├─clustering
│  │  │  │  └─strategies
│  │  │  ├─embed
│  │  │  │  └─strategies
│  │  │  ├─layout
│  │  │  │  └─methods
│  │  │  ├─merge
│  │  │  └─report
│  │  │      └─strategies
│  │  │          └─graph_intelligence
│  │  ├─overrides
│  │  └─text
│  │      ├─chunk
│  │      │  └─strategies
│  │      ├─embed
│  │      │  └─strategies
│  │      ├─replace
│  │      └─translate
│  │          └─strategies
│  └─workflows
│      └─v1
├─llm
│  ├─base
│  ├─limiting
│  ├─mock
│  ├─openai
│  └─types
├─model
├─prompt_tune
│  ├─generator
│  ├─loader
│  ├─prompt
│  └─template
├─query
│  ├─context_builder
│  ├─input
│  │  ├─loaders
│  │  └─retrieval
│  ├─llm
│  │  └─oai
│  ├─question_gen
│  └─structured_search
│      ├─global_search
│      └─local_search
└─vector_stores

```

### 3.1索引 Index

graphRAG第一步：构建索引，下面是执行语句

```
python -m graphrag.index --init --root ./ragtest
python -m graphrag.index --root ./ragtest
```

对应的源码在graphrag/index/\_\_main\_\_.py，其中\_\_main\_\_.py通过用户给定的参数去调用cli.py中的index_cli函数;

我们使用code2flow库生成index_cli的调用链的流程图，如下图所示

![index](images/index.svg)

（1）该函数的_initialize_project_at函数首先检查settings.yaml、.env、prompts等文件是否存在，没有就创建，其中settings.yaml主要是记录llm的api_key、embedding的api_key,并行化的配置、异步模型的配置、文本分块、本地搜索、全局搜索配置等一些信息；env中就是graph_rag的api_key，prompts中主要是实体的提取、实体描述的总结、提取实体的声明、社区报告的生成的提示词

```
实体提取的提示词内容概括：
描述了一个任务，目的是从给定的文本文档中识别并提取指定实体类型的所有实体，以及这些实体之间的关系。具体步骤如下：

1. **识别实体**：从文本中识别出所有符合指定类型的实体。对于每个识别出的实体，提取以下信息：
   - **实体名称**：实体的名称（首字母大写）
   - **实体类型**：实体所属的类型（从给定的实体类型列表中选择）
   - **实体描述**：对实体的属性和活动的全面描述

   每个实体将以特定格式表示。

2. **识别关系**：从步骤1中识别出的实体中，找出所有明确相关的实体对（source_entity和target_entity）。对于每对相关实体，提取以下信息：
   - **源实体**：来源实体的名称
   - **目标实体**：目标实体的名称
   - **关系描述**：解释为何认为这两个实体之间存在关系
   - **关系强度**：一个数字评分，用于表示源实体和目标实体之间关系的强度

   每个关系也将以特定格式表示。

3. **生成输出**：最终的输出将是一个包含所有识别出的实体和关系的列表，格式化为指定的分隔符形式。

该任务的目的是从文本中结构化地提取出实体及其关系，以便进一步分析或使用。
```

```
实体描述总结的提示词内容概括：
下面描述了你的任务：你需要生成一个关于所提供数据的综合总结。具体步骤如下：

1. 给定一个或两个实体，以及与这些实体相关的一系列描述。
2. 将所有这些描述连接起来，形成一个完整的综合描述。
3. 确保包含所有描述中的信息，并解决其中可能存在的矛盾，提供一个连贯的总结。
4. 确保使用第三人称来撰写，并且要包括实体名称，以便提供完整的上下文。

最终输出将是一个完整的、无矛盾的描述，涵盖所有提供的描述信息。
```

```
提取声明的提示词内容概括：
这段话描述了一项任务，目标是帮助从文本文档中分析针对特定实体的声明。任务的具体步骤如下：

1. **提取实体**：从文本中提取与预定义的实体规范匹配的所有命名实体。实体规范可以是实体名称列表或实体类型列表。
  
2. **提取声明**：对于每个识别出的实体，提取与该实体相关的所有声明。这些声明需要与指定的声明描述相匹配，并且实体应为声明的主题。每个声明需要提取以下信息：
   - **主题实体**：作为声明主题的实体名称，需大写，且必须是步骤1中识别出的实体。
   - **对象实体**：声明中涉及的对象实体名称，需大写。如果对象实体未知，使用 **NONE**。
   - **声明类型**：声明的整体类别，需大写，并能在多个文本输入中重复使用，以便类似的声明共享相同的声明类型。
   - **声明状态**：标记为 **TRUE**（确认）、**FALSE**（虚假）或 **SUSPECTED**（怀疑）。TRUE 表示声明已确认，FALSE 表示声明被证伪，SUSPECTED 表示声明尚未验证。
   - **声明描述**：详细说明声明的依据，以及所有相关的证据和参考资料。
   - **声明日期**：声明提出的期间（开始日期和结束日期），需使用 ISO-8601 格式。如果声明只在某一天提出，则开始日期和结束日期相同。如果日期未知，返回 **NONE**。
   - **声明来源文本**：原始文本中与声明相关的所有引述的列表。

   每个声明按照特定的格式表示。

3. **生成输出**：最终输出将是所有识别出的声明的列表，格式化为指定的分隔符形式。

任务的总体目标是从文本中结构化地提取出与指定实体相关的声明及其详细信息，以便进一步分析和使用。
```

```
社区报告的提示词的内容概括：
这段话描述了一项任务，你作为一个AI助手，帮助人类进行信息发现。信息发现是指在一个网络中识别和评估与某些实体（例如组织和个人）相关的相关信息。

### 目标
编写一个关于某个社区的综合报告，该社区包含一组实体及其关系，以及可能相关的声明。报告将用于向决策者传达与该社区相关的信息及其潜在影响。报告的内容包括社区主要实体的概述、法律合规性、技术能力、声誉以及值得注意的声明。

### 报告结构

报告应包括以下部分：

- **标题**：代表社区关键实体的社区名称，标题应简短且具体。如果可能，应在标题中包含代表性的命名实体。
- **摘要**：社区整体结构的执行摘要，描述实体之间的关系以及与这些实体相关的重要信息。
- **影响严重性评分**：一个介于0到10之间的浮动分数，表示社区内实体可能带来的影响严重性。影响评分代表社区的重要性。
- **评分解释**：用一句话解释影响严重性评分。
- **详细发现**：列出关于社区的5到10个关键见解。每个见解应包括一个简短摘要，后跟多个段落的解释性文本，依据特定的规则进行支持。要求内容全面。

报告最终以JSON格式返回，包含标题、摘要、评分、评分解释和详细发现部分。

### 参考规则
支持数据的论点应列出数据引用。引用中最多列出5个记录ID，多于5个时使用“+more”表示有更多相关数据。

报告应基于提供的输入文本，不能编造信息。

这项任务的目的是为决策者提供一个全面的、基于数据的社区报告，以帮助他们理解社区内实体的动态及其潜在影响。
```

（2）然后_create\_default\_config函数创建pipelineConfig，创建管道，以及其中的workflow工作流

具体步骤为首先检查配置文件，在\_read\_config\_parameters函数内部读取配置文件，主要就是加载yaml中定义的一些配置

然后根据当前配置参数，创建一个pipeline，在函数create_pipeline_config中，这个函数首先确定要跳过的工作流（workflows），获取嵌入字段，然后根据设置构建输入、报告、存储和缓存配置，下面是创建管道的核心代码

```python
result = PipelineConfig(
        root_dir=settings.root_dir,
        input=_get_pipeline_input_config(settings),
        reporting=_get_reporting_config(settings),
        storage=_get_storage_config(settings),
        cache=_get_cache_config(settings),
        workflows=[
            *_document_workflows(settings, embedded_fields),
            *_text_unit_workflows(settings, covariates_enabled, embedded_fields),
            *_graph_workflows(settings, embedded_fields),
            *_community_workflows(settings, covariates_enabled, embedded_fields),
            *(_covariate_workflows(settings) if covariates_enabled else []),
        ],
    )
```

这里涉及到微软的dataShaper库，DataShaper 提供了一组常用的数据转换操作，这些操作可以被链式调用，以形成复杂的数据处理流程。这些操作包括基本的数据表操作，如过滤、排序、分组等。

这里我们举代码中的create_base_documents的函数build_steps来对Datashaper库进行说明：

这个函数 `build_steps` 使用了 DataShaper 库来构建一个数据处理流水线，该流水线主要用于处理和转换文档表的数据。下面是对这个函数及其各个步骤的详细解释：

函数目的

`build_steps` 函数接受一个 `PipelineWorkflowConfig` 对象作为输入，并返回一个由多个 `PipelineWorkflowStep` 组成的列表。每个步骤定义了在数据转换流水线中需要执行的操作，这些操作使用 DataShaper 库的各种“动词”（verbs）来完成特定的任务。

主要步骤解释

1. **Unroll 操作**：
   
   ```python
   {
       "verb": "unroll",
       "args": {"column": "document_ids"},
       "input": {"source": "workflow:create_final_text_units"},
   }
   ```
   - **功能**：`unroll` 操作用于展开（flatten）一个数组列。这里是将 `document_ids` 列中的数组展开为多个行。
   - **输入**：使用了 `workflow:create_final_text_units` 作为数据源，这意味着此步骤依赖于前面的处理结果。
   
2. **Select 操作**：
   
   ```python
   {
       "verb": "select",
       "args": {
           "columns": ["id", "document_ids", "text"]
       },
   }
   ```
   - **功能**：`select` 操作用于选择特定的列，这里选择了 `id`, `document_ids`, 和 `text` 列。这是为了后续处理只保留必要的数据。
   
3. **Rename 操作**：
   ```python
   {
       "id": "rename_chunk_doc_id",
       "verb": "rename",
       "args": {
           "columns": {
               "document_ids": "chunk_doc_id",
               "id": "chunk_id",
               "text": "chunk_text",
           }
       },
   }
   ```
   - **功能**：`rename` 操作用于重命名列名，使数据更有意义或适合后续操作。在这里，`document_ids` 被重命名为 `chunk_doc_id`，`id` 被重命名为 `chunk_id`，`text` 被重命名为 `chunk_text`。

4. **Join 操作**：
   ```python
   {
       "verb": "join",
       "args": {
           "on": ["chunk_doc_id", "id"]
       },
       "input": {"source": "rename_chunk_doc_id", "others": [DEFAULT_INPUT_NAME]},
   }
   ```
   - **功能**：`join` 操作用于连接两个表。在这里，使用 `chunk_doc_id` 和 `id` 进行连接，将不同来源的数据合并到一起。

5. **Aggregate 操作**：
   ```python
   {
       "id": "docs_with_text_units",
       "verb": "aggregate_override",
       "args": {
           "groupby": ["id"],
           "aggregations": [
               {
                   "column": "chunk_id",
                   "operation": "array_agg",
                   "to": "text_units",
               }
           ],
       },
   }
   ```
   - **功能**：`aggregate_override` 用于对数据进行聚合。在这里，按照 `id` 分组，并将 `chunk_id` 聚合为数组，存储在 `text_units` 列中。

6. **Convert 操作**：
   ```python
   {
       "verb": "convert",
       "args": {"column": "id", "to": "id", "type": "string"}
   }
   ```
   - **功能**：`convert` 操作用于将数据类型进行转换。在这里，将 `id` 列的数据类型转换为字符串类型。
   
     

现在我们对创建的workflows进行一个细致的解释，PipelineConfig中根据workflows/v1下的模板进行创建的，一共包括五个workflow，分别为：

- _document_workflows：实现了文本分块和添加索引(doc_id)
- _text_unit_workflows：
- _graph_workflows：
- _community_workflows：
- _covariate_workflows：



![](images/pipeline.svg)

