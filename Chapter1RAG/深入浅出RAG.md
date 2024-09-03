# 深入浅出RAG

RAG（Retrieval-Augmented Generation）检索增强生成是指从某些数据源检索到问题相关信息，提升大模型生成的效果的技术，其模式为**检索+LLM Prompt**，检索到的信息作为上下文注入到提示词prompt中。

RAG从2023年以来一直是非常流行的一套基于大模型的架构，其主要解决了三个方面的问题：（1）大模型在处理超出其能力范围的问题时会出现幻觉，给出一些格式正确但与真理相差很多的误导答案（2）在回答和私有数据相关或者垂直领域较深的问题时，大模型无法给出令人满意的答案。(3)由于训练成本高，参数量大的大模型存在知识时效性问题，没有办法更新最新的知识

用通俗的话来说，我们可以把用户给出的问题当作一场考试，每一个预训练好的大模型就是考场上的学生，RAG相当于考场上开卷考试提供的资料，优秀的RAG会针对考场的题目用最快的速度给出最相关的资料，然后让我们的大模型根据这些资料去回答。

## 1.普通RAG的流程与局限

![RAG原理图](images/RAG原理图.png)

普通的RAG的流程如下：

（1）构建文本向量数据库：对各种格式的文档PDF、HTML、Word 和 Markdown进行清理和提取，最终转换为纯文本文档，然后将文本切割为合适大小、段落尽量完整的块，然后使用嵌入模型转换为向量，存储在向量数据库中

（2）用相同的embedding模型将用户的问题转为向量（暂且称之为查询向量）

（3）在向量数据库中搜索和匹配与查询向量相似性高的，选取最高的前k个文本块，作为上下文，和用户的问题一起送入大模型，然后大模型作出回答

根据上面普通RAG流程，我们可以思考下面几个问题：

### （1）查询向量和向量数据库的向量的相似性搜索有多准？

讨论这个问题之前，先做一个实验，选取的是openai的text-embedding-3-small（目前较为先进的embedding模型），我们给出了一个查询问题，以及相关的一些语句（该实验的测试题目来自[4]）

```python
import os
os.environ["OPENAI_API_KEY"]= ""

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()

def get_embedding(s:str):
    if len(s)==0:
        return
    else:
        return client.embeddings.create(
            input=s, model="text-embedding-3-small" # nomic-embed-text text-embedding-3-small
            ).data[0].embedding

query = "I like apple"

# 相关语句
related_sentences = [
    "i don't like apple",
    "i don't like apples",
    "i dislike fruits",
    "i am a vagan",
    "apple makes me happy"
]

# 获取查询语句和相关语句的嵌入
query_embedding = get_embedding(query)
related_embeddings = [get_embedding(sentence) for sentence in related_sentences]

# 将嵌入转换为 NumPy 数组
query_embedding_np = np.array(query_embedding).reshape(1, -1)
related_embeddings_np = np.array(related_embeddings)

# 计算余弦相似度
similarities = cosine_similarity(query_embedding_np, related_embeddings_np)[0]

# 打印结果
for sentence, similarity in zip(related_sentences, similarities):
    print(f"句子: {sentence}\n相似度: {similarity}\n")
```

下面是输出

```python
句子: i don't like apple
相似度: 0.809324201675432

句子: i don't like apples
相似度: 0.7381684275506182

句子: i dislike fruits
相似度: 0.45008424256851837

句子: i am a vagan
相似度: 0.25186223466431923

句子: apple makes me happy
相似度: 0.7087407861105871
```

从这个例子可以看到，向量相似性最高选出的往往不一定是最佳选项，它对于词语情感上的差异排序优先级不高，且对名词的单数、复数比较敏感。

### （2）文本切割的块的大小和top-k怎么设置最佳？

目前主流的文本切割方法有直接按token数量进行暴力切割，有按照段落进行切割的，有递归切割的等等，但是无论哪一种切割文本的方法，都不可避免的会出现一个文本块中包含两个或多个不同的话题，试想一下，如果对于查询的问题而言，如果相关性很高的语句被其他不相关的语句淹没了，它能否在检索的过程中被找到呢？我们进行一个实验，下面是代码

```python
import os
os.environ["OPENAI_API_KEY"]= ""

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()

def get_embedding(s:str):
    if len(s)==0:
        return
    else:
        return client.embeddings.create(
            input=s, model="text-embedding-3-small" # nomic-embed-text text-embedding-3-small
            ).data[0].embedding

query = "i like apple."

# 相关语句
related_sentences = [
    """
The journey of human history is a tale of relentless exploration, from the earliest days when our ancestors roamed the African savannas, to the modern age where we traverse the skies and oceans, uncovering the secrets of our world and beyond. In this grand narrative, every era has been marked by significant discoveries and the courage to venture into the unknown. The curiosity that drove early humans to migrate across continents is the same force that today compels us to probe the depths of the oceans and the far reaches of outer space.

In ancient civilizations, the quest for knowledge was intertwined with survival. People studied the stars to predict the seasons, enabling them to cultivate crops and sustain their communities. The development of writing systems allowed for the preservation and dissemination of knowledge, ensuring that each generation could build upon the discoveries of the past. This accumulation of wisdom laid the groundwork for the great empires of antiquity, where science, philosophy, and art flourished in a symbiotic relationship.

As time passed, the Renaissance era brought a resurgence of learning and culture, fueled by the rediscovery of classical texts and the invention of the printing press. This period saw an explosion of creativity and inquiry, as scholars and artists alike sought to understand the world and express the human experience in new ways. The spirit of the Renaissance was embodied by figures such as Leonardo da Vinci and Galileo Galilei, whose contributions spanned multiple disciplines, pushing the boundaries of what was known and understood.

The Scientific Revolution that followed was a direct result of this renewed focus on observation and experimentation. Thinkers like Isaac Newton and Johannes Kepler sought to explain the natural world through the lens of mathematics and empirical evidence. Their work laid the foundation for modern science, where hypotheses are tested rigorously, and theories are constantly refined in light of new data. The technological advancements of the Industrial Revolution, which came next, further accelerated human progress, transforming societies and economies at an unprecedented pace.

Today, we live in a world shaped by the innovations of the past, yet still driven by the same insatiable curiosity. The digital age has ushered in an era of information, where knowledge is more accessible than ever before. The internet has become a global repository of human thought, a vast network where ideas are shared and debated across borders and cultures. This interconnectedness has fostered a new level of collaboration and innovation, as scientists, engineers, and thinkers from around the world work together to solve the pressing challenges of our time.

Climate change, for instance, represents one of the most significant threats to our planet's future. The scientific consensus is clear: human activity is altering the Earth's climate, with potentially devastating consequences. However, this challenge also presents an opportunity for innovation. Researchers are developing new technologies to reduce carbon emissions, harness renewable energy, and mitigate the impacts of climate change. The global community is beginning to recognize the importance of sustainability, with governments, corporations, and individuals taking steps to protect our planet for future generations.

In the realm of space exploration, humanity continues to push the boundaries of what is possible. The successful landing of rovers on Mars, the ongoing research aboard the International Space Station, and the ambitious plans for manned missions to other planets all speak to our enduring desire to explore the cosmos. These endeavors are not just about discovering new worlds; they also have the potential to unlock new technologies and resources that could benefit life on Earth.

Yet, even as we look to the stars, it is essential to remember the importance of understanding and preserving our own world. The study of ecology, for example, has revealed the intricate web of relationships that sustain life on our planet. Each species, no matter how small, plays a role in maintaining the balance of our ecosystems. The loss of biodiversity, driven by habitat destruction, pollution, and climate change, threatens to unravel this delicate balance, with far-reaching consequences for all life on Earth.

The advancement of artificial intelligence (AI) represents another frontier of human exploration. AI has the potential to revolutionize industries, from healthcare to finance, by automating tasks and providing insights that were previously unimaginable. However, it also raises ethical questions about the role of machines in society, the future of work, and the nature of consciousness itself. As we continue to develop and integrate AI into our lives, it is crucial to consider these implications and ensure that this technology is used for the betterment of humanity.

At the same time, the study of human psychology and neuroscience is shedding new light on the mysteries of the mind. Understanding how the brain works, how we think, feel, and make decisions, has profound implications for fields such as education, mental health, and personal development. By exploring the inner workings of our minds, we can better understand ourselves and improve our lives in meaningful ways.

In reflecting on these themes, it is evident that the quest for knowledge and discovery is deeply embedded in the human spirit. Throughout history, we have sought to understand the world around us, to improve our circumstances, and to explore the unknown. Whether through the study of ancient texts, the observation of the stars, or the development of cutting-edge technology, our pursuit of knowledge has been a constant driving force.

As we look to the future, it is clear that this pursuit will continue to shape our world in profound ways. From addressing global challenges such as climate change and biodiversity loss to exploring new frontiers in space and technology, the path forward will require collaboration, innovation, and a deep commitment to the betterment of humanity. In moments of reflection, I often find myself considering the simple pleasures of life, like how I enjoy eating apple during a quiet afternoon. Such moments remind us of the importance of balance, of finding joy in the everyday while continuing to reach for the stars.

In conclusion, the journey of human history is a testament to our relentless curiosity and desire for progress. As we continue to explore the mysteries of the universe and our own minds, we must do so with a sense of responsibility and a commitment to ensuring that the knowledge we gain is used to create a better world for all.
""",
    "i dont like apple",
]

# 获取查询语句和相关语句的嵌入
query_embedding = get_embedding(query)
related_embeddings = [get_embedding(sentence) for sentence in related_sentences]

# 将嵌入转换为 NumPy 数组
query_embedding_np = np.array(query_embedding).reshape(1, -1)
related_embeddings_np = np.array(related_embeddings)

# 计算余弦相似度
similarities = cosine_similarity(query_embedding_np, related_embeddings_np)[0]

# 打印结果
idx=0
# 打印结果
for sentence, similarity in zip(related_sentences, similarities):
    print(f"句子{idx}: \n相似度: {similarity}\n")
    idx+=1
```

下面是输出结果

```
句子0: 
相似度: 0.14242785109432549

句子1: 
相似度: 0.7438104997372591
```

其中句子0中是包含一句“I enjoy eating apple during a quiet afternoon.”但是完全被一些无关信息给淹没了，导致其与查询语句的向量相似度很低，因此可见，当文本切割块的size过大，会导致命中率降低

还有topk这一参数设置，因为我们知道大模型的上下文token是有限制的，且如果上下文很长，模型消耗的算力也比较大，推理时间也会较长，在上一个问题中我们不能保证相似性最高的就是最想要的结果，topk参数设置和文本块切割设置多少，能保证RAG系统发挥最佳性能呢？超参数的调整和验证方案可能能发挥作用，但是当文档数量越多，其调整的成本也就越大。

### （3）多跳问答（Multi-hop Q&A ）应该如何解决？

多跳问答（Multi-hop Q&A）是一种自然语言处理（NLP）技术，涉及到在回答复杂问题时，需要通过多步推理，从多个不同的信息源或文本片段中逐步获取和整合相关信息，最终得出正确答案的过程。与传统的单步问答系统不同，多跳问答要求系统能够在理解和连接多个相关的信息点之间进行推理，以完成跨越多个逻辑或语义层次的推理任务。这个过程通常涉及多个推理步骤或“跳跃”，以便在分散的上下文中找到最终答案。

比如我询问这样一个问题

```
谁是写《哈利·波特》的作者的导师？
```

**解答过程**:

1. **第一步**：从问题中提取信息，“《哈利·波特》的作者是谁？”
   - **答案**: 《哈利·波特》的作者是J.K.罗琳（J.K. Rowling）。
2. **第二步**：接着提取信息，“J.K.罗琳的导师是谁？”
   - **答案**: J.K.罗琳的导师是伊恩·班克斯（Ian Rankin）（假设在某个具体场景中是她的导师）。

**最终答案**: J.K.罗琳的导师是伊恩·班克斯。

在这个例子中，需要通过多个推理步骤，先找出《哈利·波特》的作者是谁，然后再找到该作者的导师是谁。这种多步推理的过程就是“多跳问答”的核心特征。然而普通一步式的RAG很难回答这种多跳问答，需要进行改进。

### （4）总结性问题如何解决？

总结性问题比如：“这篇文章主要讲了什么内容？”，“这篇文章的top5主题是什么？"，普通的RAG或者传统RAG都无法解决这类问题，需要使用graphRAG，关于graphRAG的相关内容可以看我另一篇笔记。

## 2.RAG分类

根据检索器是通过什么方式增强生成器的生成效果可以讲RAG分成四类，其中主要还是基于查询的RAG。

![RAG分类](images/RAG分类.png)

### （1）基于查询的RAG

基于查询的RAG将用户的问题和检索到的信息结合在一起，直接输入大模型得到回答，基于查询的RAG是最常见的类型。其中检索的方法多种多样。Lewis等人$^{[6]}$利用 DPR 进行信息检索,RALM $^{[7]}$ 使用 BM25 进行文档检索等等

### （2）基于潜在表征的RAG

基于潜在表征的RAG将检索到的对象以潜在表征的形式融合到生成模型，增强了模型的理解能力并提高了生成内容的质量。FiD和RETRO是这种框架的两种代表RAG，FiD$^{[8]}$将问题和每个检索到的文本片段分别通过编码器独立编码，然后在解码器中联合处理。这种“Fusion-in-Decoder”的方法使得模型能够有效地从多个片段中聚合和组合证据

### （3）基于Logit的RAG

在基于Logit的 RAG 中，生成模型在解码过程中通过 Logit 整合检索信息。通常，通过简单的求和或模型将 Logit 组合起来，以计算分步生成的概率。kNN-LM 及其变体将语言模型生成的概率与每个解码步骤中从相似前缀的检索距离得出的概率相结合。

### （4）投机性RAG

投机性 RAG 寻求使用检索代替纯生成的机会，旨在节省资源并加快响应速度。GPTCache通过构建用于存储 LLM 响应的语义缓存解决了使用 LLM API 时的高延迟问题。COG 将文本生成过程分解为一系列复制粘贴操作，从文档中检索单词或短语而不是生成

## 3.RAG改进方法





## 4.RAG开源项目介绍



## 5.对比和思考



## 6.参考文献

[1]Zhao, Penghao, et al. "Retrieval-augmented generation for ai-generated content: A survey." *arXiv preprint arXiv:2402.19473* (2024).

[2]Gao, Yunfan, et al. "Retrieval-augmented generation for large language models: A survey." *arXiv preprint arXiv:2312.10997* (2023).

[3]https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6

[4]https://medium.com/@kelvin.lu.au/disadvantages-of-rag-5024692f2c53

[5]https://luxiangdong.com/2023/09/25/ragone/

[6]P. S. H. Lewis, E. Perez, A. Piktus et al., “Retrieval-augmented generation for knowledge-intensive NLP tasks,” in NeurIPS, 2020.

[7]O. Ram, Y. Levine, I. Dalmedigos et al., “In-context retrievalaugmented language models,” arXiv:2302.00083, 2023.

[8]G. Izacard and E. Grave, “Leveraging passage retrieval with generative models for open domain question answering,” in EACL, 2021.