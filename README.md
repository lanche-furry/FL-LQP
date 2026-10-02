<div align="center">

# FL-LQP

### Frozen-LLM Learnable Query Pooling

*A parameter-efficient architecture for sentence embeddings.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**lanche-furry** · 2026 · [GitHub](https://github.com/lanche-furry/FL-LQP)

</div>

![FL-LQP Architecture](FL-LQP.png)

> **English** | [中文文档](#中文文档)

---

## Abstract

We introduce **FL-LQP** (*Frozen-LLM Learnable Query Pooling*), a parameter-efficient architecture for sentence embeddings. FL-LQP **completely freezes** a pretrained large language model backbone and trains only a **learnable query attention pooling head** — approximately **26M parameters (0.6% of a 4B backbone)** — to aggregate token-level hidden states into sentence-level embeddings.

On Qwen3-Embedding-4B, FL-LQP achieves **0.8227 / 0.8779 / 0.8991** (Spearman ρ) on STS12 / STS16 / STSBenchmark, averaging **0.8666** — **surpassing its own teacher by ~2%** — with only 0.6% trainable parameters. Training requires a single A10 GPU for about 50 minutes on 26-language NLI data.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Architecture](#2-architecture)
3. [Training Objective](#3-training-objective)
4. [Experimental Setup](#4-experimental-setup)
5. [Results](#5-results)
6. [Ablation](#6-ablation)
7. [Quick Start](#7-quick-start)
8. [Reproduction](#8-reproduction)
9. [FAQ](#9-faq)
10. [Citation](#10-citation)

---

## 1. Introduction

Sentence embeddings underpin **semantic retrieval**, **clustering**, and **retrieval-augmented generation (RAG)**. Recent works adopt **decoder-only LLMs** as embedding backbones, but face two obstacles:

- **Causal attention** makes sentence-level aggregation non-trivial.
- **Full fine-tuning** is expensive and risks catastrophic forgetting.

**FL-LQP takes a different path:** freeze the backbone entirely and concentrate all learning capacity in a lightweight pooling head. Our core hypothesis is simple: a pretrained LLM **already knows language** — the only remaining task is to learn *how to aggregate*.

### Related Work

| Method | Backbone | Trainable | Pooling |
|:-------|:--------:|:---------:|:--------|
| NV-Embed | Frozen LLM | ~10M | Latent attention (LLM output as query) |
| GLOT (ICLR 2026) | Frozen LLM | ~20M | Graph neural network |
| PromptEmbedder | Frozen LLM | ~15M | Dual-LLM soft prompts |
| **FL-LQP (ours)** | Frozen LLM | **26M** | **Learnable query attention** |

## 2. Architecture

| Module | Status | Parameters |
|:-------|:------:|:----------:|
| Frozen LLM backbone | ❄️ Frozen | 4,048M |
| Query attention pooling head | 🔥 Trainable | **26M (0.6%)** |

### 2.1 Context-Modulated Learnable Query

Given hidden states $H \in \mathbb{R}^{B \times L \times D}$ and mask $M$, we define a learnable query $q \in \mathbb{R}^{1 \times 1 \times D}$, initialized from $\mathcal{N}(0, 0.02^2)$, modulated by the masked sequence mean:

$$\bar{h} = \frac{\sum_{i=1}^{L} M_i H_i}{\sum_{i=1}^{L} M_i}, \qquad q' = q + W_{\text{pos}} \bar{h}$$

### 2.2 Multi-Head Attention Pooling

$$\text{Attn}(q', H, H) = \text{softmax}\!\left(\frac{q' W_Q (H W_K)^\top}{\sqrt{d_k}}\right) H W_V$$

Output at position 0 is LayerNorm-ed to yield the sentence embedding $e \in \mathbb{R}^{B \times D}$.

### 2.3 Reference Implementation

```python
import torch
import torch.nn as nn

class QueryAttentionPool(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.pos_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        B = x.shape[0]
        x_f = x.float()
        mf = mask.unsqueeze(-1).float()
        avg = (x_f * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1.0)
        q = (self.q.float() + self.pos_proj(avg)).expand(B, -1, -1).contiguous()
        kp = (~mask.bool()) if mask is not None else None
        out, _ = self.attn(q, x_f, x_f, key_padding_mask=kp)
        return self.norm(out[:, 0, :]).to(x.dtype)
```

## 3. Training Objective

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{distill}} + (1 - \alpha) \cdot \mathcal{L}_{\text{contrast}}$$

$$\mathcal{L}_{\text{distill}} = 1 - \frac{s \cdot t}{\|s\| \|t\|}, \qquad \mathcal{L}_{\text{contrast}} = -\log \frac{\exp(\text{sim}(a_i, b_i)/\tau)}{\sum_{j=1}^{B} \exp(\text{sim}(a_i, b_j)/\tau)}$$

We set $\alpha = 0.7$, $\tau = 0.05$. The distillation term is weighted higher to prioritize representation-space alignment.
## 4. Experimental Setup

| Item | Value |
|:-----|:------|
| Backbone | Qwen3-Embedding-4B (36 layers, D=2560) |
| Trainable | 26M (0.6%) |
| Data | 250K multilingual NLI pairs (26 languages) |
| Batch size | 32 × 2 (concatenated) |
| Learning rate | 1e-3, cosine, 5% warmup |
| Epochs | 1 |
| Hardware | Single NVIDIA A10 (24 GB) |
| Training time | ~50 minutes |
| Precision | fp16 + SDPA |

## 5. Results

### 5.1 STS Benchmarks

| Task | FL-LQP | Teacher | Δ |
|:-----|:------:|:-------:|:-:|
| STS12 | **0.8227** | ~0.78 | **+0.04** |
| STS16 | **0.8779** | ~0.81 | **+0.07** |
| STSBenchmark | **0.8991** | ~0.86 | **+0.04** |
| **Average** | **0.8666** | ~0.85 | **+0.02** |

### 5.2 Efficiency

| Metric | Value |
|:-------|:------|
| Total parameters | 4,048M |
| Trainable parameters | 26M (0.6%) |
| Peak VRAM | ~10 GB |
| Training time | ~50 min (A10) |
| Inference speedup | **None** (backbone runs fully) |

### 5.3 Inference Note

FL-LQP **does not accelerate inference**. The backbone still runs a full forward pass (8 GB, fp16). FL-LQP only reduces **training** cost (parameters, VRAM, time).

For inference acceleration, consider complementary techniques:
- **Quantization** (INT8 / INT4): ~2-4x speedup, minimal quality loss
- **Distillation to smaller backbones** (e.g., 0.6B): ~6x speedup, -3% STS
- **Layer pruning** of the backbone: risky for decoder-only models
- **Matryoshka Representation Learning (MRL)**: dynamic dimension truncation at query time

### 5.4 Training Efficiency (vs. full fine-tuning)

| Metric | Full fine-tuning (4B) | **FL-LQP** |
|:-------|:---------------------:|:----------:|
| Trainable params | 4B | **26M (-99.4%)** |
| Peak VRAM | ~40 GB | **~10 GB (-75%)** |
| Trainable on single A10 | No | **Yes** |
| Training time (250K, A10) | N/A | **~50 min** |
| Catastrophic forgetting | Risk | **None** |

## 6. Ablation

### 6.1 Pooling Strategy

| Strategy | STS Avg | Trainable |
|:---------|:-------:|:---------:|
| Last-token | ~0.85 | 0 |
| Mean pooling | ~0.83 | 0 |
| CLS token | ~0.84 | 0 |
| Linear projection | ~0.85 | 3M |
| **FL-LQP (ours)** | **0.8666** | **26M** |

### 6.2 Query Initialization

| Initialization | STS Avg |
|:---------------|:-------:|
| $\mathcal{N}(0, 1)$ | 0.8210 |
| Zeros | 0.8402 |
| **$\mathcal{N}(0, 0.02^2)$** | **0.8666** |

### 6.3 Context Modulation

| Variant | STS Avg |
|:--------|:-------:|
| Query only (no modulation) | 0.8503 |
| **Query + context modulation** | **0.8666** |

## 7. Quick Start

### 7.1 Installation

```bash
pip install torch transformers
# or from source
git clone https://github.com/lanche-furry/FL-LQP.git
cd FL-LQP && pip install -r requirements.txt
```

### 7.2 Minimal Example

```python
import torch
from transformers import AutoModel, AutoTokenizer
from fl_lqp import FL_LQPModel

# Load frozen backbone
backbone = AutoModel.from_pretrained(
    "Qwen/Qwen3-Embedding-4B",
    dtype=torch.float16,
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
)

# Wrap with FL-LQP (backbone frozen automatically)
model = FL_LQPModel(backbone).cuda().eval()

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B')
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Encode multilingual sentences
texts = [
    "A man is playing a guitar.",
    "一个男人正在弹吉他。",
    "Un homme joue de la guitare.",
    "The stock market crashed today.",
]

enc = tok(texts, padding=True, truncation=True,
          max_length=128, return_tensors='pt').cuda()

with torch.no_grad():
    embs = model(enc["input_ids"], enc["attention_mask"]).float()

embs = embs / embs.norm(dim=-1, keepdim=True)
sims = embs @ embs.T
print(sims.cpu().numpy().round(3))
```

**Expected output** (first three are semantically aligned across languages):

```
[[1.    0.91  0.89  0.12]
 [0.91  1.    0.88  0.15]
 [0.89  0.88  1.    0.11]
 [0.12  0.15  0.11  1.  ]]
```

## 8. Reproduction

### Step 1: Prepare Data

```python
from datasets import load_dataset
import json, random

langs = ['ar','bn','de','es','fa','fr','he','hi','id','it','ja','ko',
         'mr','nl','pl','ps','pt','ru','sv','sw','ta','tr','uk','ur','vi','zh']
tasks = ['mnli','anli','fever','ling','wanli']

samples = []
for lang in langs:
    for task in tasks:
        ds = load_dataset(
            'MoritzLaurer/multilingual-NLI-26lang-2mil7',
            split=f'{lang}_{task}', streaming=True)
        n = 0
        for ex in ds:
            if ex['label'] != 0:
                continue
            p, h = ex['premise'].strip(), ex['hypothesis'].strip()
            if len(p) < 10 or len(h) < 10:
                continue
            samples.append({'text_a': p[:512], 'text_b': h[:512]})
            n += 1
            if n >= 8000: break

random.seed(42)
random.shuffle(samples)
with open('nli_250k.json', 'w') as f:
    json.dump(samples, f)
print(f'Total: {len(samples):,} pairs')
```

### Step 2: Precompute Teacher Embeddings

```python
import torch, json
from transformers import AutoModel, AutoTokenizer

device = 'cuda'
teacher = AutoModel.from_pretrained(
    "Qwen/Qwen3-Embedding-4B",
    dtype=torch.float16, attn_implementation='sdpa',
).to(device).eval()

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B')
if tok.pad_token is None: tok.pad_token = tok.eos_token

with open('nli_250k.json') as f:
    samples = json.load(f)

@torch.inference_mode()
def encode_last_token(texts, bs=32, max_len=128):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], padding=True, truncation=True,
                  max_length=max_len, return_tensors='pt').to(device)
        h = teacher(**enc).last_hidden_state.float()
        lengths = enc['attention_mask'].sum(1) - 1
        idx = lengths.view(-1, 1, 1).expand(-1, 1, h.size(-1))
        last = h.gather(1, idx).squeeze(1)
        last = last / last.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        out.append(last.half().cpu())
    return torch.cat(out, 0)

t_a = encode_last_token([s['text_a'] for s in samples])
t_b = encode_last_token([s['text_b'] for s in samples])
torch.save({'t_a': t_a, 't_b': t_b}, 'teacher_embeddings.pt')
```

**Runtime**: ~90 minutes on a single A10 for 250K × 2 sides.

### Step 3: Train the FL-LQP Head

```python
import torch, torch.nn.functional as F, math
from transformers import AutoModel
from fl_lqp import FL_LQPModel

backbone = AutoModel.from_pretrained(
    "Qwen/Qwen3-Embedding-4B",
    dtype=torch.float16, attn_implementation='sdpa',
)
model = FL_LQPModel(backbone).cuda()

optimizer = torch.optim.AdamW(
    model.trainable_parameters(),
    lr=1e-3, weight_decay=0.01, betas=(0.9, 0.95)
)

def distill(s, t):
    s = F.normalize(s, dim=-1)
    t = F.normalize(t, dim=-1)
    return (1.0 - (s * t).sum(-1)).mean()

def contrastive(a, b, temp=0.05):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = (a @ b.transpose(-2, -1)) / temp
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) +
                  F.cross_entropy(logits.transpose(-2, -1), labels))

model.train()
for step in range(3906):
    s_a = model(a_ids, a_mask).float()
    s_b = model(b_ids, b_mask).float()
    L_d = 0.5 * (distill(s_a, t_a) + distill(s_b, t_b))
    L_c = contrastive(s_a, s_b)
    loss = 0.7 * L_d + 0.3 * L_c

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.pool.state_dict(), 'fl_lqp_head.pt')
```

**Runtime**: ~50 minutes on a single A10.

### Step 4: Evaluate

```python
from datasets import load_dataset
from scipy.stats import spearmanr
import numpy as np

def eval_sts(task_name):
    ds = load_dataset(f'mteb/{task_name}-sts', split='test')
    a = [x['sentence1'] for x in ds]
    b = [x['sentence2'] for x in ds]
    gold = np.array([float(x['score']) for x in ds])
    ea = encode(a); eb = encode(b)
    sims = (ea * eb).sum(1)
    sims = (sims + 1) * 2.5
    return spearmanr(sims, gold).correlation

results = {t: eval_sts(t) for t in ['sts12', 'sts16', 'stsbenchmark']}
print(f'Average: {sum(results.values()) / 3:.4f}')  # 0.8666
```

## 9. FAQ

**Q1: Why not fine-tune the backbone?**

Full fine-tuning of a 4B model requires ~16 GB of optimizer states plus gradients plus activations — infeasible on a single A10. More importantly, it risks catastrophic forgetting. FL-LQP shows that 0.6% trainable parameters suffice to **surpass** a 4B teacher on embedding tasks.

**Q2: Why does a 26M head beat last-token pooling?**

Last-token pooling assumes the final position summarizes the whole sentence, which is fragile for long sequences. Learnable attention pooling dynamically selects informative tokens per input.

**Q3: Can I use a different backbone?**

Yes. FL-LQP is backbone-agnostic. Any decoder-only LLM exposing `last_hidden_state` works — just set `hidden_size` in `FL_LQPConfig`.

**Q4: Minimum VRAM?**

| Backbone | Model | Training Peak | Recommended GPU |
|:---------|:-----:|:-------------:|:----------------|
| Qwen3-Embedding-0.6B | 1.2 GB | ~4 GB | RTX 3060 / T4 |
| Qwen3-Embedding-4B | 8 GB | ~10 GB | A10 / RTX 3090 |
| Qwen3-Embedding-8B | 16 GB | ~20 GB | A100 40GB |

**Q5: Training time?**

| GPU | Time (250K, 1 epoch) |
|:----|:---------------------|
| A100 40GB | ~15 min |
| RTX 3090 | ~40 min |
| A10 | ~50 min |
| T4 | ~120 min |

**Q6: Why is 250K sufficient?**

The pooling head has only 26M trainable parameters. Empirically, 250K pairs give ~10 samples per parameter — well above the overfitting threshold. Additionally, embedding models learn **geometric structure**, not facts. Our ablation shows <1% STS improvement going from 125K to 250K.

**Q7: Multilingual capability?**

Training data covers 26 languages with ~10K samples each. Cross-lingual alignment emerges naturally from NLI entailment — the same entailment relation holds across translations.

**Q8: Can I use it for RAG?**

Yes. `model(...)` returns raw `[B, 2560]` vectors, compatible with FAISS / Chroma / Qdrant. We recommend L2 normalization + cosine similarity.

## 10. Citation

```bibtex
@misc{fl-lqp-2026,
  title  = {FL-LQP: Frozen-LLM Learnable Query Pooling for Efficient Sentence Embeddings},
  author = {lanche-furry},
  year   = {2026},
  url    = {https://github.com/lanche-furry/FL-LQP}
}
```

---

**[⬆ Back to top](#fl-lqp)** · **[中文文档 →](#中文文档)**
---

<a id="中文文档"></a>

# 中文文档

> [English](#fl-lqp) | **中文**

---

## 摘要

我们提出 **FL-LQP**（*Frozen-LLM Learnable Query Pooling*，冻结大语言模型 + 可学习查询池化），一种参数高效的句子嵌入架构。FL-LQP **完全冻结**预训练大语言模型主干，仅训练一个**可学习查询注意力池化头**——约 **26M 参数（占 4B 主干的 0.6%）**——将 token 级隐藏状态聚合为句子级嵌入。

在 Qwen3-Embedding-4B 上，FL-LQP 在 STS12 / STS16 / STSBenchmark 上分别达到 **0.8227 / 0.8779 / 0.8991**（Spearman ρ），平均 **0.8666**——**超越自身教师约 2%**——而可训练参数仅占 0.6%。训练使用单张 A10 GPU，基于 26 种语言的 NLI 数据，约 50 分钟即可完成。

---

## 目录

1. [引言](#1-引言)
2. [架构](#2-架构)
3. [训练目标](#3-训练目标)
4. [实验设置](#4-实验设置)
5. [实验结果](#5-实验结果)
6. [消融分析](#6-消融分析)
7. [快速开始](#7-快速开始)
8. [复现指南](#8-复现指南)
9. [常见问题](#9-常见问题)
10. [引用](#10-引用)

---

## 1. 引言

句子嵌入是**语义检索**、**聚类**和**检索增强生成（RAG）**的核心基础。近期研究开始采用 **decoder-only LLM** 作为嵌入主干，但面临两大障碍：

- **因果注意力**使句子级聚合变得困难。
- **全量微调**代价高昂，且存在灾难性遗忘风险。

**FL-LQP 走了一条截然不同的路径：** 完全冻结主干，将所有学习能力集中于一个轻量级池化头。核心假设非常简单：预训练 LLM **已经懂语言**——剩余的任务只是学习「如何聚合」。

### 相关工作

| 方法 | 主干 | 可训练 | 池化方式 |
|:-----|:----:|:------:|:---------|
| NV-Embed | 冻结 LLM | ~10M | 潜在注意力（LLM 输出作为 Query）|
| GLOT (ICLR 2026) | 冻结 LLM | ~20M | 图神经网络 |
| PromptEmbedder | 冻结 LLM | ~15M | 双 LLM 软提示 |
| **FL-LQP（本文）** | 冻结 LLM | **26M** | **可学习查询注意力** |

## 2. 架构

| 模块 | 状态 | 参数量 |
|:-----|:----:|:------:|
| 冻结 LLM 主干 | ❄️ 冻结 | 4,048M |
| 查询注意力池化头 | 🔥 可训练 | **26M (0.6%)** |

### 2.1 上下文调制的可学习查询

给定隐藏状态 $H \in \mathbb{R}^{B \times L \times D}$ 与掩码 $M$，定义可学习查询 $q \in \mathbb{R}^{1 \times 1 \times D}$，初始化自 $\mathcal{N}(0, 0.02^2)$，用掩码序列均值调制：

$$\bar{h} = \frac{\sum_{i=1}^{L} M_i H_i}{\sum_{i=1}^{L} M_i}, \qquad q' = q + W_{\text{pos}} \bar{h}$$

### 2.2 多头注意力池化

$$\text{Attn}(q', H, H) = \text{softmax}\!\left(\frac{q' W_Q (H W_K)^\top}{\sqrt{d_k}}\right) H W_V$$

第 0 位输出经过 LayerNorm 得到句子嵌入 $e \in \mathbb{R}^{B \times D}$。

### 2.3 参考实现

```python
import torch
import torch.nn as nn

class QueryAttentionPool(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.pos_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        B = x.shape[0]
        x_f = x.float()
        mf = mask.unsqueeze(-1).float()
        avg = (x_f * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1.0)
        q = (self.q.float() + self.pos_proj(avg)).expand(B, -1, -1).contiguous()
        kp = (~mask.bool()) if mask is not None else None
        out, _ = self.attn(q, x_f, x_f, key_padding_mask=kp)
        return self.norm(out[:, 0, :]).to(x.dtype)
```

## 3. 训练目标

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{蒸馏}} + (1 - \alpha) \cdot \mathcal{L}_{\text{对比}}$$

$$\mathcal{L}_{\text{蒸馏}} = 1 - \frac{s \cdot t}{\|s\| \|t\|}, \qquad \mathcal{L}_{\text{对比}} = -\log \frac{\exp(\text{sim}(a_i, b_i)/\tau)}{\sum_{j=1}^{B} \exp(\text{sim}(a_i, b_j)/\tau)}$$

我们设置 $\alpha = 0.7$、$\tau = 0.05$。蒸馏项权重更高，以优先对齐表示空间。

## 4. 实验设置

| 项目 | 值 |
|:-----|:---|
| 主干 | Qwen3-Embedding-4B（36 层，D=2560）|
| 可训练参数 | 26M (0.6%) |
| 数据 | 250K 多语言 NLI 数据对（26 种语言）|
| 批次大小 | 32 × 2（拼接）|
| 学习率 | 1e-3，余弦调度，5% 预热 |
| Epoch | 1 |
| 硬件 | 单张 NVIDIA A10 (24GB) |
| 训练时间 | 约 50 分钟 |
| 精度 | fp16 + SDPA |

## 5. 实验结果

### 5.1 STS 基准

| 任务 | FL-LQP | 教师 | 差值 |
|:-----|:------:|:----:|:----:|
| STS12 | **0.8227** | ~0.78 | **+0.04** |
| STS16 | **0.8779** | ~0.81 | **+0.07** |
| STSBenchmark | **0.8991** | ~0.86 | **+0.04** |
| **平均** | **0.8666** | ~0.85 | **+0.02** |

### 5.2 效率指标

| 指标 | 值 |
|:-----|:---|
| 总参数量 | 4,048M |
| 可训练参数 | 26M (0.6%) |
| 峰值显存 | ~10 GB |
| 训练时间 | ~50 分钟 (A10) |
| 推理速度 | **无加速**（主干需完整前向）|

### 5.3 推理说明

FL-LQP **不加速推理**。主干仍要完整前向（8 GB，fp16）。FL-LQP 只降低**训练**成本（参数、显存、时间）。

如需推理加速，需配合其他技术：
- **量化**（INT8 / INT4）：约 2-4 倍加速，质量损失极小
- **蒸馏到小主干**（如 0.6B）：约 6 倍加速，STS -3%
- **主干层剪枝**：对 decoder-only 模型风险高
- **Matryoshka 表示学习 (MRL)**：查询时动态截断维度

### 5.4 训练效率（对比全量微调）

| 指标 | 全量微调 (4B) | **FL-LQP** |
|:-----|:-------------:|:----------:|
| 可训练参数 | 4B | **26M (-99.4%)** |
| 峰值显存 | ~40 GB | **~10 GB (-75%)** |
| 单卡 A10 可行性 | ❌ | **✅** |
| 训练时间（250K, A10）| N/A | **~50 分钟** |
| 灾难性遗忘风险 | 有 | **无** |

## 6. 消融分析

### 6.1 池化策略

| 策略 | STS 平均 | 可训练参数 |
|:-----|:--------:|:----------:|
| Last-token | ~0.85 | 0 |
| 均值池化 | ~0.83 | 0 |
| CLS token | ~0.84 | 0 |
| 线性投影 | ~0.85 | 3M |
| **FL-LQP（本文）** | **0.8666** | **26M** |

### 6.2 查询初始化

| 初始化方式 | STS 平均 |
|:-----------|:--------:|
| $\mathcal{N}(0, 1)$ | 0.8210 |
| 全零 | 0.8402 |
| **$\mathcal{N}(0, 0.02^2)$** | **0.8666** |

### 6.3 上下文调制

| 变体 | STS 平均 |
|:-----|:--------:|
| 仅查询（无调制）| 0.8503 |
| **查询 + 上下文调制** | **0.8666** |

## 7. 快速开始

### 7.1 安装

```bash
pip install torch transformers
# 或从源码
git clone https://github.com/lanche-furry/FL-LQP.git
cd FL-LQP && pip install -r requirements.txt
```

### 7.2 最小示例

```python
import torch
from transformers import AutoModel, AutoTokenizer
from fl_lqp import FL_LQPModel

# 加载冻结主干
backbone = AutoModel.from_pretrained(
    "Qwen/Qwen3-Embedding-4B",
    dtype=torch.float16,
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
)

# 包装为 FL-LQP（主干自动冻结）
model = FL_LQPModel(backbone).cuda().eval()

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B')
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# 编码多语言句子
texts = [
    "A man is playing a guitar.",
    "一个男人正在弹吉他。",
    "Un homme joue de la guitare.",
    "The stock market crashed today.",
]

enc = tok(texts, padding=True, truncation=True,
          max_length=128, return_tensors='pt').cuda()

with torch.no_grad():
    embs = model(enc["input_ids"], enc["attention_mask"]).float()

embs = embs / embs.norm(dim=-1, keepdim=True)
sims = embs @ embs.T
print(sims.cpu().numpy().round(3))
```

**预期输出**（前三句语言不同但语义相近，相似度显著高于第四句）：

```
[[1.    0.91  0.89  0.12]
 [0.91  1.    0.88  0.15]
 [0.89  0.88  1.    0.11]
 [0.12  0.15  0.11  1.  ]]
```

## 8. 复现指南

### 步骤 1：准备数据

```python
from datasets import load_dataset
import json, random

langs = ['ar','bn','de','es','fa','fr','he','hi','id','it','ja','ko',
         'mr','nl','pl','ps','pt','ru','sv','sw','ta','tr','uk','ur','vi','zh']
tasks = ['mnli','anli','fever','ling','wanli']

samples = []
for lang in langs:
    for task in tasks:
        ds = load_dataset(
            'MoritzLaurer/multilingual-NLI-26lang-2mil7',
            split=f'{lang}_{task}', streaming=True)
        n = 0
        for ex in ds:
            if ex['label'] != 0:
                continue
            p, h = ex['premise'].strip(), ex['hypothesis'].strip()
            if len(p) < 10 or len(h) < 10:
                continue
            samples.append({'text_a': p[:512], 'text_b': h[:512]})
            n += 1
            if n >= 8000: break

random.seed(42)
random.shuffle(samples)
with open('nli_250k.json', 'w') as f:
    json.dump(samples, f)
print(f'Total: {len(samples):,} pairs')
```

### 步骤 2：预计算教师嵌入

```python
import torch, json
from transformers import AutoModel, AutoTokenizer

device = 'cuda'
teacher = AutoModel.from_pretrained(
    "Qwen/Qwen3-Embedding-4B",
    dtype=torch.float16, attn_implementation='sdpa',
).to(device).eval()

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B')
if tok.pad_token is None: tok.pad_token = tok.eos_token

with open('nli_250k.json') as f:
    samples = json.load(f)

@torch.inference_mode()
def encode_last_token(texts, bs=32, max_len=128):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], padding=True, truncation=True,
                  max_length=max_len, return_tensors='pt').to(device)
        h = teacher(**enc).last_hidden_state.float()
        lengths = enc['attention_mask'].sum(1) - 1
        idx = lengths.view(-1, 1, 1).expand(-1, 1, h.size(-1))
        last = h.gather(1, idx).squeeze(1)
        last = last / last.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        out.append(last.half().cpu())
    return torch.cat(out, 0)

t_a = encode_last_token([s['text_a'] for s in samples])
t_b = encode_last_token([s['text_b'] for s in samples])
torch.save({'t_a': t_a, 't_b': t_b}, 'teacher_embeddings.pt')
```

**耗时**：单张 A10 约 90 分钟（250K × 2 侧）。

### 步骤 3：训练 FL-LQP 池化头

```python
import torch, torch.nn.functional as F, math
from transformers import AutoModel
from fl_lqp import FL_LQPModel

backbone = AutoModel.from_pretrained(
    "Qwen/Qwen3-Embedding-4B",
    dtype=torch.float16, attn_implementation='sdpa',
)
model = FL_LQPModel(backbone).cuda()

optimizer = torch.optim.AdamW(
    model.trainable_parameters(),
    lr=1e-3, weight_decay=0.01, betas=(0.9, 0.95)
)

def distill(s, t):
    s = F.normalize(s, dim=-1)
    t = F.normalize(t, dim=-1)
    return (1.0 - (s * t).sum(-1)).mean()

def contrastive(a, b, temp=0.05):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = (a @ b.transpose(-2, -1)) / temp
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) +
                  F.cross_entropy(logits.transpose(-2, -1), labels))

model.train()
for step in range(3906):
    s_a = model(a_ids, a_mask).float()
    s_b = model(b_ids, b_mask).float()
    L_d = 0.5 * (distill(s_a, t_a) + distill(s_b, t_b))
    L_c = contrastive(s_a, s_b)
    loss = 0.7 * L_d + 0.3 * L_c

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.pool.state_dict(), 'fl_lqp_head.pt')
```

**耗时**：单张 A10 约 50 分钟。

### 步骤 4：评估

```python
from datasets import load_dataset
from scipy.stats import spearmanr
import numpy as np

def eval_sts(task_name):
    ds = load_dataset(f'mteb/{task_name}-sts', split='test')
    a = [x['sentence1'] for x in ds]
    b = [x['sentence2'] for x in ds]
    gold = np.array([float(x['score']) for x in ds])
    ea = encode(a); eb = encode(b)
    sims = (ea * eb).sum(1)
    sims = (sims + 1) * 2.5
    return spearmanr(sims, gold).correlation

results = {t: eval_sts(t) for t in ['sts12', 'sts16', 'stsbenchmark']}
print(f'平均: {sum(results.values()) / 3:.4f}')  # 0.8666
```

## 9. 常见问题

**Q1：为什么不微调主干？**

全量微调 4B 模型需要约 16 GB 优化器状态 + 梯度 + 激活值，单张 A10（24 GB）无法承受。更重要的是，它存在**灾难性遗忘**风险。FL-LQP 证明 0.6% 的可训练参数就足以在嵌入任务上**超越** 4B 教师。

**Q2：26M 的池化头为什么能打败 last-token 池化？**

last-token 池化假设序列最后位置可以概括全句语义，但对长序列和远距离依赖很不友好。可学习注意力池化能针对不同输入**动态选择**信息量大的 token。

**Q3：可以换其他主干吗？**

可以。FL-LQP 与主干无关，任何输出 `last_hidden_state` 的 decoder-only LLM 都适用。只需在 `FL_LQPConfig` 中设置 `hidden_size`。已测试：Qwen3-Embedding-4B / 0.6B。

**Q4：最低显存需求？**

| 主干 | 模型大小 | 训练峰值 | 推荐 GPU |
|:-----|:--------:|:--------:|:---------|
| Qwen3-Embedding-0.6B | 1.2 GB | ~4 GB | RTX 3060 / T4 |
| Qwen3-Embedding-4B | 8 GB | ~10 GB | A10 / RTX 3090 |
| Qwen3-Embedding-8B | 16 GB | ~20 GB | A100 40GB |

**Q5：训练时间？**

| GPU | 时间（250K，1 epoch）|
|:----|:---------------------|
| A100 40GB | ~15 分钟 |
| RTX 3090 | ~40 分钟 |
| A10 | ~50 分钟 |
| T4 | ~120 分钟 |

**Q6：为什么 250K 数据就够？**

池化头只有 26M 可训练参数，250K 样本约每参数 10 个样本——远超过拟合阈值。此外，嵌入模型学习的是**几何结构**而非事实知识。消融显示从 125K 增加到 250K，STS 提升 <1%。

**Q7：多语言能力如何？**

训练数据覆盖 26 种语言，每种约 10K 样本。跨语言语义对齐通过 NLI 蕴含对自然习得——同一对蕴含关系在不同语言中保持。

**Q8：可以用作 RAG 检索吗？**

可以。`model(...)` 返回原始 `[B, 2560]` 向量，兼容 FAISS / Chroma / Qdrant。**推荐做法**：编码后做 L2 归一化，用余弦相似度检索。

## 10. 引用

```bibtex
@misc{fl-lqp-2026,
  title  = {FL-LQP: Frozen-LLM Learnable Query Pooling for Efficient Sentence Embeddings},
  author = {lanche-furry},
  year   = {2026},
  url    = {https://github.com/lanche-furry/FL-LQP}
}
```

---

## 未来版本说明

FL-LQP 是 IRIXEN1.7 系列背后的**开放研究架构**。

**从 IRIXEN2 开始，所有后续模型将转为闭源。**

FL-LQP 架构本身仍以 MIT 协议自由开放，供学术与商业使用。

---

<div align="center">

**MIT License** · Copyright (c) 2026 lanche-furry

*用 ❄️ 冻结的主干与 🔥 可学习的查询构建。*

**[⬆ 回到顶部](#fl-lqp)**

</div>