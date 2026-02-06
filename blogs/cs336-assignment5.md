---
layout: page
permalink: /blogs/cs336-assignment5/index.html
title: CS336-Assignment5
---

# CS336-Assignment5

## Targets

- Run SFT on Qwen 2.5 Math 1.5B with reasoning traces from R1

- Run Expert Iteration on Qwen 2.5 Math 1.5B with verified rewards.

- Run GRPO on Qwen 2.5 Math 1.5B with verified rewards.

- [代码仓库](https://github.com/ShallowU/cs336-assignment/tree/main)

- 本篇文章大部分由Gemini3 Pro完成(讲得不错可以看个大概，光看不行且本节适合手敲一遍代码)

  
  
  **SFT、EI、GRPO 的联系和区分**

| **维度**          | **SFT (监督微调)**                       | **EI (专家迭代)**                                         | **GRPO (组相对策略优化)**                               |
| ----------------- | ---------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| **数据来源**      | **静态**：人工标注的标准答案 (Gold Data) | **动态**：模型自己生成的正确答案 (Synthetic Data)         | **动态**：模型生成的答案 (无论对错)                     |
| **学习方式**      | **模仿**：老师怎么写，我就怎么写         | **探索+模仿**：我自己试出来的解法，如果是对的，我就背下来 | **试错+强化**：和同伴比，分高的受奖励，分低的受惩罚     |
| **对答案的要求**  | 必须严格匹配每一个字                     | 只要最终答案数值对，过程是模型自己的风格                  | 只要最终答案数值对 (Reward高)，过程不限                 |
| **是否有 Critic** | 否                                       | 否                                                        | **否 (这是GRPO区别于PPO的关键)**                        |
| **效果上限**      | 受限于训练集质量，容易过拟合             | 可以超越训练集，泛化性好，但容易收敛到单一模式            | **上限最高**，鼓励探索多样性，DeepSeek-R1证明了其有效性 |
| **计算开销**      | 低 (只需一次前向后向)                    | 中 (需要推理生成数据 + SFT训练)                           | 高 (需要大量采样 Rollout + 复杂的梯度计算)              |

注意：它们都是为了让模型学会做数学题。EI 其实是 SFT 的一种高级形式（数据增强）。GRPO 是 RL，但不需要价值网络。

------

## 引言：当“预测下一个词”不再够用

如果你读这篇博客，我相信你已经熟悉了**预训练（Pre-training）**——把海量文本喂给模型，让它学会预测下一个 token。这时候的模型像个博学的“野孩子”，它能续写小说，但它不知道怎么解数学题，也不知道怎么遵循复杂的指令。

最近，**DeepSeek-R1** 的横空出世让我们看到了**后训练（Post-training）**的巨大威力。它证明了通过合理的强化学习（RL），模型可以涌现出惊人的推理能力（Chain-of-Thought）。

**但是，复现 R1 需要几千张卡吗？**

并不是！在本文中，我将基于**单张 A100 (80G)**，带你亲手实现大模型后训练的三大里程碑：

1. **SFT (监督微调)**：教会模型基本的做题格式。
2. **Expert Iteration (专家迭代)**：让模型通过“刷题”自我提升。
3. **GRPO (组相对策略优化)**：DeepSeek-R1 的核心算法，让模型在探索中进化。

所有代码均基于 PyTorch 和 vLLM 原生实现，拒绝黑盒。

------

## 第一部分：基石——后训练的“基础设施”

在开始训练之前，我们需要在 `utils.py` 中造几个轮子。后训练与预训练的核心区别在于：我们**只关心模型回答得好不好**，而不关心它是否记住了问题。

### 1. 怎么只让模型学习“答案”？ (`tokenize_prompt_and_output`)

在预训练中，计算 Loss 是针对整句话的。但在指令微调中，我们必须使用 **Mask（掩码）** 技术。

我们需要构建一个 `response_mask`：

- **Prompt 部分**：Mask = 0（不计算 Loss，不进行梯度更新）。
- **Output 部分**：Mask = 1（计算 Loss）。

看看代码是怎么实现的：

Python

```python
# utils.py
def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    # ... (省略 Tokenize 过程)
    
    # 构建 Mask：只在 output 部分为 1
    # output_start 是 prompt 的长度
    for j in range(output_start, output_end):
        mask[j] = 1.0
        
    return {
        "input_ids": ..., 
        "labels": ..., 
        "response_mask": ... # 关键！
    }
```

### 2. GRPO 的核心魔法：告别 Critic 模型 (`compute_group_normalized_rewards`)

传统的 PPO 强化学习需要一个 Critic 模型（价值网络）来给动作打分，但这会消耗双倍显存。DeepSeek-R1 采用的 **GRPO** 算法极其优雅地解决了这个问题：**全靠同行衬托**。

我们对同一个问题生成一组（Group）回答（比如 16 个），然后计算它们的平均分。

- 比平均分高 $\rightarrow$ **优势 (Advantage) 为正** $\rightarrow$ 鼓励。
- 比平均分低 $\rightarrow$ **优势 (Advantage) 为负** $\rightarrow$ 抑制。

代码实现非常直观：

Python

```python
# utils.py
def compute_group_normalized_rewards(..., group_size, ...):
    # ...
    # 计算这一组回答的平均分
    group_mean = group_rewards_tensor.mean()
    # 优势 = 自己的得分 - 平均分
    advantages = group_rewards_tensor - group_mean
    # 除以标准差进行归一化 (GRPO Standard)
    if normalize_by_std:
        advantages = advantages / (group_std + 1e-8)
    return advantages
```

------

## 第二部分：起步——监督微调 (SFT)

SFT 是最基础的环节。我们使用 **GSM8K** 数据集，通过简单的“模仿学习”让模型学会 `推理答案` 这种输出格式。

在 `sft.py` 中，我们的训练核心就是带 Mask 的交叉熵损失（NLL Loss）：

Python

```python
# utils.py
def sft_microbatch_train_step(policy_log_probs, response_mask, ...):
    # 只计算 response_mask 为 1 的部分的负对数似然
    loss = masked_mean(-policy_log_probs, response_mask, dim=-1).mean()
    loss.backward()
    return loss
```

**实战效果**：经过 SFT，Qwen2.5-1.5B 在 GSM8K 上的准确率从 基础**12%** 提升到了 **39%**。模型学会了做题，但还不够聪明。

------

## 第三部分：进阶——专家迭代 (Expert Iteration)

SFT 只能学现成的答案。如果让模型自己做题，把做对的题收集起来教自己，会怎么样？这就是 **Expert Iteration (EI)**。

### 自举（Bootstrapping）流程

EI 的本质是 `Generate` $\rightarrow$ `Filter` $\rightarrow$ `Train` 的循环：

1. **Generate**：使用 vLLM 对每个问题采样 4 个答案（Temperature=1.0，鼓励探索）。
2. **Filter**：使用奖励函数（Reward Function）筛选。只有**格式正确**且**答案数值正确**的样本才会被保留。
3. **Train**：把筛选出的“高分样本”加入训练集进行 SFT。

### 关键实现：生成与筛选

- **生成**：利用 `vLLM` 极速生成，设置 `n=4`（一题生成4个解）。
- **筛选**：重点讲解 `reward_fn`。只有当**格式正确**且**答案数值正确**时，才保留这条数据。
  - *代码引用*：`if result['format_reward'] == 1.0 and result['answer_reward'] == 1.0:`。
- **训练**：把筛选出来的“正确数据”当成新的 SFT 数据集进行训练。

```python
# expert_iteration.py
# 只有当格式和答案都完美时，才加入训练集
if result['format_reward'] == 1.0 and result['answer_reward'] == 1.0:
    prompts_filtered.append(sampled_prompts[j])
    answers_filtered.append(generated_text)
```

**实战效果**：准确率进一步提升至 **46%**。但 EI 有个致命弱点：如果模型根本不知道怎么做某类题（永远蒙不对），它就永远学不会。

------

## 第四部分：终极形态——GRPO (DeepSeek-R1 核心)

为了突破 EI 的瓶颈，我们需要引入**相对优势**。即使一组回答都错了，GRPO 也能告诉模型“哪个错得离谱一点”，从而提供更密集的梯度信号。

### 1. 单卡 A100 的极限挑战：显存爆炸 (OOM)

RL 训练最头疼的是：

- 我们需要 **vLLM** 进行快速推理（Rollout）。
- 我们需要 **PyTorch** 进行反向传播（Train）。
- 这两个大模型同时塞进 80G 显存？**直接 OOM！**

### 2. 解决方案：时间换空间 (CPU Offload)

我在 `grpo.py` 中实现了一个**权重搬运机制**。在 Rollout 阶段，我们将训练模型的权重卸载到 CPU，腾出显存给 vLLM；Rollout 结束后，再把权重加载回 GPU 进行训练。

Python

```python
# grpo.py
def load_policy_into_vllm_instance(policy, llm):
    torch.cuda.synchronize()
    # 关键：先移动到 CPU，避免 GPU 显存双倍占用
    state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
    gc.collect()
    torch.cuda.empty_cache()
    # 加载进 vLLM
    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())
```

### 3. PPO-Clip 损失函数

GRPO 的训练目标不仅是最大化奖励，还要防止模型“步子迈太大”。我们使用了 PPO 的 Clipping 机制：

$$L = -\min(\text{ratio} \cdot A, \text{clip}(\text{ratio}, 1-\epsilon, 1+\epsilon) \cdot A)$$

Python

```python
# utils.py
def compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange):
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    # 取最小值（悲观估计），保证训练稳定性
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    return loss
```

### 4. 实战效果：涌现！

经过 GRPO 训练，模型准确率飙升至 **67%**！

更令人惊喜的是，模型学会了在 `<think>` 标签中输出更长的思维链。它开始学会**自我纠错**，为了获得最终的正确奖励，它不得不学会把推理过程写清楚。这就是 DeepSeek-R1 背后的秘密。

## 第五部分：遇到的问题

遇到的主要问题是训练GRPO时候，由于需要有vllm进行rollout采样模型和一个反向传播的pytorch更新模型，80G的显存也耐不住。

运行几个step以后就报显存OOM或者cuda非法访问地址，这是由于两个模型和pytorch的操作是异步的，两者形成了竞争关系，vllm经常需要热加载pytorch的参数，所以导致报错。

**解决方案**
首先参数设置：`enforce_eager=True`千万不要开启，会造成由显存碎片的问题。而用于测试的vllm实例可以开启。

```python
# 创建 vLLM 实例（仅用于 rollout）
print("初始化 vLLM...")
llm = LLM(
    model=config.model_path,
    dtype="bfloat16",
    gpu_memory_utilization=config.gpu_memory_utilization,
    device=device,
    # enforce_eager=True
)
```

同步和清理机制,以及先加载到CPU中offload
```python
def load_policy_into_vllm_instance(policy: nn.Module, llm: LLM) -> None:
    # 1. 强制同步，确保训练计算图已执行完毕
    torch.cuda.synchronize()
    
    # 2. 【关键】移动到 CPU。
    #    这既避免了 GPU 显存翻倍（OOM），又彻底隔绝了 CUDA 指针冲突（Illegal Access）。
    #    虽然有数据传输开销，但对于几十步才做一次的 Rollout 来说，稳定性远比这点速度重要。
    state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
    
    gc.collect()
    torch.cuda.empty_cache()
    # 3. 加载到 vLLM (vLLM 会自动处理从 CPU 到 GPU 的搬运)
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
    
    # 4. 清理内存
    del state_dict
    gc.collect()
    # 5. 再次同步，确保 vLLM 加载完成前不进行后续操作
    torch.cuda.synchronize()
```



## 总结

从 12% 到 67%，我们用几百行代码复现了大模型对齐的完整进化史。

- **SFT** 教会了“规矩”。
- **EI** 尝试了“自学”。
- **GRPO** 通过“内卷”（组内竞争）激发了潜能。

最重要的是，这一切都可以在一张 A100 80G上完成。希望这篇博客能成为你大模型后训练之路的起点！
