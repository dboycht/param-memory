# param-memory · 思想钢印（Thought Stamp）

> **把长期记忆从「上下文」下沉到「参数」** —— 旁路低秩适配器 + 运行期梯度写入 + 可遗忘机制。
> 研究工作代码仓库（实验进行中，**当前阶段 T0**）。

**English abstract.** Long-term memory in today's LLM agents lives in the context window. That channel is
zero-sum: every remembered token competes with the tokens needed for reasoning, and remembered content is
indistinguishable from the current working set, so a topic switch drags stale context along. We study an
alternative: keep the backbone frozen and attach a **side-path low-rank memory** that is written at runtime by
gradient updates, read implicitly through the forward pass, and **forgotten** through capacity-bounded
eviction, utility decay and time decay. The central methodological requirement is an **eviction-isolation
protocol**: memory content must be provably absent from the context before we may claim that the parameters
are doing the remembering.

---

## 1. 问题：上下文是零和通道，而且甩不掉

不改参数的前提下，模型唯一能"写入"新信息的地方就是上下文。这带来三个结构性缺陷：

1. **零和**：记忆占用的额度 = 从推理/工作上下文里抢走的额度。记得越多，可用上下文越短。
2. **无差别驻留**：上下文里的每一段都同等"在场"，没有"长期记忆 vs 当前工作集"的分层，重要与否都占预算。
3. **惯性 / 锚定**：上下文中的内容被当作"正在发生的现实"而非"可调用的历史"。话题跳转时，模型被上文
   的主题、语气、甚至**错误前提**拖着走 —— 记忆与工作记忆没有分离。

## 2. 方案：旁路参数记忆（thought stamp）

- **主干冻结**，在主干旁路挂一个专用记忆模块（低秩适配器），承担「写 — 存 — 忘 — 读」四件事。
- **写入**：运行期由梯度更新完成，不经过上下文（因此不占上下文预算）。
- **读取**：通过前向传播隐式生效，不需要把记忆内容重新检索进上下文。
- **遗忘**：容量上限 + 效用淘汰 + 时间衰减，多机制可插拔；目标是**只忘该忘的，不掉基座能力**。
- **可归因**：任何"参数记住了 X"的结论，都必须在 **X 已被驱逐出上下文** 的条件下取得。

## 3. 与"上下文记忆 / 外挂向量库"的区别

| | 占上下文预算 | 话题跳转时被拖拽 | 可被遗忘 | 需要检索步骤 |
| --- | --- | --- | --- | --- |
| 长上下文 / 历史拼接 | 是（零和） | 是 | 否 | 否 |
| 外挂向量库 / RAG 记忆 | 检索命中后才占 | 较弱 | 是（删条目） | 是 |
| 知识编辑（ROME/MEMIT 类） | 否 | 否 | 难（易伤基座） | 否 |
| **本项目：旁路参数记忆** | **否** | **待验证（核心假设）** | **是（核心设计）** | **否** |

## 4. 研究问题（RQ）

- **RQ1（写得进吗）**：把一段对话中的事实/经验写进旁路参数后，在**该信息已完全离开上下文**时，模型能否正确读出？
- **RQ2（忘得对吗）**：容量受限时，效用淘汰 + 时间衰减能否在保留重要记忆的同时清掉噪声，且不损伤基座能力？
- **RQ3（救得了惯性吗）**：参数记忆是否比"把同样内容放进上下文"更能抵抗话题跳转带来的上下文惯性/锚定？
- **RQ4（代价多少）**：写入-读取的显存/时间开销，与上下文方案相比的净收益在哪里？

## 5. 阶段路线（T0 → T4）

| 阶段 | 内容 | 验收判据 |
| --- | --- | --- |
| **T0** | 问题陈述、与现有工作的差异、指标与**驱逐隔离协议** | 设计文档定稿；协议可实现、可写成断言 |
| **T1** | 最小闭环：旁路记忆写入 + 上下文驱逐后读出 | 受控合成任务上，驱逐后命中率显著高于"无记忆"基线 |
| **T2** | 遗忘：容量上限 + 效用淘汰 + 时间衰减 | 容量–保留率–干扰曲线；基座能力退化在预算内 |
| **T3** | 上下文惯性实验；双层时间尺度（会话快记忆 / 跨会话慢固化） | 惯性指标相对上下文基线下降 |
| **T4** | 评测矩阵 + 公开长对话基准对照 + 论文 | 英文 arXiv 预印本 + 中文技术报告 + 全部脚本可复现 |

## 6. 目标设定（诚实边界）

- 算力：**单卡 RTX 5060 Laptop 8 GB**。
- 基座：**Qwen3-0.6B**（主，bf16 + LoRA 旁路）；Qwen3-1.7B 作**规模对照**。
  （2026-09-29 用户决定：用尽可能小的模型，只要能证明思路正确且有效即可——设备较弱。
  实测两个规模上"写得进 / 擦得净 / 读时叠加是瓶颈"这组结论一致，而 0.6B 快一倍、显存省一个数量级。）
- 因此本工作的定位是 **机制验证 + 小规模受控实验**，不声称"训出一个会记忆的大模型"；结论的可迁移性
  由"机制是否随规模单调"这一步单独讨论（T4 的 limitation 节）。

## 7. 目录

```
docs/          设计与协议文档（问题陈述、指标、隔离协议）
src/parammem/  记忆模块、写入判据、遗忘策略
bench/         受控合成基准生成器
experiments/   T1~T4 的实验入口脚本
paper/         英文论文（LaTeX）
```

## 8. 状态

- [x] **T0-a** 项目骨架与仓库
- [x] **T0-b** 三路文献侦察（快权重/TTT、持续学习与遗忘、记忆评测与惯性）并回填定位
- [x] **T0-c** 设计文档：问题陈述、系统设计、**P1–P5 驱逐隔离协议**
- [ ] T0-d 文献差异表（`docs/03-related-work.md`，撰写中）
- [x] **T1-a** 协议原语 + 虚构世界 + episode 生成器 + rank-blocked LoRA 槽位 + 写入器（**85 例单测全绿**）
- [ ] **T1-b** 端到端跑通：写入 → 上下文驱逐 → 读出（等基座权重就绪）
- [ ] T2 遗忘 / T3 惯性 / T4 论文

### 当前实现要点

| 文件 | 作用 |
| --- | --- |
| `src/parammem/bench/protocol.py` | P1 驱逐检查 / P2 反事实写入 / P3 负对照 / P4 适配器消融 / P5 提示词对照 + bootstrap CI |
| `src/parammem/bench/entities.py` | 可证明不碰撞的虚构实体生成（编码式，跨种子必然不同） |
| `src/parammem/memory/slots.py` | 秩块划分的 LoRA 槽位：读掩码做消融、`erase` 做**精确遗忘** |
| `src/parammem/memory/writer.py` | 写入器：每次写新建优化器（避免冻结槽漂移），并在运行时**断言隔离** |
| `src/parammem/memory/store.py` | 槽位元数据 + 四种淘汰策略对比（fifo/lru/lfu/utility） |

## License

MIT — 见 `LICENSE`。
