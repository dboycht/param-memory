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

## 5. 阶段与实际结果（T1 → T8）

| 阶段 | 内容 | 结果 |
| --- | --- | --- |
| **T1** | 写入 → 上下文驱逐 → 读出（最小闭环） | 本槽召回 4/4；目标 CE ≈ 0；`mem_off` 与"从未写过"**逐字符相同**；隔离**逐张量断言** |
| **T2** | 读时组合：**选，而不是和** | 求和 0/16、选一槽 14/16（上限 15/16）、选两槽 4/16；路由 0.938，且**差距 = 路由错误** |
| **T3** | 写入判据（always / surprise / selfcheck / explicit） | 无标签自检在同等召回下 **5 次写入 vs 18 次**、基座损伤 **0.54 vs 2.59 nats**；**似然判据被否证** |
| **T4** | 容量与遗忘 | **30/30 次驱逐逐位回初始态**；"钢印"硬保护有效；只有 FIFO 会丢掉用过的记忆 |
| **T5** | 上下文惯性（**假设被否证**） | 上下文臂与参数臂**完全相同**（1.00 vs 1.00），控制臂精确塌回地板 |
| **T6** | 持久化 | 加载前 `virgin=True`、重启后召回 6/6、擦除仍逐位精确；8 槽快照 18.4 MB |
| **T7** | 公开基准受控诊断 + 外部基线 + LLM 判官 | 参数记忆 **30/30**、配对 **8 胜 0 负**、**0 额外 prompt token**；上下文臂 0.867–0.967 但要付 117–1688 token |
| **T8** | B1：写入目标加"已写记忆保留项" | **假设未复现**：小样本（n=24）看着像修好了 k=2（p=0.070），**四倍样本（n=64）没有复现**（11 帮助 / 10 损害，p=1.000）。站得住的是基座损伤 **−79%** 与**全开**档的单向小改善（0 → 0.109，p=0.016）|

> 详录：`docs/04-roadmap.md`（路线与验收）、`docs/06-public-benchmark-feasibility.md`（公开基准、判官、
> 两次自我更正）、`runs/RESULTS.md`（由各阶段 JSON 自动汇总，含 provenance）。

## 6. 目标设定（诚实边界）

- 算力：**单卡 RTX 5060 Laptop 8 GB**。
- 基座：**Qwen3-0.6B**（主，bf16 + LoRA 旁路）；Qwen3-1.7B 作**规模对照**。
  （2026-09-29 用户决定：用尽可能小的模型，只要能证明思路正确且有效即可——设备较弱。
  实测两个规模上"写得进 / 擦得净 / 读时叠加是瓶颈"这组结论一致，而 0.6B 快一倍、显存省一个数量级。）
- 因此本工作的定位是 **机制验证 + 小规模受控实验**，不声称"训出一个会记忆的大模型"；结论的可迁移性
  由"机制是否随规模单调"这一步单独讨论（论文的 limitation 节）。

## 7. 目录

```
docs/          设计与协议文档（问题陈述、指标、隔离协议、公开基准可行性）
src/parammem/  记忆模块：槽位 / 写入器 / 路由 / 判据 / 淘汰 / 持久化 / LLM 判官 / 配对检验
bench/         受控合成基准生成器 + 判官人工标注（入库以便审计）
experiments/   T1~T8 实验入口，run_all.py 统一驱动（full / quick / collect-only）
paper/         英文论文（LaTeX）+ 中文技术报告（report-zh.md）+ 表格生成器
tests/         单测
runs/          各阶段 JSON 与自动汇总（不入库：仓库只放源码）
```

## 8. 状态

- ✅ **T1–T8 全部跑通**；`experiments/run_all.py` 一条命令驱动，`--collect-only` 零算力重建汇总。
- ✅ 单测**全绿**；论文**真编译**（TeX Live 2025，12 页），论文与中文报告里的数字**全部由
  `runs/summary.json` 生成**（88 个宏 / 10 张表，**无手抄**，并有测试守住引用完整性）。
- ✅ 英文论文 + 中文技术报告 + 可复现脚本；公开基准受控诊断带**预注册的留出集复制**（另 30 题，0.917）。
- 🔬 **两个负结果**（上下文惯性假设、似然型写入判据）与**三次方法学自我更正**
  （含式判据偏向逐字复现；生成预算截断把结论翻了个面；**小样本的"修复"在大样本下消失**）
  都如实写入论文，而不是藏起来。
- ⬜ 未做：公开基准的完整适配（需"轮→问答对"抽取 + 跨条推理读路径）、多种子全矩阵重跑、
  4B 规模、双层时间尺度。
- ⬜ **预印本暂不发布**（2026-09-29 用户决定：等成果更强再发；投稿链路已备好，
  `python paper/build_arxiv_bundle.py` 一键重建源文件包、纯文本摘要与提交清单）。

### 当前实现要点

| 文件 | 作用 |
| --- | --- |
| `src/parammem/bench/protocol.py` | P1 驱逐检查 / P2 反事实写入 / P3 负对照 / P4 适配器消融 / P5 提示词对照 + bootstrap CI |
| `src/parammem/bench/entities.py` | 可证明不碰撞的虚构实体生成（编码式，跨种子必然不同） |
| `src/parammem/memory/slots.py` | 秩块划分的 LoRA 槽位：读掩码做消融、`erase` 做**精确遗忘** |
| `src/parammem/memory/writer.py` | 写入器：每次写新建优化器（避免冻结槽漂移），运行时**断言隔离** |
| `src/parammem/memory/retention.py` | **保留项**该保护哪些旧记忆（纯函数，带单测） |
| `src/parammem/memory/router.py` | 查询键余弦路由（取键时关闭记忆，避免键随已写记忆漂移） |
| `src/parammem/eval/judge.py` | LLM 判官：一候选一次调用、凭据**不进日志/不入库**、传输层可注入以便离线测试 |
| `src/parammem/eval/paired.py` | 配对符号检验（同一批题/探针的比较，不靠重叠区间下结论） |
| `src/parammem/report.py` | 各阶段 JSON → 论文宏与表格（**数字的单一来源**） |

## License

MIT — 见 `LICENSE`。
