# 03 · 相关工作与定位（T0 交付物 4/4 · 文献侦察汇总）

> 日期：**2026-09-29**（跑 `Get-Date -Format 'yyyy-MM-dd'` 读得，非记忆）。
> 本文由**三路文献侦察汇总**：分支 A（测试时训练与快权重）、分支 B（持续学习与知识编辑）、分支 C（记忆评测与上下文惯性），
> 原始文件见 `D:\code\DeepSeekHarness\.scratch\param-memory-recon\` 下的 `A-fast-weights-ttt.md`、`B-forgetting-continual.md`、`C-memory-eval-and-inertia.md`。
> **证据分级见 §6**：三份侦察文件把每条信息标为【已读原文】/【二手转述】/【未核实】，本文档照抄其分级，不升格。
> **仓库许可（§9）是本文档唯一新增的一手核实结果**：由本机直连 GitHub REST API 实测（`github.com` 被本机加速器改写到 `127.0.0.1`，`web_fetch` 会被判非公网 IP 拒；`api.github.com` 经 `Invoke-RestMethod` 可用）。

---

## 1. 定位一句话

已有工作分四类：**① 上下文层记忆系统**（MemGPT/Letta、Mem0、A-MEM、MemoryBank、MemoRAG、Zep）、**② 外挂向量库/RAG**、**③ 测试时训练与快权重**（TTT 系、Titans 系、DeltaNet 系、GradMem、Cartridges）、**④ 知识编辑与持续微调**（ROME/MEMIT/GRACE/WISE/AlphaEdit、EWC/O-LoRA/InfLoRA）。

**本项目占据的空位是：旁路 + 冻结主干 + 运行期梯度写入 + 可精确擦除的容量受限槽位。**

拆开说，这四个定语各自都有先例，但**没有任何一篇同时具备四者**：

- **旁路 + 冻结主干 + 运行期梯度写入**：GradMem（arXiv:2603.13875）已做到，但它**没有遗忘机制**；
- **旁路 + 写参数 + 有遗忘**：Titans / ATLAS / Miras 这条线做到了，但它们**是架构级、要重训**，不是给现成 LLM 挂旁路；
- **可精确擦除 + 容量受限槽位**：知识编辑侧的 AlphaEdit（零空间投影）与 GRACE（外挂 codebook）各自解决了"不弄坏基座"和"不改权重"，但**没有"槽位容量 + 淘汰 + 精确擦除"这一套**；
- **上下文惯性的量化**：三份侦察的一致负面结论是**没有专门基准**（§5a）。

侦察文件 A 的读表结论可以直接引用：同时满足「旁路 + 不动主干 + 真写参数 + 有遗忘」的**只有 Titans/ATLAS/Miras 这一条线**；GradMem 满足「旁路 + 冻结 + 写参数 + 实时」但缺遗忘。
⇒ **把 Miras 式 retention gate 补进 GradMem 式写入、并把载体换成可枚举可擦除的秩块槽位，就是本项目的技术定位。**

---

## 2. 定位表（核心）

| 工作 | 出处（year / venue / arXiv） | 一句话思想 | 与本项目的关系 | 我们不同在哪 |
| --- | --- | --- | --- | --- |
| **Titans** | 2024-12-31，arXiv:2501.00663（CC BY 4.0；发表状态待核实） | 注意力旁挂神经长期记忆模块，测试时在线学习；attention=短期记忆、neural memory=长期记忆 | **最同构的先例，related work 排第一位**；"测试时学习 + 神经记忆 + 衰减遗忘"三件套就是我们想做的事 | Titans **架构级、需预训练**；我们给**现成冻结 LLM** 挂旁路。Titans 记忆是固定尺寸 MLP 隐参数，我们是**可枚举、可单独擦除的秩块槽位** |
| **GradMem** | 2026（Comments 标 ICML 2026），arXiv:2603.13875 | 冻结主干，用 per-sample 测试时梯度下降把上下文写进少量 prefix memory tokens，目标函数是 context 重建损失 | **"旁路 + 冻结主干 + 真写参数 + 读时不需原上下文"的第一个正式论文**，直接可复现蓝本；其"更多梯度步 > 重复前向写入"支持梯度写入路线 | GradMem **无遗忘机制**（长期累积会溢出）、载体是软提示式 token（容量/可解释性受 token 数限制）、评测是 bAbI/SQuAD 变体而非对话；我们**显式容量 K + 淘汰 + 精确擦除** |
| **TTT layers** | 2024-07-05，arXiv:2407.04620 | 把 RNN 隐状态本身做成一个模型，隐状态更新 = 一步自监督学习 | "记忆=参数"的最纯粹实现；证明"压进参数"能随 token 增长持续降困惑度 | 隐状态固定尺寸、更新在主干内；我们**不动主干**、记忆是外挂且可逐条删除 |
| **TTT-E2E** | 2025-12-29/31，arXiv:2512.23675 | 长上下文重述为持续学习：标准 Transformer + 滑窗注意力，推理时用 NTP 对权重做梯度更新，并用 meta-learning 学初始化 | 最强"写参数 > 塞上下文"工程证据：3B/164B tokens 下扩展性与全注意力一致（Mamba-2、Gated DeltaNet 做不到），128K 时比全注意力快 2.7×、延迟与长度无关 | TTT-E2E **更新主干权重**（与"不动主干/可回滚/多用户隔离"冲突）、meta-learning 管线重、不涉及遗忘与对话记忆 |
| **qTTT（Let's (not) just put things in Context）** | 2025-12-15，arXiv:2512.13898（ICLR 2026 接收为二手、未核实） | 静态自注意力有 score dilution，"能吞下的远多于能可靠使用的"；改为对给定上下文做定向梯度更新 | 本项目"记忆下沉、不占上下文预算"的**最重要外部背书**；其"仅 query 驱动更新"还是 §3 隔离协议的一个可借手法 | 每次新 context 都要重训（context-specific，**非累积式长期记忆**）、同样更新主干、只测长文本 QA |
| **ATLAS** | 2025-05-29，arXiv:2505.23735 | 诊断 Titans 三病根：容量受架构/特征映射限制、在线更新只优化最后一个输入、固定尺寸记忆管理表达力弱；用"当前+过去 token 共同优化"等三管齐下 | **"只按最新一轮更新记忆"这个坑的直接警告**，其"取一小段历史一起优化"是可借鉴的批量写入策略；容量是该路线第一瓶颈 | 仍是架构层研究（要重训），评测为长上下文理解/召回；我们是旁路 + 现成模型 |
| **Miras** | 2025-04-17，arXiv:2504.13173 | 把 Transformer/Titans/线性 RNN 统一为"用 attentional bias 学 key→value 的联想记忆"，分解为四正交选择（架构 / 目标 / retention gate / 学习算法） | 给了**记忆模块设计空间的 checklist**；**(iii) retention gate 正是"不用的记忆衰减"的理论化形态** | Miras 是要重训的框架、四类选择最优组合依赖任务；我们取其 retention gate 作为**槽位淘汰**的判据来源 |
| **HOPE / Nested Learning** | 2025-12-31，arXiv:2512.24695（Comments 注明 NeurIPS 2025） | 模型 = 一组嵌套的多层级优化问题（各有目标/学习率/更新频率/上下文窗口）；CMS 推广快慢分层记忆 | CMS = "多时间尺度记忆 + 各自更新频率"的正式表述，与"有的常更新、有的长期冻结、不用的衰减"同构 | 框架性/偏理论、无官方代码、评测无对话记忆；我们只取"多时间尺度槽位"这一条落到工程 |
| **DeltaNet / 快权重程序员** | 2021 ICML，arXiv:2102.11174；并行化版 2024-06，arXiv:2406.06484 | 线性注意力 ≡ 90 年代快权重程序员；把加法外积换成 delta rule（先减旧值再写新值）可修正 key→value 映射 | 给出"写记忆"的正确数学形式（**修正式写入而非只增不改**），是我们写入算子的数学依据 | 层内机制、不解决"何时写"；遗忘是门控式的，不是条目级语义淘汰 |
| **Mamba-2 / SSD** | 2024 ICML，arXiv:2405.21060 | SSM 与注意力共享结构化半可分矩阵理论，Mamba-2 核心层比 Mamba 快 2–8× | 若旁路记忆用 SSM/线性注意力实现，SSD 是当前最优工程底座 | SSD **不承诺容量足够**（见 Zoology / Repeat After Me）；不含测试时写入语义与淘汰设计 |
| **Memory Layers at Scale** | 2024-12-12，arXiv:2412.09764 | 可训练 key–value 查表（product-key）记忆层：加参数几乎不加 FLOPs，专存事实 | "加参数不加算力"的旁路记忆载体；淘汰 = 删某个 key，比改权重更可解释 | 稀疏查表**非权重式记忆**，写入需可微训练、**无遗忘/时效机制**、需预训练期就插入（不能事后挂冻结模型） |
| **Cartridges** | 2025-06-06，arXiv:2506.06266（ICLR 2026 接收为二手、未核实） | 为每份语料离线训练一份更小 KV cache，配 self-study（合成对话 + context distillation） | "记忆→可训练参数"的第二条载体；**不同语料的 Cartridge 可在推理时组合而不需重训**（多用户/多主题记忆并存的直接借鉴）；合成对话+蒸馏是可行配方 | 离线、per-corpus，**不是每轮对话实时写**；**无遗忘**；评测面向语料检索而非人格化长期记忆 |
| **SEAL** | 2025，arXiv:2506.10943（S2 venue = NeurIPS 2025） | LLM 自生成微调数据与更新指令，用 SFT 固化为持久权重更新，RL 以下游表现为奖励 | 唯一"真改主干且由模型自己决定怎么写"的路线；回答了最难的"写什么" | **改主干** ⇒ 与可回滚/隔离冲突；RL+SFT 成本极高、有灾难性遗忘与安全风险；评测为知识注入，未验多轮对话 |
| **Adapters / LoRA / Prefix-Tuning** | 1902.00751（2019）；2106.09685（2021）；2101.00190（2021） | 冻结原权重，只训练旁路小参数（adapter / 低秩矩阵 / 连续前缀），可增删切换 | "旁路可写参数"的成熟形态，也是我们**记忆载体的物理形式**（低秩矩阵） | 标准用法是**离线监督微调**；低秩容量有限；稠密 LoRA 没有"条数"概念、无法单条因果消融 |
| **EWC** | PNAS 2017；arXiv:1612.00796 | 用 Fisher 信息给重要权重加二次惩罚（弹性弹簧），只放不重要的权重改 | "写进去但不弄坏基座"的**正则化派祖师爷**，我们写入损失里的 KL 锚定与之同源 | EWC 要**按任务存 Fisher 与 θ\***，内存随任务数线性增长；**需要任务边界**，而对话写入没有天然边界 |
| **Synaptic Intelligence** | ICML 2017，arXiv:1703.04200 | 在线累积每个突触对损失下降的贡献作为重要性，无需事后估 Fisher | 比 EWC 更"在线"，天然匹配流式对话 | 仍属正则化派，容量硬上限未解；重要性路径依赖训练轨迹 |
| **PackNet** | CVPR 2018，arXiv:1711.05769 | 每学完一个任务剪枝并打 mask 冻结，各任务用互不重叠的参数子集 | 架构隔离派代表；**mask 冻结 = "思想钢印"的工程实现** | **冻结参数无法回收**，与本项目"淘汰回收容量"直接冲突；需任务边界与推理期 task ID |
| **Progressive Networks** | 2016-06，arXiv:1606.04671 | 每个新任务新开一列网络，旧列全冻结，靠侧向连接取特征，按构造免疫遗忘 | **"旁路挂专用网络"这一架构选型的先例，可作为架构论证的引用锚点** | 参数量随任务数**线性增长**、无遗忘/回收；旁路若无淘汰必然膨胀 |
| **O-LoRA** | 2023-10，arXiv:2310.14152（作者名单待核实） | 每任务一套 LoRA，新任务低秩子空间与所有旧任务正交，旧 LoRA 冻结求和 | **最贴合"参数即记忆 + 不弄坏基座"的路线**：低秩增量为可插拔小块，正交约束为互不干扰的独立记忆槽 | 每任务留一份 LoRA ⇒ 内存随任务数增长、仍缺淘汰；**需任务边界**；正交子空间会耗尽（推断，未核实） |
| **InfLoRA** | **CVPR 2024**（非 ICLR），arXiv:2404.00228 | 把微调注入的参数重述为"在预训练权重某子空间内微调"，设计该子空间以消除新任务对旧任务的干扰 | 与 O-LoRA 同属子空间隔离，且**更明确地处理 stability–plasticity trade-off** | 同 O-LoRA 的按任务累积与任务边界问题；对"基座通用能力退化"是否有效，转述中未见数据（未核实） |
| **Merge before Forget** | 2025-12-28，arXiv:2512.23017（CC BY 4.0） | 正交基初始化新任务 LoRA 后顺序合并进同一 LoRA，用 A/B 非对称性做时间感知缩放 ⇒ **内存相对任务数恒定** | 正面命中本项目"容量满了怎么办"的墙；证明"常数内存 + 持续合并"可行 | **合并后无法按条撤销**（分支 B 基于摘要的推断，**原文是否讨论可撤销性未核实**），与"可精确擦除"存在张力；我们**保留逐槽可撤销**，代价是容量硬上限 K |
| **FADE（Learning to Forget）** | 2026-04-29，arXiv:2604.27063（Schmidhuber 组；arXiv 非独占许可） | 权重衰减作为**显式遗忘机制**，用近似元梯度逐参数自适应衰减率，让稳定参数慢忘、跟踪参数快忘 | **分支 B 里"遗忘"环节最直接可用的一篇**：把遗忘从外部操作变成优化过程的内生行为，无需淘汰事件 | FADE 推导在**在线线性设定**、明确只作用于**最后一层**、**无现成代码**；不做条目级淘汰。我们做**槽位级显式淘汰 + 精确擦除** |
| **语义缓存淘汰策略研究** | 2026-08-20，arXiv:2608.20280 | 18 个设定下评测 FIFO/LRU/LFU/ARC/GDSF/streaming-SISO/semantic-redundancy | 给本项目淘汰策略**唯一可引用的硬数字**：无策略比 LFU 好超过 **0.041 个百分点**；紧容量下 FIFO 落后 LFU **8.67 个百分点**；命中率 51–60% 但质量调整后仅 **1.1–2.2%** | 负载是**语义缓存**而非参数记忆，外推属类比；我们的淘汰对象是秩块槽位，必须自己测 |
| **D-MEM** | 2026-03-15，arXiv:2603.14597（CC BY 4.0） | Critic Router 按"惊讶度 + 效用"打分：低 RPE 走 O(1) 快缓存，高 RPE（事实矛盾/偏好变化）才触发 O(N) 记忆演化 | **"写什么"最贴合的一篇**：可量化的写入门控；"高 RPE = 事实矛盾/偏好变化"正是值得钢印化的内容 | 属 **agent memory 系统层**（知识图谱/外存），**不是参数级写入**；我们需把门控信号接到梯度/更新强度上 |
| **CLS 互补学习系统** | Psychological Review 1995, 102(3):419–457 | 快学习破坏旧知识、慢学习记不住单次经历 ⇒ 需要海马体（快、稀疏、绑定具体经历）+ 新皮层（慢、分布式、提取统计结构），经重放巩固 | **本项目架构选型的第一性原理**：主干=新皮层，旁路记忆=海马体；"巩固"对应我们的离线回写步骤 | 原论文是认知科学论证、基于当年浅层网络实验，**当作 LLM 蓝图属类比式论证，需实验支撑而非直接引用** |
| **ROME** | NeurIPS 2022，arXiv:2202.05262 | causal tracing 定位中层 MLP 为 key–value 记忆，对其做秩一更新插入新事实 | **事实级写入的最小可行基线**，写入成本极低（秩一） | ROME **一次只能写一条**、依赖逐模型层定位、连续编辑会退化；我们是批量、在线、无标签、可淘汰 |
| **MEMIT** | 2022-10/2023-08，arXiv:2210.07229 | 秩一更新分散到多个中层 MLP 并批量求解，一次写入数千条关联（GPT-J 6B / GPT-NeoX 20B） | **批量写入**的证明；"分散到多层以缓解单点扰动"可借鉴到写入布局 | 仍依赖 locate-then-edit 的层定位假设；顺序编辑退化问题依旧 |
| **GRACE** | NeurIPS 2023，arXiv:2211.11031 | **不改模型权重**，在某层旁挂离散 codebook：key=需修正的激活，value=修正后激活；推理时落进邻域就改写激活 | **"旁路挂专用记忆网络"的一个已验证实例**（不动基座 + 外挂 + 相似度路由）；其 add/expand/split 三种操作回答了"记忆条目如何增长与合并" | GRACE 是**激活/工作记忆级**编辑而非参数级（与"下沉到 weights"部分偏离）；codebook 持续增长、**摘要中未提供容量淘汰策略**（推断，未核实） |
| **WISE** | NeurIPS 2024，arXiv:2405.14768 | 主记忆（预训练参数，不动）+ 侧记忆（所有编辑写这里）+ router；知识分片把不同批编辑放进不同参数子空间再无冲突合并 | **理论层面最重要的一条**：其结构 = **main memory + side memory**，与本项目"主干 + 旁路"同构；**"不可能三角"（reliability / generalization / locality 不可兼得）应作为我们的评价框架** | WISE 仍是"旁路"，未真正"下沉到基座权重"；router 引入额外组件与训练成本；知识分片的容量行为（子空间何时耗尽）摘要未说明（未核实） |
| **AlphaEdit** | ICLR 2025 Oral，arXiv:2410.02355 | 把编辑扰动 Δ 先投影到"待保留知识"的零空间（Δ′=PΔ）再施加：零空间方向不影响保留知识的输出 | **防污染的最优性价比工具**：把"不弄坏基座"从软惩罚升级为**硬约束**，实现只需一行投影；本设计 §3.1 已采纳为 λ_cap 的实现 | 零空间维数有限、**随写入量增大会耗尽**（基于原理的推断，未核实）⇒ "防污染"与"容量淘汰"耦合；需知道"要保护什么"，对话场景边界模糊 |
| **RippleEdits** | TACL vol.12 pp.283–298 (2024)，DOI 10.1162/tacl_a_00644，**arXiv:2307.12976** | 编辑一条事实会牵连一批相关事实；据此建 5K 条事实编辑的诊断基准 | **对本项目最不利但必须正面引用的一条**：主流编辑方法在 ripple 一致性上普遍失败，而**简单的上下文编辑基线得分最高** ⇒ 我们必须说明"下沉到参数"在哪个维度换取了什么 | 该结论直接质疑"参数编辑"路线的一致性；我们的应对是**把一致性代价显式报告**（§4），而不是回避 |
| **连续编辑退化的正则化** | 2025-02-03/2025-05-21，arXiv:2502.01636 | 把 locate-then-edit 形式化为两步微调；退化源于内部激活过度优化 + **被编辑矩阵范数持续增长**；提 MPES 与 Frobenius 范数约束，可扩展至 10,000 次编辑、编辑时间 −42~61% | **直接工程含义：任何持续往参数里写的机制都必须监控权重范数**，否则写入本身会自毁。本设计 §3.2 已把"范数增长"列为必监控量 | 其对象是事实编辑而非对话记忆；正则化方案的有效性我们未复核（未核实） |
| **MemGPT / Letta** | arXiv:2310.08560 | 把 LLM 当 OS，函数调用在"主上下文/外部上下文"间换页，用 interrupt 控制流 | **上下文层记忆系统的代表**；其"外部上下文换页"隐喻是隔离协议里"信息驱逐"手法的一个借用来源 | **无参数更新**：记忆仍占上下文预算（只是分页），且**上下文惯性未被解决**——这正是本项目要打的点 |
| **Mem0** | arXiv:2504.19413 | 两阶段（抽取 → LLM 判定 ADD/UPDATE/DELETE/NOOP）动态维护显著事实的 memory store；图变体存三元组 | 最强工程 baseline（报称对 OpenAI 记忆方案 26% 相对提升、p95 延迟 −91%、token 成本 −90%+）；其 ADD/UPDATE/DELETE 语义与我们的槽位淘汰可对照 | 记忆在外部库、**无参数更新**；指标依赖 LLM-as-a-Judge（可被质疑）；"26% 相对提升"的参照口径原文只写 "over OpenAI"（未核实） |
| **A-MEM** | NeurIPS 2025，arXiv:2502.12110 | Zettelkasten 式互联笔记网络：新记忆生成结构化笔记并与历史动态建链，新记忆还反向演化旧记忆的表示 | "记忆之间互相关联"这一维度我们目前没有；可作为讨论章节的对照面 | 外部库、**无参数更新**；无显式容量回收；评测为 6 个基座上的对比 |
| **MemoryBank** | arXiv:2305.10250（**未核实**，仅二手摘要） | 用艾宾浩斯遗忘曲线驱动记忆的强化/衰减，并可合成用户人格画像；模型无关 | **直接冲击本项目"支持遗忘"卖点**：外部库方案已有遗忘机制（虽然是启发式曲线）。必须在 related work 里正面比较"遗忘的对象是文本条目还是参数槽位" | 遗忘是**对文本条目的启发式衰减**，不是参数擦除，无法保证"擦除后输出逐比特退化"；论文原文本次未读到 |
| **MemoRAG** | arXiv:2409.05591（**未核实**） | 双系统：轻量长程记忆（KV cache 压缩）产出 draft answer 当检索线索，重型模型做最终回答 | "用压缩记忆产生检索线索"的思路；说明"记忆"未必是原文条目 | 记忆是 KV cache + 检索线索，**无参数写入**；v3 标题已改名（版本差异未核实） |
| **Zep / Graphiti** | arXiv:2501.13956（**未核实**，3 次抓取失败） | 时序知识图谱（episodic/semantic/community 子图），记录"事实随时间如何变化"而非静态检索 | **"事实随时间变化"正是本项目需要的能力**（偏好变化、事实修正）；报称 DMR 94.8% vs MemGPT 93.4%、LongMemEval +18.5% —— **这些数字全部未核实，禁止引用** | 记忆在外部图库、**无参数更新**；所有性能数字本次无法一手核验 |
| **Evo-Memory** | 2025-11/2026-05，arXiv:2511.20857 | 把数据集组织成序列化任务流，要求 LLM 每次交互后检索/整合/更新记忆；统一实现 10+ 记忆模块、10 个数据集，给出 ExpRAG 与 ReMem 基线 | **"我们这套东西该用什么指标衡量"的现成对标基准**；且它测的是**记忆状态演化（无梯度）**，与我们写权重正好构成对照组 | 不涉及参数写入；评测是任务流而非长期陪伴；官方 repo 本次未核实 |
| **ProCL** | arXiv:2605.13162（**父代理一手核实**：abs 页 HTTP 200 + Semantic Scholar 交叉确认） | 持续 LoRA 框架 + **Program memory**（灵感取自神经科学互补学习系统）：把 LoRA 适配器组织成结构化 **program memory 槽位**，由 input-conditioned attention 动态检索；相似输入复用共享适配器区域、并为未来数据**保留未用容量**；槽位再与底层适配器结合（后者作分布式表示、跨任务逐步积累），以平衡可塑性与稳定性 | 🔴 **本项目最近的先验工作**：它的"LoRA 槽位 + 复用 + 保留容量"与我们的"秩块槽位"**直接同域** ⇒ 必须在正文正面比较，不能回避 | 差异三条：① 它面向**任务增量微调**（有任务数据集、离线批量），我们面向**对话运行期的无标签写入**；② 它**没有遗忘 / 精确擦除 / 容量淘汰**，而这正是我们的核心贡献；③ 它是**多个独立适配器**，我们是**单个适配器内的秩块划分**（擦除粒度与代价不同）。出处细节：Hung Le, Svetha Venkatesh；2026-05-13 提交；cs.LG；CC BY 4.0；18 页 preprint |
| **LongMemEval** | ICLR 2025，arXiv:2410.10813（arXiv 页 CC BY 4.0） | 五项长时记忆能力：信息抽取 / 跨会话推理 / 时序推理 / 知识更新 / 弃答；历史长度可缩放，含负样本干扰证据 | **最贴近本项目的现成基准**：跨会话 + 知识更新 + 弃答三块与我们的 RQ1/RQ2/RQ3 直接对应 | 它测的是"读得到吗"，不区分"记在参数里还是上下文里"⇒ 仍需 §5b 的隔离协议；S/M 档具体 token 数未核实 |
| **LoCoMo** | arXiv:2402.17753 | 超长期对话记忆（QA / 事件摘要 / 多模态生成），含时序题与"没提过"对抗题；QA 共 7,512 对 | 对话记忆的现成评测域；其"长上下文 LLM 与 RAG 都显著落后人类"是一个**双刃证据** | 仅 10 段对话、每段均 300 turns / 9K tokens ⇒ **统计噪声大，厂商自报分不可横向比较**（二手批评，未核实） |
| **MemoryAgentBench** | arXiv:2507.05257 | 增量多轮交互下四能力：精确检索 / **test-time learning** / 长程理解 / **选择性遗忘**；结论：当前所有方法无法四项全占 | 🔴 **最贴近本课题的现成评测骨架**：它把 "test-time learning" 与 "selective forgetting" 单列为能力项 ⇒ 建议作为主评测骨架，本项目的贡献正好落在它指出的"无法全占"处 | 该基准本身不含参数写入方法；四能力分离的判据需要我们自己接上 P1–P5 |

---

## 3. 三类质疑的正面回答（补齐证据）

### 3.1 "这不就是 O(1)/线性注意力的快权重吗？"

**支持这一质疑的证据（必须承认）**：
- Schlag et al. 2021（arXiv:2102.11174）**证明**线性化自注意力与 90 年代快权重程序员形式等价，key×value 外积就是写指令；DeltaNet（arXiv:2406.06484）把它扩展到 1.3B/100B tokens；
- Titans（arXiv:2501.00663）已经在做"旁路神经记忆 + 测试时学习 + 衰减遗忘"，且其记忆模块本身就带 surprise 写入与 weight decay 遗忘；
- Miras（arXiv:2504.13173）把 retention gate 理论化为遗忘的正规形态。

**反对"就是同一件事"的证据**：
- **固定尺寸状态的容量有硬上限**：Zoology（arXiv:2312.04927）在 17 个模型对照中发现门控卷积落后注意力最多 2.1 困惑度、**其中 82% 由联想召回能力解释**，70M 注意力模型在 AR 上打败 1.4B 门控卷积；"Repeat After Me"（arXiv:2402.01032）理论上证明两层 Transformer 可复制指数长字符串而**固定尺寸隐状态的 GSSM 做不到**；
- 因此把记忆塞进 RNN 隐状态与"记忆放在可枚举的低秩槽位里"，在**可归因性**上是两类东西：隐状态无法回答"第 7 条记忆对本次回答贡献了多少"，而槽位可以（本设计 §2 的 `MEM_SHUFFLE` 臂就是为此设计的）。

**我们的差异（三条）**：① 主干预**冻结**；② 记忆载体是**显式秩块槽位**（可枚举、可单独擦除、可单条因果消融）；③ 遗忘是**精确擦除**（$B_k \leftarrow 0$），不是隐状态的门控衰减。

### 3.2 "这不就是知识编辑（ROME/MEMIT/AlphaEdit）吗？"

**支持这一质疑的证据**：ROME（arXiv:2202.05262）已证明"事实可以写进参数"，MEMIT（arXiv:2210.07229）已证明可批量写入，AlphaEdit（arXiv:2410.02355）把"不弄坏基座"做成了硬约束，GRACE（arXiv:2211.11031）更是**不改权重、外挂 codebook**——与本项目架构最接近。

**反对"就是同一件事"的证据（全部来自已读原文摘要，可安全引用）**：
- AlphaEdit：locate-then-edit 的扰动 "**inevitably disrupt the originally preserved knowledge**, especially in sequential editing scenarios"；
- GRACE：既有编辑器 "**degrade model performance quickly across multiple, sequential edits**"；
- arXiv:2502.01636：退化源于**内部激活过度优化 + 被编辑矩阵范数持续增长**；
- WISE（arXiv:2405.14768）：终身编辑的**不可能三角**（reliability / generalization / locality 不可兼得）——直接编辑长期记忆伤可靠性与局部性，改用工作记忆则伤泛化；
- RippleEdits（TACL 2024）：主流编辑方法在 ripple 一致性上普遍失败。

**我们的差异（四条）**：① **运行期、无标签、增量**（编辑器需要预先构造的编辑样本与层定位）；② **可淘汰**（编辑器几乎都无容量管理）；③ **多记忆并存且可单条擦除**（GRACE 是激活级 codebook，无法证明参数里无残留）；④ 我们**采纳** AlphaEdit 的零空间投影作为写入保护层，而不是与之竞争。

### 3.3 "长上下文足够强时还有必要吗？为什么不直接放上下文？"

**这条最难回答，必须先摆出对我方不利的证据**：
- **RippleEdits**（arXiv:2307.12976）：一个"简单的上下文编辑基线"在 ripple 一致性上**得分最高** —— 在"编辑一致性"这个维度上，上下文内编辑胜过参数编辑；
- **Long Context vs. RAG**（arXiv:2501.01880）：**长上下文在 QA 上普遍优于 RAG**（尤其 Wikipedia 类），摘要式检索≈长上下文，chunk 式检索落后；RAG 的优势只出现在**对话类与通用问句**；
- **Sufficient Context**（arXiv:2411.06037）：大模型在上下文充分时很准，不充分时倾向硬答而非弃答；**~20–25% 的 RAG 失败发生在上下文已经充分时**（=模型不会用，不是没检索到）。

**对我方有利的证据（全部是"长上下文读不动"场景）**：
- qTTT（arXiv:2512.13898）：静态自注意力有 **score dilution**，改做定向梯度更新后 Qwen3-4B 在 LongBench-v2 子集 **+12.6pt**、ZeroScrolls 子集 **+14.1pt**；
- TTT-E2E（arXiv:2512.23675）：128K 上下文下比全注意力**快 2.7×**，且**推理延迟与上下文长度无关**；
- GradMem（arXiv:2603.13875）：同记忆尺寸下，**梯度写入优于前向写入**；
- LoCoMo（arXiv:2402.17753）：长上下文 LLM 与 RAG 在超长期对话上**都显著落后人类**（双刃：既支持"上下文不够"，也提示"外挂也未必行"）。

**结论**：三类质疑里，第 1、2 类可以靠"机制差异"回答清楚；**第 3 类不能靠论证回答，只能靠 02 文档 §3.3 的三臂对照（A_无记忆 / B_上下文记忆 / C_参数记忆）把它变成数据**。在拿到数据之前，本项目的正文**不得**出现"参数记忆优于上下文"这种无条件表述（§4）。

---

## 4. 必须限定的主张边界（重要）

**已有证据明确显示长上下文在 QA 上普遍优于 RAG**（arXiv:2501.01880；arXiv:2411.06037 的"上下文充分时大模型很准、~20–25% 失败发生在已充分时"从另一侧印证同一结论）。

因此：

1. ❌ **不得声称**"参数记忆普遍优于上下文记忆"；
2. ❌ **不得声称**"检索/外挂记忆会伤害性能"——该说法**不在** Sufficient Context 的摘要中，来自二手描述（未核实）；
3. ✅ 主张必须**限定在三个条件下同时成立**时才提出：
   - **跨会话持久**（信息已离开上下文、且没有检索触发词可用）；
   - **需要遗忘**（容量受限、噪声必须被回收，且要能证明擦除后无残留）；
   - **抗上下文惯性**（话题跳转后不被上文主题/语气/错误前提拖拽）。
4. ✅ 表述模板：*"在**跨会话持久 + 需要遗忘 + 抗上下文惯性**这三类任务上，我们把记忆写入旁路参数槽位，相对把同样内容放进上下文，在 I1–I3 惯性指标上取得 X；代价是在 ripple 一致性上不如上下文内编辑（RippleEdits, arXiv:2307.12976），且受容量 K 限制。"*
5. ⚠️ 记忆指标必须定义**"召回之后改善了什么"**，不能只报命中率：arXiv:2608.20280 显示语义缓存命中率 51–60%、**质量调整后仅 1.1–2.2%**。

---

## 5. 文献缺口（本文档最有价值的部分）

### 5.1 (a) 未找到"上下文惯性 / 锚定"的专门基准 ⇒ 自建

**负面结论（三路侦察一致）**：话题跳转后被上文拖着走这一现象，**没有专门的评测基准**；也未找到"多轮错误前提继承"或"话题切换鲁棒性"的命名基准。

**可借的三块拼图**（均已读原文）：
| 拼图 | 出处 | 借什么 |
| --- | --- | --- |
| **Lost in the Middle** | arXiv:2307.03172 | **位置置换协议**：改变相关信息位置而不改变内容，看答案是否随之改变——证明"上文在影响答案"的干净模板 |
| **GSM-IC（无关上下文干扰）** | arXiv:2302.00093 | 只加**一句**语义相关但无关的话就能大幅拉低解题准确率；自洽解码与"忽略无关信息"指令只能缓解、不能根治 |
| **Sycophancy** | arXiv:2310.13548 | 多轮中被 pushback 后立场漂移，是"话题/立场被上文带偏"最直接的既有评测范式 |
| 补充：Anchoring bias | arXiv:2412.06593 | 初始信息不成比例地影响判断；CoT / ToP / 忽略锚点提示 / Reflection **都不足以消除**锚定 |
| 补充：BiasBuster | arXiv:2403.00811 | 把序列式偏置操作化为 13,465 条 prompt 的评测/缓解框架（注意 **CC BY-NC-SA 4.0，非商用限制**） |

**我们打算怎么做**：在 02 文档的 `inertia` 子集上实现"旧话题给错误前提 → 切新话题 → 测 I1/I2/I3"，并把**位置置换**（Lost-in-the-Middle 式）与**无关句注入**（GSM-IC 式）作为两个可参数化旋钮，使"惯性强度"成为连续可调的自变量；同时报告任务成功率，防止"为了不纠结而不干活"。**这套基准本身是本项目的一个贡献点**，也是审稿人必问"怎么测"的地方。

### 5.2 (b) 未找到"参数记忆 vs 上下文记忆"的隔离协议基准 ⇒ 自建

**负面结论**：没有任何现成基准能证明"是参数在记而不是上下文在记"。

**可借手法（"机制清晰、可用于设计" ≠ 已有人这么做）**：
| 手法 | 为何能隔离 | 来源 |
| --- | --- | --- |
| **信息驱逐 / eviction** | 写入后清空全部对话上下文，答对 ⇒ 信息不在上下文里 | MemGPT 的外部上下文换页隐喻（arXiv:2310.08560）；StreamingLLM 驱逐中间 token（arXiv:2309.17453） |
| **权重冻结对照（最强）** | 同模型同评测，**唯一自变量 = 是否更新权重** | TTT-E2E 的"把上下文压进权重"表述（arXiv:2512.23675） |
| **仅 query 更新** | 梯度只由 query 驱动、不接触上下文 token ⇒ 增益只能来自已写入的参数 | qTTT（arXiv:2512.13898） |
| **置换 / 反事实控制** | 只改变位置/顺序，不改变内容 | Lost in the Middle（arXiv:2307.03172）；BiasBuster sequential bias（arXiv:2403.00811） |
| **消融** | 关旁路 / 回滚参数 ⇒ 性能应回基线 | MemoryAgentBench 四能力分测（arXiv:2507.05257） |
| **遗忘验证** | 写入 A 后显式遗忘 A：A 应失效、B 应保留 | MemoryAgentBench 的 selective forgetting competency（arXiv:2507.05257） |
| **充分性分层** | 区分"没记住"与"记住了但没用上" | Sufficient Context（arXiv:2411.06037） |

**我们打算怎么做**：在 02 文档 P1–P5 之上落地自拟骨架
`同模型 × 同问题 × {参数更新 / 上下文拼接 / 检索式外部库} 三臂 × {上下文保留 / 上下文驱逐} 两条件 × {话题同域 / 话题跳转}`，
一次实验同时回答"记忆在哪一层"与"是否缓解惯性"。**关键是把 P4（适配器消融）作为最强因果证据**：`MEM_ON > MEM_OFF ≈ MEM_SHUFFLE`，且 `MEM_OFF` 若仍很高即判定泄漏、实验作废。

### 5.3 (c) 主动可控遗忘的文献极薄 ⇒ 作为核心贡献

**负面结论（分支 B 的核心判断）**：**防遗忘**的文献极多（EWC / SI / LwF / Replay / PackNet / Progressive / O-LoRA / InfLoRA / AlphaEdit / Merge before Forget，
以及 arXiv:1811.11682、2302.00487、2505.12512、2404.12526、1606.09282、1703.04200 等）；
而**主动可控遗忘**本次只找到 **FADE（arXiv:2604.27063）**这一篇把遗忘当一等公民处理，且它**只作用于最后一层、无现成代码**。

其余可引用证据：
- 淘汰策略方面唯一带硬数字的是**语义缓存**上的研究（arXiv:2608.20280）：无策略比 LFU 好超过 **0.041 个百分点**，紧容量下 FIFO 落后 LFU **8.67 个百分点**；并明确警告**命中率（51–60%）≠ 有用性（1.1–2.2%）**；
- **结论尚未统一**：另有转述称 agent-memory 轨迹上 LRU/LFU 反而输给 FIFO（二手，双方均未读原文，未核实）⇒ **淘汰策略在文献中尚无共识，必须自己测**。

**我们打算怎么做**：把"可控遗忘 / 容量回收"写成第一贡献：显式容量 $K$ + 效用/时间淘汰 + **精确擦除**，并给出「容量–保留率–基座漂移」三维曲线；同时把"判别力 = retention_important − retention_noise"作为主报告量（02 §3.2）。**注意不要重蹈 FIFO 的坑**：默认策略取 LFU 式频次，并与其他策略在自建负载上对比后再下结论。

### 5.4 (d) 找不到小规模 LLM 上 stability–plasticity 的可核实实测数字 ⇒ T2 把它变成可测量的贡献

**诚实结论**：本次检索**没有**找到一篇能核实原文、且专门给出"小规模 LLM 在 stability–plasticity 权衡上的实测数字"的论文。只有**方向性、二手转述、数字一律不可引用**的说法：

| 二手方向性观察 | 状态 |
| --- | --- |
| 朴素微调即产生可测量的遗忘（适应新目标时早期任务性能显著下滑） | 未核实 |
| 遗忘严重程度取决于任务差异度、学习率、新数据量、顺序适应步数 | 未核实 |
| 持续预训练的遗忘通常比任务特定微调更轻 | 未核实 |
| **大模型往往比小模型遗忘更少**（小模型冗余容量少 ⇒ 更脆弱） | 未核实 |
| 小模型配方：低学习率 + 混入通用数据（replay）+ 参数高效更新 | 未核实 |

**我们打算怎么做**：这正是 T2 的机会——**在 Qwen3-1.7B + 秩块 LoRA 上给出可核实的 stability–plasticity 曲线**（写入条数 × 基座能力探针漂移 × 通用能力），把"小模型更脆弱"从二手断言变成可测量的贡献。设计上对应 02 §3.2 的 `capability_drift`（先设"漂移 ≤ 2%"通过线）与 01 §3.2 的必监控量（**被更新矩阵范数增长**、槽间串扰、基座能力漂移）。

### 5.5 (e) 附加缺口："写入"与"遗忘"从未在同一系统里被验证过

- GradMem（arXiv:2603.13875）证明"冻结主干 + 梯度写入"可行，但**无遗忘**；
- FADE（arXiv:2604.27063）证明"内生遗忘"可行，但**只作用最后一层、无代码、不做条目级淘汰**；
- Merge before Forget（arXiv:2512.23017）解决了常数内存，但**合并后按条撤销困难**（分支 B 的推断，原文是否讨论可撤销性未核实）；
- AlphaEdit（arXiv:2410.02355）的零空间维数会随写入量耗尽（推断，未核实）⇒ **防污染与容量淘汰是耦合的**，文献未处理这个耦合。

**我们打算怎么做**：把"写入 + 淘汰 + 精确擦除 + 基座不退化"这四件事放进**同一个受控实验**里同时报告，并把"零空间容量 vs 槽位容量 K"作为一条独立的量在 T2 容量实验里测出来。⇒ 这就是本项目"三件事同时成立仍缺干净实验结果"（`00` §5）的具体落点。

---

## 6. 证据分级表（三份侦察文件的汇总）

分级沿用侦察文件原标注，**本文档不升格**：
**【已读正文】** > **【已读摘要】**（抓取到 abs 页/出版社页原文，但未读全文） > **【二手转述】**（仅搜索摘要/综述转述，数字一律视为未核实） > **【未核实】**（禁止引用）。

### 6.1 【已读正文】或【已读正文（关键词/原句级）】

| 条目 | 已核实的原句/要点 |
| --- | --- |
| **Titans（2501.00663）正文** | "treat its training as an online learning problem…compress the past information into the parameters"；更新式 $M_t = M_{t-1} - \theta_t \nabla \ell$（∇ℓ 即 Surprise）；momentum $S_t$ "act as a memory of surprise across time"；weight decay "closely related to the gating mechanism in modern RNNs"；**不做遗忘会导致 memory overflow**；变体缩写 MAG/MAC/MAL |
| **Memory Layers at Scale（2412.09764）正文** | 关键词计数核实：`memory layer` 59 次、`product-key` 3 次、`product key` 2 次、`sparse` 7 次、`key-value` 5 次 |
| ⚠️ **Titans 摘要本身** | **不含** "surprise" / "momentum" / "weight decay" 字样（已核对该 abs 页）⇒ 这三个机制细节来自**第三方解读**，故在 §2 表中标为二手（见 6.2） |

### 6.2 【已读摘要】（abs 页/出版社页原文，未读全文）

按分支归类（条目 = 工作(arXiv 号)）：

| 分支 | 已读摘要的条目 |
| --- | --- |
| **A 测试时训练 / 快权重** | TTT 2020(1909.13231)、TTT layers(2407.04620)、TTT-E2E(2512.23675)、qTTT(2512.13898)、Ba et al. 2016(1610.06258)、Dynamic Evaluation(1709.07432) |
| **A 神经记忆 / 线性注意力 / 容量** | Titans(2501.00663)、ATLAS(2505.23735)、Miras(2504.13173)、HOPE(2512.24695)、Schlag FWP(2102.11174)、DeltaNet 并行化(2406.06484)、Mamba-2(2405.21060)、Zoology(2312.04927)、Repeat After Me(2402.01032) |
| **A 旁路参数 / 参数化上下文** | Adapters(1902.00751)、LoRA(2106.09685)、Prefix-Tuning(2101.00190)、SEAL(2506.10943)、Cartridges(2506.06266) |
| **A 记忆评测** | Evo-Memory(2511.20857)、LTM 立场报告(2410.15665) |
| **B 知识编辑** | ROME(2202.05262)、MEMIT(2210.07229)、GRACE(2211.11031)、WISE(2405.14768)、AlphaEdit(2410.02355)、Merge before Forget(2512.23017)、连续编辑退化正则化(2502.01636) |
| **B 遗忘 / 写入判据** | FADE(2604.27063)、语义缓存淘汰(2608.20280)、D-MEM(2603.14597) |
| **C 记忆系统** | MemGPT(2310.08560)、Mem0(2504.19413)、A-MEM(2502.12110) |
| **C 评测基准** | LongMemEval(2410.10813)、LoCoMo(2402.17753)、LongBench v2(2412.15204)、∞Bench(2402.13718)、RULER(2404.06654)、MemoryAgentBench(2507.05257) |
| **C 上下文 vs 参数 / 惯性** | Long Context vs RAG(2501.01880)、Sufficient Context(2411.06037)、Anchoring(2412.06593)、BiasBuster(2403.00811)、Lost in the Middle(2307.03172)、StreamingLLM(2309.17453)、GSM-IC(2302.00093)、Sycophancy(2310.13548) |

**RippleEdits** 为 **ACL Anthology 正式版页面（TACL 2024）** 已读，关键结论"简单上下文编辑基线得分最高"直接引自摘要。

### 6.3 【二手转述】（数字一律不可引用）

| 条目 | 二手内容 | 本文档处理 |
| --- | --- | --- |
| **Hinton & Plaut 1987** | 页码/出版社细节（CMU 摘要页 + eScholarship 条目） | 只作为"快权重/遗忘是默认行为"的源头引用，**不引页码** |
| **Schmidhuber 1992** | 全文；"技术报告版 1991-03-26" | 同上，**日期细节标未核实** |
| **EWC / SI / LwF / Replay / PackNet / Progressive / O-LoRA / MEND / CLS 1995 / sleep replay** | 全部机制描述与结论 | "问题背景"可引，"具体数字/有效性"不可引；**PackNet 无 LICENSE 已由 §9 一手确认** |
| **Titans 的 surprise / 动量 S_t / weight decay / memory overflow 细节** | 来自 Google Research 博客、alphaXiv、Medium 等**第三方解读** | **标二手**；若要写进论文必须抓 PDF 正文核实后再引（§6.1 只核实过正文关键词级表述） |
| **InfLoRA 的"稳定-可塑"论述与 GLUE 退化有无数据** | 检索转述（CVPR 页面证据已一手确认） | 会议与全称已纠错（§8），**效果数据标未核实** |
| **小规模 LLM stability–plasticity 的全部结论** | 见 §5.4 表 | **全部未核实，一个数字都不引用** |
| **RippleEdits 的"组合性/逻辑蕴含/别名/逆关系普遍失败"** | 二手转述 | 只引已核实的那 4 条摘要结论 |
| **淘汰策略的反例（agent-memory 上 LRU/LFU 输给 FIFO）** | 二手转述，双方未读原文 | 作为"文献无共识"的证据，**不引具体结论** |
| **AlphaEdit 的"Outstanding/Best Paper Award"** | 检索转述 | 我方只写 **ICLR 2025 Oral**（原文摘要可确认） |
| **Titans 是否为 NeurIPS 2025 / Cartridges、qTTT 是否被 ICLR 2026 接收** | 会议信息为二手 | 发表状态标**待核实** |

### 6.4 【未核实】（禁止引用）

| 条目 | 状态与处理 |
| --- | --- |
| **Zep/Graphiti（2501.13956）** | arXiv 页 3 次抓取失败（429+timeout）⇒ **DMR 94.8% vs MemGPT 93.4%、LongMemEval +18.5%、延迟 −90% 全部未核实，禁止引用** |
| **MemoryBank（2305.10250）/ MemoRAG（2409.05591）/ Titans（2501.00663，分支 C 视角）** | 未读原页；**注意 Titans 在分支 A 已读 abs + 正文关键词，故以 A 的分级为准**（跨分支分级不一致处一律取更保守者用于引用） |
| **"LongMem" 作为记忆系统名** | 未找到与 MemGPT/Mem0 并列的同名系统论文，疑与基准 **LongMemEval** 混淆 |
| **MSC / Multi-Session Chat** | 未定位到独立基准论文 |
| **LongMemEval-S/M 的具体 token 数（~115k/~500k）** | 二手，未核实 |
| **Mem0 "26% 相对提升"的参照口径** | 原文只写 "over OpenAI"，具体口径未核实 |
| **LoCoMo "LLM-as-judge 可被 gaming"、"10 段对话噪声大"** | 二手批评 |
| **Sufficient Context "retrieval 反而伤害性能"** | **摘要中无此句** —— 关键防误引项 |
| **Titans / ATLAS / Miras / HOPE / GradMem 官方代码** | 实测 `google-research/titans` **API 返回 404**（§9 复核一致）；GradMem 仅二手项目页 |
| **`sustcsonglin/flash-linear-attention` 与 `fla-org/flash-linear-attention` 的关系** | §9 实测两者返回**逐字段相同**（含 pushed_at 秒级），疑为重定向/迁移；**未取到 full_name 字段最终确认** |
| **所有仓库许可（侦察文件中的二手值）** | ⇒ 已由本文档 §9 **一手核实**，分级升格为【API 实测】 |

---

## 7. 集市（URL 清单，按主题归类，去重后）

> 每个类目给代表条目；**所有在侦察文件中出现过的 arXiv 编号均予保留**（含反例与禁止引用项，集中在 7.12）。

### 7.1 快权重与测试时训练（源头）
1. Hinton & Plaut 1987 PDF: https://www.cs.toronto.edu/~hinton/absps/fastweights87.pdf ｜ 摘要页: https://ni.cmu.edu/~plaut/papers/abstracts/HintonPlaut87CogSciConf.fastWeights.html
2. Schmidhuber 1992（Neural Computation 4(1):131–139）: https://direct.mit.edu/neco/article/4/1/131/5620/Learning-to-Control-Fast-Weight-Memories-An
3. Schmidhuber FWP 官方整理页: https://people.idsia.ch/~juergen/fast-weight-programmer-1991-transformer-bigrefs.html
4. Ba et al. 2016 — arXiv:1610.06258: https://arxiv.org/abs/1610.06258
5. Dynamic Evaluation 2017 — arXiv:1709.07432: https://arxiv.org/abs/1709.07432
6. TTT 2020 — arXiv:1909.13231: https://arxiv.org/abs/1909.13231
7. TTT layers 2024 — arXiv:2407.04620: https://arxiv.org/abs/2407.04620
8. TTT-E2E — arXiv:2512.23675: https://arxiv.org/abs/2512.23675
9. qTTT — arXiv:2512.13898: https://arxiv.org/abs/2512.13898
10. GradMem — arXiv:2603.13875: https://arxiv.org/abs/2603.13875

### 7.2 神经长期记忆模块（Titans 系）
1. Titans — arXiv:2501.00663: https://arxiv.org/abs/2501.00663 ｜ 正文核实版: https://arxiv.org/html/2501.00663v1
2. ATLAS — arXiv:2505.23735: https://arxiv.org/abs/2505.23735
3. Miras — arXiv:2504.13173: https://arxiv.org/abs/2504.13173
4. HOPE / Nested Learning — arXiv:2512.24695: https://arxiv.org/abs/2512.24695
5. Google Research 博客（Titans/Miras）: https://research.google/blog/titans-miras-helping-ai-have-long-term-memory
6. Nested Learning 博客（**未核实**）: https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning

### 7.3 线性注意力 / 状态空间 / 记忆层 / 容量天花板
1. Schlag, Irie, Schmidhuber 2021 — arXiv:2102.11174: https://arxiv.org/abs/2102.11174
2. DeltaNet 并行化 — arXiv:2406.06484: https://arxiv.org/abs/2406.06484
3. Mamba-2 / SSD — arXiv:2405.21060: https://arxiv.org/abs/2405.21060
4. Memory Layers at Scale — arXiv:2412.09764: https://arxiv.org/abs/2412.09764
5. Zoology（MQAR）— arXiv:2312.04927: https://arxiv.org/abs/2312.04927
6. Repeat After Me — arXiv:2402.01032: https://arxiv.org/abs/2402.01032
7. Titans/HOPE 第三方复现（未验证）: https://github.com/aryateja2106/neural-memory-reproduction

### 7.4 旁路可写参数与"参数化上下文"
1. Adapters — arXiv:1902.00751: https://arxiv.org/abs/1902.00751
2. LoRA — arXiv:2106.09685: https://arxiv.org/abs/2106.09685
3. Prefix-Tuning — arXiv:2101.00190: https://arxiv.org/abs/2101.00190
4. SEAL — arXiv:2506.10943: https://arxiv.org/abs/2506.10943 ｜ 论文主页: https://jyopari.github.io/posts/seal
5. Cartridges — arXiv:2506.06266: https://arxiv.org/abs/2506.06266

### 7.5 持续学习与防遗忘（防遗忘侧）
1. EWC — arXiv:1612.00796: https://arxiv.org/abs/1612.00796 ｜ PNAS: https://www.pnas.org/doi/10.1073/pnas.1611835114
2. Synaptic Intelligence — arXiv:1703.04200: https://arxiv.org/abs/1703.04200
3. LwF — arXiv:1606.09282: https://arxiv.org/abs/1606.09282
4. Experience Replay — arXiv:1811.11682: https://arxiv.org/abs/1811.11682
5. 持续学习综述 — arXiv:2302.00487: https://arxiv.org/abs/2302.00487
6. Replay 现代组件化 — arXiv:2505.12512: https://arxiv.org/html/2505.12512v1 ｜ 自适应 replay — arXiv:2404.12526: https://arxiv.org/html/2404.12526v1
7. PackNet — arXiv:1711.05769: https://arxiv.org/abs/1711.05769
8. Progressive Networks — arXiv:1606.04671: https://arxiv.org/abs/1606.04671
9. CLS 1995 PDF: https://stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf ｜ PubMed: https://pubmed.ncbi.nlm.nih.gov/7624455
10. Deep Generative Replay — arXiv:1705.08690: https://arxiv.org/abs/1705.08690 ｜ Brain-inspired replay: https://www.nature.com/articles/s41467-020-17866-2 ｜ sleep-like replay: https://www.nature.com/articles/s41467-022-34938-7

### 7.6 LoRA 持续学习 / 合并 / 子空间隔离
1. O-LoRA — arXiv:2310.14152: https://arxiv.org/abs/2310.14152
2. InfLoRA — arXiv:2404.00228: https://arxiv.org/abs/2404.00228
3. Merge before Forget — arXiv:2512.23017: https://arxiv.org/abs/2512.23017
4. SCOPE / continual-lora: https://github.com/luk-st/continual-lora
5. 模型合并综述（NVIDIA）: https://developer.nvidia.com/blog/an-introduction-to-model-merging-for-llms

### 7.7 知识编辑与副作用
1. ROME — arXiv:2202.05262: https://arxiv.org/abs/2202.05262 ｜ 项目页: https://rome.baulab.info
2. MEMIT — arXiv:2210.07229: https://arxiv.org/abs/2210.07229 ｜ https://memit.baulab.info
3. MEND — arXiv:2110.11309: https://arxiv.org/abs/2110.11309 ｜ SERAC — arXiv:2206.06520: https://arxiv.org/abs/2206.06520
4. GRACE — arXiv:2211.11031: https://arxiv.org/abs/2211.11031
5. WISE — arXiv:2405.14768: https://arxiv.org/abs/2405.14768
6. AlphaEdit — arXiv:2410.02355: https://arxiv.org/abs/2410.02355
7. RippleEdits（TACL 2024）— arXiv:2307.12976: https://arxiv.org/abs/2307.12976 ｜ 正式版: https://aclanthology.org/2024.tacl-1.16/
8. 连续编辑退化正则化 — arXiv:2502.01636: https://arxiv.org/abs/2502.01636

### 7.8 主动遗忘与淘汰（本项目的差异化区）
1. FADE — arXiv:2604.27063: https://arxiv.org/abs/2604.27063
2. 语义缓存淘汰策略 — arXiv:2608.20280: https://arxiv.org/abs/2608.20280
3. D-MEM（RPE 门控写入）— arXiv:2603.14597: https://arxiv.org/abs/2603.14597
4. 选择性剪枝遗忘 — arXiv:2403.01267: https://arxiv.org/abs/2403.01267 ｜ 联邦类判别剪枝 — arXiv:2110.11794: https://arxiv.org/pdf/2110.11794 ｜ Un-pruning — arXiv:2507.18725: https://arxiv.org/abs/2507.18725
5. LLM-Eraser（KDD 2025）: https://dl.acm.org/doi/10.1145/3690624.3709312
6. 机器遗忘综述 — arXiv:2404.01206: https://arxiv.org/html/2404.01206v1
7. Redis LFU vs LRU: https://redis.io/blog/lfu-vs-lru-how-to-choose-the-right-cache-eviction-policy

### 7.9 记忆系统（上下文层 / 外挂库）
1. MemGPT — arXiv:2310.08560: https://arxiv.org/abs/2310.08560
2. Mem0 — arXiv:2504.19413: https://arxiv.org/abs/2504.19413
3. A-MEM — arXiv:2502.12110: https://arxiv.org/abs/2502.12110
4. MemoryBank — arXiv:2305.10250: https://arxiv.org/abs/2305.10250
5. MemoRAG — arXiv:2409.05591: https://arxiv.org/abs/2409.05591
6. Zep / Graphiti — arXiv:2501.13956: https://arxiv.org/abs/2501.13956
7. Evo-Memory — arXiv:2511.20857: https://arxiv.org/abs/2511.20857
8. LTM 立场报告 — arXiv:2410.15665: https://arxiv.org/abs/2410.15665

### 7.10 记忆评测基准
1. LongMemEval — arXiv:2410.10813: https://arxiv.org/abs/2410.10813
2. LoCoMo — arXiv:2402.17753: https://arxiv.org/abs/2402.17753
3. LongBench v2 — arXiv:2412.15204: https://arxiv.org/abs/2412.15204
4. ∞Bench — arXiv:2402.13718: https://arxiv.org/abs/2402.13718
5. RULER — arXiv:2404.06654: https://arxiv.org/abs/2404.06654
6. MemoryAgentBench — arXiv:2507.05257: https://arxiv.org/abs/2507.05257
7. LongBench 数据集镜像: https://huggingface.co/datasets/THUDM/LongBench

### 7.11 上下文 vs 参数 · 惯性 / 锚定 / 干扰
1. Long Context vs. RAG — arXiv:2501.01880: https://arxiv.org/abs/2501.01880
2. Sufficient Context — arXiv:2411.06037: https://arxiv.org/abs/2411.06037
3. Lost in the Middle — arXiv:2307.03172: https://arxiv.org/abs/2307.03172
4. Attention Sink / StreamingLLM — arXiv:2309.17453: https://arxiv.org/abs/2309.17453
5. GSM-IC — arXiv:2302.00093: https://arxiv.org/abs/2302.00093
6. Sycophancy — arXiv:2310.13548: https://arxiv.org/abs/2310.13548
7. Anchoring bias — arXiv:2412.06593: https://arxiv.org/abs/2412.06593
8. BiasBuster — arXiv:2403.00811: https://arxiv.org/abs/2403.00811
9. Mermillod et al. 2013（stability–plasticity）: https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2013.00504/full

### 7.12 明确"禁止引用 / 仅作线索"的编号（**保留编号以便后续核验，不得进正文**）
- **arXiv:2307.12995** —— 被搜索引擎误当作 RippleEdits；实为物理论文（见 §8）
- **arXiv:2508.19597** —— 被误当作 Nested Learning；实为车辆运动预测持续学习论文（见 §8）
- **arXiv:2605.13162（ProCL）** —— ⚠️ **不在此清单**：已由父代理**一手核实**（abs 页 HTTP 200 + Semantic Scholar 交叉确认），
  三份侦察文件虽未覆盖它，但本文档已据一手来源补入 §2 定位表（见 §8 第 6 条）
- **arXiv:2602.02543 / 2606.26783 / 2509.22072** —— 知识编辑线索，未核验
- **arXiv:2505.18343**（Graph-Based External Memory 编辑）—— 未核验
- **arXiv:2607.11696 / 2608.30650 / 2605.08563 / 2609.03436** —— 带 `26xx` 前缀的搜索结果编号，**未核实且未读原文**
- **arXiv:2006.03340（MANTRA）** —— 仅二手转述
- **ELDER / ELISTER / WikiBigEdit / AnyEdit** 基准 —— 未核验，仅线索
- **"An Empirical Study of Catastrophic Forgetting in LLMs During Continual Fine-Tuning"（IEEE）** —— URL/DOI 未获取，**该文献是否存在待核实**

---

## 8. 纠错记录（三路侦察发现的假编号与错引）

| # | 错误说法 | 事实 | 证据来源 |
| --- | --- | --- | --- |
| 1 | RippleEdits 的 arXiv 号 = **2307.12995** | ❌ 假。**2307.12995** 实为物理学科论文 *Measurement of the high-energy γ-rays from heavy ion reactions using Čerenkov detector*（Si et al., 2023, physics.ins-det）。**正确号 = arXiv:2307.12976**，并经 ACL Anthology TACL 正式版页面（https://aclanthology.org/2024.tacl-1.16/ ，TACL vol.12 pp.283–298）交叉确认 | 分支 B 实抓两个 URL 比对标题 |
| 2 | InfLoRA 发表在 **ICLR 2024**，全称 "**inter**ventional LoRA" | ❌ 假。全称 = **Interference-Free Low-Rank Adaptation for Continual Learning**，发表在 **CVPR 2024**（arXiv:2404.00228），有 CVPR 开放获取页与 IEEE 文档号 10658274 为证 | 分支 B（CVPR 页面证据一致）；本文档核查会议信息时保持"二手"标注 |
| 3 | GRACE = "**general graph-based** codebook model editing" | ❌ 假。GRACE = *Aging with GRACE: Lifelong Model Editing with **Discrete Key-Value Adaptors***（arXiv:2211.11031, NeurIPS 2023）。graph-based 方向是另一篇（arXiv:2505.18343，**未核实**） | 分支 B 已读 abs 页 |
| 4 | Nested Learning 的 arXiv 号 = **2508.19597** | ❌ 假。**2508.19597** 实为 *Complementary Learning System Empowers Online Continual Learning of Vehicle Motion Forecasting in Smart Cities*（Zirui Li 等，2025-08-27），与本工作无关。**正确号 = arXiv:2512.24695**（Comments 注明 NeurIPS 2025） | 分支 A 实测打开 2508.19597 比对标题 |
| 5 | Titans 摘要里写了 surprise / momentum / weight decay | ❌ 不准确。**该 abs 页不含这三个词**；机制细节来自第三方解读 ⇒ 引用前必须抓 PDF 正文核实 | 分支 A（按摘要原文核对） |
| 6 | **ProCL（arXiv:2605.13162）** | ⚠️ **信息缺口已由父代理补救**：三份侦察文件确实**均未出现**该工作（全文检索确认），所以侦察阶段"无证据"的判断**对侦察范围而言没错**。但父代理在 T0 期间用 `web_fetch` 抓 `https://arxiv.org/abs/2605.13162` 得到 **HTTP 200**，页面为 *Continual Fine-Tuning of Large Language Models via Program Memory*（Hung Le / Svetha Venkatesh，2026-05-13 提交，cs.LG，CC BY 4.0，18 页 preprint），并用 `https://api.semanticscholar.org/graph/v1/paper/arXiv:2605.13162` 交叉确认标题与年份 ⇒ **该工作确实存在，本文档按一手来源引用（§2）** | 父代理一手核实（abs 页 + S2 API）；**教训见下方 ⑤** |
| 7 | MemoRAG 仓库 = `Qian250/MemoRAG` | ❌ 实测 **404**。GitHub 搜索 API 命中的同名高星仓库为 **`qhjqhj00/MemoRAG`**（Apache-2.0，2267★，描述 "Empowering RAG with a memory-based data interface for all-purpose applications!"）；**是否即论文官方仓库仍以论文页为准**（未从论文页确认） | §9 API 实测 |
| 8 | MemoryBank 代码仓库只写 `MemoryBank-SiliconFriend`（无 owner） | ⚠️ 缺 owner；搜索 API 命中 **`zhongwanjun/MemoryBank-SiliconFriend`**（MIT，451★，描述 "Source code and demo for memory bank and SiliconFriend"），**归属未从论文页确认** | §9 API 实测 |
| 9 | Titans 官方仓库 `google-research/titans` | ✅ 侦察文件称 404 —— **§9 复核一致（HTTP 404）**，Google 官方仓库确实不存在 | §9 API 实测 |

**教训（必须写进本项目的检索纪律）**：
> **搜索引擎的"摘要式答案"会伪造 arXiv 编号。** 本次三路侦察各自独立撞到至少一例（2307.12995 / 2508.19597），
> 说明这不是偶发失误而是系统性风险。因此：
> ① 任何 arXiv 编号在引用前必须**抓 abs 页比对标题**（不是抓搜索摘要）；
> ② 会议/venue/奖项信息同样必须回到出版方页面确认（InfLoRA 的 ICLR↔CVPR 混淆即此类）；
> ③ 未核验的编号**一律进 7.12 禁引清单**，不得进入正文与定位表；
> ④ 仓库许可同理：**二手转述不算**，必须用 GitHub REST API 或 LICENSE 正文核实（本文档 §9 即为该纪律的一次执行——侦察文件里那些二手许可值，经实测**大部分正确、少数需要更正或补缺口**）；
> ⑤ **子代理的"没找到"不是"不存在"**：子代理的负结论只覆盖**它检索过的范围**。本次 ProCL 就是实例——
> 三路侦察都没提到它，但父代理换个入口（直接抓 abs 页）一次就核实到了。⇒
> **凡是要写进结论的"文献缺口"，必须由父代理复核一遍再落笔**，否则会把"检索覆盖不足"误写成"文献空白"。

---

## 9. 仓库许可核实附表（本文档一手实测）

**方法**：本机 `github.com` 被加速器改写到 `127.0.0.1`，`web_fetch` 抓 github.com 会被判非公网 IP 而拒绝；
改用 `pwsh` 的 `Invoke-RestMethod` 直连 **GitHub REST API**（匿名，无 token，未打印任何凭据）：
`https://api.github.com/repos/<owner>/<repo>` 读 `license.spdx_id` / `stargazers_count` / `archived` / `pushed_at`，
对 `NOASSERTION` 的再读 `https://api.github.com/repos/<owner>/<repo>/license`。
**核实时间：2026-09-29。星数与 pushed_at 为当日快照。**

### 9.1 核实成功（逐条为 API 实测）

| # | owner/repo | SPDX | ★ | archived | pushed_at | 结论与可用性 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `test-time-training/ttt-lm-pytorch` | **MIT** | 1405 | False | 2024-07-14 | 可用（保留声明） |
| 2 | `test-time-training/e2e` | **NONE** | 708 | False | 2026-02-15 | ⚠️ **无 LICENSE ⇒ 默认保留所有权利，不可抄代码**，只可参考思路 |
| 3 | `lucidrains/titans-pytorch` | **MIT** | 1986 | False | 2026-07-13 | 可用（非官方实现） |
| 4 | `aryateja2106/neural-memory-reproduction` | **NOASSERTION**(Other) | 6 | False | 2025-12-31 | ⚠️ 有 LICENSE 文件但非标准 SPDX；**许可正文未读到（API 限流）⇒ 不可假定宽松** |
| 5 | `shayme92/Titans-NNX` | **MIT** | 8 | False | 2025-11-10 | 可用（个人项目，质量未验证） |
| 6 | `facebookresearch/memory` | **NOASSERTION** = **CC BY-NC 4.0** | 387 | False | 2024-12-12 | 🔴 **署名-非商业性使用 4.0** ⇒ **本项目不可商用/不可直接抄入商用代码**；正文已解码确认（"Attribution-NonCommercial 4.0 International"） |
| 7 | `fla-org/flash-linear-attention` | **MIT** | 5798 | False | 2026-09-28 | 可用 |
| 8 | `sustcsonglin/flash-linear-attention` | **MIT** | 5798 | False | 2026-09-28 | 与 #7 **逐字段完全相同**（含 pushed_at 秒级）⇒ 疑为重定向/迁移；**关系未最终确认** |
| 9 | `state-spaces/mamba` | **Apache-2.0** | 18874 | False | 2026-07-22 | 可用 |
| 10 | `HazyResearch/zoology` | **Apache-2.0** | 287 | False | 2026-03-22 | 可用（MQAR 容量单测可直接用） |
| 11 | `Continual-Intelligence/SEAL` | **MIT** | 1863 | False | 2025-08-01 | 可用 |
| 12 | `HazyResearch/cartridges` | **Apache-2.0** | 339 | False | 2026-03-23 | 可用 |
| 13 | `microsoft/LoRA` | **MIT** | 13817 | False | 2024-12-17 | 可用 |
| 14 | `xanderdavies/elastic_weight_consolidation` | **NONE** | 1 | False | 2022-06-09 | ⚠️ 无 LICENSE（仅 1★ 的复现仓库，本就不建议依赖） |
| 15 | `ganguli-lab/pathint`（SI 官方） | **MIT** | 112 | False | 2018-08-14 | 可用 |
| 16 | `arunmallya/packnet` | **NONE** | 245 | False | 2018-10-07 | ✅ **一手确认侦察文件 B 的二手转述**：确实无 LICENSE ⇒ 默认保留所有权利，**只可参考思路** |
| 17 | `liangyanshuo/InfLoRA` | **MIT** | 115 | False | 2025-03-13 | 可用 |
| 18 | `luk-st/continual-lora` | **NONE** | 6 | False | 2025-07-02 | ⚠️ 无 LICENSE |
| 19 | `kmeng01/rome` | **MIT** | 780 | False | 2024-04-20 | ✅ **一手确认** B 的二手值（MIT）**正确** |
| 20 | `kmeng01/memit` | **MIT** | 562 | False | 2024-01-31 | 可用 |
| 21 | `eric-mitchell/mend` | **MIT** | 260 | False | 2023-08-30 | 可用 |
| 22 | `thartvigsen/grace` | **NONE** | 85 | False | 2024-12-21 | ⚠️ **GRACE 官方代码无 LICENSE** ⇒ 只可参考架构思路，不可抄代码 |
| 23 | `zjunlp/EasyEdit`（含 WISE） | **MIT** | 2927 | False | 2026-09-24 | ✅ 确认 B 的二手值正确；可用 |
| 24 | `jianghoucheng/AlphaEdit` | **MIT** | 460 | False | 2025-10-15 | 可用（零空间投影一行实现可直接落地） |
| 25 | `edenbiran/RippleEdits` | **MIT** | 57 | False | 2024-04-15 | 可用（论文本体 TACL 2024 为 CC BY 4.0） |
| 26 | `GMvandeVen/brain-inspired-replay` | **MIT** | 252 | False | 2023-07-06 | ✅ 确认 B 的二手值正确 |
| 27 | `ContinualAI/continual-learning-papers` | **MIT** | 740 | False | 2024-04-22 | 可用（论文清单，非代码） |
| 28 | `letta-ai/letta`（MemGPT） | **Apache-2.0** | 24962 | False | 2026-09-10 | 可用 |
| 29 | `mem0ai/mem0` | **Apache-2.0** | 66256 | False | 2026-09-25 | 可用（工程 baseline） |
| 30 | `WujiangXu/A-mem` | **MIT** | 974 | False | 2026-03-05 | 可用 |
| 31 | `WujiangXu/A-mem-sys` | **MIT** | 398 | False | 2026-03-15 | 可用 |
| 32 | `getzep/graphiti`（Zep） | **Apache-2.0** | 31284 | False | 2026-09-28 | 可用（**注意：Zep 论文数字未核实，勿引**） |
| 33 | `hsiehjackson/RULER` | **Apache-2.0** | 1621 | False | 2026-07-22 | 可用 |
| 34 | `xiaowu0162/LongMemEval` | **MIT** | 1115 | False | 2026-05-11 | 可用（主评测候选） |
| 35 | `snap-research/locomo` | **NOASSERTION**(Other, `LICENSE.txt`) | 1195 | False | 2024-08-13 | ⚠️ 有 LICENSE 文件但非标准 SPDX；**正文未读到（API 限流）⇒ 不可假定宽松** |
| 36 | `OpenBMB/InfiniteBench` | **MIT** | 393 | False | 2024-09-25 | 可用 |
| 37 | `mit-han-lab/streaming-llm` | **MIT** | 7268 | False | 2024-07-11 | 可用 |
| 38 | `qhjqhj00/MemoRAG`（**替代 404 的 `Qian250/MemoRAG`**） | **Apache-2.0** | 2267 | — | — | 由搜索 API 命中；**归属未从论文页确认** |
| 39 | `zhongwanjun/MemoryBank-SiliconFriend` | **MIT** | 451 | — | — | 由搜索 API 命中；**归属未从论文页确认** |

### 9.2 核实失败

| owner/repo | 结果 | 处理 |
| --- | --- | --- |
| `google-research/titans` | **HTTP 404**（API 实测） | 与分支 A 结论一致：**Titans 无 Google 官方仓库**；只能用第三方复现（#3/#4/#5） |
| `Qian250/MemoRAG` | **HTTP 404** | 侦察文件 C 给出的 owner/repo 有误 ⇒ 见 §8 第 7 条 |
| `aryateja2106/neural-memory-reproduction` 许可正文 | **未核实（API 失败）**：核心配额耗尽（`core: 0/60`，重置 12:46），仅取到 `spdx=NOASSERTION / name=Other` | 表内已如实标注；**落地前需在有配额时复核 LICENSE 正文** |
| `snap-research/locomo` 许可正文 | **未核实（API 失败）**：同上 | 同上 |

### 9.3 许可纪律小结（可直接抄进项目文档）

1. **可用（宽松许可，需保留声明）**：MIT 23 个 + Apache-2.0 8 个 = **31 个仓库**，含本项目最关键的 `jianghoucheng/AlphaEdit`(MIT)、`kmeng01/rome`(MIT)、`HazyResearch/zoology`(Apache-2.0)、`xiaowu0162/LongMemEval`(MIT)。
2. 🔴 **非商用限制，禁止直接抄入**：`facebookresearch/memory` = **CC BY-NC 4.0**（想让 Memory Layers 进产品线的话，只能参考思路）。
3. ⚠️ **无 LICENSE（默认保留所有权利，只可参考思路、不可抄代码）**：`test-time-training/e2e`、`thartvigsen/grace`、`arunmallya/packnet`、`luk-st/continual-lora`、`xanderdavies/elastic_weight_consolidation`。
   ⇒ **尤其注意 `test-time-training/e2e`（TTT-E2E 官方实现）与 `thartvigsen/grace`（GRACE 官方实现）都无许可**，尽管它们是最相关的两份先例代码。
4. ⚠️ **NOASSERTION（非标准许可，正文未读到）**：`aryateja2106/neural-memory-reproduction`、`snap-research/locomo`。
5. **教训**：侦察文件里由搜索转述得来的许可值（ROME=MIT、EasyEdit=MIT、brain-inspired-replay=MIT）经实测**均正确**；
   但 **PackNet 无 LICENSE** 这一条虽是二手、**实测同样成立** ——
   ⇒ 二手许可信息**不可作为依据**，但也不能假定它们一定是错的；**唯一可靠做法是 API/正文核实**。
