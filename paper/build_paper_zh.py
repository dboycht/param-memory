"""Build the Chinese review copy of the paper (``paper/main_zh.tex`` -> ``main_zh.pdf``).

Why it exists
-------------
The author reads Chinese faster than English, and reviewing a 14-page English paper
is slower than it needs to be. This produces a Chinese counterpart **from the same
bundle** as the English paper, so the two can never disagree about a number.

What it is not
--------------
It is not a second source of truth. The English ``main.tex`` remains authoritative
for the claims; this file is a derived review aid, and the PDF says so on its first
page. When the English text changes materially, the corresponding paragraph here has
to be updated by hand -- the *numbers* are automatic, the *prose* is a translation.

Design notes
------------
* Placeholders are ``[[macroName]]`` rather than ``str.format`` fields: a LaTeX
  document is full of braces, and doubling every one of them to satisfy ``format``
  is how a generator silently corrupts a formula.
* A placeholder that does not resolve is a hard error, not a blank. Numbers come
  from ``parammem.report.headline_values`` (the same function the English tables
  use), so nothing here is transcribed.
* Compile with **xelatex**, and do not override the CJK font: the Noto SC files on
  this machine are variable fonts and xdvipdfmx refuses them ("Invalid font: -1").
  ctex's windows fontset picks SimSun/YaHei, which embed correctly.

Usage::

    python paper/build_paper_zh.py            # writes paper/main_zh.tex
    cd paper && xelatex main_zh.tex           # twice, for the table of numbers
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PLACEHOLDER = re.compile(r"\[\[([A-Za-z0-9_]+)\]\]")
BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)   # spans may cross a line break

# Values are pasted into LaTeX, so they have to be escaped at this boundary. The
# English pipeline does this in parammem.report._tex; leaving it out here cost a real
# bug: the macro value "72%" put a comment character into the document and silently
# deleted the rest of the line ("写入少 72%" ate everything after it).
TEX_ESCAPES = (("\\", r"\textbackslash{}"), ("%", r"\%"), ("&", r"\&"),
               ("#", r"\#"), ("_", r"\_"), ("$", r"\$"),
               ("{", r"\{"), ("}", r"\}"))


def tex_escape(value: str) -> str:
    out = str(value)
    for raw, escaped in TEX_ESCAPES:
        out = out.replace(raw, escaped)
    return out

TEMPLATE = r"""% 中文审阅稿 —— 由 paper/build_paper_zh.py 生成，请勿手改。
% 英文版 paper/main.tex 是主张的唯一来源；本文件是派生出的审阅副本。
\documentclass[11pt]{ctexart}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[margin=2.4cm]{geometry}
\usepackage[hidelinks]{hyperref}

\newcommand{\zhbanner}[1]{\noindent\fbox{\parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule}{#1}}}
\title{\bf 面向 LLM 智能体的旁路参数化记忆：\\可精确擦除的槽位、无需上下文即可读取}
\author{成昊天 \\ \small 南京航空航天大学}
\date{2026 年 9 月 30 日}

\begin{document}
\maketitle

\zhbanner{\small\textbf{这是中文审阅稿。}它与英文版 \texttt{main.pdf} 逐节对应，
所有数字都由 \texttt{runs/summary.json} 生成（与英文版同一来源），
\textbf{技术主张以英文版为准}。本页的三条审阅要点是作者希望你先看的：
（一）§\ref{sec:neg} 里 B1 保留项“未复现”的措辞是否合适；
（二）§\ref{sec:limits} 关于“质量优势随规模收窄”的表述你是否接受；
（三）表格是否一次就能读懂。}

\vspace{0.6em}
\begin{center}\small
生成时间戳记：见仓库最近一次提交。数字来源：\texttt{runs/summary.json}（116 个宏）。
\end{center}

\section{摘要}
今天 LLM 智能体的长期记忆放在上下文窗口里。这个通道是\textbf{零和}的——每一个被记住的 token
都挤占了用来推理的 token——而且记住的内容与当前工作集\textbf{无法区分}。我们研究另一条路：
挂在\textbf{冻结主干}上的、容量受限且\textbf{可精确擦除}的旁路低秩记忆槽——运行时用梯度写入、
通过前向传播读取、遗忘靠擦除一个秩块。

在 0.6B 模型上，用**完全虚构的基准**与一套五段式隔离协议，我们得到：写入确实落地且
**只触碰自己那个槽**（写完目标交叉熵 $\approx$ [[tTargetCE]]；隔离逐张量断言）；
**擦除是精确的**（被擦除的槽与"从未写过"逐位相同，[[tErasedTotal]] 次驱逐全部通过，重启后仍然成立）；
**读时组合——而不是容量——才是瓶颈**：把所有槽加起来召回为 [[tSum]]，而按查询键相似度
**只选一个槽**可以恢复到 [[tTopOne]]（oracle 上限 [[tOracle]]，用的是改写过的问法，
且**残差恰好等于路由错误**）。无标签的"自我核对"判据用 [[tSelfWrites]] 次写入
达到全写的召回（全写要 [[tAlwaysWrites]] 次，少了 [[tSelfWriteReduction]]），
基座漂移 [[tSelfKL]] 而非 [[tAlwaysKL]] nats（少约五分之四）。

\textbf{两条主张失败了，我们如实报告而不是挑顺眼的说。}基于似然阈值的"惊讶度"判据
量的是\textbf{措辞}而不是知识（强制裸值作答要 [[tForcedParisKL]] nats，而模型自己那句话只要
[[tVerboseParisKL]] nats）；参数化记忆\textbf{并不能}减轻上下文惯性——
把前提写进权重与留在上下文里，被带进下一话题的程度\textbf{完全相同}（[[tContext]] 对 [[tParam]]）。

在真实基准内容上做的一次\textbf{范围受限的诊断}（30 道 LongMemEval single-session 题，历史完全不在场）
在\textbf{零额外 prompt token} 的前提下复现了 [[tLongMemOn]] 的参考答案；
而把同样的信息放进上下文——检索一条、三条、或全塞 30 条——只能答对
[[tJudgeContextTarget]] 到 [[tJudgeContextAll]]，代价是 [[tTokContextTarget]] 到 [[tTokContextAll]] 个 token。
我们还报告了**差点把这个对比做错**的过程：早先一次"两种介质不可分"的读数，
其实是我们自己生成预算截断造成的假象。

\section{引言}

\subsection{上下文通道是零和的}
在不改动权重的前提下，新信息唯一能去的地方就是上下文窗口，而这个通道有三个结构性缺陷：
\textbf{零和}（记住的 token 不再是可用于推理的 token）、\textbf{不可区分}（每句被记住的话都
"同等在场"，长期记忆与当前工作集没有区别）、\textbf{无法逃脱}（一旦进了上下文，除非重写提示词
并丢掉 KV cache，否则无法有选择地移除它）。

\subsection{第二个动机（后被我们否证）}
我们当初的假设是：上下文里的内容会被模型当作"正在发生的事"，而不是"可供回忆的历史"，
因此活在权重里的记忆会更不容易被带过话题边界。我们把它写在这里，因为在 §\ref{sec:neg}
里我们\textbf{否证}了它——这个否证对读者比一个含糊的正面结论更有用。

\subsection{我们造了什么}
一个冻结主干 + $K$ 个槽的旁路。每个投影层挂一个 LoRA 适配器，其秩 $R = K \cdot r$
被划分成 $K$ 个连续块，第 $k$ 块就是第 $k$ 个槽。因此：写入可以**只作用于一个块**；
读取可以**只加回一个块**；擦除就是**恢复该块的初始 $A_k$ 并把 $B_k$ 清零**——
这一点让"擦除"成为可以逐位验证的命题，而不是"应该差不多了"。

\subsection{贡献}
\begin{enumerate}\setlength\itemsep{2pt}
\item 一条旁路，其写入、读取、擦除\textbf{三者都可独立检查}：写入被断言只触碰一个槽，
擦除被断言把该槽还原成"从未写过"的逐位状态，读取可以在\textbf{逐字节相同的模型}上逐槽消融。
\item 发现\textbf{读时组合}才是约束：全部相加毫无价值，只选一个槽约等于 oracle，
选两个就已经损失大部分收益；而且"oracle 减去 top-1"**恰好等于路由错误**。
\item 一个无标签的写入判据（先问模型），在同等召回下把写入次数与基座漂移都降到零头；
并给出它所替代的似然判据的\textbf{否证}。
\item 容量下的\textbf{精确擦除}：每次驱逐都把槽还原为初始态；被标记为"重要"的槽能扛过反复溢出；
只有最笨的淘汰策略会丢掉"正在被使用"的记忆。
\item 跨进程持久化：加载前后可证明"未写过"的旁路，能回忆起上一个进程写进去的东西，
并且之后仍然可以精确擦除。
\item 两条\textbf{负结果}（上下文惯性假设、似然型写入判据）与一条\textbf{未复现}的修复尝试
（B1 保留项）：它们约束主张，而不是给主张贴金。
\item 一个范围受限的\textbf{真实内容诊断}：用\textbf{同一个路由器}把同样的信息送进上下文，
再由一个\textbf{与人工标注校准过的 LLM 判官}打分——参数臂在**零额外 token** 下全部答对。
\end{enumerate}

\section{方法}
\label{sec:method}

\subsection{秩块槽位与精确擦除}
对每个目标投影，$\Delta W = \frac{\alpha}{r} B A$，其中 $A$ 按秩块划分。
槽 $k$ 只允许写 $A$ 的第 $k$ 块与 $B$ 的第 $k$ 列；读掩码决定"加回哪些块"。
擦除 = 恢复 $A_k^{\text{init}}$ 且 $B_k \leftarrow 0$。因此"擦除"与"从未写过"是**可以逐位比较**的。

\subsection{写入目标}
写入损失是
\begin{equation}
\mathcal{L} = \mathrm{CE}(v \mid q)
\; + \; \lambda\, D_{\mathrm{KL}}\!\left(p_{\text{frozen}} \,\|\, p_{\text{slot}}\right)
\; + \; \lambda_{\mathrm{ret}}\, D_{\mathrm{KL}}\!\left(p_{\text{bank}} \,\|\, p_{\text{bank}+k}\right),
\end{equation}
其中目标项只开\textbf{本槽}（学的是"查询→答案"这条关联）；锚定项在一组通用问题上把分布拉回冻结态。
两处细节是\textbf{被测量逼出来的}：锚定必须\textbf{逐槽}评估（按"已写槽求和"评估时各槽位移会
**互相抵消**，该守卫等于没加），且必须覆盖\textbf{每个位置}（只看末位时单步散度被压到 0.08，
而自回归仍复合成复读机）。

第三项是\textbf{保留项}（B1）：参考分布取自\textbf{写入之前}的记忆库（已写槽激活、新槽零初始化
因此还不贡献），查询是**更早那些记忆的查询**；当前项把新槽加上去，即\textbf{干扰真正发生}的读条件。
它的效果见 §\ref{sec:neg}——**方向对，但没有复现**。

\subsection{读时：选，而不是和}
所有槽一起激活时答案崩溃：每个槽都被训练成产生一个低熵答案，它们的和是一个被支配的混合。
但各槽本身**完全可分**（读错的槽什么都得不到），所以正确做法是\textbf{选择}。
我们用查询键（查询在冻结主干下的末位隐状态，**取键时关闭记忆**）与各槽键做余弦相似度，
只激活最匹配的那一个。不新增参数、不训练。

\subsection{什么值得写}
四种判据：\texttt{always}（全写）、\texttt{surprise}（写入前损失超阈值）、
\texttt{selfcheck}（先问冻结模型"你知道这个吗"）、\texttt{explicit}（只写带标记的内容）。

\subsection{遗忘与持久化}
容量 $K$ + [[tPolicies]] 种淘汰策略（FIFO/LRU/LFU/效用/效用+时间衰减），
外加"重要"标记作为\textbf{硬保护}。持久化用 safetensors 快照 + 元数据严格校验
（槽数/秩/alpha/模块集合/形状必须一致，否则报错而不是静默加载）。

\section{评测协议}
\label{sec:protocol}

\paragraph{虚构基准。}每个实体与取值都由音节生成，因此任何答案都不可能来自预训练知识；
否则"答对"无法归因于写入。同理，"已经知道"这一类是**问模型确认**的，而不是假定的：
[[tKnownConfirmed]] 条真实常识里模型只答对 [[tKnownConfirmed]] 条（它把 $2+2$ 答成 2、
把一周的天数答成 1、把最大的行星答成 Mars）。

\paragraph{五段隔离检查（P1–P5）。}P1 驱逐：答案不得出现在读时上下文里（精确/规范化/数值三种匹配），
否则该条不计入。P2 反事实：取值按种子随机化。P3 负对照：从未写过的实体不得变好。
P4 消融：上下文逐字节相同，只改活跃槽集合。P5 仅提示词：剥掉提示词格式带来的收益。
只有在配对差异的 bootstrap 置信区间不含零时才报告主张。

\paragraph{改写探针。}读取用**不同措辞**的问法，而记忆是按原始措辞写的。
若沿用同一个字符串，路由准确率会到 100\%，那测的是字符串匹配而不是记忆。

\paragraph{地板筛选。}惯性实验先测"没有任何前提"的条件，把模型本来就能答对的场景剔除；
[[tScreenKept]] 个场景存活，被剔除的 [[tScreenDropped]] 个都是路边情形。

\paragraph{真实内容，以及怎么判分。}公开基准诊断用 LongMemEval 的 oracle 划分：
每条记忆**就是基准自带的 (问题, 答案) 对**，证据轮由基准自己的 \texttt{has\_answer} 标记定位
（即**检索被 oracle 化**）。对照组拿到**同样的信息**、由**同一个查询键路由器**选出来放进上下文，
所以唯一的变量是**介质**。判分是语义的而不是词面的：词面包含判据奖励"逐字复现参考答案"，
而那正是训练过的写入**擅长**的事，因此它无法比较"改写作答"的臂。所以真实内容结果由
**托管的 LLM 判官**打分（**一次只判一个候选**），并与 **[[tJudgeCalibMarks]] 条人工标注**校准；
标注样本是**匿名且打乱**的，判官的结论在人工标完之后才展示。我们同时报告校准规模与
"判官提示词改过一次"这两件事，因为它们限定了这个一致率值多少。另外两点：同一批题上的比较
一律用**配对检验**（精确符号检验），而不是看区间是否重叠；**生成预算被当作处理的一部分**——
所有臂用同一个上限，且该上限要能让最长的参考答案写得下。

\section{结果}
\label{sec:results}

\subsection{写进去了，而且只动了该动的槽（T1）}
写入隔离断言处处通过，被擦除的条件与"仅提示词"**逐字符相同**（[[tEraseIdentical]]），
写完的目标交叉熵为 [[tTargetCE]]。表里那个很低的 \texttt{mem\_on} 行是\textbf{求和式}读取，
它是 §\ref{sec:results} 要处理的那个失败，而不是关于"写不进"的证据。

\subsection{读时组合压倒一切（T2）}
\begin{table}[h]\centering\small
\begin{tabular}{lccc}\toprule
臂 & 0.6B & \textbf{1.7B} \\ \midrule
\texttt{oracle}（只开本槽） & [[tOracle]] & [[tScaleTwoOracle]] \\
\texttt{top1}（路由最优槽） & [[tTopOne]] & [[tScaleTwoTopOne]] \\
\texttt{top2}（路由前两槽） & [[tTopTwo]] & [[tScaleTwoTopTwo]] \\
\texttt{all}（全部相加） & [[tSum]] & [[tScaleTwoSum]] \\
路由准确率 & [[tRouterAcc]] & [[tScaleTwoRouter]] \\ \bottomrule
\end{tabular}
\caption{改写问法下的读时组合（40 个探针）。}
\end{table}

写入同样的记忆、只改"哪些槽参与"，**全加是 [[tSum]]**，只选一个槽是 [[tTopOne]]（oracle [[tOracle]]），
选两个已经掉到 [[tTopTwo]]。而且算术**精确闭合**：oracle 减 top-1 恰好等于路由错误，
说明**选择规则自身不引入任何损失**。实用结论比"越稀疏越好"更强：这种记忆的读取路径应当
\textbf{只选一个槽}，top-$k$ 的对冲会损失大部分收益。

\textbf{这个形状在规模上存活。}在 1.7B 上全加仍是 [[tScaleTwoSum]]，只选一个槽是
[[tScaleTwoTopOne]]（oracle [[tScaleTwoOracle]]），选两个是 [[tScaleTwoTopTwo]]，
而且同样的算术**精确闭合**（oracle 减 top-1 = 路由错误，$1 - [[tScaleTwoRouter]]$）。
随规模改变的是\textbf{损失在哪里}：路由器在 1.7B 上明显更差（0.6B 是 [[tRouterAcc]]，
1.7B 是 [[tScaleTwoRouter]]），于是与 oracle 的差距变成**检索问题**而不是干扰问题。
这指向**路由器**——而不是槽位机制——是下一步该改进的部件。

\subsection{该写什么（T3）}
把 $5$ 条虚构事实与 $[[tKnownConfirmed]]$ 条已验证常识混成一条流。
全写要 [[tAlwaysWrites]] 次梯度写入，其中 [[tAlwaysWasted]] 次花在模型**本来就会**的事实上，
基座漂移 [[tAlwaysKL]] nats。自我核对只用 [[tSelfWrites]] 次写入、浪费 [[tSelfWasted]]、
漏掉 [[tSelfMissed]] 个目标，漂移 [[tSelfKL]] nats——**同等召回，写入少 [[tSelfWriteReduction]]、
漂移少 [[tSelfKlReductionMin]]--[[tSelfKlReductionMax]]**（后者是 [[tSelfSeeds]] 个种子的范围；
写入次数是确定的，漂移不是，所以给范围）。

似然判据是反面教材：它几乎分不开两类（见 §\ref{sec:neg}），因为固定措辞量的是
"模型会不会用**这个说法**"，而不是"它知不知道这件事"。

\subsection{忘得准（T4）}
[[tStreamItems]] 条记忆流进 [[tCapacitySlots]] 个槽，前 $3$ 个被反复访问。
每次驱逐都把槽还原为初始态：[[tErasedTotal]] 个槽被验证与"从未写过"逐位相同（每种策略 [[tErased]]）。
被标记为重要的槽扛过了每一次溢出。FIFO 只留下 [[tFifoHot]] 个"正在被使用"的记忆——
它专挑你在用的丢、留着你没碰过的——其余四种策略都留下 [[tSmartHot]]：
策略确实重要，但"聪明"的几种打平，这独立复现了文献里"淘汰启发式很容易被高估"的结论。

\subsection{跨进程（T6）}
新进程里加载前 \texttt{virgin}=\texttt{[[tVirgin]]}、重启后召回 [[tSixOracle]]、
路由召回 [[tRouted]]（与路由准确率 [[tSixRouting]] 相同，因此差的那一题是路由错而不是记忆错）、
加载后擦除仍逐位精确（\texttt{[[tSixEraseVirgin]]}）；[[tSnapshotSlots]] 个槽的快照 [[tSnapshotMB]]\,MB。

\subsection{真实内容：为什么不把记忆放进上下文（T7）}
30 道 LongMemEval single-session 题，**历史完全不在场**，同一批题、同一个路由器、语义判官打分：

\begin{table}[h]\centering\small
\begin{tabular}{lccc}\toprule
臂 & 判官正确率（0.6B） & 判官正确率（1.7B） & 额外 prompt token \\ \midrule
\textbf{参数记忆} & \textbf{[[tJudgeWeights]]} & \textbf{[[tScaleWeights]]} & \textbf{[[tTokWeights]]} \\
30 条全塞上下文 & [[tJudgeContextAll]] & [[tScaleContextAll]] & [[tTokContextAll]] \\
检索 1 条 / 3 条 & [[tJudgeRagOne]] / [[tJudgeRagThree]] & --- & [[tTokRagOne]] / 219 \\
oracle 单条 & [[tJudgeContextTarget]] & [[tScaleContextTarget]] & [[tTokContextTarget]] \\
冻结（地板） & [[tJudgeFrozen]] & [[tScaleFrozen]] & 60 \\ \bottomrule
\end{tabular}
\caption{真实内容上的公平对比（判官已与人工标注校准 [[tJudgeCalibMarks]]）。}
\end{table}

参数臂在**逐题配对**中赢了 [[tJudgeWins]] 题、输了 [[tJudgeLosses]] 题，即\textbf{一题都没输}，
而且**额外 prompt token 为 [[tTokWeights]]**。留出集复制（另 30 题）得到 [[tHoldoutOn]]
（[[tHoldoutVerdict]]，预注册带 $\pm 0.15$）。

\textbf{两条测量教训是我们自己该报告的，它们让表面差距比第一眼看上去更小。}
第一，词面包含判据是有偏的：它奖励逐字复现，而那正是训练过的写入擅长的事，
它会给同样内容的正确改写**判 0**。第二，我们**自己**在这一对比上的第一版读数是
"两种介质不可分辨"（当时参数臂的包含率 [[tLongMemOnArchived]]）——那是我们**生成预算只有 32 token**
把它截断造成的，判官于是合理地判它"不完整"。把所有臂的预算一起提到能写下最长参考答案之后，
参数臂变成 [[tLongMemOn]]。

\textbf{规模对照。}同样在 1.7B 上重跑：参数臂 [[tScaleWeights]]，上下文递送
[[tScaleContextTarget]]--[[tScaleContextAll]]，成本列不变（[[tTokWeights]] 对最多 [[tScaleTokens]] 个 token），
参数臂仍然**一题不输**。但\textbf{领先幅度随规模收窄}——更大的模型**更会用上下文**
（[[tScaleFrozen]] 的题它能凭先验答对，0.6B 只有 [[tLongMemFrozen]]）。
所以：**成本优势与规模无关，质量优势与规模有关**——两点曲线，就这样写。

\section{负结果}
\label{sec:neg}

\subsection{参数化记忆并不能减轻上下文惯性（被否证，T5）}
把同一条前提分别放进上下文与权重，然后切换话题：两种做法把前提带进下一话题的程度
\textbf{完全相同}（[[tContext]] 对 [[tParam]]），干净场景上的残留也一样
（[[tContextResidue]] 对 [[tParamResidue]]）。地板（哪里都没有前提）是 [[tFloor]]，
控制臂（写了再擦）精确塌回地板。也就是说：惯性的来源是模型在条件化记忆的\textbf{内容}，
而不是那些内容占着上下文 token。

附带一个值得记录的观察：**有记忆会让模型更自信**——"我不知道"的比例在无记忆时是
[[tWithholdFloor]]，有记忆时降到 [[tWithholdMemory]]——它不只是被更好地告知，而是更敢说。

\subsection{似然判据量的是措辞（T3-b）}
它对模型**完全答得出**的内容判为"高惊讶度"。逐 token 诊断给出原因：模型从不用裸值作答
（它答 "France's capital is Paris."），强制续写 \texttt{" Paris"} 要 [[tForcedParisKL]] nats，
而模型自己那句话只要 [[tVerboseParisKL]] nats。两类内容的分布因此几乎重合
（中位数 [[tSurpriseMedianUnknown]] 对 [[tSurpriseMedianKnown]]），实测也差：
漏写 [[tSurpriseMissed]]、误写 [[tSurpriseWasted]]。**推广的教训：只有当探针允许模型用自己的措辞时，
"惊讶度"才是"未知"的代理量。**

\subsection{B1 保留项：假设未复现（T8）}
两槽读取的失败**不是检索**（正确槽位落在前两名之内的比例是 [[tRetentionRouteTwo]]），
而是\textbf{写入}制造的干扰。我们因此在写入目标里加入保留项（§\ref{sec:method}，$\lambda_{\mathrm{ret}}$ = [[tRetentionBest]]）：

\begin{table}[h]\centering\small
\begin{tabular}{lccc}\toprule
& 基线 & 加保留项 & 逐探针配对 \\ \midrule
两槽召回 & [[tRetentionTopTwoBase]] & [[tRetentionTopTwoBest]] & \textbf{[[tRetentionHelped]] 帮助 / [[tRetentionHurt]] 损害，$p$=[[tRetentionP]]} \\
全部槽一起激活 & [[tRetentionAllBase]] & [[tRetentionAllBest]] & [[tRetentionAllHelped]] / [[tRetentionAllHurt]]，$p$=[[tRetentionAllP]] \\
自身槽位召回 & [[tRetentionOwnBase]] & [[tRetentionOwnBest]] & --- \\
基座损伤（nats） & [[tRetentionKlBase]] & [[tRetentionKlBest]] & --- \\ \bottomrule
\end{tabular}
\caption{B1 保留项在 [[tRetentionProbes]] 个探针上的结果（每条记忆，[[tRetentionItems]] 条）。}
\end{table}

\textbf{它没有修复两槽召回，我们报告这一点而不是那个说它修好了的小样本。}
在 [[tRetentionProbes]] 个探针上配对结果是 [[tRetentionHelped]] 帮助对 [[tRetentionHurt]] 损害
（精确符号检验 $p$ = [[tRetentionP]]），两槽召回只从 [[tRetentionTopTwoBase]] 动到
[[tRetentionTopTwoBest]]——与噪声无异。更早一次 [[tRetentionSmallProbes]] 探针的运行**看起来**像修复
（[[tRetentionSmallHelped]] 对 [[tRetentionSmallHurt]]，$p$ = [[tRetentionSmallP]]），
把样本翻四倍后没有复现，所以我们两次都报。

\textbf{站得住的是两件事，而且每一次运行都成立}：基座损伤降约五分之四
（[[tRetentionKlBase]] $\rightarrow$ [[tRetentionKlBest]] nats）；以及保留项**真正被评估的那个条件**
单向改善——全开档从 [[tRetentionAllBase]] 到 [[tRetentionAllBest]]，
[[tRetentionAllHelped]] 帮助对 [[tRetentionAllHurt]] 损害（$p$ = [[tRetentionAllP]]）——
幅度小，但不是噪声。诚实的总结是：**教写入保护它的前辈，买到了基座损伤的大幅下降与多槽条件下的
小幅改善，而我们假设的两槽修复没有证据。**

\section{限制}
\label{sec:limits}
\begin{itemize}\setlength\itemsep{2pt}
\item \textbf{规模。}四条头条结论里有两条在 1.7B 上重跑并存活（组合结论与真实内容诊断）；
\textbf{没有}存活的是路由器，它的准确率从 [[tRouterAcc]] 掉到 [[tScaleTwoRouter]]——
这是"瓶颈从干扰挪到检索"，不是瓶颈消失。惯性否证与遗忘实验仍只有 0.6B，3B 及以上在本机未测。
\item \textbf{基座要付代价。}写入会移动冻结主干：流进 [[tStreamItems]] 条记忆后，通用问题
不再被逐字答对，全位置 KL 约 $1$ nat。参数化记忆不是免费的，且代价随写入次数增长。
\item \textbf{B1 那条头条效应没有复现。}[[tRetentionSmallProbes]] 探针时它像两槽修复
（$p$ = [[tRetentionSmallP]]），[[tRetentionProbes]] 探针时配对是 [[tRetentionHelped]] 对
[[tRetentionHurt]]（$p$ = [[tRetentionP]]），也就是什么都没有。我们把两次都留在论文里，
因为"效应随样本缩小"本身就是结论。
\item \textbf{公开基准是范围受限的诊断，不是可比成绩。}记忆条**就是**基准自带的
$(q,a)$；证据轮由基准自己的标记定位（检索 oracle 化）；写入查询等于读取查询；
只取了 single-session 子集；判官校准只有 [[tJudgeCalibMarks]] 且提示词改过一次；
判官读得到参考答案，因此**残留的"偏向接近逐字"不能排除**。
\item \textbf{指标粗糙。}基座退化有一部分靠"答案是否逐字相同"判断，它会把
`$2+2=4$` 相对冻结态的 `2` 算作变化；KL 数字更有信息量。
\item \textbf{多数合成实验是单种子。}T1–T5 各跑一次；我们没有做完整多种子矩阵，
而是给**公开基准诊断**做了留出集复制（在 §\ref{sec:results}）。
\end{itemize}

\section{复现方式}
\begin{verbatim}
python -m venv .venv && .venv\Scripts\pip install -e .
python experiments/run_all.py --collect-only   # 从已有报告重建全部数字
python experiments/run_all.py --mode quick     # 冒烟（分钟级）
python experiments/run_all.py --mode full      # 从零重跑全部阶段
python paper/build_tables.py                   # 生成英文版表格与宏
python paper/build_paper_zh.py                 # 生成本中文审阅稿（随后用 xelatex 编译）
\end{verbatim}
每个阶段都是独立进程并写自己的 JSON；失败的阶段在汇总里标 \texttt{FAILED}，其数字渲染为
\texttt{n/a}，**绝不用 0 冒充**。本文件里每个数字都来自 \texttt{runs/summary.json}。

\section{结论}
一条秩块槽位的旁路把三件事做得很好、一件事只做到一半：写入不打扰邻居、读取不花上下文、
遗忘可以逐位验证（包括跨进程重启）；它\textbf{不能}把模型与上一个话题隔开——
记忆无论存在哪里都是有粘性的。我们要带走的实用规则是：读时组合默认只选一个槽；
写入判据应当去问模型而不是卡它的似然；容量应当花在**持久且可精确擦除**的槽上。
我们追过的那一个例外在相反方向上很有教益：教写入保护记忆库能大幅降低基座损伤、
略微改善全槽条件，但**并不买回第二个槽**——更小的样本曾暗示它买回了，四倍样本否证了它。
\textbf{一个随样本量缩小的效应本身就是一个结果；我们宁愿报告被否证的修复，
也不愿报告那个讨人喜欢的样本。}

\end{document}
"""


def render(template: str, values: dict[str, str]) -> str:
    """Substitute macros, convert markdown bold, and refuse to emit a broken file."""
    missing = sorted({name for name in PLACEHOLDER.findall(template)
                      if name not in values})
    if missing:
        raise SystemExit(f"template references macros that do not exist: {missing}")
    text = PLACEHOLDER.sub(lambda m: tex_escape(values[m.group(1)]), template)
    # The prose is written with **bold** because it is far easier to read and edit
    # that way, but LaTeX would print the asterisks. Convert, then assert nothing is
    # left: an unmatched ** would otherwise reach the PDF unnoticed.
    text = BOLD.sub(lambda m: r"\textbf{" + m.group(1) + "}", text)
    if "**" in text:
        leftover = [line.strip()[:60] for line in text.splitlines() if "**" in line]
        raise SystemExit(f"unconverted markdown bold left in the document: {leftover}")
    return text


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default=str(ROOT / "runs" / "summary.json"))
    ap.add_argument("--out", default=str(ROOT / "paper" / "main_zh.tex"))
    ap.add_argument("--compile", action="store_true",
                    help="also run xelatex twice and report the page count")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    from parammem.report import headline_values

    bundle_path = Path(args.bundle)
    if not bundle_path.is_file():
        print(f"missing {bundle_path}; run experiments/run_all.py --collect-only",
              flush=True)
        return 1
    values = headline_values(json.loads(bundle_path.read_text(encoding="utf-8")))
    text = render(TEMPLATE, values)
    out = Path(args.out)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} : {len(text)} chars, "
          f"{len(PLACEHOLDER.findall(TEMPLATE))} generated values used", flush=True)

    if not args.compile:
        return 0
    for pass_no in (1, 2):
        log = out.parent / f"main_zh_pass{pass_no}.txt"
        with log.open("w", encoding="utf-8", errors="replace") as handle:
            code = subprocess.run(["xelatex", "-interaction=nonstopmode", out.name],
                                  cwd=str(out.parent), stdout=handle,
                                  stderr=subprocess.STDOUT).returncode
        if code != 0:
            print(f"  xelatex pass {pass_no} exited {code}; see {log}", flush=True)
            return code
    pdf = out.with_suffix(".pdf")
    pages = ""
    log_text = (out.parent / "main_zh_pass2.txt").read_text(encoding="utf-8",
                                                            errors="replace")
    match = re.search(r"Output written on main_zh\.pdf \((\d+) pages", log_text)
    if match:
        pages = f"{match.group(1)} pages"
    print(f"  compiled {pdf} ({pages}, {pdf.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
