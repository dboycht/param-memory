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
to be updated by hand. The *numbers* are automatic; the *prose* is a translation.

Writing style
-------------
The prose is written as a narrative rather than as a sequence of contrastive slogans.
Published analyses of machine-generated text single out the "not X, but Y" antithesis
(artificial epanorthosis), em-dash density, rule-of-three lists and metronomic
sentence rhythm as the reliable tells, and the guidance is to count them and fix them
in passes. ``_dev/check_ai_tells.py`` counts them: the first draft of this file used
"不是" 26 times and 30 em dashes in 19k characters, and this version stays far below
that. Sentence lengths here are meant to be uneven.

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
    cd paper && xelatex main_zh.tex           # twice, for the tables
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
\title{\bf 面向 LLM 智能体的旁路参数化记忆：\\可精确擦除的槽位，以及不占上下文的读取}
\author{成昊天 \\ \small 南京航空航天大学}
\date{2026 年 9 月 30 日}

\begin{document}
\maketitle

\zhbanner{\small\textbf{这是中文审阅稿。}它与英文版 \texttt{main.pdf} 逐节对应，
所有数字都由 \texttt{runs/summary.json} 生成，与英文版同一来源；
\textbf{技术主张以英文版为准}。请先看三处：（一）§\ref{sec:neg} 里 B1 保留项
“未复现”的措辞；（二）§\ref{sec:limits} 关于“质量优势随规模收窄”的表述；
（三）表格是否一次就能读懂。}

\vspace{0.6em}
\begin{center}\small
数字来源：\texttt{runs/summary.json}。生成脚本：\texttt{paper/build\_paper\_zh.py}。
\end{center}

\section{摘要}
LLM 智能体的长期记忆放在上下文窗口里，既要挤占推理所需的 token，又与当前工作集无法区分。
我们换了一条路：在冻结主干上挂一组容量受限的低秩记忆槽，用梯度写入、经前向传播读取、
靠擦除一个秩块遗忘。在 0.6B 模型上，写入只落到自己的槽；被擦除的槽与从未写过的槽逐位相同，
[[tErasedTotal]] 次驱逐与一次进程重启都如此；真正的约束是读时组合：把槽全加起来召回为
[[tSum]]，按查询键相似度只选一个则恢复到 [[tTopOne]]（oracle [[tOracle]]），两者之差恰好等于
路由错误。无标签的自我核对用 [[tSelfWrites]] 次写入达到全写的召回（全写 [[tAlwaysWrites]] 次），
基座漂移 [[tSelfKL]] nats（全写 [[tAlwaysKL]] nats）。有两个假设失败了，我们如实报告：
基于似然阈值的惊讶度判据量到的是措辞，参数化记忆也没有减少上下文惯性。在 [[tLongMemSubset]]
道 LongMemEval single-session 题（历史完全不在场）上，参数臂复现 [[tLongMemOn]] 的参考答案
且不额外占用 prompt token，而把同样信息放进上下文只能答对 [[tJudgeContextTarget]] 到
[[tJudgeContextAll]]，并付出 [[tTokContextTarget]] 到 [[tTokContextAll]] 个 token。

\section{引言}

\subsection{上下文通道的两个代价}
在不改动权重的前提下，新信息只能进上下文窗口。这条通道有三个结构性限制。
它是零和的：记住的 token 不再是可用来推理的 token。它不可区分：每句被记住的话
都同等在场，长期记忆与当前工作集没有分别。它也退不出去：内容一旦进入上下文，
除非重写提示词并丢掉 KV cache，否则无法有选择地移除。

\subsection{一个后来被我们否证的动机}
我们当初的假设是：上下文里的内容会被模型当作当前正在发生的事。我们想知道，活在权重里的记忆是否更不容易
被带过话题边界。这个假设写在 §\ref{sec:neg}，
因为我们在那里否证了它。对一个读者来说，这个否证比一个含糊的正面结论更有用。

\subsection{我们造了什么}
一个冻结主干，加一条 $K$ 个槽的旁路。每个投影层挂一个 LoRA 适配器，总秩
$R = K \cdot r$ 被切成 $K$ 个连续块，第 $k$ 块就是第 $k$ 个槽。写入可以只作用于一个块，
读取可以只加回一个块，擦除就是把该块的初始 $A_k$ 恢复，把 $B_k$ 清零。
擦除因此成了一个可以逐位验证的命题，而不再是一个大致的感觉。

\subsection{贡献}
\begin{enumerate}\setlength\itemsep{2pt}
\item 一条旁路，其写入、读取、擦除三者都能独立检查：写入被断言只触碰一个槽，
擦除被断言把槽还原成从未写过的逐位状态，读取可以在逐字节相同的模型上逐槽消融。
\item 读时组合是约束所在。全部相加毫无价值，只选一个槽约等于 oracle，
选两个就损失大部分收益，而 oracle 与 top-1 之差恰好等于路由错误。
\item 一个无标签的写入判据（先问模型），在同等召回下把写入次数与基座漂移都降到零头，
并给出它所替代的似然判据的否证。
\item 容量下的精确擦除。每次驱逐都把槽还原为初始态，被标记为重要的槽能扛过反复溢出，
只有最笨的淘汰策略会丢掉正在被使用的记忆。
\item 跨进程持久化。加载前后可证明未写过的旁路，能回忆起上一个进程写进去的东西，
并且之后仍然可以精确擦除。
\item 两条负结果（上下文惯性假设、似然型写入判据）与一条未复现的修复尝试（B1 保留项）。
它们限定了我们主张的边界。
\item 一个范围受限的真实内容诊断。同一个路由器把同样的信息送进上下文，
再由一个与人工标注校准过的 LLM 判官打分，参数臂在零额外 token 下全部答对。
\end{enumerate}

\section{方法}
\label{sec:method}

\subsection{秩块槽位与精确擦除}
对每个目标投影，$\Delta W = \frac{\alpha}{r} B A$，其中 $A$ 按秩块划分。
槽 $k$ 只能写 $A$ 的第 $k$ 块与 $B$ 的第 $k$ 列，读掩码决定加回哪些块。
擦除就是恢复 $A_k^{\text{init}}$ 并把 $B_k$ 置零，所以擦除与从未写过
是可以逐位比较的两件事。

\subsection{写入目标}
写入损失有三项：
\begin{equation}
\mathcal{L} = \mathrm{CE}(v \mid q)
\; + \; \lambda\, D_{\mathrm{KL}}\!\left(p_{\text{frozen}} \,\|\, p_{\text{slot}}\right)
\; + \; \lambda_{\mathrm{ret}}\, D_{\mathrm{KL}}\!\left(p_{\text{bank}} \,\|\, p_{\text{bank}+k}\right).
\end{equation}
第一项只开本槽，学的是查询到答案这条关联。第二项在一组通用问题上把分布拉回冻结态，
两处细节都是被测量逼出来的：它必须逐槽评估，因为按已写槽求和评估时各槽的位移会互相抵消，
这一项等于没加；它也必须覆盖每个位置，因为只看末位时单步散度被压到 $0.08$，
自回归解码仍然复合成复读机。第三项是保留项（B1）。参考分布取自写入之前的记忆库，
那时已写槽激活、新槽零初始化因而还不贡献；查询用的是更早那些记忆的查询，
当前项把新槽加上去，也就是干扰真正发生的那个读条件。它的效果见 §\ref{sec:neg}：
方向对了，但没有复现。

\subsection{读时选择}
所有槽一起激活时答案会崩。原因不难理解：每个槽都被训练成产生一个低熵答案，
它们的和是一个被支配的混合。各槽本身完全可分，读错的槽什么都得不到，
所以读取路径应当选择。我们用查询键（查询在冻结主干下的末位隐状态，取键时关闭记忆，
避免键随已写记忆漂移）与各槽键做余弦相似度，只激活最匹配的那一个。这一步不新增参数，
也不需要训练。

\subsection{什么值得写}
四种判据：\texttt{always} 全写；\texttt{surprise} 看写入前损失是否超阈值；
\texttt{selfcheck} 先问冻结模型是否知道这个内容；\texttt{explicit} 只写带标记的内容。

\subsection{遗忘与持久化}
容量 $K$ 加上 [[tPolicies]] 种淘汰策略（FIFO、LRU、LFU、效用、效用加时间衰减），
另有重要标记作为硬保护。持久化用 safetensors 快照加元数据严格校验：
槽数、秩、alpha、模块集合与形状必须一致，不一致就报错，不做静默加载。

\section{评测协议}
\label{sec:protocol}

\paragraph{虚构基准。}每个实体与取值都由音节生成，因此答案不可能来自预训练知识；
否则答对无法归因于写入。同理，已经知道这一类要先问过模型：
[[tKnownConfirmed]] 条真实常识里，模型把 $2+2$ 答成 2，把一周的天数答成 1，
把最大的行星答成 Mars，最后只有 [[tKnownConfirmed]] 条可用。

\paragraph{五段隔离检查（P1 到 P5）。}P1 驱逐：答案不得出现在读时上下文里
（精确、规范化、数值三种匹配），否则该条不计入。P2 反事实：取值按种子随机化。
P3 负对照：从未写过的实体不得变好。P4 消融：上下文逐字节相同，只改活跃槽集合。
P5 仅提示词：剥掉提示词格式带来的收益。只有在配对差异的 bootstrap 置信区间不含零时，
我们才报告主张。

\paragraph{改写探针。}读取用不同措辞的问法，记忆仍按原始措辞写入。
若沿用同一个字符串，路由准确率会到 100\%，那测到的只是字符串匹配。

\paragraph{地板筛选。}惯性实验先测没有任何前提的条件，把模型本来就能答对的场景剔除。
[[tScreenKept]] 个场景存活，被剔除的 [[tScreenDropped]] 个都是路边情形。

\paragraph{真实内容，以及怎么判分。}公开基准诊断用 LongMemEval 的 oracle 划分。
每条记忆就是基准自带的问题与答案对，证据轮由基准自己的 \texttt{has\_answer} 标记定位，
检索因此被 oracle 化。对照组拿到同样的信息，由同一个查询键路由器选出来放进上下文，
唯一的变量是介质。判分走语义。词面包含判据奖励逐字复现参考答案，
而这正是训练过的写入擅长的事，它无法比较改写作答的臂。所以真实内容结果由一个托管的
LLM 判官打分，一次只判一个候选，并与 [[tJudgeCalibMarks]] 条人工标注校准。
标注样本匿名且打乱，判官的结论在人工标完之后才展示。我们同时报告校准规模与
判官提示词改过一次这两件事，因为它们限定了这个一致率值多少。另有两点：
同一批题上的比较一律用配对检验，也就是精确符号检验；生成预算被当作处理的一部分，
所有臂用同一个上限，且这个上限要能让最长的参考答案写得下。

\section{结果}
\label{sec:results}

\subsection{写入只落到该落的槽（T1）}
写入隔离断言处处通过，被擦除的条件与仅提示词的条件逐字符相同（[[tEraseIdentical]]），
写完的目标交叉熵是 [[tTargetCE]]。表里那个很低的 \texttt{mem\_on} 行是求和式读取的结果，
它属于 §\ref{sec:results} 要处理的问题，不能当作写入失败的证据。

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
\caption{改写问法下的读时组合，每种设置 40 个探针。}
\end{table}

写入同样的记忆、只改哪些槽参与：全部相加是 [[tSum]]，只选一个槽是 [[tTopOne]]
（oracle [[tOracle]]），选两个已经掉到 [[tTopTwo]]。算术在这里精确闭合，
oracle 减 top-1 恰好等于路由错误，说明选择规则本身不引入损失。实用结论比
越稀疏越好更强：这种记忆的读取路径应当只选一个槽，top-$k$ 的对冲会损失大部分收益。

这个形状在 1.7B 上同样出现：全部相加仍是 [[tScaleTwoSum]]，只选一个槽是
[[tScaleTwoTopOne]]（oracle [[tScaleTwoOracle]]），选两个是 [[tScaleTwoTopTwo]]，
同样的算术依然闭合，oracle 减 top-1 等于 $1 - [[tScaleTwoRouter]]$。
随规模改变的是损失出现的位置。路由器在 1.7B 上明显更差，0.6B 是 [[tRouterAcc]]，
1.7B 是 [[tScaleTwoRouter]]，与 oracle 的差距于是主要来自检索。
下一步该改进的部件是路由器。

\subsection{该写什么（T3）}
把 $5$ 条虚构事实与 [[tKnownConfirmed]] 条已验证常识混成一条流。全写需要
[[tAlwaysWrites]] 次梯度写入，其中 [[tAlwaysWasted]] 次花在模型本来就会的事实上，
基座漂移 [[tAlwaysKL]] nats。自我核对只用 [[tSelfWrites]] 次写入，浪费 [[tSelfWasted]]，
漏掉 [[tSelfMissed]] 个目标，漂移 [[tSelfKL]] nats。召回相同，写入少 [[tSelfWriteReduction]]，
漂移少 [[tSelfKlReductionMin]] 到 [[tSelfKlReductionMax]]。后一个范围来自 [[tSelfSeeds]] 个种子，
因为写入次数是确定的，漂移随种子变化。似然判据是反面教材，它几乎分不开两类
（见 §\ref{sec:neg}），因为固定措辞量到的是模型会不会用这个说法，与它是否知道这件事无关。

\subsection{遗忘是精确的（T4）}
[[tStreamItems]] 条记忆流进 [[tCapacitySlots]] 个槽，前 $3$ 个被反复访问。
每次驱逐都把槽还原为初始态：[[tErasedTotal]] 个槽被验证与从未写过逐位相同，
每种策略各 [[tErased]] 个。被标记为重要的槽扛过了每一次溢出。FIFO 只留下
[[tFifoHot]] 个正在被使用的记忆，它专挑你在用的丢、留着你没碰过的；
其余四种策略都留下 [[tSmartHot]]。策略确实重要，但几种聪明策略打平，
这独立复现了文献里淘汰启发式容易被高估的结论。

\subsection{跨进程（T6）}
新进程里，加载前 \texttt{virgin} 为 \texttt{[[tVirgin]]}；重启后召回 [[tSixOracle]]，
路由召回 [[tRouted]]，与路由准确率 [[tSixRouting]] 相同，所以差的那一题属于路由错误；
加载后擦除仍逐位精确（\texttt{[[tSixEraseVirgin]]}）。[[tSnapshotSlots]] 个槽的快照占
[[tSnapshotMB]]\,MB。

\subsection{真实内容：为什么不把记忆放进上下文（T7）}
30 道 LongMemEval single-session 题，历史完全不在场，同一批题、同一个路由器、语义判官打分：

\begin{table}[h]\centering\small
\begin{tabular}{lccc}\toprule
臂 & 判官正确率（0.6B） & 判官正确率（1.7B） & 额外 prompt token \\ \midrule
\textbf{参数记忆} & \textbf{[[tJudgeWeights]]} & \textbf{[[tScaleWeights]]} & \textbf{[[tTokWeights]]} \\
30 条全塞上下文 & [[tJudgeContextAll]] & [[tScaleContextAll]] & [[tTokContextAll]] \\
检索 1 条 / 3 条 & [[tJudgeRagOne]] / [[tJudgeRagThree]] & --- & [[tTokRagOne]] / 219 \\
oracle 单条 & [[tJudgeContextTarget]] & [[tScaleContextTarget]] & [[tTokContextTarget]] \\
冻结（地板） & [[tJudgeFrozen]] & [[tScaleFrozen]] & 60 \\ \bottomrule
\end{tabular}
\caption{真实内容上的对比，判官已与人工标注校准 [[tJudgeCalibMarks]]。}
\end{table}

参数臂在逐题配对中赢了 [[tJudgeWins]] 题、输了 [[tJudgeLosses]] 题，一题都没输，
额外 prompt token 为 [[tTokWeights]]。留出集复制（另 30 题）得到 [[tHoldoutOn]]，
落在预注册带内（[[tHoldoutVerdict]]，带宽 $0.15$）。

有两条测量教训是我们自己该报告的，它们让表面差距看起来比实际更大。第一，
词面包含判据有偏，它奖励逐字复现，会给同样内容的正确改写判 0。
第二，我们在这个对比上的第一版读数是两种介质分辨不出高下，
当时参数臂的包含率是 [[tLongMemOnArchived]]。原因是我们自己的生成预算只有 32 token，
把答案截断了，判官于是合理地判它不完整。把所有臂的预算一起提到能写下最长参考答案之后，
参数臂变成 [[tLongMemOn]]。

在 1.7B 上重跑，参数臂是 [[tScaleWeights]]，上下文递送是 [[tScaleContextTarget]]
到 [[tScaleContextAll]]，成本列不变（[[tTokWeights]] 对最多 [[tScaleTokens]] 个 token），
参数臂仍然一题不输。领先幅度随规模收窄，因为更大的模型更会用上下文：
[[tScaleFrozen]] 的题它能凭先验答对，0.6B 只有 [[tLongMemFrozen]]。
成本优势与规模无关，质量优势与规模有关。这是一条两点曲线，我们按它本来的样子写。

\section{负结果}
\label{sec:neg}

\subsection{参数化记忆没有减轻上下文惯性（T5）}
把同一条前提分别放进上下文与权重，然后切换话题。两种做法把前提带进下一话题的程度
相同（[[tContext]] 对 [[tParam]]），干净场景上的残留也相同（[[tContextResidue]] 对
[[tParamResidue]]）。地板条件是 [[tFloor]]，也就是哪里都没有前提；控制臂写了再擦，
精确塌回地板。惯性的来源是模型在条件化记忆的内容，与那些内容是否占着上下文 token 无关。

附带一个值得记下的观察：有记忆会让模型更自信。我不知道的比例在无记忆时是
[[tWithholdFloor]]，有记忆时降到 [[tWithholdMemory]]。它不只是被更好地告知，
也更敢直接作答。

\subsection{似然判据量到的是措辞（T3-b）}
这个判据对模型完全答得出的内容判出高惊讶度。逐 token 诊断给出原因：模型从不用裸值作答，
它答 “France's capital is Paris.”，强制续写 \texttt{" Paris"} 要 [[tForcedParisKL]] nats，
模型自己那句话只要 [[tVerboseParisKL]] nats。两类内容的分布因此几乎重合
（中位数 [[tSurpriseMedianUnknown]] 对 [[tSurpriseMedianKnown]]），实测也差：
漏写 [[tSurpriseMissed]]、误写 [[tSurpriseWasted]]。推广的教训是：只有当探针允许模型
用自己的措辞时，惊讶度才是未知的代理量。

\subsection{B1 保留项：假设未复现（T8）}
两槽读取的失败与检索无关，正确槽位落在前两名之内的比例是 [[tRetentionRouteTwo]]，
问题出在写入制造的干扰。我们在写入目标里加入保留项（§\ref{sec:method}，
$\lambda_{\mathrm{ret}}$ 取 [[tRetentionBest]]）：

\begin{table}[h]\centering\small
\begin{tabular}{lccc}\toprule
& 基线 & 加保留项 & 逐探针配对 \\ \midrule
两槽召回 & [[tRetentionTopTwoBase]] & [[tRetentionTopTwoBest]] & \textbf{[[tRetentionHelped]] 帮助 / [[tRetentionHurt]] 损害，$p$ 为 [[tRetentionP]]} \\
全部槽一起激活 & [[tRetentionAllBase]] & [[tRetentionAllBest]] & [[tRetentionAllHelped]] / [[tRetentionAllHurt]]，$p$ 为 [[tRetentionAllP]] \\
自身槽位召回 & [[tRetentionOwnBase]] & [[tRetentionOwnBest]] & --- \\
基座损伤（nats） & [[tRetentionKlBase]] & [[tRetentionKlBest]] & --- \\ \bottomrule
\end{tabular}
\caption{B1 保留项在 [[tRetentionProbes]] 个探针上的结果，每条记忆，共 [[tRetentionItems]] 条。}
\end{table}

它没有修复两槽召回。在 [[tRetentionProbes]] 个探针上，配对结果是
[[tRetentionHelped]] 帮助对 [[tRetentionHurt]] 损害，精确符号检验的 $p$ 为 [[tRetentionP]]，
两槽召回从 [[tRetentionTopTwoBase]] 动到 [[tRetentionTopTwoBest]]，与噪声没有区别。
更早一次 [[tRetentionSmallProbes]] 探针的运行看起来像修复（[[tRetentionSmallHelped]] 对
[[tRetentionSmallHurt]]，$p$ 为 [[tRetentionSmallP]]），把样本翻四倍后没有复现，
所以两次都留在这里。

站得住的是两件事，而且每一次运行都成立。基座损伤降了约五分之四，
从 [[tRetentionKlBase]] 到 [[tRetentionKlBest]] nats。保留项真正被评估的那个条件
单向改善：全开档从 [[tRetentionAllBase]] 到 [[tRetentionAllBest]]，
[[tRetentionAllHelped]] 帮助对 [[tRetentionAllHurt]] 损害，$p$ 为 [[tRetentionAllP]]。
幅度小，但它在符号检验下是可靠的。总结起来，教写入保护它的前辈，买到了基座损伤的大幅下降
与多槽条件下的小幅改善；我们假设的两槽修复没有证据。

\section{限制}
\label{sec:limits}
\begin{itemize}\setlength\itemsep{2pt}
\item \textbf{规模。}四条头条结论里有两条在 1.7B 上重跑并存活，也就是组合结论与真实内容诊断。
没有存活的是路由器，它的准确率从 [[tRouterAcc]] 掉到 [[tScaleTwoRouter]]，
瓶颈因此从干扰挪到了检索，而没有消失。惯性否证与遗忘实验仍然只有 0.6B，
3B 及以上受本机 8 GB 显存限制未测。
\item \textbf{基座要付代价。}写入会移动冻结主干。流进 [[tStreamItems]] 条记忆后，
通用问题不再被逐字答对，全位置 KL 约 $1$ nat。参数化记忆是有代价的，
而且代价随写入次数增长。
\item \textbf{B1 那条头条效应没有复现。}[[tRetentionSmallProbes]] 探针时它像两槽修复
（$p$ 为 [[tRetentionSmallP]]），[[tRetentionProbes]] 探针时配对是 [[tRetentionHelped]]
对 [[tRetentionHurt]]（$p$ 为 [[tRetentionP]]），也就是什么都没有。我们把两次都留在论文里，
因为效应随样本缩小这件事本身就是结论。
\item \textbf{公开基准只提供范围受限的诊断。}记忆条就是基准自带的问题与答案对，
证据轮由基准自己的标记定位，检索被 oracle 化；写入查询等于读取查询；
只取了 single-session 子集；判官校准只有 [[tJudgeCalibMarks]]，提示词还改过一次；
判官读得到参考答案，因此残留的偏向接近逐字无法排除。
\item \textbf{指标粗糙。}基座退化有一部分靠答案是否逐字相同来判断，
它会把 $2+2=4$ 相对冻结态的 2 算作变化；KL 数字更有信息量。
\item \textbf{多数合成实验是单种子。}T1 到 T5 各跑一次。我们没有做完整多种子矩阵，
而是给公开基准诊断做了留出集复制（在 §\ref{sec:results}）。
\end{itemize}

\section{复现方式}
\begin{verbatim}
python -m venv .venv && .venv\Scripts\pip install -e .
python experiments/run_all.py --collect-only   # 从已有报告重建全部数字
python experiments/run_all.py --mode quick     # 冒烟，分钟级
python experiments/run_all.py --mode full      # 从零重跑全部阶段
python paper/build_tables.py                   # 生成英文版表格与宏
python paper/build_paper_zh.py                 # 生成本审阅稿，随后用 xelatex 编译
\end{verbatim}
每个阶段都是独立进程并写自己的 JSON。失败的阶段在汇总里标 \texttt{FAILED}，
其数字渲染为 \texttt{n/a}，绝不用 0 冒充。本文件里的每个数字都来自
\texttt{runs/summary.json}。

\section{结论}
一条秩块槽位的旁路把三件事做得很好，一件事只做到一半。它写入时不打扰邻居，
读取时不花上下文，遗忘可以逐位验证，跨进程重启也成立。它不能把模型与上一个话题隔开：
记忆无论存在哪里都有粘性。

我们带走的实用规则有四条。读时组合默认只选一个槽。写入判据应当去问模型；
给似然设阈值会量到措辞。容量应当花在持久且可精确擦除的槽上。我们追过的那一个例外
在相反方向上很有教益：教写入保护记忆库能大幅降低基座损伤、略微改善全槽条件，
但没有买回第二个槽。更小的样本曾暗示它买回了，四倍样本否证了它。
一个随样本量缩小的效应本身就是一个结果。
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
