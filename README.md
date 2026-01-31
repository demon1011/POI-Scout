# POI-Scout

> "周末想带孩子去个好玩的地方" —— 只需一句话，POI-Scout 帮你搜遍全网、筛选出最合适的目的地。

智能旅行目的地搜索 Agent，支持模糊需求理解、自我优化搜索和交互式筛选。

<table>
<tr>
<td width="50%">

**主界面**：创建新任务或加载已有任务
![主界面](docs/images/main_interface.png)

</td>
<td width="50%">

**筛选界面**：通过问答快速收敛到理想 POI
![交互式筛选界面](docs/images/select_interface.png)

</td>
</tr>
</table>

## 核心特性

| 特性 | 说明 |
|-----|------|
| **多步骤搜索规划** | 自动将模糊需求分解为多个搜索主题，最大化覆盖潜在需求 |
| **在线优化 (On-policy)** | 基于 [TextGrad](https://arxiv.org/abs/2406.07496) 的 Reflection 机制，实时迭代改进搜索方案 |
| **离线经验复用 (Off-policy)** | 借鉴 [GEPA](https://arxiv.org/abs/2507.19457) 思路，从历史任务提取 skills 用于新任务 |
| **贝叶斯决策树筛选** | 参考[贝叶斯实验方法](https://arxiv.org/abs/2508.21184)，通过多轮问答快速收敛到理想 POI |

> **关于 On/Off-policy 术语**：这里的"策略"指 Agent 针对特定请求生成的搜索方案。On-policy 用当前方案的反馈改进同一方案；Off-policy 用其他任务的经验辅助当前任务。

## 快速开始

```bash
# 克隆并安装
git clone https://github.com/demon1011/POI-Scout.git
cd POI-Scout
pip install -r requirements.txt
playwright install chromium

# 配置 API 密钥
cp config.example.py config.py
# 编辑 config.py，填入 SiliconFlow 和博查搜索的 API 密钥

# 运行
python main.py                          # 基础搜索
python main.py --online-opt             # 启用在线优化
python main.py --use-skill              # 使用历史经验
python main.py --online-opt --use-skill --create-skill  # 完整模式
```

**API 依赖**：[SiliconFlow](https://siliconflow.cn)（LLM + Embedding）、[博查搜索](https://bochaai.com)（网页搜索）

## 技术方案

### 在线优化（On-policy）

搜索工具对模型来说是黑盒——模型无法预知什么样的搜索请求能获得更多有效 POI。为此，我们设计了基于 Reflection 的迭代优化机制：

1. **执行与评估**：Agent 按初始方案执行搜索，记录每个步骤的 POI 召回数量和质量
2. **问题定位**：通过 Reflection 让模型分析哪些步骤效率较低，识别搜索请求的不足之处
3. **方案改进**：针对低效步骤生成具体的修改建议，调整搜索关键词、扩展搜索角度
4. **迭代执行**：用改进后的方案重新搜索，对比效果，持续优化

为防止优化陷入局部最优，我们引入了**正则化策略**：当某个步骤连续多次优化效果不佳时，触发重新采样机制，从新的方向探索。

![优化过程](docs/images/poi_optimization.png)
*优化曲线示例（请求："金华适合带娃的地方"）*

### 离线经验总结（Off-policy）

在线优化积累了大量"什么改进有效"的隐性知识，如何将其保留供后续任务使用？

我们借鉴强化学习中 **GAE（Generalized Advantage Estimation）** 的思想：通过对比优化前后的方案和效果差异，计算每次改进的"优势"。不同于数值形式的 GAE，我们以**自然语言**让模型总结这些优势，形成可读、可复用的 skill：

- 对比优化前后的搜索方案差异
- 分析 POI 召回数量和质量的变化
- 提炼出通用的经验规则（如"搜索亲子景点时应同时考虑室内和户外场景"）

通过 Embedding 对经验去重和多样性筛选，避免冗余。

**实验结果**（n=15）：使用 skills 后 POI 召回提升 **17.3%**（30.1→35.3），p=0.035，Cohen's d=0.815。

![经验效果对比](docs/images/boxplot_comparison.png)

## 项目结构

```
POI-Scout/
├── main.py              # 入口
├── config.py            # API 密钥配置
├── src/
│   ├── agent/           # ReAct Agent 实现
│   ├── search/          # 搜索与优化逻辑
│   ├── selector/        # 决策树筛选
│   └── tools/           # 爬虫与搜索工具
└── data/                # skills 和决策树存储
```

## License

MIT License
