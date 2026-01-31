# POI-Scout

> "周末想带孩子去个好玩的地方" —— 只需一句话，POI-Scout 帮你搜遍全网、筛选出最合适的目的地。

智能旅行目的地搜索 Agent，支持模糊需求理解、自我优化搜索和交互式筛选。

![主界面](docs/images/main_interface.png)

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

## 方法与实验

### 在线优化流程

通过 Reflection 定位低效搜索步骤 → 生成改进建议 → 重新执行 → 迭代优化。采用重新采样等正则化手段防止陷入局部最优。

![优化过程](docs/images/poi_optimization.png)
*优化曲线示例（请求："金华适合带娃的地方"）*

### 离线经验总结

对比优化前后效果差异（借鉴 GAE 思想），以自然语言总结"什么样的改进是有效的"，形成可复用 skill。

**实验结果**（n=15）：使用 skills 后 POI 召回提升 **17.3%**（30.1→35.3），p=0.035，Cohen's d=0.815。

![经验效果对比](docs/images/boxplot_comparison.png)

### 交互式筛选

基于贝叶斯决策树的多轮 A/B 选择，快速收敛到用户偏好的 POI。

![交互式筛选界面](docs/images/select_interface.png)

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
