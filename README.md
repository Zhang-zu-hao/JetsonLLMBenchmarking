# JetsonLLMBenchmarking

**专为 NVIDIA Jetson 边缘设备打造的 LLM 推理性能基准测试工具**

基于 Ollama 的 Jetson LLM 基准测试项目，支持交互式模型选择、自定义 Prompt JSON、按设备内存推荐 token 长度（`num_predict`）。

> 💡 **项目定位**：目前市面上针对通用平台的 LLM benchmark 工具较多，但专门面向 Jetson 等边缘设备、集成系统级监控的开源工具相对稀缺。本项目填补了这一空白，为边缘 AI 开发者提供开箱即用的性能评估方案。

## 核心优势

- 🎯 **边缘设备优化**：专为 Jetson 设计，根据设备内存智能推荐推理参数
- 📊 **系统级监控**：集成 CPU/GPU/RAM/温度采集，全面掌握设备状态
- 🚀 **Ollama 原生支持**：无缝对接本地 Ollama 服务，支持多模型批量测试
- 🔧 **灵活配置**：支持交互式/非交互式模式，自定义 Prompt 集
- 📈 **专业报告**：输出 CSV + JSON 格式，含 P95 延迟等汇总指标

## 功能特性

- 交互输入要测试的模型，支持 `空格 / 逗号 / 分号 / | / 中文逗号` 分隔，按输入顺序执行
- 交互时会先显示本地已下载的 Ollama 模型列表
- 支持默认 Prompt 集（`prompts/default_prompts.json`）或自定义 JSON 文件
- 根据设备内存自动给出偏积极的 `num_predict` 推荐值，支持用户自定义全局覆盖（会提示过大可能爆内存）
- 采集推理吞吐（tok/s）与系统指标（CPU/GPU/RAM/温度，若可用）
- 输出 CSV + JSON 报告，便于后续分析
- 计算模型汇总指标（含 P95 延迟），方便横向比较

## 环境要求

- **操作系统**：Linux（Jetson Orin/Nano/Xavier 系列推荐）
- **Python 版本**：3.8+
- **推理框架**：本地已安装并运行 Ollama（默认地址 `http://localhost:11434`）

### 依赖安装

```bash
cd /home/seeed/JetsonLLMBenchmarking
python3 -m pip install -r requirements.txt
```

## 快速开始（交互模式）

```bash
cd /home/seeed/JetsonLLMBenchmarking
python3 benchmark.py
```

交互流程：

1. 输入模型，例如：
   - `qwen2.5:7b llama3.1:8b`
   - `qwen2.5:7b,deepseek-r1:7b`
   - `qwen2.5:7b|llama3.1:8b;phi3`
2. 选择 Prompt 文件：
   - `1` 使用默认 `prompts/default_prompts.json`
   - `2` 输入你自己的 JSON 路径
3. 选择 `num_predict`：
   - 回车：采用系统推荐（偏积极，适合较长推理）
   - 输入正整数：全局覆盖
   - 输入 `0`：使用 JSON 各条 `num_predict`

## 非交互模式

```bash
python3 benchmark.py \
  --non-interactive \
  --models qwen2.5:7b llama3.1:8b \
  --prompt-file ./prompts/default_prompts.json \
  --num-predict 256 \
  --rounds 1
```

## 默认 Prompt JSON 格式

项目默认文件：`prompts/default_prompts.json`

结构要求（顶层数组）：

```json
[
  {
    "key": "short_qa",
    "label": "短问答",
    "prompt": "请用一句话解释什么是量子计算。",
    "num_predict": 512
  }
]
```

字段说明：

- `key`: 唯一标识
- `label`: 展示名称
- `prompt`: 提示词文本
- `num_predict`: 本条测试最大生成 token 数

也可以用下面命令快速查看格式提示：

```bash
python3 benchmark.py --show-prompt-format
```

## 输出文件

默认输出到 `results/` 目录：

- `jetson_llm_benchmark_时间戳.csv` - 详细测试数据（CSV 格式）
- `jetson_llm_benchmark_时间戳.json` - 完整报告（含汇总指标）

可通过 `-o/--output` 自定义前缀，例如：

```bash
python3 benchmark.py -o results/my_bench
```

## 典型应用场景

- 📌 **模型选型评估**：对比不同模型在 Jetson 设备上的推理性能
- 🔍 **性能调优**：通过系统监控数据定位瓶颈，优化配置参数
- 📊 **横向对比**：多轮测试 + P95 延迟统计，科学评估稳定性
- 🎓 **教学演示**：交互式体验，快速展示边缘 LLM 推理能力

## 常见问题

**Q: 为什么需要 `tegrastats`？**  
A: `tegrastats` 是 NVIDIA Jetson 平台的系统监控工具，可提供精确的 GPU/CPU 负载、温度、功耗等数据。如果未检测到，脚本会自动降级到部分进程级指标，不影响主流程。

**Q: 如何查看 Ollama 已下载的模型？**  
A: 运行 `ollama list` 或在交互模式启动时自动显示。

**Q: 推荐值过于积极导致内存不足怎么办？**  
A: 脚本会提示风险，可选择手动输入更保守的 `num_predict` 值。

**Q: 连接不到 Ollama 服务？**  
A: 请先执行 `ollama serve` 启动本地服务。

## 项目说明

- 如果系统未检测到 `tegrastats`，脚本会自动降级到部分进程级指标，不影响主流程
- 若连接不到 Ollama，请先执行 `ollama serve`
- 项目启动会提示 GitHub 反馈地址：`https://github.com/Zhang-zu-hao`，感谢大佬们来指导和反馈

## 相关资源

- [Ollama 官方文档](https://ollama.ai/)
- [NVIDIA Jetson 开发者中心](https://developer.nvidia.com/embedded-computing)
- [LLM Benchmark 工具对比](https://blog.csdn.net/gitblog_00387/article/details/149038534)
