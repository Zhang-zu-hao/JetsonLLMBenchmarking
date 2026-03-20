# JetsonLLMBenchmarking

基于 Ollama 的 Jetson LLM 基准测试项目，支持交互式模型选择、自定义 Prompt JSON、按设备内存推荐 token 长度（`num_predict`）。

## 功能

- 交互输入要测试的模型，支持 `空格 / 逗号 / 分号 / | / 中文逗号` 分隔，按输入顺序执行。
- 交互时会先显示本地已下载的 Ollama 模型列表（与参考脚本一致的体验）。
- 支持默认 Prompt 集（`prompts/default_prompts.json`）或自定义 JSON 文件。
- 根据设备内存自动给出偏积极的 `num_predict` 推荐值，也支持用户自定义全局覆盖（会提示过大可能爆内存）。
- 采集推理吞吐（tok/s）与系统指标（CPU/GPU/RAM/温度，若可用）。
- 输出 CSV + JSON 报告，便于后续分析。
- 计算模型汇总指标（含 P95 延迟），方便横向比较。

## 环境要求

- Linux（Jetson 推荐）
- Python 3.8+
- 本地已安装并运行 Ollama（默认地址 `http://localhost:11434`）

安装依赖：

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

- `jetson_llm_benchmark_时间戳.csv`
- `jetson_llm_benchmark_时间戳.json`

可通过 `-o/--output` 自定义前缀，例如：

```bash
python3 benchmark.py -o results/my_bench
```

## 说明

- 如果系统未检测到 `tegrastats`，脚本会自动降级到部分进程级指标，不影响主流程。
- 若连接不到 Ollama，请先执行 `ollama serve`。
- 项目启动会提示 GitHub 反馈地址：`https://github.com/Zhang-zu-hao`，欢迎提 Issue 或一起共建。
