#!/usr/bin/env python3
"""
Jetson LLM Benchmarking - 主程序

专为 NVIDIA Jetson 边缘设备打造的 LLM 推理性能基准测试工具

功能特性:
- 交互式模型选择，支持编号和模型名
- 自动拉取缺失的 Ollama 模型
- 支持默认/自定义 prompt JSON
- 基于内存推荐 token 长度（num_predict）
- 采集推理吞吐（tok/s）与系统指标（CPU/GPU/RAM/温度）
- 输出 CSV + JSON 报告（JSON 包含完整回答内容）
- 计算模型汇总指标（含 P95 延迟）

使用方法:
    交互模式：python3 benchmark.py
    非交互模式：python3 benchmark.py --non-interactive --models qwen2.5:7b --prompt-file ./prompts/default_prompts.json

GitHub: https://github.com/Zhang-zu-hao/JetsonLLMBenchmarking
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn
)

# 导入核心模块
from core.models import OllamaClient
from core.monitor import TegrastatsMonitor
from core.inference import benchmark_single, PromptCase
from core.results import print_run_table, print_model_summary, save_outputs

# 导入工具模块
from utils.prompts import PromptManager, load_prompt_cases
from utils.cli import (
    get_interactive_models,
    choose_prompt_file,
    ask_num_predict_override,
    print_welcome_panel,
    parse_models_input
)

# 常量定义
console = Console()
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS_PATH = PROJECT_DIR / "prompts" / "default_prompts.json"
DEFAULT_OUTPUT_PREFIX = PROJECT_DIR / "results" / "jetson_llm_benchmark"


def maybe_override_num_predict(cases: list, override: int) -> list:
    """
    根据用户指定的值覆盖 Prompt 的 num_predict 配置
    
    Args:
        cases: PromptCase 列表
        override: 用户指定的 num_predict 值，None 表示不覆盖
        
    Returns:
        更新后的 PromptCase 列表
    """
    if override is None:
        return cases
    
    updated = []
    for c in cases:
        updated.append(
            PromptCase(
                key=c.key,
                label=c.label,
                prompt=c.prompt,
                num_predict=override
            )
        )
    return updated


def benchmark(
    models: list,
    prompts: list,
    rounds: int,
    warmup: bool,
    interval_ms: int
) -> list:
    """
    执行完整的基准测试流程
    
    Args:
        models: 模型列表
        prompts: Prompt 列表
        rounds: 测试轮数
        warmup: 是否预热模型
        interval_ms: 系统监控采样间隔（毫秒）
        
    Returns:
        InferenceResult 结果列表
    """
    # 初始化监控器和客户端
    monitor = TegrastatsMonitor(interval_ms=interval_ms)
    client = OllamaClient()
    
    total_tasks = len(models) * len(prompts) * rounds
    all_results = []
    
    # 检查 tegrastats 可用性
    if not monitor.enabled:
        console.print("[yellow]提示：当前环境未检测到 tegrastats，将仅采集进程级指标。[/yellow]")
    
    # 进度条
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("基准测试进行中", total=total_tasks)
        
        for model in models:
            # 预热模型
            if warmup:
                progress.update(task, description=f"预热 {model}")
                client.warmup(model)
                import time
                time.sleep(1)
            
            for round_idx in range(1, rounds + 1):
                for p in prompts:
                    progress.update(
                        task,
                        description=f"{model} · {p.label} · R{round_idx}"
                    )
                    
                    # 执行单次测试
                    result = benchmark_single(
                        model=model,
                        prompt_case=p,
                        monitor=monitor,
                        client=client
                    )
                    
                    all_results.append(result)
                    
                    # 实时显示结果
                    if result.error:
                        console.print(
                            f"  [red]✗ {model} · {p.label} | {result.error[:72]}[/red]"
                        )
                    else:
                        console.print(
                            f"  [green]✓[/green] {model} · {p.label}"
                            f" | Gen {result.gen_tokens_per_sec:.1f} tok/s"
                            f" | {result.eval_count} tokens"
                            f" | GPU {result.avg_gpu_percent:.0f}%"
                            f" | RAM {result.avg_ram_used_mb:.0f}MB"
                        )
                    
                    progress.advance(task)
    
    return all_results


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="Jetson LLM Benchmarking - 专为 NVIDIA Jetson 边缘设备打造的 LLM 推理性能基准测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  交互模式：
    python3 benchmark.py
  
  非交互模式:
    python3 benchmark.py --non-interactive --models qwen2.5:7b llama3.1:8b --prompt-file ./prompts/default_prompts.json --num-predict 256 --rounds 1
  
  查看 Prompt 格式:
    python3 benchmark.py --show-prompt-format
        """
    )
    
    parser.add_argument(
        "-m", "--models",
        nargs="*",
        help="模型列表；未提供时进入交互输入"
    )
    parser.add_argument(
        "-p", "--prompt-file",
        help="prompt JSON 文件路径"
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=None,
        help="全局覆盖 num_predict"
    )
    parser.add_argument(
        "-r", "--rounds",
        type=int,
        default=1,
        help="重复轮数，默认 1"
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="跳过预热"
    )
    parser.add_argument(
        "--tegrastats-interval",
        type=int,
        default=500,
        help="tegrastats 采样间隔 (ms)"
    )
    parser.add_argument(
        "-o", "--output",
        default=str(DEFAULT_OUTPUT_PREFIX),
        help="输出文件前缀"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="禁用交互，缺少参数则用默认值"
    )
    parser.add_argument(
        "--show-prompt-format",
        action="store_true",
        help="打印默认 prompt JSON 格式示例并退出"
    )
    
    return parser


def main():
    """主函数入口"""
    args = build_parser().parse_args()
    
    # 显示 Prompt 格式
    if args.show_prompt_format:
        prompt_manager = PromptManager(DEFAULT_PROMPTS_PATH)
        prompt_manager.show_format()
        return
    
    # 初始化客户端并检查连接
    client = OllamaClient()
    client.check_connection()
    
    # 打印欢迎信息
    print_welcome_panel()
    
    # 获取可用模型
    available = client.get_available_models()
    if not available:
        raise RuntimeError("未检测到任何 Ollama 模型，请先 `ollama pull <model>`")
    
    console.print(f"\n检测到 Ollama 模型：[cyan]{', '.join(available)}[/cyan]")
    
    # 确定是否交互模式
    interactive = not args.non_interactive and not args.models
    
    # 获取模型列表
    if args.models:
        models = args.models
    else:
        models = get_interactive_models(available)
    
    # 检查并拉取缺失的模型
    missing_models = [m for m in models if m not in available]
    if missing_models:
        console.print(f"\n[yellow]发现缺失模型：{missing_models}[/yellow]")
        console.print("[yellow]开始自动拉取...[/yellow]\n")
        
        for model in missing_models:
            success = client.pull_model(model)
            if success:
                available.append(model)
            else:
                console.print(f"[red]模型 {model} 拉取失败，已跳过[/red]")
                models.remove(model)
    
    if not models:
        raise RuntimeError("可测试模型为空")
    
    # 选择 Prompt 文件
    prompt_file = choose_prompt_file(
        interactive=interactive,
        cli_prompt_path=args.prompt_file
    )
    
    # 加载 Prompt
    prompt_cases = load_prompt_cases(prompt_file)
    
    # 询问 num_predict 覆盖
    num_predict_override = ask_num_predict_override(
        interactive=interactive,
        cli_num_predict=args.num_predict
    )
    
    # 应用覆盖
    prompt_cases = maybe_override_num_predict(prompt_cases, num_predict_override)
    
    # 计算总任务数
    total_tasks = len(models) * len(prompt_cases) * max(1, args.rounds)
    
    # 打印测试计划
    console.print(f"\n使用模型：[cyan]{models}[/cyan]")
    console.print(f"Prompt 文件：[cyan]{prompt_file}[/cyan]")
    console.print(f"Prompt 数量：[cyan]{len(prompt_cases)}[/cyan]")
    console.print(f"轮数：[cyan]{args.rounds}[/cyan]")
    console.print(f"Warmup: [cyan]{'否' if args.no_warmup else '是'}[/cyan]\n")
    console.print(
        f"测试计划：[yellow]{len(models)}[/yellow] 个模型 × "
        f"[yellow]{len(prompt_cases)}[/yellow] 类提示词 × "
        f"[yellow]{max(1, args.rounds)}[/yellow] 轮 = "
        f"[bold yellow]{total_tasks}[/bold yellow] 次推理\n"
    )
    
    # 执行基准测试
    results = benchmark(
        models=models,
        prompts=prompt_cases,
        rounds=max(1, args.rounds),
        warmup=not args.no_warmup,
        interval_ms=max(100, args.tegrastats_interval),
    )
    
    # 打印结果表格
    print_run_table(results)
    console.print()
    
    # 打印模型汇总
    print_model_summary(results)
    
    # 保存结果
    save_outputs(results, Path(args.output))
    
    # 完成提示
    console.print(Panel.fit("[bold green]测试完成[/bold green]", border_style="green"))


if __name__ == "__main__":
    main()
