#!/usr/bin/env python3
"""
命令行交互模块
负责与用户交互，包括模型选择、Prompt 文件选择和参数配置
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel

console = Console()

GITHUB_CONTACT_URL = "https://github.com/Zhang-zu-hao/JetsonLLMBenchmarking"


def print_welcome_panel():
    """打印欢迎信息面板"""
    console.print(
        Panel.fit(
            "[bold cyan]Jetson LLM Benchmarking[/bold cyan]\n"
            "[dim]交互式模型选择 · 自定义 prompts · token 推荐[/dim]\n"
            f"[dim]欢迎大佬指点：{GITHUB_CONTACT_URL}[/dim]",
            border_style="cyan",
        )
    )


def parse_models_input(text: str) -> List[str]:
    """
    解析用户输入的模型字符串
    
    支持的分隔符：空格、逗号、分号、|、中文逗号、顿号
    
    Args:
        text: 用户输入的字符串
        
    Returns:
        模型名称列表
    """
    parts = [p.strip() for p in re.split(r"[,\s;|，、]+", text.strip()) if p.strip()]
    return parts


def get_interactive_models(available: List[str]) -> List[str]:
    """
    交互式选择模型
    
    支持两种输入方式：
    1. 直接输入模型名称（支持多个，用分隔符分开）
    2. 输入编号（支持多个，用分隔符分开）
    
    Args:
        available: 本地已下载的模型列表
        
    Returns:
        选中的模型列表
        
    Raises:
        ValueError: 未选择任何模型
    """
    console.print("\n[bold]本地已下载模型（Ollama）:[/bold]")
    console.print(f"[cyan]{', '.join(available)}[/cyan]")
    console.print("[dim]可用以下分隔符输入多个模型：空格 / 逗号 / 分号 / | / 中文逗号[/dim]")
    
    # 显示编号列表
    for idx, m in enumerate(available, start=1):
        console.print(f"  {idx:>2}. {m}")
    
    # 获取用户输入
    raw = input("\n请输入要测试的模型（支持编号或直接输入模型名，多个用分隔符分开）: ").strip()
    
    if not raw:
        raise ValueError("未输入任何模型")
    
    # 解析输入
    selected = parse_models_input(raw)
    resolved_models = []
    
    for item in selected:
        # 尝试解析为编号
        if item.isdigit():
            idx = int(item) - 1
            if 0 <= idx < len(available):
                resolved_models.append(available[idx])
            else:
                console.print(f"[yellow]警告：编号 {idx + 1} 超出范围，已跳过[/yellow]")
        else:
            # 直接作为模型名
            resolved_models.append(item)
    
    if not resolved_models:
        raise ValueError("未选择任何有效模型")
    
    return resolved_models


def choose_prompt_file(interactive: bool, cli_prompt_path: Optional[str]) -> Path:
    """
    选择 Prompt 文件
    
    Args:
        interactive: 是否进入交互模式
        cli_prompt_path: 命令行指定的路径
        
    Returns:
        Prompt 文件路径
        
    Raises:
        ValueError: 未提供有效路径
    """
    from utils.prompts import DEFAULT_PROMPTS_PATH
    
    if cli_prompt_path:
        return Path(cli_prompt_path).expanduser().resolve()
    
    if not interactive:
        return DEFAULT_PROMPTS_PATH
    
    console.print("\nPrompt 文件选择：")
    console.print("  1) 使用默认 prompts/default_prompts.json")
    console.print("  2) 自定义 JSON 文件路径")
    
    mode = input("请选择 [1/2] (默认 1): ").strip() or "1"
    
    if mode == "2":
        p = input("请输入自定义 JSON 路径：").strip()
        if not p:
            raise ValueError("未输入 JSON 路径")
        return Path(p).expanduser().resolve()
    
    return DEFAULT_PROMPTS_PATH


def recommend_num_predict() -> Tuple[int, str]:
    """
    根据系统内存推荐 num_predict 值
    
    Returns:
        (推荐值，推荐理由)
    """
    import psutil
    
    vm = psutil.virtual_memory()
    total_gb = vm.total / (1024 ** 3)
    avail_gb = vm.available / (1024 ** 3)
    
    # 根据总内存确定基础推荐值
    if total_gb <= 8:
        rec = 512
    elif total_gb <= 16:
        rec = 768
    elif total_gb <= 32:
        rec = 1024
    else:
        rec = 1536
    
    # 根据可用内存调整
    if avail_gb < 2:
        rec = min(rec, 384)
    elif avail_gb < 4:
        rec = min(rec, 512)
    
    reason = (
        f"内存总量约 {total_gb:.1f}GB，可用约 {avail_gb:.1f}GB，推荐 num_predict={rec}。"
        "如设置过大，可能触发显存/内存压力导致 OOM 或推理中断。"
    )
    
    return rec, reason


def ask_num_predict_override(interactive: bool, cli_num_predict: Optional[int]) -> Optional[int]:
    """
    询问用户是否要覆盖 num_predict 配置
    
    Args:
        interactive: 是否进入交互模式
        cli_num_predict: 命令行指定的值
        
    Returns:
        用户指定的 num_predict 值，None 表示使用 JSON 中的配置
    """
    # 命令行已指定
    if cli_num_predict is not None:
        return max(1, cli_num_predict)
    
    # 获取推荐值
    rec, reason = recommend_num_predict()
    console.print(f"\n[cyan]Token 长度建议：[/cyan]{reason}")
    
    if not interactive:
        return None
    
    # 交互模式询问用户
    raw = input(
        "请输入全局 num_predict（回车用推荐值，输入 0 按 JSON 各条配置；"
        "可设大一点，但过大可能爆内存）: "
    ).strip()
    
    if not raw:
        return rec
    
    try:
        val = int(raw)
    except ValueError as exc:
        raise ValueError("num_predict 必须是整数") from exc
    
    if val == 0:
        return None
    
    if val < 0:
        raise ValueError("num_predict 不能为负数")
    
    return val
