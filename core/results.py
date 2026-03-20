#!/usr/bin/env python3
"""
结果处理和输出模块
负责格式化展示测试结果并保存到文件
"""

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .inference import InferenceResult

console = Console()


def print_run_table(results: List[InferenceResult]):
    """
    打印详细的测试结果表格
    
    Args:
        results: 推理结果列表
    """
    table = Table(title="逐项测试结果", show_lines=True, expand=True)
    table.add_column("模型", style="cyan")
    table.add_column("Prompt", style="magenta")
    table.add_column("num_predict", justify="right")
    table.add_column("Prompt tok/s", justify="right", style="green")
    table.add_column("Gen tok/s", justify="right", style="green bold")
    table.add_column("生成 Tokens", justify="right")
    table.add_column("总耗时 (s)", justify="right")
    table.add_column("GPU% avg/max", justify="right", style="yellow")
    table.add_column("RAM MB avg/max", justify="right")
    table.add_column("状态", style="red")
    
    for r in results:
        table.add_row(
            r.model,
            r.prompt_label,
            str(r.planned_num_predict),
            f"{r.prompt_tokens_per_sec:.1f}" if not r.error else "-",
            f"{r.gen_tokens_per_sec:.1f}" if not r.error else "-",
            str(r.eval_count) if not r.error else "-",
            f"{r.total_duration_ns / 1e9:.2f}" if not r.error else "-",
            f"{r.avg_gpu_percent:.0f}/{r.max_gpu_percent:.0f}" if not r.error else "-",
            f"{r.avg_ram_used_mb:.0f}/{r.max_ram_used_mb:.0f}" if not r.error else "-",
            "OK" if not r.error else f"ERR: {r.error[:28]}",
        )
    console.print(table)


def print_model_summary(results: List[InferenceResult]):
    """
    打印模型汇总统计表格
    
    Args:
        results: 推理结果列表
    """
    # 按模型分组
    grouped: Dict[str, List[InferenceResult]] = {}
    for r in results:
        if not r.error:
            grouped.setdefault(r.model, []).append(r)
    
    table = Table(title="模型汇总", show_lines=True)
    table.add_column("模型", style="cyan")
    table.add_column("测试数", justify="right")
    table.add_column("平均 Gen tok/s", justify="right", style="green bold")
    table.add_column("P95 总耗时 (s)", justify="right")
    table.add_column("平均 GPU%", justify="right", style="yellow")
    table.add_column("平均 RAM MB", justify="right")
    
    for model, rs in grouped.items():
        # 计算 P95 延迟
        total_seconds = sorted([r.total_duration_ns / 1e9 for r in rs])
        p95_idx = min(len(total_seconds) - 1, int(len(total_seconds) * 0.95))
        
        table.add_row(
            model,
            str(len(rs)),
            f"{statistics.mean([r.gen_tokens_per_sec for r in rs]):.1f}",
            f"{total_seconds[p95_idx]:.2f}",
            f"{statistics.mean([r.avg_gpu_percent for r in rs]):.0f}",
            f"{statistics.mean([r.avg_ram_used_mb for r in rs]):.0f}",
        )
    console.print(table)


def save_outputs(results: List[InferenceResult], output_prefix: Path):
    """
    保存测试结果到 CSV 和 JSON 文件
    
    Args:
        results: 推理结果列表
        output_prefix: 输出文件前缀路径
        
    输出文件：
        - {prefix}_timestamp.csv: CSV 格式详细数据
        - {prefix}_timestamp.json: JSON 格式完整报告（含回答内容）
    """
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_prefix.parent / f"{output_prefix.name}_{ts}.csv"
    json_path = output_prefix.parent / f"{output_prefix.name}_{ts}.json"
    
    # CSV 字段（不包含长文本）
    csv_fields = [
        "model",
        "prompt_key",
        "prompt_label",
        "planned_num_predict",
        "prompt_eval_count",
        "eval_count",
        "prompt_tokens_per_sec",
        "gen_tokens_per_sec",
        "total_duration_s",
        "avg_cpu_percent",
        "avg_gpu_percent",
        "avg_ram_used_mb",
        "avg_gpu_temp",
        "error"
    ]
    
    # 写入 CSV
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "model": r.model,
                "prompt_key": r.prompt_key,
                "prompt_label": r.prompt_label,
                "planned_num_predict": r.planned_num_predict,
                "prompt_eval_count": r.prompt_eval_count,
                "eval_count": r.eval_count,
                "prompt_tokens_per_sec": round(r.prompt_tokens_per_sec, 2),
                "gen_tokens_per_sec": round(r.gen_tokens_per_sec, 2),
                "total_duration_s": round(r.total_duration_ns / 1e9, 2),
                "avg_cpu_percent": round(r.avg_cpu_percent, 2),
                "avg_gpu_percent": round(r.avg_gpu_percent, 2),
                "avg_ram_used_mb": round(r.avg_ram_used_mb, 2),
                "avg_gpu_temp": round(r.avg_gpu_temp, 2),
                "error": r.error,
            })
    
    # JSON 包含完整数据（含回答内容）
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "results": []
    }
    
    for r in results:
        result_dict = {
            "model": r.model,
            "prompt_key": r.prompt_key,
            "prompt_label": r.prompt_label,
            "prompt_text": r.prompt_text,
            "planned_num_predict": r.planned_num_predict,
            "response_text": r.response_text,  # 模型回答
            "thinking_content": r.thinking_content,  # 思考过程（如果有）
            "prompt_eval_count": r.prompt_eval_count,
            "prompt_eval_duration_ns": r.prompt_eval_duration_ns,
            "eval_count": r.eval_count,
            "eval_duration_ns": r.eval_duration_ns,
            "total_duration_ns": r.total_duration_ns,
            "load_duration_ns": r.load_duration_ns,
            "prompt_tokens_per_sec": round(r.prompt_tokens_per_sec, 2),
            "gen_tokens_per_sec": round(r.gen_tokens_per_sec, 2),
            "avg_cpu_percent": round(r.avg_cpu_percent, 2),
            "max_cpu_percent": round(r.max_cpu_percent, 2),
            "avg_gpu_percent": round(r.avg_gpu_percent, 2),
            "max_gpu_percent": round(r.max_gpu_percent, 2),
            "avg_ram_used_mb": round(r.avg_ram_used_mb, 2),
            "max_ram_used_mb": round(r.max_ram_used_mb, 2),
            "avg_gpu_temp": round(r.avg_gpu_temp, 2),
            "max_gpu_temp": round(r.max_gpu_temp, 2),
            "avg_cpu_temp": round(r.avg_cpu_temp, 2),
            "max_cpu_temp": round(r.max_cpu_temp, 2),
            "error": r.error
        }
        json_data["results"].append(result_dict)
    
    # 写入 JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    console.print(f"[green]结果已保存:[/green] {csv_path}")
    console.print(f"[green]结果已保存:[/green] {json_path}")
    
    # 打印输出文件说明
    console.print(Panel.fit(
        "[dim]CSV 文件：包含性能指标数据，适合 Excel 分析[/dim]\n"
        "[dim]JSON 文件：包含完整数据（含模型回答），适合详细分析[/dim]",
        border_style="green"
    ))
