#!/usr/bin/env python3
"""
Jetson LLM Benchmarking
- 交互式选择模型（支持多分隔符）
- 支持默认/自定义 prompt JSON
- 支持基于内存推荐 token 长度（num_predict）或用户自定义
"""

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()
OLLAMA_BASE = "http://localhost:11434"
GITHUB_CONTACT_URL = "https://github.com/Zhang-zu-hao"
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS_PATH = PROJECT_DIR / "prompts" / "default_prompts.json"
DEFAULT_OUTPUT_PREFIX = PROJECT_DIR / "results" / "jetson_llm_benchmark"


@dataclass
class PromptCase:
    key: str
    label: str
    prompt: str
    num_predict: int


@dataclass
class SystemSnapshot:
    timestamp: float = 0.0
    cpu_percent: List[float] = field(default_factory=list)
    ram_used_mb: int = 0
    ram_total_mb: int = 0
    swap_used_mb: int = 0
    swap_total_mb: int = 0
    gpu_freq_percent: int = 0
    gpu_temp: float = 0.0
    cpu_temp: float = 0.0


@dataclass
class InferenceResult:
    model: str
    prompt_key: str
    prompt_label: str
    prompt_text: str
    planned_num_predict: int
    response_text: str = ""
    prompt_eval_count: int = 0
    prompt_eval_duration_ns: int = 0
    eval_count: int = 0
    eval_duration_ns: int = 0
    total_duration_ns: int = 0
    load_duration_ns: int = 0
    prompt_tokens_per_sec: float = 0.0
    gen_tokens_per_sec: float = 0.0
    avg_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    avg_gpu_percent: float = 0.0
    max_gpu_percent: float = 0.0
    avg_ram_used_mb: float = 0.0
    max_ram_used_mb: float = 0.0
    avg_gpu_temp: float = 0.0
    max_gpu_temp: float = 0.0
    avg_cpu_temp: float = 0.0
    max_cpu_temp: float = 0.0
    error: str = ""


class TegrastatsMonitor:
    """后台采样 tegrastats；若系统无 tegrastats，将优雅降级。"""

    def __init__(self, interval_ms: int = 500):
        self.interval_ms = interval_ms
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.snapshots: List[SystemSnapshot] = []
        self.enabled = self._check_tegrastats()

    @staticmethod
    def _check_tegrastats() -> bool:
        try:
            proc = subprocess.run(
                ["which", "tegrastats"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return proc.returncode == 0
        except Exception:
            return False

    def start(self):
        self._stop.clear()
        self.snapshots.clear()
        if not self.enabled:
            return

        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self) -> List[SystemSnapshot]:
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            self._proc.wait()
        if self._thread:
            self._thread.join(timeout=3)
        return list(self.snapshots)

    def _reader(self):
        while not self._stop.is_set() and self._proc and self._proc.stdout:
            line = self._proc.stdout.readline()
            if not line:
                break
            snap = self._parse_line(line.strip())
            if snap:
                with self._lock:
                    self.snapshots.append(snap)

    @staticmethod
    def _parse_line(line: str) -> Optional[SystemSnapshot]:
        s = SystemSnapshot(timestamp=time.time())
        try:
            ram = re.search(r"RAM (\d+)/(\d+)MB", line)
            if ram:
                s.ram_used_mb = int(ram.group(1))
                s.ram_total_mb = int(ram.group(2))

            swap = re.search(r"SWAP (\d+)/(\d+)MB", line)
            if swap:
                s.swap_used_mb = int(swap.group(1))
                s.swap_total_mb = int(swap.group(2))

            cpus = re.findall(r"(\d+)%@\d+", line)
            if cpus:
                s.cpu_percent = [float(c) for c in cpus]

            gpu = re.search(r"GR3D_FREQ (\d+)%", line)
            if gpu:
                s.gpu_freq_percent = int(gpu.group(1))

            gpu_temp = re.search(r"GPU@([\d.]+)C", line)
            if gpu_temp:
                s.gpu_temp = float(gpu_temp.group(1))

            cpu_temp = re.search(r"CPU@([\d.]+)C", line)
            if cpu_temp:
                s.cpu_temp = float(cpu_temp.group(1))
            return s
        except Exception:
            return None


def parse_models_input(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"[,\s;|，、]+", text.strip()) if p.strip()]
    return parts


def load_prompt_cases(prompt_file: Path) -> List[PromptCase]:
    with open(prompt_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("prompt JSON 顶层必须是数组")

    cases: List[PromptCase] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"第 {idx + 1} 条 prompt 不是对象")
        missing = [k for k in ("key", "label", "prompt") if k not in item]
        if missing:
            raise ValueError(f"第 {idx + 1} 条 prompt 缺少字段: {missing}")

        num_predict = int(item.get("num_predict", 256))
        cases.append(
            PromptCase(
                key=str(item["key"]),
                label=str(item["label"]),
                prompt=str(item["prompt"]),
                num_predict=max(1, num_predict),
            )
        )
    return cases


def get_available_models() -> List[str]:
    resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def check_ollama_ready() -> None:
    try:
        requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5).raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"无法连接 Ollama ({OLLAMA_BASE})，请先启动服务：`ollama serve`。原始错误: {exc}"
        ) from exc


def warmup_model(model: str):
    try:
        requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": "hi", "stream": False, "options": {"num_predict": 1}},
            timeout=180,
        )
    except Exception:
        pass


def recommend_num_predict() -> Tuple[int, str]:
    vm = psutil.virtual_memory()
    total_gb = vm.total / (1024 ** 3)
    avail_gb = vm.available / (1024 ** 3)

    if total_gb <= 8:
        rec = 512
    elif total_gb <= 16:
        rec = 768
    elif total_gb <= 32:
        rec = 1024
    else:
        rec = 1536

    if avail_gb < 2:
        rec = min(rec, 384)
    elif avail_gb < 4:
        rec = min(rec, 512)

    reason = (
        f"内存总量约 {total_gb:.1f}GB，可用约 {avail_gb:.1f}GB，推荐 num_predict={rec}。"
        "如设置过大，可能触发显存/内存压力导致 OOM 或推理中断。"
    )
    return rec, reason


def run_inference(model: str, prompt: str, num_predict: int) -> dict:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": num_predict, "temperature": 0.7},
        },
        timeout=900,
    )
    resp.raise_for_status()
    return resp.json()


def benchmark_single(model: str, prompt_case: PromptCase, monitor: TegrastatsMonitor) -> InferenceResult:
    result = InferenceResult(
        model=model,
        prompt_key=prompt_case.key,
        prompt_label=prompt_case.label,
        prompt_text=prompt_case.prompt,
        planned_num_predict=prompt_case.num_predict,
    )

    monitor.start()
    process_cpu_samples = []
    process_mem_samples_mb = []

    try:
        proc = psutil.Process()
        start_t = time.time()
        data = run_inference(model, prompt_case.prompt, prompt_case.num_predict)
        end_t = time.time()

        # 记录一次最基础的进程指标（补充 tegrastats 不可用场景）
        process_cpu_samples.append(proc.cpu_percent(interval=None))
        process_mem_samples_mb.append(proc.memory_info().rss / (1024 * 1024))

        result.response_text = data.get("response", "")
        result.prompt_eval_count = data.get("prompt_eval_count", 0)
        result.prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)
        result.eval_count = data.get("eval_count", 0)
        result.eval_duration_ns = data.get("eval_duration", 0)
        result.total_duration_ns = data.get("total_duration", int((end_t - start_t) * 1e9))
        result.load_duration_ns = data.get("load_duration", 0)

        if result.prompt_eval_duration_ns > 0:
            result.prompt_tokens_per_sec = result.prompt_eval_count / (result.prompt_eval_duration_ns / 1e9)
        if result.eval_duration_ns > 0:
            result.gen_tokens_per_sec = result.eval_count / (result.eval_duration_ns / 1e9)
    except Exception as exc:
        result.error = str(exc)
    finally:
        snapshots = monitor.stop()

    if snapshots:
        cpu_avgs = [sum(s.cpu_percent) / len(s.cpu_percent) for s in snapshots if s.cpu_percent]
        cpu_maxes = [max(s.cpu_percent) for s in snapshots if s.cpu_percent]
        gpu_pcts = [s.gpu_freq_percent for s in snapshots]
        ram_useds = [s.ram_used_mb for s in snapshots if s.ram_used_mb > 0]
        gpu_temps = [s.gpu_temp for s in snapshots if s.gpu_temp > 0]
        cpu_temps = [s.cpu_temp for s in snapshots if s.cpu_temp > 0]

        if cpu_avgs:
            result.avg_cpu_percent = statistics.mean(cpu_avgs)
            result.max_cpu_percent = max(cpu_maxes)
        if gpu_pcts:
            result.avg_gpu_percent = statistics.mean(gpu_pcts)
            result.max_gpu_percent = max(gpu_pcts)
        if ram_useds:
            result.avg_ram_used_mb = statistics.mean(ram_useds)
            result.max_ram_used_mb = max(ram_useds)
        if gpu_temps:
            result.avg_gpu_temp = statistics.mean(gpu_temps)
            result.max_gpu_temp = max(gpu_temps)
        if cpu_temps:
            result.avg_cpu_temp = statistics.mean(cpu_temps)
            result.max_cpu_temp = max(cpu_temps)

    if process_cpu_samples and result.avg_cpu_percent == 0:
        result.avg_cpu_percent = statistics.mean(process_cpu_samples)
        result.max_cpu_percent = max(process_cpu_samples)
    if process_mem_samples_mb and result.avg_ram_used_mb == 0:
        result.avg_ram_used_mb = statistics.mean(process_mem_samples_mb)
        result.max_ram_used_mb = max(process_mem_samples_mb)

    return result


def print_run_table(results: List[InferenceResult]):
    table = Table(title="逐项测试结果", show_lines=True, expand=True)
    table.add_column("模型", style="cyan")
    table.add_column("Prompt", style="magenta")
    table.add_column("num_predict", justify="right")
    table.add_column("Prompt tok/s", justify="right", style="green")
    table.add_column("Gen tok/s", justify="right", style="green bold")
    table.add_column("生成Tokens", justify="right")
    table.add_column("总耗时(s)", justify="right")
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
    grouped: Dict[str, List[InferenceResult]] = {}
    for r in results:
        if not r.error:
            grouped.setdefault(r.model, []).append(r)

    table = Table(title="模型汇总", show_lines=True)
    table.add_column("模型", style="cyan")
    table.add_column("测试数", justify="right")
    table.add_column("平均 Gen tok/s", justify="right", style="green bold")
    table.add_column("P95 总耗时(s)", justify="right")
    table.add_column("平均 GPU%", justify="right", style="yellow")
    table.add_column("平均 RAM MB", justify="right")

    for model, rs in grouped.items():
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
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_prefix.parent / f"{output_prefix.name}_{ts}.csv"
    json_path = output_prefix.parent / f"{output_prefix.name}_{ts}.json"

    fields = [
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
        "error",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
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
                }
            )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "results": [r.__dict__ for r in results],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    console.print(f"[green]结果已保存:[/green] {csv_path}")
    console.print(f"[green]结果已保存:[/green] {json_path}")


def maybe_override_num_predict(cases: List[PromptCase], override: Optional[int]) -> List[PromptCase]:
    if override is None:
        return cases
    updated: List[PromptCase] = []
    for c in cases:
        updated.append(PromptCase(key=c.key, label=c.label, prompt=c.prompt, num_predict=override))
    return updated


def get_interactive_models(available: List[str]) -> List[str]:
    console.print("\n[bold]本地已下载模型（Ollama）:[/bold]")
    console.print(f"[cyan]{', '.join(available)}[/cyan]")
    console.print("[dim]可用以下分隔符输入多个模型：空格 / 逗号 / 分号 / | / 中文逗号[/dim]")
    for idx, m in enumerate(available, start=1):
        console.print(f"  {idx:>2}. {m}")
    raw = input("\n请输入要测试的模型（支持空格/逗号/分号/| 分隔，按输入顺序执行）: ").strip()
    selected = parse_models_input(raw)
    if not selected:
        raise ValueError("未输入任何模型")
    return selected


def choose_prompt_file(interactive: bool, cli_prompt_path: Optional[str]) -> Path:
    if cli_prompt_path:
        return Path(cli_prompt_path).expanduser().resolve()
    if not interactive:
        return DEFAULT_PROMPTS_PATH

    console.print("\nPrompt 文件选择：")
    console.print("  1) 使用默认 prompts/default_prompts.json")
    console.print("  2) 自定义 JSON 文件路径")
    mode = input("请选择 [1/2] (默认1): ").strip() or "1"
    if mode == "2":
        p = input("请输入自定义 JSON 路径: ").strip()
        if not p:
            raise ValueError("未输入 JSON 路径")
        return Path(p).expanduser().resolve()
    return DEFAULT_PROMPTS_PATH


def ask_num_predict_override(interactive: bool, cli_num_predict: Optional[int]) -> Optional[int]:
    if cli_num_predict is not None:
        return max(1, cli_num_predict)
    rec, reason = recommend_num_predict()
    console.print(f"\n[cyan]Token 长度建议：[/cyan]{reason}")
    if not interactive:
        return None

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


def benchmark(models: List[str], prompts: List[PromptCase], rounds: int, warmup: bool, interval_ms: int) -> List[InferenceResult]:
    monitor = TegrastatsMonitor(interval_ms=interval_ms)
    total_tasks = len(models) * len(prompts) * rounds
    all_results: List[InferenceResult] = []

    if not monitor.enabled:
        console.print("[yellow]提示：当前环境未检测到 tegrastats，将仅采集部分进程级指标。[/yellow]")

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
            if warmup:
                progress.update(task, description=f"预热 {model}")
                warmup_model(model)
                time.sleep(1)

            for round_idx in range(1, rounds + 1):
                for p in prompts:
                    progress.update(task, description=f"{model} · {p.label} · R{round_idx}")
                    result = benchmark_single(model, p, monitor)
                    all_results.append(result)
                    if result.error:
                        console.print(f"  [red]✗ {model} · {p.label} | {result.error[:72]}[/red]")
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
    parser = argparse.ArgumentParser(description="Jetson LLM Benchmarking")
    parser.add_argument("-m", "--models", nargs="*", help="模型列表；未提供时进入交互输入")
    parser.add_argument("-p", "--prompt-file", help="prompt JSON 文件路径")
    parser.add_argument("--num-predict", type=int, default=None, help="全局覆盖 num_predict")
    parser.add_argument("-r", "--rounds", type=int, default=1, help="重复轮数，默认 1")
    parser.add_argument("--no-warmup", action="store_true", help="跳过预热")
    parser.add_argument("--tegrastats-interval", type=int, default=500, help="tegrastats 采样间隔(ms)")
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT_PREFIX), help="输出前缀")
    parser.add_argument("--non-interactive", action="store_true", help="禁用交互，缺少参数则用默认值")
    parser.add_argument("--show-prompt-format", action="store_true", help="打印默认 prompt JSON 格式示例并退出")
    return parser


def print_prompt_format():
    console.print(
        Panel.fit(
            "默认 prompt JSON 是数组，每项包含：\n"
            "- key: 唯一标识\n"
            "- label: 显示名\n"
            "- prompt: 提示词正文\n"
            "- num_predict: 生成 token 上限\n\n"
            "示例：\n"
            "[\n"
            '  {"key":"short_qa","label":"短问答","prompt":"什么是量子计算？一句话回答。","num_predict":128},\n'
            '  {"key":"reasoning","label":"逻辑推理","prompt":"...","num_predict":512}\n'
            "]",
            title="Prompt JSON 格式",
            border_style="cyan",
        )
    )
    console.print(f"默认文件路径：{DEFAULT_PROMPTS_PATH}")


def main():
    args = build_parser().parse_args()

    if args.show_prompt_format:
        print_prompt_format()
        return

    check_ollama_ready()

    console.print(
        Panel.fit(
            "[bold cyan]Jetson LLM Benchmarking[/bold cyan]\n"
            "[dim]交互式模型选择 · 自定义 prompts · token 推荐[/dim]\n"
            f"[dim]遇到问题欢迎到 GitHub 反馈：{GITHUB_CONTACT_URL}[/dim]\n"
            "[dim]欢迎一起建设这个项目[/dim]",
            border_style="cyan",
        )
    )

    interactive = not args.non_interactive and not args.models
    available = get_available_models()
    if not available:
        raise RuntimeError("未检测到任何 Ollama 模型，请先 `ollama pull <model>`")
    console.print(f"\n检测到 Ollama 模型: [cyan]{', '.join(available)}[/cyan]")

    models = args.models if args.models else get_interactive_models(available)
    missing = [m for m in models if m not in available]
    if missing:
        console.print(f"[yellow]以下模型本机不可用，将跳过: {missing}[/yellow]")
    models = [m for m in models if m in available]
    if not models:
        raise RuntimeError("可测试模型为空")

    prompt_file = choose_prompt_file(interactive=not args.non_interactive, cli_prompt_path=args.prompt_file)
    if not prompt_file.exists():
        raise FileNotFoundError(f"prompt 文件不存在: {prompt_file}")

    prompt_cases = load_prompt_cases(prompt_file)
    num_predict_override = ask_num_predict_override(interactive=not args.non_interactive, cli_num_predict=args.num_predict)
    prompt_cases = maybe_override_num_predict(prompt_cases, num_predict_override)

    total_tasks = len(models) * len(prompt_cases) * max(1, args.rounds)
    console.print(f"\n使用模型: [cyan]{models}[/cyan]")
    console.print(f"Prompt 文件: [cyan]{prompt_file}[/cyan]")
    console.print(f"Prompt 数量: [cyan]{len(prompt_cases)}[/cyan]")
    console.print(f"轮数: [cyan]{args.rounds}[/cyan]")
    console.print(f"Warmup: [cyan]{'否' if args.no_warmup else '是'}[/cyan]\n")
    console.print(
        f"测试计划: [yellow]{len(models)}[/yellow] 个模型 × "
        f"[yellow]{len(prompt_cases)}[/yellow] 类提示词 × "
        f"[yellow]{max(1, args.rounds)}[/yellow] 轮 = "
        f"[bold yellow]{total_tasks}[/bold yellow] 次推理\n"
    )

    results = benchmark(
        models=models,
        prompts=prompt_cases,
        rounds=max(1, args.rounds),
        warmup=not args.no_warmup,
        interval_ms=max(100, args.tegrastats_interval),
    )

    print_run_table(results)
    console.print()
    print_model_summary(results)
    save_outputs(results, Path(args.output))
    console.print(Panel.fit("[bold green]测试完成[/bold green]", border_style="green"))


if __name__ == "__main__":
    main()
