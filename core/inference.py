#!/usr/bin/env python3
"""
推理执行模块
负责执行单次推理并记录结果
"""

import time
import statistics
from typing import Optional
from dataclasses import dataclass, field

from .models import OllamaClient
from .monitor import TegrastatsMonitor


@dataclass
class PromptCase:
    """Prompt 测试用例"""
    key: str
    label: str
    prompt: str
    num_predict: int


@dataclass
class InferenceResult:
    """
    单次推理测试结果
    
    Attributes:
        model: 模型名称
        prompt_key: Prompt 唯一标识
        prompt_label: Prompt 显示名称
        prompt_text: 完整的输入提示词
        planned_num_predict: 计划生成的 token 数
        response_text: 模型生成的回答内容
        thinking_content: 模型的思考过程（如果有）
        prompt_eval_count: 输入 token 数量
        prompt_eval_duration_ns: 输入处理时间（纳秒）
        eval_count: 输出 token 数量
        eval_duration_ns: 输出生成时间（纳秒）
        total_duration_ns: 总耗时（纳秒）
        load_duration_ns: 模型加载时间（纳秒）
        prompt_tokens_per_sec: 输入处理速度（tok/s）
        gen_tokens_per_sec: 生成速度（tok/s）
        avg_cpu_percent: 平均 CPU 使用率
        max_cpu_percent: 最大 CPU 使用率
        avg_gpu_percent: 平均 GPU 使用率
        max_gpu_percent: 最大 GPU 使用率
        avg_ram_used_mb: 平均内存使用（MB）
        max_ram_used_mb: 最大内存使用（MB）
        avg_gpu_temp: 平均 GPU 温度
        max_gpu_temp: 最大 GPU 温度
        avg_cpu_temp: 平均 CPU 温度
        max_cpu_temp: 最大 CPU 温度
        error: 错误信息（如果有）
    """
    model: str
    prompt_key: str
    prompt_label: str
    prompt_text: str
    planned_num_predict: int
    response_text: str = ""
    thinking_content: str = ""
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


def run_inference(
    client: OllamaClient,
    model: str,
    prompt: str,
    num_predict: int,
    temperature: float = 0.7
) -> dict:
    """
    执行单次推理
    
    Args:
        client: Ollama 客户端
        model: 模型名称
        prompt: 输入提示词
        num_predict: 最大生成 token 数
        temperature: 温度参数
        
    Returns:
        Ollama API 响应字典
    """
    return client.generate(
        model=model,
        prompt=prompt,
        num_predict=num_predict,
        temperature=temperature
    )


def benchmark_single(
    model: str,
    prompt_case: PromptCase,
    monitor: TegrastatsMonitor,
    client: Optional[OllamaClient] = None
) -> InferenceResult:
    """
    执行单次基准测试
    
    Args:
        model: 模型名称
        prompt_case: Prompt 测试用例
        monitor: 系统监控器
        client: Ollama 客户端，如果为 None 则创建新的
        
    Returns:
        InferenceResult 包含完整的测试结果
    """
    if client is None:
        client = OllamaClient()
    
    result = InferenceResult(
        model=model,
        prompt_key=prompt_case.key,
        prompt_label=prompt_case.label,
        prompt_text=prompt_case.prompt,
        planned_num_predict=prompt_case.num_predict,
    )
    
    # 启动系统监控
    monitor.start()
    
    # 用于记录进程级指标（tegrastats 不可用时的备选）
    process_cpu_samples = []
    process_mem_samples_mb = []
    
    try:
        proc = psutil.Process()
        start_t = time.time()
        
        # 执行推理
        data = run_inference(
            client=client,
            model=model,
            prompt=prompt_case.prompt,
            num_predict=prompt_case.num_predict
        )
        end_t = time.time()
        
        # 记录进程指标
        process_cpu_samples.append(proc.cpu_percent(interval=None))
        process_mem_samples_mb.append(proc.memory_info().rss / (1024 * 1024))
        
        # 提取响应内容（适配 Ollama 新版本 API）
        # Ollama 新版本可能使用 'response' 或 'thinking' 字段
        result.response_text = data.get("response", "") or data.get("thinking", "")
        result.thinking_content = data.get("thinking", "")
        
        # 提取性能指标
        result.prompt_eval_count = data.get("prompt_eval_count", 0)
        result.prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)
        result.eval_count = data.get("eval_count", 0)
        result.eval_duration_ns = data.get("eval_duration", 0)
        result.total_duration_ns = data.get("total_duration", int((end_t - start_t) * 1e9))
        result.load_duration_ns = data.get("load_duration", 0)
        
        # 计算吞吐量
        if result.prompt_eval_duration_ns > 0:
            result.prompt_tokens_per_sec = result.prompt_eval_count / (result.prompt_eval_duration_ns / 1e9)
        if result.eval_duration_ns > 0:
            result.gen_tokens_per_sec = result.eval_count / (result.eval_duration_ns / 1e9)
            
    except Exception as exc:
        result.error = str(exc)
    finally:
        # 停止监控并获取快照
        snapshots = monitor.stop()
    
    # 处理监控数据
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
    
    # 如果 tegrastats 不可用，使用进程级指标
    if process_cpu_samples and result.avg_cpu_percent == 0:
        result.avg_cpu_percent = statistics.mean(process_cpu_samples)
        result.max_cpu_percent = max(process_cpu_samples)
    if process_mem_samples_mb and result.avg_ram_used_mb == 0:
        result.avg_ram_used_mb = statistics.mean(process_mem_samples_mb)
        result.max_ram_used_mb = max(process_mem_samples_mb)
    
    return result


# 需要在这里导入 psutil，因为在 InferenceResult 的 benchmark_single 中用到
import psutil
