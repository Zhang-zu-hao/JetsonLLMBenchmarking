#!/usr/bin/env python3
"""
Ollama 客户端模块
负责与 Ollama API 交互，包括模型列表获取、模型拉取和推理请求
"""

import requests
from typing import List, Dict, Any, Optional
from pathlib import Path

OLLAMA_BASE = "http://localhost:11434"


class OllamaClient:
    """Ollama API 客户端，提供模型管理和推理功能"""
    
    def __init__(self, base_url: str = OLLAMA_BASE):
        """
        初始化 Ollama 客户端
        
        Args:
            base_url: Ollama API 基础 URL，默认为本地服务
        """
        self.base_url = base_url
    
    def check_connection(self) -> None:
        """
        检查 Ollama 服务是否可用
        
        Raises:
            RuntimeError: 如果无法连接到 Ollama 服务
        """
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=5).raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"无法连接 Ollama ({self.base_url})，请先启动服务：`ollama serve`。原始错误：{exc}"
            ) from exc
    
    def get_available_models(self) -> List[str]:
        """
        获取本地已下载的模型列表
        
        Returns:
            模型名称列表
            
        Raises:
            requests.RequestException: API 请求失败时
        """
        resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    
    def pull_model(self, model_name: str) -> bool:
        """
        从 Ollama 拉取模型
        
        Args:
            model_name: 要拉取的模型名称
            
        Returns:
            拉取是否成功
        """
        try:
            console = None
            try:
                from rich.console import Console
                console = Console()
            except ImportError:
                pass
            
            if console:
                console.print(f"[yellow]正在拉取模型：{model_name}...[/yellow]")
            
            # 使用流式 API 拉取模型
            resp = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=3600
            )
            resp.raise_for_status()
            
            # 处理流式响应
            for line in resp.iter_lines():
                if line:
                    data = line.decode('utf-8')
                    try:
                        status = eval(data)  # 简单解析 JSON
                        if console and 'status' in status:
                            if 'completed' in status and 'total' in status:
                                progress = status['completed'] / status['total'] * 100
                                console.print(f"\r  下载进度：{progress:.1f}%", end="")
                            elif 'status' in status:
                                console.print(f"\r  {status['status']}", end="")
                    except:
                        pass
            
            if console:
                console.print()  # 换行
                console.print(f"[green]✓ 模型 {model_name} 拉取成功[/green]")
            
            return True
            
        except Exception as e:
            if console:
                console.print(f"[red]✗ 模型拉取失败：{e}[/red]")
            return False
    
    def generate(
        self,
        model: str,
        prompt: str,
        num_predict: int = 256,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        执行推理生成请求
        
        Args:
            model: 模型名称
            prompt: 输入提示词
            num_predict: 最大生成 token 数
            temperature: 温度参数，控制随机性
            stream: 是否使用流式输出
            
        Returns:
            Ollama API 响应字典，包含：
            - response: 生成的文本
            - prompt_eval_count: 输入 token 数
            - eval_count: 输出 token 数
            - prompt_eval_duration: 输入处理时间 (ns)
            - eval_duration: 输出生成时间 (ns)
            - total_duration: 总耗时 (ns)
            - load_duration: 模型加载时间 (ns)
            
        Raises:
            requests.RequestException: API 请求失败
        """
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "num_predict": num_predict,
                    "temperature": temperature
                }
            },
            timeout=900
        )
        resp.raise_for_status()
        return resp.json()
    
    def warmup(self, model: str) -> None:
        """
        预热模型，减少首次推理延迟
        
        Args:
            model: 要预热的模型名称
        """
        try:
            self.generate(
                model=model,
                prompt="hi",
                num_predict=1,
                stream=False
            )
        except Exception:
            # 预热失败不影响主流程
            pass
