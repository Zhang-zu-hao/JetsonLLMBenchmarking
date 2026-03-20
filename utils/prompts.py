#!/usr/bin/env python3
"""
Prompt 管理模块
负责加载和验证 Prompt JSON 文件
"""

import json
from pathlib import Path
from typing import List
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel

console = Console()

# 常量定义
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS_PATH = PROJECT_DIR / "prompts" / "default_prompts.json"


@dataclass
class PromptCase:
    """
    Prompt 测试用例
    
    Attributes:
        key: 唯一标识符
        label: 显示名称
        prompt: 提示词正文
        num_predict: 最大生成 token 数
    """
    key: str
    label: str
    prompt: str
    num_predict: int


class PromptManager:
    """Prompt 文件管理器"""
    
    def __init__(self, default_path: Path):
        """
        初始化 Prompt 管理器
        
        Args:
            default_path: 默认 Prompt 文件路径
        """
        self.default_path = default_path
    
    def load(self, prompt_file: Path) -> List[PromptCase]:
        """
        加载 Prompt 文件
        
        Args:
            prompt_file: Prompt JSON 文件路径
            
        Returns:
            PromptCase 列表
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: JSON 格式错误
        """
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt 文件不存在：{prompt_file}")
        
        return load_prompt_cases(prompt_file)
    
    def show_format(self):
        """打印 Prompt JSON 格式说明"""
        console.print(
            Panel.fit(
                "默认 prompt JSON 是数组，每项包含：\n"
                "- key: 唯一标识\n"
                "- label: 显示名\n"
                "- prompt: 提示词正文\n"
                "- num_predict: 生成 token 上限\n\n"
                "示例：\n"
                "[\n"
                '  {"key":"short_qa","label":"短问答","prompt":"什么是量子计算？一句话回答。","num_predict":512},\n'
                '  {"key":"reasoning","label":"逻辑推理","prompt":"...","num_predict":1024}\n'
                "]",
                title="Prompt JSON 格式",
                border_style="cyan",
            )
        )
        console.print(f"默认文件路径：{self.default_path}")


def load_prompt_cases(prompt_file: Path) -> List[PromptCase]:
    """
    加载 Prompt JSON 文件并解析为 PromptCase 列表
    
    Args:
        prompt_file: Prompt JSON 文件路径
        
    Returns:
        PromptCase 列表
        
    Raises:
        ValueError: JSON 格式错误或字段缺失
    """
    with open(prompt_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    # 验证顶层结构
    if not isinstance(raw, list):
        raise ValueError("Prompt JSON 顶层必须是数组")
    
    cases: List[PromptCase] = []
    for idx, item in enumerate(raw):
        # 验证每个元素
        if not isinstance(item, dict):
            raise ValueError(f"第 {idx + 1} 条 prompt 不是对象")
        
        # 检查必需字段
        missing = [k for k in ("key", "label", "prompt") if k not in item]
        if missing:
            raise ValueError(f"第 {idx + 1} 条 prompt 缺少字段：{missing}")
        
        # 提取字段，num_predict 可选，默认 256
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
