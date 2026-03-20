#!/usr/bin/env python3
"""
系统监控模块
负责采集 GPU/CPU/RAM/温度等系统指标
支持 tegrastats 和进程级指标两种方式
"""

import subprocess
import threading
import time
import re
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

import psutil


@dataclass
class SystemSnapshot:
    """系统状态快照"""
    timestamp: float = 0.0
    cpu_percent: List[float] = field(default_factory=list)
    ram_used_mb: int = 0
    ram_total_mb: int = 0
    swap_used_mb: int = 0
    swap_total_mb: int = 0
    gpu_freq_percent: int = 0
    gpu_temp: float = 0.0
    cpu_temp: float = 0.0


class TegrastatsMonitor:
    """
    后台采样 tegrastats 系统监控工具
    如果系统没有 tegrastats，会自动降级到进程级指标
    """
    
    def __init__(self, interval_ms: int = 500):
        """
        初始化监控器
        
        Args:
            interval_ms: 采样间隔（毫秒），默认 500ms
        """
        self.interval_ms = interval_ms
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.snapshots: List[SystemSnapshot] = []
        self.enabled = self._check_tegrastats()
    
    @staticmethod
    def _check_tegrastats() -> bool:
        """检查系统是否安装了 tegrastats"""
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
        """启动监控线程"""
        self._stop.clear()
        self.snapshots.clear()
        
        if not self.enabled:
            return
        
        # 启动 tegrastats 进程
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        
        # 启动读取线程
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
    
    def stop(self) -> List[SystemSnapshot]:
        """
        停止监控并返回采集的快照
        
        Returns:
            系统快照列表
        """
        self._stop.set()
        
        if self._proc:
            self._proc.terminate()
            self._proc.wait()
        
        if self._thread:
            self._thread.join(timeout=3)
        
        return list(self.snapshots)
    
    def _reader(self):
        """后台读取 tegrastats 输出"""
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
        """
        解析 tegrastats 输出行
        
        Args:
            line: tegrastats 输出一行
            
        Returns:
            SystemSnapshot 对象，解析失败返回 None
        """
        s = SystemSnapshot(timestamp=time.time())
        try:
            # 解析内存信息：RAM 7960/16384MB
            ram = re.search(r"RAM (\d+)/(\d+)MB", line)
            if ram:
                s.ram_used_mb = int(ram.group(1))
                s.ram_total_mb = int(ram.group(2))
            
            # 解析交换分区：SWAP 0/0MB
            swap = re.search(r"SWAP (\d+)/(\d+)MB", line)
            if swap:
                s.swap_used_mb = int(swap.group(1))
                s.swap_total_mb = int(swap.group(2))
            
            # 解析 CPU 使用率：cpu [6%,0%,0%,0%@998]
            cpus = re.findall(r"(\d+)%@\d+", line)
            if cpus:
                s.cpu_percent = [float(c) for c in cpus]
            
            # 解析 GPU 频率：GR3D_FREQ 90%
            gpu = re.search(r"GR3D_FREQ (\d+)%", line)
            if gpu:
                s.gpu_freq_percent = int(gpu.group(1))
            
            # 解析 GPU 温度：GPU@56.3C
            gpu_temp = re.search(r"GPU@([\d.]+)C", line)
            if gpu_temp:
                s.gpu_temp = float(gpu_temp.group(1))
            
            # 解析 CPU 温度：CPU@55.4C
            cpu_temp = re.search(r"CPU@([\d.]+)C", line)
            if cpu_temp:
                s.cpu_temp = float(cpu_temp.group(1))
            
            return s
            
        except Exception:
            return None
    
    def get_process_metrics(self) -> dict:
        """
        获取当前进程的 CPU 和内存使用
        
        Returns:
            包含 cpu_percent 和 memory_mb 的字典
        """
        proc = psutil.Process()
        return {
            'cpu_percent': proc.cpu_percent(interval=0.1),
            'memory_mb': proc.memory_info().rss / (1024 * 1024)
        }
