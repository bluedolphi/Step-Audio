import time
from datetime import datetime
import torch
import os
import psutil

class StatusTracker:
    def __init__(self, version="1.0.0"):
        """初始化状态跟踪器"""
        self.start_time = time.time()
        self.version = version
        self.model_status = {
            "tokenizer": {"loaded": False, "path": None, "load_time": None},
            "tts": {"loaded": False, "path": None, "load_time": None},
            "llm": {"loaded": False, "path": None, "load_time": None}
        }
        
    def update_model_status(self, model_type, loaded=True, path=None):
        """更新模型加载状态
        
        Args:
            model_type: 模型类型，如'tokenizer', 'tts', 'llm'
            loaded: 是否已加载
            path: 模型路径
        """
        self.model_status[model_type] = {
            "loaded": loaded,
            "path": path,
            "load_time": datetime.now().isoformat()
        }
        
    def get_health_info(self):
        """获取健康状态信息"""
        return {
            "status": "ok",
            "version": self.version,
            "uptime": int(time.time() - self.start_time)
        }
        
    def get_model_status(self):
        """获取模型状态信息"""
        all_loaded = all(m["loaded"] for m in self.model_status.values())
        return {
            "models": self.model_status,
            "ready": all_loaded
        }
        
    def get_gpu_info(self):
        """获取GPU资源信息"""
        gpu_count = torch.cuda.device_count()
        gpus = []
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            mem_total = props.total_memory / (1024**3)  # GB
            mem_used = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            mem_free = mem_total - mem_used
            
            gpus.append({
                "id": i,
                "name": props.name,
                "memory_total": round(mem_total, 2),
                "memory_used": round(mem_used, 2),
                "memory_free": round(mem_free, 2),
                "utilization": round(mem_used / mem_total * 100, 2)
            })
            
        return {
            "gpu_count": gpu_count,
            "gpus": gpus
        }

    def get_system_info(self):
        """获取系统资源信息"""
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu": {
                "percent": cpu_percent,
                "cores": psutil.cpu_count()
            },
            "memory": {
                "total": round(memory.total / (1024**3), 2),  # GB
                "available": round(memory.available / (1024**3), 2),  # GB
                "percent": memory.percent
            },
            "disk": {
                "total": round(disk.total / (1024**3), 2),  # GB
                "free": round(disk.free / (1024**3), 2),  # GB
                "percent": disk.percent
            }
        } 