#!/usr/bin/env python3
"""
Step-Audio 统一启动脚本
可以启动完整的服务或单独模块
"""

import argparse
import os
import subprocess
import sys
import time

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Step-Audio 服务启动器")
    
    parser.add_argument(
        "--model-path", 
        type=str, 
        default=os.environ.get("MODELS_DIR", "./models"),
        help="模型路径，默认使用MODELS_DIR环境变量或'./models'"
    )
    
    parser.add_argument(
        "--server-name", 
        type=str, 
        default=os.environ.get("SERVER_NAME", "0.0.0.0"),
        help="服务器地址，默认使用环境变量或'0.0.0.0'"
    )
    
    parser.add_argument(
        "--server-port", 
        type=int, 
        default=int(os.environ.get("SERVER_PORT", "7860")),
        help="服务器端口，默认使用环境变量或7860"
    )
    
    parser.add_argument(
        "--gpu-ids", 
        type=str, 
        default=os.environ.get("GPU_IDS", ""),
        help="GPU ID列表，以逗号分隔，例如'0,1'"
    )
    
    parser.add_argument(
        "--device-map", 
        type=str, 
        default=os.environ.get("DEVICE_MAP", "auto"),
        help="模型加载设备映射策略(auto, balanced, sequential)"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["all", "chat", "tts"], 
        default="all",
        help="启动模式：all(全部), chat(仅聊天), tts(仅TTS)"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="是否启用调试模式"
    )
    
    return parser.parse_args()

def build_command(args, mode):
    """构建启动命令"""
    cmd = [sys.executable]
    
    if mode == "chat":
        cmd.append("app.py")
    elif mode == "tts":
        cmd.append("tts_app.py")
    
    cmd.extend([
        "--model-path", args.model_path,
        "--server-name", args.server_name,
        "--server-port", str(args.server_port)
    ])
    
    # 添加GPU相关参数
    if args.gpu_ids:
        cmd.extend(["--gpu-ids", args.gpu_ids])
    
    if args.device_map:
        cmd.extend(["--device-map", args.device_map])
    
    return cmd

def start_process(cmd, debug=False):
    """启动子进程"""
    if debug:
        print(f"执行命令: {' '.join(cmd)}")
        return subprocess.Popen(cmd)
    else:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def main():
    """主函数"""
    args = parse_args()
    
    # 确保目录存在
    os.makedirs(os.path.join(args.model_path), exist_ok=True)
    
    # 根据模式启动不同的服务
    processes = []
    
    if args.mode in ["all", "chat"]:
        chat_cmd = build_command(args, "chat")
        if args.mode == "all" and args.server_port:
            # 如果是全模式，聊天服务使用指定端口
            chat_cmd[-1] = str(args.server_port)
        
        print(f"启动聊天服务: {' '.join(chat_cmd)}")
        chat_process = start_process(chat_cmd, args.debug)
        processes.append(("聊天服务", chat_process))
    
    if args.mode in ["all", "tts"]:
        tts_cmd = build_command(args, "tts")
        if args.mode == "all" and args.server_port:
            # 如果是全模式，TTS服务使用指定端口+1
            tts_cmd[-1] = str(args.server_port + 1)
        
        print(f"启动TTS服务: {' '.join(tts_cmd)}")
        tts_process = start_process(tts_cmd, args.debug)
        processes.append(("TTS服务", tts_process))
    
    # 监控进程并输出日志
    try:
        print(f"所有服务已启动，按Ctrl+C结束...")
        while True:
            time.sleep(1)
            for name, process in processes:
                if process.poll() is not None:
                    print(f"{name}异常退出，退出代码: {process.returncode}")
                    return process.returncode
    except KeyboardInterrupt:
        print("接收到中断信号，正在关闭服务...")
    finally:
        # 关闭所有进程
        for name, process in processes:
            print(f"正在关闭{name}...")
            process.terminate()
            process.wait(timeout=5)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 