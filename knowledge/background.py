"""Windows 后台入口：读取用户环境、保存日志，然后运行 MCP 服务。"""
import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    root = args.root.resolve()
    os.chdir(root)
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    # Task Scheduler 的父进程可能持有旧环境；每次启动直接读取当前用户的持久化值。
    if os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as environment:
                secret, _ = winreg.QueryValueEx(environment, "DEEPSEEK_API_KEY")
            if secret:
                os.environ["DEEPSEEK_API_KEY"] = secret
        except FileNotFoundError:
            pass
    logs = root / "RAG" / "library" / "service_logs"
    logs.mkdir(parents=True, exist_ok=True)
    # 按启动时间分文件，避免重启覆盖之前的故障日志。
    path = logs / (datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}.log")
    with path.open("a", encoding="utf-8", buffering=1) as output:
        sys.stdout = sys.stderr = output
        logging.basicConfig(stream=output, level=logging.INFO, force=True)
        print(f"Starting local knowledge MCP; pid={os.getpid()}; port={args.port}", flush=True)
        print("DEEPSEEK_API_KEY configured: " + str(bool(os.environ.get("DEEPSEEK_API_KEY"))), flush=True)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        sys.argv = ["mcp_server", "--root", str(root), "--port", str(args.port)]
        try:
            from knowledge.mcp_server import main as serve
            serve()
        except BaseException:
            traceback.print_exc(file=output)
            raise


if __name__ == "__main__":
    main()
