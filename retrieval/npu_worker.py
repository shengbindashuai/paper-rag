"""每个驻留档位独立进程，隔离本机驱动多图共存时的 BufferInfo 冲突。

父进程串行请求；传输的是问题和小向量，不复制整份模型。关闭只作用于
本对象创建的子进程。Windows 使用 spawn，应用入口须有 __main__ 保护。
"""
from contextlib import redirect_stdout
import multiprocessing as mp
import sys


def _serve(connection, model_dir, tokenizer_dir, instruction, cache_dir):
    model = None
    try:
        from rag_core.encoders.npu import NpuQueryEmbedder
        with redirect_stdout(sys.stderr):
            model = NpuQueryEmbedder(model_dir, tokenizer_dir, instruction, cache_dir)
        connection.send((True, model.device_info))
        while True:
            command, text = connection.recv()
            if command == "close":
                break
            try:
                connection.send((True, model.encode_one(text)))
            except Exception as exc:
                connection.send((False, f"{type(exc).__name__}: {exc}"))
    except EOFError:
        pass
    except Exception as exc:
        try: connection.send((False, f"{type(exc).__name__}: {exc}"))
        except (OSError, EOFError): pass
    finally:
        if model is not None: model.close()
        connection.close()


class IsolatedNpuEncoder:
    def __init__(self, model_dir, tokenizer_dir, instruction, cache_dir):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_serve, args=(child, model_dir, tokenizer_dir, instruction, cache_dir), daemon=True)
        self.process.start()
        child.close()
        try:
            self.device_info = self._receive(timeout=300)
        except Exception:
            self.close()
            raise

    def _receive(self, timeout=120):
        if not self.connection.poll(timeout):
            raise RuntimeError("NPU 工作进程超时")
        success, value = self.connection.recv()
        if not success: raise RuntimeError(value)
        return value

    def encode_one(self, query):
        self.connection.send(("encode", query))
        vector, timing = self._receive()
        timing["worker_pid"] = self.process.pid
        return vector, timing

    def close(self):
        if self.process.is_alive():
            try: self.connection.send(("close", None))
            except (OSError, EOFError): pass
            self.process.join(5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(5)
        self.connection.close()
