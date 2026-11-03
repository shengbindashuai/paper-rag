"""只替换业务层的 STDIO 协议测试进程。"""
import asyncio
import sys

from knowledge.mcp_server import build_server, run_stdio


class API:
    def call(self, name, arguments):
        return {"ok": True}


if __name__ == "__main__":
    import anyio
    asyncio.run(run_stdio(build_server(API()), anyio.wrap_file(sys.stdout)))
