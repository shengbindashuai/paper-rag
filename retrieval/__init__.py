"""在线检索：四档 NPU/GPU 查询编码、只读快照召回、GPU 重排及可选 API 门控。

Python 调用使用 query.PaperRetriever；未来 MCP 使用 tool_api 的 schema 与分发。
模型准备、命令行和真实硬件联调也归入本目录，配置和自动测试分别位于同级
config/ 与 tests/，建库逻辑保持在 ingestion/。
"""
