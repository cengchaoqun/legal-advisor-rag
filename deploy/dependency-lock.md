# 依赖与环境锁定（V0.4 开发记录）

记录日期：2026-10-21。所有条目均为本机实际执行结果，不是计划值。

当日Qwen3更新：当前uv.lock为transformers4.57.6、peft0.21.2、torch2.11.0+cu128、datasets3.6.0、accelerate1.15.0。锁文件约束已离线更新；本机GPU为RTX5060 Laptop约8GiB，下文“无GPU/CPU torch”是旧环境记录。Qwen3-8B训练目标是用户外部24GB服务器，其实际torch/CUDA/训练显存尚未验证；不要把本机锁定记录当作外部服务器已验证。

## 1. 运行时与工具

| 项目 | 锁定值 | 说明 |
| --- | --- | --- |
| Python | 3.11.9（uv 托管 cpython-3.11-windows-x86_64） | `pyproject.toml` 要求 `>=3.11,<3.12` |
| uv | 0.11.28 | `uv sync --extra embedding --no-editable` |
| Node | 不用于本项目 | 前端为原生 HTML/CSS/JS |
| Docker | 29.2.1 / Compose v5.0.2 | Windows + Docker Desktop |

## 2. 中间件镜像（deploy/compose.yml）

| 服务 | 镜像 | 端口（仅本机） |
| --- | --- | --- |
| MongoDB | mongo:7.0.14 | 127.0.0.1:27017 |
| 对象存储 | chrislusf/seaweedfs:3.71 | 127.0.0.1:9000（S3 端口 8333 映射） |
| etcd | quay.io/coreos/etcd:v3.5.16 | 内部 2379 |
| Milvus | milvusdb/milvus:v2.4.15 | 127.0.0.1:19530（健康检查 9091） |
| Neo4j | neo4j:5.26.0-community | 127.0.0.1:7687 / 7474 |

## 3. 关键 Python 依赖（uv.lock 实际解析）

| 包 | 版本 |
| --- | --- |
| fastapi | 0.156.5 |
| uvicorn | 0.54.0 |
| pydantic | 2.14.1 |
| pymongo | 4.19.0（AsyncMongoClient） |
| pymilvus | 2.4.15 |
| neo4j | 5.28.8 |
| minio | 7.2.20 |
| pypdf | 6.x（随 lock） |
| langgraph | 0.2.x/0.3.x（随 lock） |
| jsonschema | 4.25.x（随 lock） |
| openapi-spec-validator | 0.7.x（dev 组） |
| torch | 2.14.1（CPU） |
| transformers | 4.57.6 |
| FlagEmbedding | 1.3.x（随 lock） |

## 4. 模型快照

| 用途 | 模型 | 快照 |
| --- | --- | --- |
| 向量化 | BAAI/bge-m3 | `5617a9f61b028005a4858fdac845db406aefb181`（EMBED_MODEL_REVISION） |
| 最终回答 | legal-labor-sft-v1（程序外微调后部署） | 未训练、未部署（阻塞项） |
| 辅助抽取/HyDE | qwen-flash | 未配置密钥（阻塞项） |
| 重排序 | qwen3-rerank | 未配置密钥（阻塞项） |

## 5. 与设计的偏差（均为环境强制，可回退）

1. **对象存储由 MinIO 改为 SeaweedFS**：MinIO 开源版已归档，官方二进制返回 `410 Gone`，Docker Hub/镜像源均无法拉取（2026-10-21 实测）。应用侧使用标准 S3 客户端，替换回 MinIO 只需改 endpoint/凭据。
2. **Python 包以非可编辑方式安装**：项目路径含中文，Python 的 `.pth` 以 GBK 读取导致路径乱码、`import legal_rag` 失败。改为 `uv sync --no-editable`，修改代码后需 `uv sync --extra embedding --no-editable --reinstall-package legal-rag`。
3. **MongoDB 曾在官方 Windows 二进制上运行**：Docker 镜像源长时间阻塞时用官方 zip 启动 `mongod` 完成集成测试；镜像拉取成功后已切回 `mongo:7.0.14` 容器，`docker compose ps` 可核对。
4. **前端只有三个页面**：按 V0.4 移除“模型与评估”入口；训练/评估产物留在 `artifacts/`，不进入业务数据库。

## 6. 未锁定/未验证项

- GPU 训练环境（CUDA/torch 版本、显存）未验证：本机无可用 GPU，也未租赁。
- 在线模型账户、地域、单价未配置：`deploy/pricing.json` 未创建，付费调用默认关闭。
- 联网检索服务（MCP_URL）未配置：`web_enabled=true` 时返回 unavailable。
