# exam-qa-stack（Docker 版）

拍照 → 识别题目 → 检索知识库 → 生成答案（含依据引用）；另带网页版资料库。三个服务各自独立成镜像，`docker compose up -d` 一键起。

> 与 2026-11-08 线上版（121 挂载源码 + 启动时 pip install）等价，但依赖已烘焙进镜像，**离线也能重启**。

## 架构

```
试题照片 ─▶ exam-solver(8092) ─▶ ocr-service(8093→8090)   视觉模型读图出题
                │
                ├─▶ kb-service(8090)  检索：查询向量化 → SQLite 内比对
                │        └─▶ DashScope text-embedding-v4（1536 维）
                └─▶ 答题模型（网页「模型配置」卡管理，llm_models.json）
```

| 服务 | 宿主端口 | 作用 | 数据卷 |
|---|---|---|---|
| exam-solver | 8092 | 答题 API + 上传页 `/` + 只读看板 `/live` | `./data/solver`（answers.db、llm_models.json） |
| ocr-service | 8093 | 图片 → 结构化题目 JSON（需视觉模型 key） | 无 |
| kb-service | 8090 | 知识库：建库/录入/检索（`X-API-Key` 鉴权） | `./data/kb`（config.json、index.html、kb_*.db） |

## 快速开始

```bash
cp .env.example .env            # 填 EXAM_AUTH_PASS / HS_PLAN_API_KEY / KB_API_KEY
mkdir -p data/kb data/solver
cp kb-service/config.example.json data/kb/config.json   # 填 embed_token（DashScope key）
docker compose up -d --build
```

- 答题页：`http://<host>:8092/`（Basic Auth：`EXAM_AUTH_USER/PASS`）
- 知识库页：`http://<host>:8090/`（页内用 `api_key`，即 `KB_API_KEY`）
- 健康检查：`/healthz`（solver）、`/health`（kb）、`/healthz`（ocr）

## 模型配置

- **答题模型**：登录答题页 → 「⚙️ 模型配置」卡 → 添加（名称/调用地址/Key/模型名/用途），可存多条一键切换；不添加则回落到 env 的 `HS_PLAN_API_KEY` + Ark plan 端点。OCR 识图模型同样在该卡配置。
- **向量模型**：kb 的 `data/kb/config.json`（`embed_url` / `embed_token` / `embed_model` / `embed_dims`）。⚠️ 换模型或维度 = 该库全部向量作废重建。

## 数据与升级

- 全部状态在 `./data/`（SQLite），升级镜像 = `git pull && docker compose up -d --build`，数据不动。
- 备份：拷走 `./data/` 即可整栈迁移。

## 安全须知

- 真实密钥只放 `.env` 与 `data/kb/config.json`，两者均已 gitignore；`data/llm_models.json` 含模型 key，同样不入库。
- 公网暴露请套反向代理 + HTTPS（线上用 Lucky 3377）。
