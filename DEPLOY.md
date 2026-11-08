# 部署说明（Docker Compose）

三服务（exam-solver / ocr-service / kb-service）各自独立成镜像，`docker compose up -d --build` 一条命令起全栈。依赖已烘焙进镜像，**断网也能重启**（调用外部模型时才需要网络）。

## 一、前置条件

| 项 | 要求 |
|---|---|
| Docker | 24+，含 Compose v2（`docker compose version` 可用即可） |
| 系统 | Linux x86_64；约 1.5 GB 镜像空间 + 数据目录 |
| 内存 | ≥ 1 GB（三容器实测合计约 300–500 MB） |
| 网络 | 能访问模型端点：火山方舟 Ark（`ark.cn-beijing.volces.com`）与 DashScope（`dashscope.aliyuncs.com`）；纯内网可换成自建兼容 OpenAI 协议的端点 |
| 密钥 | 至少一个视觉/文本模型 Key（答题 + OCR），一个 embedding Key（知识库检索） |

端口占用：默认 8090（kb）、8092（solver）、8093（ocr）。冲突时在 `.env` 里改 `KB_PORT/SOLVER_PORT/OCR_PORT`，容器内端口不变。

## 二、五步起栈

```bash
# 1. 取代码
git clone https://github.com/zhaoleijack-cn/exam-qa-stack.git
cd exam-qa-stack

# 2. 环境变量（必填三项：EXAM_AUTH_PASS、HS_PLAN_API_KEY、KB_API_KEY）
cp .env.example .env
vi .env

# 3. 数据目录 + 知识库配置（embed_token 填 DashScope key）
mkdir -p data/kb data/solver
cp kb-service/config.example.json data/kb/config.json
vi data/kb/config.json

# 4. 构建并启动
docker compose up -d --build

# 5. 看状态
docker compose ps
docker compose logs -f --tail=50
```

`.env` 权限建议 `chmod 600 .env`；它和 `data/kb/config.json` 都已被 `.gitignore` 排除，**不要提交**。

## 三、验证

```bash
curl -s localhost:8092/healthz            # 应见 ok:true、datasets>0、keys.ark:true
curl -s localhost:8090/health             # {"status":"ok","datasets":N,"dim":1536,...}
curl -s localhost:8093/healthz            # key_configured:true
curl -s -o /dev/null -w '%{http_code}\n' localhost:8092/    # 无凭据应为 401
```

浏览器：

- 答题页 `http://<host>:8092/` —— Basic Auth 用 `EXAM_AUTH_USER/PASS`；上传照片即出答案
- 只读看板 `http://<host>:8092/live` —— 适合投屏/手机盯屏
- 知识库页 `http://<host>:8090/` —— 页面内用 `data/kb/config.json` 的 `api_key` 登录

## 四、首次使用：把资料灌进知识库

新装时知识库是空的（`/api/datasets` 返回空），答题会命中不到依据，需先录入。

```bash
# 建库（首次写入自动创建 dataset，名字限小写字母/数字/下划线）
curl -s -X POST localhost:8090/ingest/text \
  -H 'X-API-Key: <KB_API_KEY>' -H 'Content-Type: application/json' \
  -d '{"dataset":"bank","title":"示例","text":"任意一段法规文本……"}'

# 文件录入（pdf/docx/txt/md，按 chunk 切分并向量化）
curl -s -X POST localhost:8090/ingest/file \
  -H 'X-API-Key: <KB_API_KEY>' \
  -F 'file=@/path/制度.pdf' -F 'dataset=bank' -F 'title=某制度'

# 检索自检（score < 0.5 视为弱相关）
curl -s -X POST localhost:8090/search \
  -H 'X-API-Key: <KB_API_KEY>' -H 'Content-Type: application/json' \
  -d '{"query":"支票提示付款期限","dataset":"exam","top_k":5}'
```

也可以用知识库网页 `http://<host>:8090/` 上传。题库类资料和行内制度建议分开建库（如 `exam` / `bank`），答题页下拉里可切换。

> 数据量提示：embedding 走外部 API，几千块文本大约几分钟；大批量录入前先确认 Key 额度。

## 五、模型配置

- **答题模型**：登录答题页 → 「⚙️ 模型配置」卡 → 添加（名称 / 调用地址 / Key / 模型名 / 用途 answer｜ocr）→ 一键切换。未配置时回落到 `.env` 的 `HS_PLAN_API_KEY` + Ark plan 端点。
- **OCR 识图模型**：同一张卡的 `ocr` 用途条目。⚠️ 当前 compose 只把 `llm_models.json` 挂在 solver 容器里，ocr-service 读的是 `/shared/llm_models.json`（未挂载），因此**线上以 `.env` 的 `HS_PLAN_API_KEY`/`OCR_MODEL` 为准**；要让网页里切换 OCR 模型生效，给 `ocr-service` 加一行 `volumes: - ./data/solver:/shared:ro` 并重建（文档下方「已知缺口」）。
- **向量模型**：`data/kb/config.json` 的 `embed_url/embed_token/embed_model/embed_dims`。换模型或维度 = 该库全部向量作废，必须重新录入。

## 六、升级 / 备份 / 回滚

```bash
# 升级（数据在 ./data，不受影响）
git pull
docker compose up -d --build

# 备份：整个 ./data 拷走即可整栈迁移
tar czf exam-data-$(date +%F).tgz data/ .env

# 回滚：回到旧提交重新构建
git checkout <旧提交> && docker compose up -d --build
```

`data/` 内含：`solver/answers.db`（答题记录）、`solver/llm_models.json`（模型配置+Key）、`kb/config.json`、`kb/kb_*.db`（向量库）。迁移到新机器时连同 `.env` 一起拷，改完端口直接起。

SQLite 是 WAL 模式：**停容器后再拷**才是干净快照（停容器会自动 checkpoint，`-wal/-shm` 消失）。

## 七、公网暴露

- 必须前面套反向代理 + HTTPS（Nginx / Caddy / Lucky 都行），不要把 8092 直接映射到公网。
- 反代路径保持原样即可；`/`、`/live` 已带 `Cache-Control: no-store`，不必额外加缓存头。
- Basic Auth 是应用层校验，反代层再叠一层 Basic Auth 也无妨（双层不会冲突）。
- 若用 Lucky：新建子规则，后端地址填 `http://<docker宿主机>:8092`，前端域名配好证书；知识库页单独一条指向 8090。

## 八、故障排查

| 现象 | 原因 / 处理 |
|---|---|
| 所有请求 401 | `.env` 的 `EXAM_AUTH_PASS` 为空 → app 直接拒（`if not AUTH_PASS: return False`）。检查 `docker compose config` 里的 env 是否注入 |
| `/healthz` 里 `datasets:0` | kb 未起或 `KB_API_KEY` 不一致 → `docker compose logs kb-service`；确认 solver 与 kb 用的是同一个 key |
| 拍照报「上游调用失败 / Illegal header value」 | 模型 Key 没注入 → `curl localhost:8093/healthz` 看 `key_configured`，补 `.env` 的 `HS_PLAN_API_KEY` 后重建 ocr-service |
| 录入报 embedding 失败 | `data/kb/config.json` 的 `embed_token` 错/额度用尽；或 `embed_url` 不可达 |
| 页面还是旧版 | 浏览器缓存：加 `?v=2` 或用无痕窗口；服务端已发 no-store |
| 容器起不来且日志无异常 | `docker compose config` 校验 YAML；`docker compose up` 不带 `-d` 看构建输出 |
| 改了源码不生效 | 只 `restart` 不会重建镜像，必须 `up -d --build` |

## 九、从源码版（非 Docker）迁移

1. 旧服务停掉（systemd / 直接进程）。
2. 拷 `answers.db`、`llm_models.json` 到 `data/solver/`；拷 `config.json`、`kb_*.db` 到 `data/kb/`。
3. 把旧 `.env` 里的密钥按 `.env.example` 的键名整理（键名不一致时逐个对应）。
4. `docker compose up -d --build`，按第三节验证。
5. 反向代理后端地址从「宿主机 IP:端口」保持不变（容器端口映射与旧版一致），无需改规则。

## 十、已知缺口

- `ocr-service` 未挂载 `/shared/llm_models.json`（见第五节），网页切换 OCR 模型暂不生效，实际用 `.env` 配置。
- 向量检索是纯 COSINE 相似度，同义改写友好，但精确编号/条款号不如 BM25；需要时可加混合检索。
- 没有内置用户体系，只有一个共享的 Basic Auth 账号；多人使用靠反代层再加鉴权。
