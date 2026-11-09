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

安全默认：**ocr-service 只绑 `127.0.0.1`**（容器间走内部网络，不需要对外暴露）；solver 的 `/api/items` 需登录，匿名只能读 `/live` 与 `/api/live/items`。

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
curl -s localhost:8093/healthz            # key_configured:true（该端口只在宿主机回环，外部访问会被拒）
curl -s -o /dev/null -w '%{http_code}\n' localhost:8092/    # 无凭据应为 401
curl -s -o /dev/null -w '%{http_code}\n' localhost:8092/api/items         # 401（管理接口要登录）
curl -s -o /dev/null -w '%{http_code}\n' localhost:8092/api/live/items    # 200（看板专用公开接口）
```

浏览器：

- 答题页 `http://<host>:8092/` —— Basic Auth 用 `EXAM_AUTH_USER/PASS`；上传照片即出答案
- 只读看板 `http://<host>:8092/live` —— 适合投屏/手机盯屏
- 知识库页 `http://<host>:8090/` —— 页面内用 `data/kb/config.json` 的 `api_key` 登录

## 三半、知识库的存储与检索（v3）

```
data/kb/
  kb_<库名>.db     ← 正文：documents / chunks / FTS5 全文索引 / vec_map(label↔chunk)
  vec/
    <库名>.npy     ← 向量源（float32，行号=label，append-only）
    <库名>.usearch ← HNSW 派生索引（可 mmap，损坏可由 npy 重建，不必重新 embedding）
    <库名>.meta.json
```

- **正文与向量分离**：SQLite 只管正文/元数据，向量独立成文件，备份=拷 `data/kb`。
- **向量检索**：≤ `VEC_EXACT_MAX`（默认 2 万）条时用精确线性扫描（与暴力检索等价，几百条 <1ms）；超过才走 HNSW 近似（`HNSW_EF_SEARCH` 默认 128）。写入是增量 upsert，删除只打墓碑，墓碑超 30% 自动压缩（也可 `POST /compact/<库名>` 手动触发）。
- **全文检索**：FTS5 `trigram` 分词（中文按 3 字窗口），BM25 排序；查询先按精确片段匹配，全部落空再退化为 3 字滑窗（照顾用户改写）；整句都是 2 字短词时会拼成 3 字滑窗兜底。**单个少于 3 字的词（如「灾备」）无法匹配 trigram**，这类短查询只走向量。
- **索引自愈**：`.usearch` 缺失或损坏时启动自动用 `.npy` 重建；`meta.json` 丢失时按 `.npy` 非零前缀行推断规模、按索引存活数反推墓碑并继续服务（`count/capacity/tombstones` 无需重算 embedding）；`/health` 的 `vectors` 是活跃向量数（不含墓碑）。
- **故障降级**：`hybrid` 模式下 embedding 服务故障时自动降级为仅全文（响应带 `degraded: true` 与 `embed_error`），答题链路不因 embedding 抖动整体不可用；显式 `mode=embedding` 仍如实报错。
- **embedding 分批**：入库按 `EMBED_BATCH`（默认 64）分批请求，大文档不会因单批过大被上游拒绝。
- **融合**：RRF（`RRF_K=60`），默认向量:全文 = 0.5:0.5（`HYBRID_EMBED_WEIGHT` 可调，单次请求也能传 `embedding_weight`）。返回的 `score` 仍是向量余弦（仅全文命中的是 0.5 占位），融合排序看 `rrf_score`、`rank_vector`、`rank_fulltext`；`min_score` 对两路统一生效（仅全文命中的占位分同样受阈值约束）。
- **升级到 v3 会自动迁移**：首次启动检测到旧库（`chunks` 表里有 `embedding` 列）就地迁移，原库备份为 `kb_xxx.db.pre-vecsplit-<时间戳>`；665 块的库约 2 秒，**不需要重新 embedding**。

实测（bank 库 665 块，20 条自然语言 + 45 条原文片段，hit 按命中文档计）：

| 模式 | 自然语言 hit@1/hit@3 | 原文片段 hit@1/hit@3 | 延迟 |
|---|---|---|---|
| 仅向量 | 20/20 · 20/20 | 29/45 · 32/45 | ≈200ms（几乎全是 embedding） |
| 混合 | 20/20 · 20/20 | 39/45 · 43/45 | ≈200ms |
| 仅全文 | 15/20 · 17/20 | 43/45 · 45/45 | 2–3ms |

> 原文片段未命中的 2 条经核对是标注歧义（同一段文字被另一份文件收录 / 章标题过于通用），不是检索故障。查条款号、文号这类字面查询用 `mode=fulltext` 最快最准；用户改写的问题走 `hybrid`（默认）。

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

- **答题模型**：登录答题页 → 「⚙️ 模型配置」卡 → 添加（名称 / 调用地址 / Key / 模型名 / 用途 answer｜ocr）→ 一键切换。未配置时回落到 `.env` 的 `HS_PLAN_API_KEY` + Ark plan 端点。编辑已有模型时 **Key 留空 = 沿用原值**（接口不回显明文），`extra`（如关闭 thinking）也会保留。
- **OCR 识图模型**：同一张卡的 `ocr` 用途条目。compose 已把 `./data/solver` 只读挂到 ocr-service 的 `/shared`，`SHARED_MODELS=/shared/llm_models.json`，所以网页里保存/切换 `ocr` 条目后**下一次识图立即生效**，不用重建容器；文件里没有启用的 ocr 条目时才回落到 `.env` 的 `HS_PLAN_API_KEY`/`OCR_MODEL`。
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
| 上传后提示「已跳过」但没答案 | 同库已答过同一张图（去重按库隔离）；勾选页面「强制重答」再传 |
| 一页只出部分题目 | OCR 单题校验失败不再整页丢弃：可用题目照常作答，失败题在响应 `errors` 里，页面会弹提示 |
| 网页写操作被 403「拒绝跨站请求」 | 跨站 CSRF 防护（浏览器带 `Sec-Fetch-Site: cross-site`）；本站操作不会触发，curl 不受影响 |
| 换 embedding 模型后分数异常 | 新向量与旧库维度/模型不一致 → 需重建该库并重新录入（服务启动/录入时已校验维度） |
| kb 启动日志报「模型/维度变化，需重建」 | `vec/*.meta.json` 与当前 config 不一致：换过 `embed_model`/`embed_dims`。要么改回原模型，要么删库重录（向量无法直接换算） |
| 全文检索查不到 | ①查询少于 3 个字（trigram 限制）→ 用向量路或换 mode=hybrid；②该库是迁移前的老数据且无向量列 → 迁移时已标为仅全文，检查 `vec_map` 映射 |
| 升级 v3 后首次启动稍慢 | 正常：正在迁移旧库并重建向量索引，665 块约 2 秒；日志里会有 `[migrate]` 行 |

## 九、从源码版（非 Docker）迁移

1. 旧服务停掉（systemd / 直接进程）。
2. 拷 `answers.db`、`llm_models.json` 到 `data/solver/`；拷 `config.json`、`kb_*.db` 到 `data/kb/`。
3. 把旧 `.env` 里的密钥按 `.env.example` 的键名整理（键名不一致时逐个对应）。
4. `docker compose up -d --build`，按第三节验证。
5. 反向代理后端地址从「宿主机 IP:端口」保持不变（容器端口映射与旧版一致），无需改规则。

## 十、已知缺口

- 向量检索是纯 COSINE 相似度，同义改写友好，但精确编号/条款号不如 BM25；需要时可加混合检索。
- 没有内置用户体系，只有一个共享的 Basic Auth 账号；多人使用靠反代层再加鉴权。
