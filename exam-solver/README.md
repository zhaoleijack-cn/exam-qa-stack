# exam-solver（答题服务）

拍照 → OCR 识别题目 → 检索知识库 → 生成答案（含题干、选项、依据）。

- 端口：8092；入口 `/`（Basic Auth 上传页）、`/live`（只读看板）、`/healthz`（健康检查）
- 依赖：`ocr-service`（识图）、`kb-service`（向量检索）、答题模型（网页「模型配置」卡或 `HS_PLAN_API_KEY`）
- 数据：`./data/solver/answers.db`（题目清单）、`./data/solver/llm_models.json`（模型配置）

## 机制要点

1. **检索输入带选项**：query = 题干 + 全部选项。
2. **top6 证据**：每题取 6 段，prompt 编号，答案须能在证据中找到出处。
3. **强约束 prompt**：只依据资料、资料不足降置信度，输出结构化 JSON（answer/confidence/evidence/reason）。
4. **答案归一校验**：单选取首个合法字母；多选去重排序并过滤越界字母；判断题归一「正确/错误」。
5. **低置信度复核**：confidence < 0.6 再生成一次，两次一致加分，不一致取高置信度并封顶。
6. **人工复核标记**：confidence < 0.55、检索最高分 < 0.3、答案为空 → `needs_review=true`。

正确率上限由知识库覆盖决定。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/datasets` | 可用知识库（取自 kb-service） |
| POST | `/api/solve` | multipart：`files`、`dataset_id`、`concurrency`、`only_ocr` |
| POST | `/api/solve/json` | `{"dataset_id":"...","images":[{"name":"a.jpg","base64":"..."}]}` |
| GET | `/api/items` | 全部题目 + 汇总 |
| DELETE | `/api/items/{id}` | 删除单题 |
| DELETE | `/api/batches/{id}` | 删除整批 |
| POST | `/api/clear` | 清空清单 |
| POST | `/api/markdown` | 结果 JSON 转 Markdown |
| GET | `/healthz` | 依赖与密钥状态 |

## 评测

```bash
python3 eval.py --dataset <知识库id> --dir /path/to/images --answers answers.json
# 或复用已有结果：--result result.json
```

`answers.json`：`{"图片文件名": ["B"], "多题图.jpg": ["A", "AC"]}`

## 去重

每张图片按 sha256 前 16 位记指纹，已入库的图片再次上传直接跳过（返回 `skipped`）。
