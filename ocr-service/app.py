"""考试题 OCR 服务：视觉直出 + 规则校验 + 重试 + 可选双跑一致性

POST /ocr        multipart/form-data: file=<图片>, strict=<bool>
POST /ocr/b64    application/json: {"image_base64": "...", "strict": false}
POST /ocr/batch  multipart/form-data: files=<图片...>, strict=<bool>
GET  /healthz

设计原则：
- 视觉模型只负责"看"，输出结构化 JSON；
- 服务端做硬校验（字母连续、选项数、题型归一、空值、标签残留），不合规重试一次；
- strict=true 时同一张图跑两次，比对归一化文本，不一致降置信度；
- 所有 warnings 不阻断，errors 阻断并返回 422。
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import time
import unicodedata
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps

API_BASE = os.environ.get("OCR_API_BASE", "https://ark.cn-beijing.volces.com/api/plan/v3").rstrip("/")
API_KEY = os.environ.get("HS_PLAN_API_KEY", "")
MODEL = os.environ.get("OCR_MODEL", "deepseek-v4.1-flash")

SHARED_MODELS = os.environ.get("SHARED_MODELS", "/shared/llm_models.json")

def ocr_active():
    """读共享模型库中 role=ocr 的启用条目；无则回落容器环境变量。"""
    try:
        models = json.loads(open(SHARED_MODELS).read())
        act = [m for m in models if m.get("role") == "ocr" and m.get("active")]
        if act:
            m = act[0]
            return {"base": str(m["base_url"]).strip().rstrip("/"),
                    "key": str(m.get("api_key", "")), "model": str(m["model"]),
                    "extra": m.get("extra") or {}}
    except Exception:
        pass
    return {"base": API_BASE, "key": API_KEY, "model": MODEL,
            "extra": {"thinking": {"type": "disabled"}}}
MAX_SIDE = int(os.environ.get("OCR_MAX_SIDE", "1920"))
MAX_UPLOAD_MB = int(os.environ.get("OCR_MAX_UPLOAD_MB", "20"))
MAX_PIXELS = int(os.environ.get("OCR_MAX_PIXELS", "40000000"))  # 解压炸弹防护（约 40MP）
STRICT_DEFAULT = os.environ.get("OCR_STRICT", "0") == "1"
CONCURRENCY = int(os.environ.get("OCR_CONCURRENCY", "4"))

PROMPT = (
    "识别图中的考试题目。只输出 JSON，不要解释、不要代码块。"
    '格式：{"type":"单选/多选/判断","stem":"题干原文，不含题号","options":{"A":"...","B":"..."}}。'
    "如果图中有多道题，输出 JSON 数组。逐字照抄，不得改写、不得补全、不得漏选项。"
)
PROMPT_RETRY = (
    "上一条回复无法解析。请重新识别，严格遵守：只输出纯 JSON（数组或对象），"
    "不要 markdown 代码块、不要任何说明文字。每题形如 "
    '{"type":"单选|多选|判断","stem":"...","options":{"A":"...","B":"..."}}。'
    "选项字母必须从 A 开始连续，逐字照抄。"
)

TYPE_MAP = {
    "单选题": "单选", "单选": "单选", "单项选择题": "单选", "单项选择": "单选",
    "多选题": "多选", "多选": "多选", "多项选择题": "多选", "多项选择": "多选",
    "判断题": "判断", "判断": "判断", "是非题": "判断",
}

app = FastAPI(title="Exam OCR Service", version="1.0.0")
_client: httpx.AsyncClient | None = None
_sem = asyncio.Semaphore(CONCURRENCY)
STARTED = time.time()


# ---------- 基础工具 ----------

def _client_get() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0))
    return _client


def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", "", s)
    # v1.0.2: 保留小数点与负号——此前 1.5% 与 15%、-10 与 10 被视为相同，双跑比对形同虚设
    return re.sub(r"[，。、；：？！“”‘’（）()《》【】—…·,;:?\"'<>]", "", s)


def _strip_fence(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_json(s: str) -> Any:
    """容忍代码块、前后废话、单对象/数组、{"questions":[...]} 包装。"""
    s = _strip_fence(s)
    try:
        return json.loads(s)
    except Exception:
        pass
    # 找第一个 [ 或 {，做括号平衡截取
    start = min([i for i in (s.find("["), s.find("{")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("回复中没有 JSON")
    stack: list[str] = []
    pairs = {"[": "]", "{": "}"}
    for i in range(start, len(s)):
        ch = s[i]
        if ch in pairs:
            stack.append(pairs[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
            if not stack:
                return json.loads(s[start : i + 1])
    raise ValueError("JSON 括号不平衡")


def _as_list(obj: Any) -> list[dict]:
    if isinstance(obj, dict):
        if isinstance(obj.get("questions"), list):
            return [q for q in obj["questions"] if isinstance(q, dict)]
        return [obj]
    if isinstance(obj, list):
        out: list[dict] = []
        for item in obj:
            if isinstance(item, dict):
                if isinstance(item.get("questions"), list):
                    out.extend(q for q in item["questions"] if isinstance(q, dict))
                else:
                    out.append(item)
        return out
    return []


def _norm_type(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    key = re.sub(r"[\s【】\[\]]", "", raw)
    return TYPE_MAP.get(key)


# ---------- 校验 ----------

def validate_question(q: dict) -> tuple[dict | None, list[str], list[str]]:
    """返回 (清洗后的题目 | None, errors, warnings)。errors 非空即不可用。"""
    errors: list[str] = []
    warnings: list[str] = []

    stem = str(q.get("stem") or "").strip()
    # 去题型标签残留
    if re.search(r"【[^】]{0,8}题】", stem):
        stem = re.sub(r"【[^】]{0,8}题】", "", stem).strip()
        warnings.append("题干残留题型标签，已清除")
    # 去题号前缀（v1.0.2: 半角点后紧跟数字视为小数开头，不当题号误删——"1.5%的利率" 不能变成 "5%的利率"）
    if re.match(r"^\d+\s*[、．)）]\s*", stem) or re.match(r"^\d+\.(?!\d)\s*", stem):
        stem = re.sub(r"^\d+\s*[、．)）]\s*", "", stem)
        stem = re.sub(r"^\d+\.(?!\d)\s*", "", stem).strip()
        warnings.append("题干含题号，已去掉")
    if len(stem) < 3:
        errors.append("题干为空或过短")
    if len(stem) > 800:
        errors.append("题干过长（>800 字）")
    if "```" in stem or "\n\n" in stem:
        warnings.append("题干含异常换行/代码块")

    raw_opts = q.get("options")
    if not isinstance(raw_opts, dict) or not raw_opts:
        errors.append("缺少 options")
        return None, errors, warnings

    opts: dict[str, str] = {}
    for k, v in raw_opts.items():
        kk = str(k).strip().upper()
        kk = re.sub(r"[^A-Z]", "", kk)
        if not kk:
            warnings.append(f"丢弃非法选项键 {k!r}")
            continue
        vv = str(v or "").strip()
        if not vv:
            errors.append(f"选项 {kk} 为空")
            continue
        opts[kk] = vv

    if not opts:
        errors.append("所有选项为空")
        return None, errors, warnings

    keys = sorted(opts.keys())
    expect = [chr(ord("A") + i) for i in range(len(keys))]
    if keys != expect:
        errors.append(f"选项字母不连续：{keys}")
    if len(keys) > 8:
        errors.append(f"选项过多：{len(keys)} 个")

    qtype = _norm_type(q.get("type"))
    if qtype is None:
        qtype = "判断" if len(keys) == 2 else "单选"
        warnings.append(f"题型缺失或无法识别，按选项数推断为「{qtype}」")
    if qtype == "判断" and len(keys) != 2:
        errors.append(f"判断题应有 2 个选项，实际 {len(keys)}")
    if qtype in ("单选", "多选") and len(keys) < 3:
        errors.append(f"{qtype}题选项少于 3 个")

    for k, v in opts.items():
        if len(v) > 600:
            errors.append(f"选项 {k} 过长（>600 字）")
        if "```" in v:
            warnings.append(f"选项 {k} 含代码块符号")

    if errors:
        return None, errors, warnings
    return {"type": qtype, "stem": stem, "options": opts}, errors, warnings


# ---------- 上游调用 ----------

class UpstreamFatal(RuntimeError):
    """不可重试的上游错误（4xx / 缺 Key），重试只会白等。"""


async def call_vision(b64: str, mime: str, prompt: str) -> tuple[str, dict, float]:
    mo = ocr_active()
    if not mo["key"]:
        raise UpstreamFatal("未配置模型 Key：请设置 HS_PLAN_API_KEY 或在网页「模型配置」里添加识图模型")
    payload = {
        "model": mo["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 2000,
    }
    headers = {"Authorization": f"Bearer {mo['key']}", "Content-Type": "application/json"}
    last: Exception | None = None
    for attempt in range(3):
        t0 = time.time()
        try:
            async with _sem:
                r = await _client_get().post(f"{mo['base']}/chat/completions", json=payload, headers=headers)
            if r.status_code >= 400:
                msg = f"上游 HTTP {r.status_code}: {r.text[:200]}"
                # 408/409/425/429 与 5xx 才是"等一下可能好"的；其余 4xx 立即失败
                if r.status_code in (408, 409, 425, 429) or r.status_code >= 500:
                    raise RuntimeError(msg)
                raise UpstreamFatal(msg)
            d = r.json()
            return d["choices"][0]["message"]["content"], d.get("usage") or {}, time.time() - t0
        except UpstreamFatal:
            raise
        except Exception as e:  # 网络错误退避重试
            last = e
            if attempt < 2:
                await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"上游调用失败：{last}")


# ---------- 图片处理 ----------

def prepare_image(data: bytes) -> tuple[str, str, dict]:
    try:
        im = Image.open(io.BytesIO(data))
        w0, h0 = im.size
        if w0 * h0 > MAX_PIXELS:
            raise HTTPException(413, f"图片像素过大（{w0}x{h0}），上限 {MAX_PIXELS} 像素")
        im.load()
    except HTTPException:
        raise
    except Image.DecompressionBombError as e:
        raise HTTPException(413, f"图片像素过大：{e}")
    except Exception as e:
        raise HTTPException(400, f"无法解析图片：{e}")
    # 手机直出的照片常带 EXIF 方向，不摆正会横着进模型，识别率断崖下降
    try:
        im = ImageOps.exif_transpose(im) or im
    except Exception:
        pass
    orig = im.size
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    w, h = im.size
    scale = min(1.0, MAX_SIDE / float(max(w, h)))
    if scale < 1.0:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    meta = {"orig": list(orig), "sent": list(im.size), "bytes": len(buf.getvalue())}
    return b64, "image/jpeg", meta


# ---------- 核心识别流程 ----------

async def recognize(data: bytes, strict: bool) -> dict:
    b64, mime, img_meta = prepare_image(data)
    t0 = time.time()
    attempts = 0
    errors: list[str] = []
    warnings: list[str] = []
    questions: list[dict] = []
    usage: dict = {}
    raw_last = ""

    best = None   # (rank, questions, errors, warnings, usage, raw)
    for attempt in (1, 2):
        attempts = attempt
        prompt = PROMPT if attempt == 1 else PROMPT_RETRY
        try:
            raw, u, _ = await call_vision(b64, mime, prompt)
        except UpstreamFatal as e:
            errors = [str(e)]
            break
        except RuntimeError as e:
            errors = [str(e)]
            break
        raw_last = raw
        usage = u or usage
        try:
            obj = _extract_json(raw)
        except Exception as e:
            errors = [f"第 {attempt} 次输出无法解析为 JSON：{e}"]
            continue
        items = _as_list(obj)
        if not items:
            errors = [f"第 {attempt} 次输出没有题目"]
            continue
        qs, errs, warns = [], [], []
        for idx, q in enumerate(items, 1):
            clean, e_, w_ = validate_question(q)
            errs.extend(f"第{idx}题：{e}" for e in e_)
            warns.extend(f"第{idx}题：{w}" for w in w_)
            if clean:
                qs.append(clean)
        # 部分题目校验失败时，保留能用的题目（不再整张图丢弃）
        rank = (len(errs), -len(qs))
        if best is None or rank < best[0]:
            best = (rank, qs, errs, warns, u or {}, raw)
        if not errs:
            break

    if best is not None:
        _, questions, errors, warnings, u_best, raw_best = best
        usage = u_best or usage
        raw_last = raw_best or raw_last

    # 重复题检测
    seen: dict[str, int] = {}
    for i, q in enumerate(questions, 1):
        k = _norm_text(q["stem"])
        if k in seen:
            warnings.append(f"第{i}题与第{seen[k]}题题干重复")
        else:
            seen[k] = i

    confidence = "high"
    if errors:
        confidence = "low"
    elif warnings or attempts > 1:
        confidence = "medium"

    # strict：再跑一次，比较归一化文本
    if strict and questions and not errors:
        try:
            raw2, u2, _ = await call_vision(b64, mime, PROMPT)
            usage2 = u2 or {}
            obj2 = _extract_json(raw2)
            items2 = _as_list(obj2)
            q2: list[dict] = []
            for q in items2:
                c, _, _ = validate_question(q)
                if c:
                    q2.append(c)
            txt1 = "".join(q["stem"] + "".join(q["options"].values()) for q in questions)
            txt2 = "".join(q["stem"] + "".join(q["options"].values()) for q in q2)
            # v1.0.2: 按完整结构比较（题型+题数+题干+选项），不再只比拼接文本——
            # 此前题型缺失、题数差异都可能拼出相同文本被放过
            def _sig(qs: list) -> list:
                return [(q.get("type", ""), _norm_text(q.get("stem", "")),
                         sorted((k, _norm_text(v)) for k, v in (q.get("options") or {}).items())) for q in qs]
            if _sig(questions) != _sig(q2):
                confidence = "low"
                warnings.append("双跑结果不一致，建议人工复核")
            usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0) + usage2.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0) + usage2.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0) + usage2.get("total_tokens", 0),
            }
        except Exception as e:
            warnings.append(f"双跑校验失败：{e}")

    elapsed_ms = int((time.time() - t0) * 1000)
    body = {
        "ok": bool(questions),   # 有可用题目即成功；部分失败的题在 errors 里列出
        "partial": bool(questions) and bool(errors),
        "questions": questions,
        "warnings": warnings,
        "confidence": confidence,
        "meta": {
            "model": ocr_active()["model"],
            "elapsed_ms": elapsed_ms,
            "attempts": attempts,
            "strict": strict,
            "usage": usage,
            "image": img_meta,
        },
    }
    if errors:
        body["errors"] = errors
        body["raw"] = raw_last[:1500]
    return body


# ---------- 路由 ----------

@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "model": ocr_active()["model"],
        "uptime_s": int(time.time() - STARTED),
        "key_configured": bool(ocr_active()["key"]),
        "max_side": MAX_SIDE,
        "strict_default": STRICT_DEFAULT,
    }


async def _handle_one(f: UploadFile, strict: bool) -> tuple[str, dict]:
    data = await f.read()
    if not data:
        return f.filename or "?", {"ok": False, "errors": ["空文件"]}
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return f.filename or "?", {"ok": False, "errors": [f"文件超过 {MAX_UPLOAD_MB}MB"]}
    try:
        body = await recognize(data, strict)
    except HTTPException as e:
        return f.filename or "?", {"ok": False, "errors": [str(e.detail)]}
    except Exception as e:
        return f.filename or "?", {"ok": False, "errors": [f"内部错误：{e}"]}
    body["file"] = f.filename
    print(
        f"[ocr] {f.filename} ok={body['ok']} conf={body['confidence']} "
        f"{body['meta']['elapsed_ms']}ms attempts={body['meta']['attempts']}",
        flush=True,
    )
    return f.filename or "?", body


@app.post("/ocr")
async def ocr(file: UploadFile = File(...), strict: bool = Form(False)):
    _, body = await _handle_one(file, strict or STRICT_DEFAULT)
    if not body.get("ok"):
        raise HTTPException(422, detail=body)
    return body


@app.post("/ocr/batch")
async def ocr_batch(files: list[UploadFile] = File(...), strict: bool = Form(False)):
    results = await asyncio.gather(*[_handle_one(f, strict or STRICT_DEFAULT) for f in files])
    items = [b for _, b in results]
    return {
        "ok": all(i.get("ok") for i in items),
        "count": len(items),
        "results": items,
        "failed": sum(1 for i in items if not i.get("ok")),
    }


@app.post("/ocr/b64")
async def ocr_b64(payload: dict):
    b64 = (payload or {}).get("image_base64") or ""
    if not b64:
        raise HTTPException(400, "缺少 image_base64")
    if "," in b64[:80]:
        b64 = b64.split(",", 1)[1]
    try:
        data = base64.b64decode(b64, validate=False)
    except Exception as e:
        raise HTTPException(400, f"base64 解码失败：{e}")
    body = await recognize(data, bool((payload or {}).get("strict", STRICT_DEFAULT)))
    if not body.get("ok"):
        raise HTTPException(422, detail=body)
    return body


@app.on_event("shutdown")
async def _shutdown():
    if _client is not None:
        await _client.aclose()
