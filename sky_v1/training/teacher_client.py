"""5-老师知识蒸馏 API 客户端 + Qwen2.5-72B 本地 fallback。

兑现规格 §6（5 老师蒸馏方案）和 §11（Fallback 到本地 Qwen2.5-72B）。

对外提供三类组件：

* :class:`TeacherAPIClient` —— 异步/同步调用 5 个老师模型 API
  (claude_opus_4_8 / gpt_5_6_sol / kimi_k3 / mimo_v2_5 / qwen_3_8)。
  每个老师从环境变量读 API key；未配置或调用失败时标记 ``unavailable``，绝不崩溃。
  使用 httpx（不可用时 fallback 到 urllib.request）+ 3 次指数退避重试。
* :class:`Qwen72BLocalTeacher` —— 本地 Qwen2.5-72B fallback teacher，
  延迟加载 transformers，不可用时返回 stub。
* :class:`DistillSetBuilder` —— 批量构建 50K 蒸馏数据集，支持断点续传。

所有外部 API 调用均被 try/except 保护；测试中不做真实网络请求。
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from ..utils.logging import get_logger

log = get_logger("training.teacher_client")

# 默认 5 老师配置（与 configs/training/teachers.yaml 保持一致）。
# 权重顺序 [1.2, 1.3, 1.4, 1.2, 1.0] 必须与 TeacherPool.teacher_weights 一致。
DEFAULT_TEACHERS: dict[str, dict[str, Any]] = {
    "claude_opus_4_8": {
        "provider": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "model_id": "claude-opus-4-8",
        "weight": 1.2,
        "endpoint": "https://api.anthropic.com/v1/messages",
    },
    "gpt_5_6_sol": {
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "model_id": "gpt-5.6-sol",
        "weight": 1.3,
        "endpoint": "https://api.openai.com/v1/chat/completions",
    },
    "kimi_k3": {
        "provider": "moonshot",
        "api_key_env": "MOONSHOT_API_KEY",
        "model_id": "kimi-k3",
        "weight": 1.4,
        "endpoint": "https://api.moonshot.cn/v1/chat/completions",
    },
    "mimo_v2_5": {
        "provider": "mimo",
        "api_key_env": "MIMO_API_KEY",
        "model_id": "mimo-v2.5-pro",
        "weight": 1.2,
        "endpoint": "https://api.mimo.xiaomi.com/v1/chat/completions",
    },
    "qwen_3_8": {
        "provider": "dashscope",
        "api_key_env": "DASHSCOPE_API_KEY",
        "model_id": "qwen-3.8-turbo",
        "weight": 1.0,
        "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    },
}

DEFAULT_FALLBACK = {
    "local_qwen72b": {
        "model_path": "Qwen/Qwen2.5-72B-Instruct",
        "device": "auto",
    }
}

DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_S = 60.0


def _http_post_json(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    """POST JSON，优先用 httpx，缺失时 fallback 到 urllib.request。

    返回解析后的 JSON dict。任何异常向上抛出（由调用方做重试与兜底）。
    """
    try:
        import httpx

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except ImportError:
        # httpx 不可用 → urllib 兜底
        import urllib.request
        import urllib.error

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={**headers, "Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
                body = r.read().decode("utf-8", errors="ignore")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            # 让上层按 HTTP 错误处理
            raise RuntimeError(f"HTTP {e.code}: {e.reason}") from e


async def _ahttp_post_json(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    """异步 POST JSON，优先 httpx.AsyncClient，缺失时退到同步 urllib（在线程中执行）。"""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except ImportError:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, _http_post_json, url, headers, payload, timeout
        )


def _build_request(
    teacher_cfg: dict[str, Any], prompt: str, max_tokens: int
) -> tuple[dict[str, str], dict[str, Any]]:
    """根据 provider 构造 headers + body（OpenAI 兼容 / Anthropic 原生）。"""
    provider = teacher_cfg["provider"]
    api_key = os.environ.get(teacher_cfg["api_key_env"], "")
    model_id = teacher_cfg["model_id"]
    endpoint = teacher_cfg["endpoint"]

    if provider == "anthropic":
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {
            "model": model_id,
            "max_tokens": int(max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }
    else:
        # OpenAI 兼容格式（openai / moonshot / mimo / dashscope）
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model_id,
            "max_tokens": int(max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }
    return headers, body


def _parse_response(
    teacher_cfg: dict[str, Any], raw: dict[str, Any], latency_ms: float
) -> dict[str, Any]:
    """把各家 API 响应统一成 {text, logits_stub, usage, latency_ms, available}。"""
    provider = teacher_cfg["provider"]
    text = ""
    usage: dict[str, Any] = {}
    try:
        if provider == "anthropic":
            content = raw.get("content") or []
            if isinstance(content, list):
                text = " ".join(
                    str(b.get("text", "")) for b in content if isinstance(b, dict)
                )
            u = raw.get("usage") or {}
            usage = {
                "prompt_tokens": u.get("input_tokens", 0),
                "completion_tokens": u.get("output_tokens", 0),
                "total_tokens": (u.get("input_tokens", 0) or 0) + (u.get("output_tokens", 0) or 0),
            }
        else:
            choices = raw.get("choices") or []
            if choices and isinstance(choices, list):
                msg = choices[0].get("message") or {}
                text = msg.get("content", "") or ""
            u = raw.get("usage") or {}
            usage = {
                "prompt_tokens": u.get("prompt_tokens", 0),
                "completion_tokens": u.get("completion_tokens", 0),
                "total_tokens": u.get("total_tokens", 0),
            }
    except Exception as e:  # 解析失败不致命
        log.warning("Failed to parse teacher response", teacher=teacher_cfg.get("provider"), exc=str(e))

    # 远程 API 不返回 logits，这里给一个 stub（与 TeacherPool.simulate_from_student 对齐）。
    logits_stub = [0.0] * 8
    return {
        "text": text,
        "logits_stub": logits_stub,
        "usage": usage,
        "latency_ms": latency_ms,
        "available": True,
    }


def _unavailable_result(reason: str, latency_ms: float = 0.0) -> dict[str, Any]:
    return {
        "text": "",
        "logits_stub": [],
        "usage": {},
        "latency_ms": latency_ms,
        "available": False,
        "reason": reason,
    }


class TeacherAPIClient:
    """5 老师 API 客户端（异步 + 同步），支持 3 次指数退避重试。"""

    TEACHERS = DEFAULT_TEACHERS
    FALLBACK = DEFAULT_FALLBACK
    MAX_RETRIES = DEFAULT_MAX_RETRIES
    TIMEOUT_S = DEFAULT_TIMEOUT_S

    def __init__(
        self,
        config_path: str | Path | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.max_retries = int(max_retries)
        self.timeout_s = float(timeout_s)
        # 允许从 yaml 覆盖默认配置；加载失败时退回默认。
        self.teachers: dict[str, dict[str, Any]] = dict(DEFAULT_TEACHERS)
        self.fallback_cfg: dict[str, Any] = dict(DEFAULT_FALLBACK)
        if config_path is not None:
            self._load_config(config_path)

    def _load_config(self, config_path: str | Path) -> None:
        try:
            from ..utils.config import load_yaml_config

            raw = load_yaml_config(config_path)
            t = raw.get("teachers")
            if isinstance(t, dict) and t:
                merged: dict[str, dict[str, Any]] = {}
                for name, spec in DEFAULT_TEACHERS.items():
                    if isinstance(t.get(name), dict):
                        merged[name] = {**spec, **t[name]}
                    else:
                        merged[name] = spec
                self.teachers = merged
            fb = raw.get("fallback")
            if isinstance(fb, dict) and fb.get("local_qwen72b"):
                self.fallback_cfg = fb
        except Exception as e:
            log.warning("Failed to load teachers config, using defaults", path=str(config_path), exc=str(e))

    # ---- 配置访问 ----
    def teacher_names(self) -> list[str]:
        return list(self.teachers.keys())

    def teacher_weights(self) -> list[float]:
        """返回与 teacher_names() 同序的权重列表，规格 §6 要求 [1.2, 1.3, 1.4, 1.2, 1.0]。"""
        return [float(self.teachers[n].get("weight", 1.0)) for n in self.teacher_names()]

    def is_configured(self, teacher_name: str) -> bool:
        cfg = self.teachers.get(teacher_name)
        if not cfg:
            return False
        key = os.environ.get(cfg["api_key_env"], "")
        return bool(key and key.strip())

    # ---- 单老师调用（同步） ----
    def call_teacher(
        self, teacher_name: str, prompt: str, max_tokens: int = 512
    ) -> dict[str, Any]:
        cfg = self.teachers.get(teacher_name)
        if cfg is None:
            return _unavailable_result(f"unknown teacher: {teacher_name}")
        if not self.is_configured(teacher_name):
            return _unavailable_result(f"api key not set: {cfg['api_key_env']}")

        headers, body = _build_request(cfg, prompt, max_tokens)
        endpoint = cfg["endpoint"]
        last_err: Exception | None = None
        start = time.perf_counter()
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = _http_post_json(endpoint, headers, body, self.timeout_s)
                latency_ms = (time.perf_counter() - start) * 1000.0
                return _parse_response(cfg, raw, latency_ms)
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < self.max_retries:
                    backoff = 0.2 * (2 ** (attempt - 1))
                    log.warning(
                        "Teacher call failed, retrying",
                        teacher=teacher_name,
                        attempt=attempt,
                        wait_s=backoff,
                        exc=str(e)[:200],
                    )
                    time.sleep(backoff)
        latency_ms = (time.perf_counter() - start) * 1000.0
        log.warning("Teacher call exhausted retries", teacher=teacher_name, exc=str(last_err)[:200])
        return _unavailable_result(f"call failed: {last_err}", latency_ms)

    # ---- 单老师调用（异步） ----
    async def acall_teacher(
        self, teacher_name: str, prompt: str, max_tokens: int = 512
    ) -> dict[str, Any]:
        cfg = self.teachers.get(teacher_name)
        if cfg is None:
            return _unavailable_result(f"unknown teacher: {teacher_name}")
        if not self.is_configured(teacher_name):
            return _unavailable_result(f"api key not set: {cfg['api_key_env']}")

        headers, body = _build_request(cfg, prompt, max_tokens)
        endpoint = cfg["endpoint"]
        last_err: Exception | None = None
        start = time.perf_counter()
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = await _ahttp_post_json(endpoint, headers, body, self.timeout_s)
                latency_ms = (time.perf_counter() - start) * 1000.0
                return _parse_response(cfg, raw, latency_ms)
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < self.max_retries:
                    backoff = 0.2 * (2 ** (attempt - 1))
                    log.warning(
                        "Async teacher call failed, retrying",
                        teacher=teacher_name,
                        attempt=attempt,
                        exc=str(e)[:200],
                    )
                    await asyncio.sleep(backoff)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return _unavailable_result(f"call failed: {last_err}", latency_ms)

    # ---- 并行调用全部老师 ----
    def call_all_teachers(
        self,
        prompt: str,
        max_tokens: int = 512,
        teachers: Iterable[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        names = list(teachers) if teachers is not None else self.teacher_names()
        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(names))) as ex:
            futures = {ex.submit(self.call_teacher, n, prompt, max_tokens): n for n in names}
            for fut in futures:
                name = futures[fut]
                try:
                    results[name] = fut.result()
                except Exception as e:  # noqa: BLE001
                    results[name] = _unavailable_result(f"executor error: {e}")
        return results

    async def acall_all_teachers(
        self,
        prompt: str,
        max_tokens: int = 512,
        teachers: Iterable[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        names = list(teachers) if teachers is not None else self.teacher_names()
        coros = [self.acall_teacher(n, prompt, max_tokens) for n in names]
        gathered = await asyncio.gather(*coros, return_exceptions=True)
        results: dict[str, dict[str, Any]] = {}
        for name, res in zip(names, gathered):
            if isinstance(res, Exception):
                results[name] = _unavailable_result(f"async error: {res}")
            else:
                results[name] = res
        return results


class Qwen72BLocalTeacher:
    """本地 Qwen2.5-72B fallback teacher（延迟加载 transformers，不可用时返回 stub）。"""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-72B-Instruct",
        device: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._load_error: str | None = None

    def _resolve_device(self) -> str:
        if self.device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return self.device

    def _try_load(self) -> bool:
        if self._loaded:
            return True
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            device = self._resolve_device()
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map=device if device != "cpu" else None,
            )
            if device == "cpu" and self._model is not None:
                self._model = self._model.to(device)
            self._loaded = self._model is not None and self._tokenizer is not None
            if not self._loaded:
                self._load_error = "model or tokenizer is None"
        except Exception as e:  # noqa: BLE001
            self._load_error = f"{type(e).__name__}: {e}"
            self._loaded = False
        return self._loaded

    def is_available(self) -> bool:
        """检查模型是否可加载（不抛异常）。"""
        try:
            return bool(self._try_load())
        except Exception:
            return False

    def generate(self, prompt: str, max_tokens: int = 512) -> dict[str, Any]:
        if not self._try_load():
            return {
                "text": "",
                "logits": [],
                "usage": {},
                "available": False,
                "reason": self._load_error or "model not loadable",
            }
        try:
            import torch

            assert self._tokenizer is not None and self._model is not None
            device = self._resolve_device()
            inputs = self._tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            with torch.no_grad():
                out = self._model.generate(
                    input_ids,
                    max_new_tokens=int(max_tokens),
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            gen_ids = out.sequences[0][input_ids.shape[1]:]
            text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
            logits: list[float] = []
            if getattr(out, "scores", None):
                try:
                    last = out.scores[-1]
                    if last is not None:
                        probs = torch.softmax(last[0].float(), dim=-1)
                        logits = probs.topk(min(8, probs.shape[-1])).values.tolist()
                except Exception:
                    logits = []
            usage = {
                "prompt_tokens": int(input_ids.shape[1]),
                "completion_tokens": int(gen_ids.shape[0]),
                "total_tokens": int(input_ids.shape[1] + gen_ids.shape[0]),
            }
            return {"text": text, "logits": logits, "usage": usage, "available": True}
        except Exception as e:  # noqa: BLE001
            return {
                "text": "",
                "logits": [],
                "usage": {},
                "available": False,
                "reason": f"generate failed: {e}",
            }


def _load_prompts(prompts_file: str | Path, max_samples: int | None = None) -> list[str]:
    """从 JSON（list[str] 或 list[{prompt}]）或 CSV（'prompt' 列）加载 prompt 列表。"""
    p = Path(prompts_file)
    if not p.exists():
        raise FileNotFoundError(f"prompts file not found: {p}")
    prompts: list[str] = []
    suffix = p.suffix.lower()
    if suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    prompts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
                    prompts.append(item["prompt"])
        elif isinstance(data, dict) and isinstance(data.get("prompts"), list):
            for item in data["prompts"]:
                if isinstance(item, str):
                    prompts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
                    prompts.append(item["prompt"])
    elif suffix == ".csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if isinstance(row, dict) and isinstance(row.get("prompt"), str):
                    prompts.append(row["prompt"])
    else:
        # 兜底：按行读
        with p.open("r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    if max_samples is not None and max_samples > 0:
        prompts = prompts[:max_samples]
    return prompts


class DistillSetBuilder:
    """构建 50K 蒸馏数据集（5 老师 + Qwen fallback），支持断点续传。

    每条记录::

        {
            "prompt": str,
            "teacher_responses": {teacher_name: {text, usage, latency_ms, available}},
            "teacher_logits": {teacher_name: [float, ...]},
            "weights": [1.2, 1.3, 1.4, 1.2, 1.0],
        }
    """

    def __init__(
        self,
        teacher_client: TeacherAPIClient | None = None,
        qwen_fallback: Qwen72BLocalTeacher | None = None,
        config_path: str | Path | None = None,
        use_fallback: bool = True,
    ) -> None:
        self.client = teacher_client if teacher_client is not None else TeacherAPIClient(config_path)
        self.qwen = qwen_fallback if qwen_fallback is not None else Qwen72BLocalTeacher()
        self.use_fallback = bool(use_fallback)
        self.records: list[dict[str, Any]] = []

    def _completed_prompts(self) -> set[str]:
        return {r.get("prompt", "") for r in self.records if isinstance(r, dict)}

    def build_from_prompts(
        self,
        prompts_file: str | Path,
        output_path: str | Path,
        max_samples: int | None = None,
        teachers: Iterable[str] | None = None,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        """批量调 5 老师（+Qwen fallback），保存为 .pt 文件，支持断点续传。"""
        import torch

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 断点续传：已存在则加载已完成记录
        if out_path.exists():
            try:
                prev = torch.load(out_path, weights_only=False)
                if isinstance(prev, list):
                    self.records = prev
                log.info("Resuming distillset build", completed=len(self.records), path=str(out_path))
            except Exception as e:
                log.warning("Failed to load existing distillset, starting fresh", exc=str(e))
                self.records = []

        prompts = _load_prompts(prompts_file, max_samples=max_samples)
        names = list(teachers) if teachers is not None else self.client.teacher_names()
        weights = self.client.teacher_weights()
        completed = self._completed_prompts()

        total = len(prompts)
        processed = 0
        skipped = 0
        failed = 0
        for i, prompt in enumerate(prompts, 1):
            if prompt in completed:
                skipped += 1
                continue
            teacher_results = self.client.call_all_teachers(
                prompt, max_tokens=max_tokens, teachers=names
            )
            teacher_responses: dict[str, Any] = {}
            teacher_logits: dict[str, Any] = {}
            any_available = False
            for name in names:
                r = teacher_results.get(name, {})
                teacher_responses[name] = {
                    "text": r.get("text", ""),
                    "usage": r.get("usage", {}),
                    "latency_ms": r.get("latency_ms", 0.0),
                    "available": r.get("available", False),
                }
                teacher_logits[name] = r.get("logits_stub", [])
                if r.get("available"):
                    any_available = True

            # 5 老师全部 unavailable 时 fallback 到本地 Qwen72B（规格 §11）
            if not any_available and self.use_fallback:
                try:
                    fb = self.qwen.generate(prompt, max_tokens=max_tokens)
                    teacher_responses["local_qwen72b"] = {
                        "text": fb.get("text", ""),
                        "usage": fb.get("usage", {}),
                        "latency_ms": 0.0,
                        "available": fb.get("available", False),
                    }
                    teacher_logits["local_qwen72b"] = fb.get("logits", [])
                except Exception as e:
                    log.warning("Qwen72B fallback failed", exc=str(e)[:200])
                    failed += 1

            record = {
                "prompt": prompt,
                "teacher_responses": teacher_responses,
                "teacher_logits": teacher_logits,
                "weights": weights,
            }
            self.records.append(record)
            processed += 1

            if i % 50 == 0 or i == total:
                log.info(
                    "Distillset progress",
                    done=i,
                    total=total,
                    processed=processed,
                    skipped=skipped,
                )
                # 周期性落盘，便于断点续传
                try:
                    torch.save(self.records, out_path)
                except Exception as e:
                    log.warning("Intermediate save failed", exc=str(e)[:200])

        # 最终落盘
        try:
            torch.save(self.records, out_path)
        except Exception as e:
            log.error("Final save failed", exc=str(e)[:200])

        stats = {
            "total_prompts": total,
            "processed": processed,
            "skipped_resume": skipped,
            "failed": failed,
            "records": len(self.records),
            "output_path": str(out_path),
        }
        log.info("Distillset build complete", **stats)
        return stats

    @classmethod
    def from_precomputed(cls, path: str | Path) -> "DistillSetBuilder":
        """加载预计算的蒸馏数据集。空文件/损坏文件不崩溃，返回空 records 的 builder。"""
        import torch

        builder = cls(teacher_client=TeacherAPIClient())
        p = Path(path)
        if not p.exists():
            builder.records = []
            return builder
        try:
            data = torch.load(p, weights_only=False)
            if isinstance(data, list):
                builder.records = [r for r in data if isinstance(r, dict)]
            else:
                builder.records = []
        except Exception as e:
            log.warning("from_precomputed: failed to load, returning empty", path=str(p), exc=str(e)[:200])
            builder.records = []
        return builder
