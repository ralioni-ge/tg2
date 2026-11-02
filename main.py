# =====================================================================
#  main.py  --  Instagram Reel -> Telegram            REV 3 (no ffmpeg)
#
#  1) صوت دست‌نخورده به Gemini Files API می‌رود: بدون WAV، بدون فشرده‌سازی،
#     بدون base64 و بدون inlineData.
#  2) ویدیو با آدرس به تلگرام داده می‌شود (<=20MB). اگر نشود، یک‌بار دانلود و
#     آپلود می‌شود (<=50MB). بالاتر از آن به کاربر اطلاع داده می‌شود.
#  3) تنها مدل جمینای: gemini-3.5-flash-lite  (بدون زنجیره‌ی fallback)
#
#  متغیرهای محیطی لازم:  BOT_TOKEN , GEMINI_KEYS
#  متغیرهای اختیاری:    DEBUG_CHAT_ID , TRANSLATE_TO , MAX_CONCURRENT_JOBS ,
#                       ALLOW_URL_SEND , URL_LIMIT_MB , UPLOAD_LIMIT_MB ,
#                       AUDIO_CHUNK_MIN , THINKING_MINIMAL
# =====================================================================
from __future__ import annotations

import asyncio
import datetime as dt
import glob
import html
import json
import os
import re
import sys
import traceback
import uuid
import zoneinfo
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import uvicorn
import yt_dlp
from fastapi import FastAPI

# ------------------------------------------------------------------ #
#  0) پیکربندی                                                       #
# ------------------------------------------------------------------ #
MB = 1024 * 1024

def _keys(raw: str) -> List[str]:
    return [k.strip() for k in (raw or "").split(",") if k.strip()]

def _int(name: str, default: int) -> int:
    try:
        return int((os.getenv(name) or "").strip() or default)
    except (TypeError, ValueError):
        return default

def _flag(name: str, default: str = "1") -> bool:
    return (os.getenv(name) or default).strip().lower() not in ("0", "false", "no")

CFG: Dict[str, Any] = {
    "BOT_TOKEN":       (os.getenv("BOT_TOKEN") or "").strip(),
    "GEMINI_KEYS":     _keys(os.getenv("GEMINI_KEYS", "")),
    "GEMINI_MODEL":    (os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite").strip(),
    "TRANSLATE_TO":    (os.getenv("TRANSLATE_TO") or "fa").strip().lower(),
    "DEBUG_CHAT_ID":   (os.getenv("DEBUG_CHAT_ID") or "").strip(),
    "URL_LIMIT":       _int("URL_LIMIT_MB", 20) * MB,
    "UPLOAD_LIMIT":    _int("UPLOAD_LIMIT_MB", 50) * MB,
    "JOBS":            max(1, _int("MAX_CONCURRENT_JOBS", 2)),
    "AUDIO_CHUNK_MIN": _int("AUDIO_CHUNK_MIN", 0),
    "ALLOW_URL_SEND":  _flag("ALLOW_URL_SEND", "1"),
    "THINKING_MIN":    _flag("THINKING_MINIMAL", "1"),
    "TMP":             os.getenv("TMPDIR") or "/tmp",
}

CAPTION_MAX = 900      # حاشیه‌ی امن زیر سقف 1024 کاراکتری کپشن تلگرام
MSG_MAX     = 3800     # حاشیه‌ی امن زیر سقف 4096 کاراکتری پیام
ZWNJ        = "\u200c"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

def TG() -> str:
    return "https://api.telegram.org/bot" + CFG["BOT_TOKEN"]

def log(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)

async def to_thread(fn: Callable, *a: Any, **kw: Any) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*a, **kw))

# ------------------------------------------------------------------ #
#  1) لاگر دیباگ (اختیاری)                                           #
# ------------------------------------------------------------------ #
class DebugLogger:
    def __init__(self, client: httpx.AsyncClient, chat_id: str,
                 kind: str, origin_id: int):
        self.client = client
        self.chat_id = (chat_id or "").strip()
        self.on = bool(self.chat_id)
        self.kind = kind
        self.origin = origin_id
        self.t0 = dt.datetime.now(dt.timezone.utc)
        self.lines: List[str] = []
        self.mid: Optional[int] = None
        self._run_idx = -1
        self._run_name = ""
        self._run_start = self.t0

    def _el(self) -> str:
        return "%.1fs" % (dt.datetime.now(dt.timezone.utc) - self.t0).total_seconds()

    async def _flush(self) -> None:
        if not self.on:
            return
        body = "\n".join(self.lines)
        if len(body) > 3400:
            body = "...\n" + body[-3400:]
        text = "🔧 %s · msg %s\n\n%s" % (self.kind, self.origin, body)
        try:
            if self.mid:
                await self.client.post(TG() + "/editMessageText", json={
                    "chat_id": self.chat_id, "message_id": self.mid,
                    "text": text, "parse_mode": "HTML"}, timeout=8.0)
            else:
                r = await self.client.post(TG() + "/sendMessage", json={
                    "chat_id": self.chat_id, "text": "<pre>" +
                    html.escape(text) + "</pre>", "parse_mode": "HTML"}, timeout=8.0)
                if r.status_code == 200:
                    self.mid = r.json().get("result", {}).get("message_id")
        except Exception:
            pass

    def _close(self, mark: str = "✓") -> None:
        if self._run_idx >= 0:
            took = (dt.datetime.now(dt.timezone.utc) - self._run_start).total_seconds()
            self.lines[self._run_idx] = "%s %s — %.1fs (at %s)" % (
                mark, self._run_name, took, self._el())
            self._run_idx = -1

    async def step(self, name: str) -> None:
        self._close()
        self._run_name = name
        self._run_start = dt.datetime.now(dt.timezone.utc)
        self.lines.append("▶ %s … (from %s)" % (name, self._el()))
        self._run_idx = len(self.lines) - 1
        await self._flush()

    async def note(self, text: str) -> None:
        self.lines.append("• " + text)
        await self._flush()

    async def err(self, label: str, e: Any) -> None:
        msg = str(getattr(e, "message", e)).replace("\n", " ")[:280]
        self.lines.append("✗ [%s] %s: %s" % (self._el(), label, msg))
        await self._flush()

    async def end(self, ok: bool) -> None:
        self._close("✓" if ok else "✗")
        self.lines.append("🏁 %s in %s" % (
            "finished" if ok else "stopped", self._el()))
        await self._flush()

    async def fatal_end(self, e: Any) -> None:
        self._close("✗")
        self.lines.append("✗ [%s] FATAL: %s" % (self._el(), str(e)[:300]))
        await self._flush()

# ------------------------------------------------------------------ #
#  2) Gemini:  Files API + تنها مدل gemini-3.5-flash-lite            #
# ------------------------------------------------------------------ #
GEM_BASE   = "https://generativelanguage.googleapis.com/v1beta"
GEM_UPLOAD = "https://generativelanguage.googleapis.com/upload/v1beta/files"
GEM_FILE_MAX = 2 * 1024 ** 3        # 2GB سقف هر فایل در Files API
GEM_OUT_MAX  = 65536                # سقف خروجی gemini-3.5-flash-lite

class GeminiError(RuntimeError):
    pass

_key_lock = asyncio.Lock()
_key_idx = 0

async def pick_key() -> str:
    global _key_idx
    keys = CFG["GEMINI_KEYS"]
    if not keys:
        raise GeminiError("GEMINI_KEYS در متغیرهای محیطی تنظیم نشده است")
    async with _key_lock:
        k = keys[_key_idx % len(keys)]
        _key_idx += 1
    return k

class GeminiLite:
    # یک نمونه = یک کلید. فایل به پروژه‌ی همان کلیدی تعلق دارد که آپلودش
    # کرده؛ پس کلید در سازنده گرفته می‌شود و تا پایان عمر شیء ثابت می‌ماند.

    def __init__(self, client: httpx.AsyncClient, key: str,
                 dbg: Optional[DebugLogger] = None):
        self.client = client
        self.key = key
        self.dbg = dbg
        self.names: List[str] = []

    # --------------------------- آپلود --------------------------- #
    async def upload(self, path: str, mime: str,
                     display_name: Optional[str] = None,
                     timeout: float = 900.0) -> Dict[str, Any]:
        size = os.path.getsize(path)
        if size > GEM_FILE_MAX:
            raise GeminiError("حجم فایل %.2fGB از سقف 2GB بیشتر است" % (size / 2**30))
        dn = display_name or os.path.basename(path)
        with open(path, "rb") as fh:
            r = await self.client.post(
                GEM_UPLOAD, params={"key": self.key},
                headers={"x-goog-upload-protocol": "raw",
                         "x-goog-upload-header-content-length": str(size)},
                data={"metadata": json.dumps({"display_name": dn})},
                files={"file": (dn, fh, mime)}, timeout=timeout)
        if r.status_code != 200:
            raise GeminiError("upload HTTP %s: %s" % (r.status_code, r.text[:300]))
        meta = r.json()["file"]
        self.names.append(meta["name"])
        meta = await self.wait_active(meta)
        if self.dbg:
            await self.dbg.note("files ↑ %s · %.1fMB · %s" % (
                meta["name"], size / MB, mime))
        return meta

    async def wait_active(self, meta: Dict[str, Any], tries: int = 40,
                          delay: float = 2.0) -> Dict[str, Any]:
        name = meta["name"]
        for i in range(tries):
            state = (meta.get("state") or "ACTIVE").upper()
            if state == "ACTIVE":
                return meta
            if state == "FAILED":
                raise GeminiError("%s پردازش نشد (FAILED)" % name)
            await asyncio.sleep(min(delay * (i + 1), 5.0))
            r = await self.client.get("%s/%s" % (GEM_BASE, name),
                                      params={"key": self.key}, timeout=30)
            if r.status_code == 200:
                meta = r.json()
        raise GeminiError("%s هرگز ACTIVE نشد" % name)

    # --------------------------- حذف --------------------------- #
    async def delete(self, name: str) -> None:
        try:
            await self.client.delete("%s/%s" % (GEM_BASE, name),
                                     params={"key": self.key}, timeout=20)
        except Exception:
            pass

    async def cleanup(self) -> None:
        while self.names:
            await self.delete(self.names.pop())

    # ------------------------ تولید محتوا ------------------------ #
    async def generate(self, parts: List[Dict[str, Any]], system: str = "",
                       max_tokens: int = GEM_OUT_MAX, temperature: float = 0.2,
                       label: str = "gemini", retries: int = 4
                       ) -> Tuple[str, str]:
        url = "%s/models/%s:generateContent" % (GEM_BASE, CFG["GEMINI_MODEL"])
        cfg: Dict[str, Any] = {
            "maxOutputTokens": min(max_tokens, GEM_OUT_MAX),
            "temperature": temperature,
        }
        body: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": cfg,
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        use_thinking = CFG["THINKING_MIN"]
        last: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            payload = dict(body)
            if use_thinking:
                payload["generationConfig"] = dict(
                    cfg, thinkingConfig={"thinkingLevel": "minimal"})
            try:
                r = await self.client.post(url, params={"key": self.key},
                                           json=payload, timeout=300)
            except httpx.HTTPError as e:
                last = e
                await asyncio.sleep(1.5 * attempt)
                continue

            # اگر این مدل/کلید thinkingConfig را نپذیرفت، یک بار بدون آن بفرست
            if r.status_code == 400 and use_thinking and "thinking" in r.text.lower():
                use_thinking = False
                continue
            if r.status_code == 429:
                last = GeminiError("429 محدودیت نرخ")
                if self.dbg:
                    await self.dbg.err(label, last)
                await asyncio.sleep(2.0 * attempt)
                continue
            if r.status_code == 404:
                raise GeminiError("مدل در دسترس نیست: %s" % CFG["GEMINI_MODEL"])
            if r.status_code != 200:
                raise GeminiError("%s HTTP %s: %s" % (label, r.status_code, r.text[:300]))

            data = r.json()
            cand = (data.get("candidates") or [{}])[0]
            text = "".join(p.get("text", "")
                           for p in cand.get("content", {}).get("parts", [])).strip()
            finish = (cand.get("finishReason")
                      or data.get("promptFeedback", {}).get("blockReason") or "ok")
            if not text:
                raise GeminiError("%s: خروجی خالی (%s)" % (label, finish))
            if finish == "MAX_TOKENS" and self.dbg:
                await self.dbg.note("⚠ خروجی در %d توکن بریده شد" % cfg["maxOutputTokens"])
            return text, finish
        raise GeminiError("%s: %d تلاش ناموفق (%s)" % (label, retries, last))

    # ------------------------ ترنسکریپت ------------------------ #
    async def transcribe(self, path: str, mime: str) -> str:
        meta = await self.upload(path, mime)
        try:
            parts = [
                {"text": TRANSCRIBE_PROMPT},
                {"fileData": {"fileUri": meta["uri"],
                              "mimeType": meta.get("mimeType") or mime}},
            ]
            text, _ = await self.generate(parts, system=TRANSCRIBE_SYS,
                                          label="transcribe")
            clean = text.strip()
            return "" if clean.upper().startswith("NO_SPEECH") else clean
        finally:
            await self.cleanup()

    async def text(self, prompt: str, system: str = "", max_tokens: int = 4096,
                   temperature: float = 0.2, label: str = "text") -> str:
        out, _ = await self.generate([{"text": prompt}], system=system,
                                     max_tokens=max_tokens,
                                     temperature=temperature, label=label)
        return out

TRANSCRIBE_SYS = "You are a professional audio transcriber."
TRANSCRIBE_PROMPT = (
    "Listen to this media and transcribe ALL spoken words verbatim in their "
    "original language. Do not summarize, translate, or paraphrase. Keep "
    "paragraph breaks at natural pauses. Output ONLY the transcribed text. "
    "If there is no speech at all, reply exactly: NO_SPEECH"
)

# ------------------------------------------------------------------ #
#  3) مدیا: probe بدون دانلود، سنجش آدرس، دانلود فقط صوت             #
# ------------------------------------------------------------------ #
MIME_BY_EXT = {
    ".m4a": "audio/mp4", ".mp4": "video/mp4", ".mov": "video/mp4",
    ".aac": "audio/aac", ".mp3": "audio/mp3", ".ogg": "audio/ogg",
    ".opus": "audio/ogg", ".webm": "audio/webm", ".wav": "audio/wav",
    ".flac": "audio/flac", ".mkv": "video/x-matroska",
}

def mime_for(path: str) -> str:
    return MIME_BY_EXT.get(os.path.splitext(path)[1].lower(), "audio/mp4")

def normalize_mime(ct: str) -> str:
    ct = (ct or "").split(";")[0].strip().lower()
    if not ct:
        return ""
    if ct in ("video/quicktime", "video/x-m4v", "application/octet-stream"):
        return "video/mp4"
    return ct

def _size(f: Dict[str, Any]) -> int:
    try:
        return int(f.get("filesize") or f.get("filesize_approx") or 0)
    except (TypeError, ValueError):
        return 0

def _has_video(f: Dict[str, Any]) -> bool:
    return f.get("vcodec") not in (None, "none")

def _has_audio(f: Dict[str, Any]) -> bool:
    return f.get("acodec") not in (None, "none")

def probe(url: str) -> Dict[str, Any]:
    # فقط metadata — هیچ ویدیویی دانلود نمی‌شود
    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "noplaylist": True, "http_headers": {"User-Agent": UA}}
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=False)
    if info.get("entries"):
        info = info["entries"][0]

    fmts = [f for f in (info.get("formats") or []) if f.get("url")]
    prog = sorted([f for f in fmts if _has_video(f) and _has_audio(f)], key=_size)
    auds = sorted([f for f in fmts if not _has_video(f) and _has_audio(f)],
                  key=_size, reverse=True)

    pick = None
    for f in prog:
        if 0 < _size(f) <= CFG["URL_LIMIT"]:
            pick = f
            break
    if pick is None:
        for f in reversed(prog):
            if 0 < _size(f) <= CFG["UPLOAD_LIMIT"]:
                pick = f
                break
    if pick is None and prog:
        pick = prog[-1]

    best_audio = auds[0] if auds else None
    return {
        "id":          info.get("id"),
        "duration":    int(info.get("duration") or 0),
        "width":       int(info.get("width") or 0),
        "height":      int(info.get("height") or 0),
        "uploader":    info.get("uploader") or info.get("uploader_id") or "instagram",
        "uploader_id": info.get("uploader_id") or "",
        "caption":     (info.get("description") or info.get("title") or "").strip(),
        "video_url":   (pick or {}).get("url"),
        "video_size":  _size(pick or {}),
        "video_ext":   (pick or {}).get("ext", "mp4"),
        "audio_url":   (best_audio or {}).get("url"),
        "audio_ext":   (best_audio or {}).get("ext", "m4a"),
        "audio_size":  _size(best_audio or {}),
    }

async def remote_stat(client: httpx.AsyncClient,
                      url: str) -> Tuple[str, int, int]:
    # MIME و اندازه‌ی واقعی روی CDN — تلگرام به MIME درست نیاز دارد
    for headers in ({"User-Agent": UA},
                    {"User-Agent": UA, "Range": "bytes=0-0"}):
        try:
            r = await client.head(url, headers=headers, timeout=20,
                                  follow_redirects=True)
            if r.status_code < 400:
                ct = (r.headers.get("content-type") or "").split(";")[0].strip()
                cl = r.headers.get("content-length") or ""
                if ct or cl:
                    return ct.lower(), int(cl or 0), r.status_code
        except httpx.HTTPError:
            pass
    try:
        r = await client.get(url, headers={"User-Agent": UA, "Range": "bytes=0-0"},
                             timeout=25, follow_redirects=True)
        rng = r.headers.get("content-range") or ""
        cl = int(rng.split("/")[-1]) if "/" in rng else int(
            r.headers.get("content-length") or 0)
        ct = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
        return ct, cl, r.status_code
    except httpx.HTTPError:
        return "", 0, 0

# فرمت‌هایی که هم ویدیو هم صوت دارند → yt-dlp به ffmpeg برای merge نیاز ندارد
FMT_VIDEO = ("best[vcodec!=none][acodec!=none][ext=mp4]/"
             "best[vcodec!=none][acodec!=none]/best")
FMT_AUDIO = "bestaudio/best"

async def fetch_media(url: str, out_base: str, want: str = "audio"
                      ) -> Optional[str]:
    opts = {
        "outtmpl": out_base + ".%(ext)s",
        "format": FMT_AUDIO if want == "audio" else FMT_VIDEO,
        "quiet": True, "no_warnings": True, "noplaylist": True,
        "http_headers": {"User-Agent": UA},
    }
    info = await to_thread(lambda: yt_dlp.YoutubeDL(opts).extract_info(
        url, download=True))
    try:
        return info["requested_downloads"][0]["filepath"]
    except (KeyError, IndexError, TypeError):
        cands = [p for p in glob.glob(out_base + ".*") if not p.endswith(".part")]
        return max(cands, key=os.path.getsize) if cands else None

async def fetch_audio_chunks(url: str, out_base: str, total_sec: int,
                             chunk_min: int) -> List[str]:
    # برای صوت‌های خیلی بلند (لایو/پادکست) — بدون ffmpeg، با download_ranges
    try:
        from yt_dlp.utils import download_range_func
    except Exception:
        return []
    step = chunk_min * 60
    out: List[str] = []
    for i, start in enumerate(range(0, max(total_sec, step), step)):
        end = min(start + step, total_sec)
        opts = {
            "outtmpl": "%s_%03d.%%(ext)s" % (out_base, i),
            "format": FMT_AUDIO,
            "download_ranges": download_range_func(None, [(start, end)]),
            "quiet": True, "no_warnings": True, "noplaylist": True,
            "http_headers": {"User-Agent": UA},
        }
        try:
            info = await to_thread(lambda: yt_dlp.YoutubeDL(opts).extract_info(
                url, download=True))
            p = info["requested_downloads"][0]["filepath"]
            if p and os.path.exists(p):
                out.append(p)
        except Exception as e:
            log("chunk %d failed: %s" % (i, e))
    return out

# ------------------------------------------------------------------ #
#  4) تلگرام: انتخاب مسیر بر اساس اندازه، قبل از هر تلاش            #
# ------------------------------------------------------------------ #
def decide(size: int, mime: str) -> str:
    if size <= 0:
        return "unknown"
    if mime.startswith("video/") and size <= CFG["URL_LIMIT"]:
        return "url"
    if size <= CFG["UPLOAD_LIMIT"]:
        return "upload"
    return "impossible"

async def send_video(client: httpx.AsyncClient, chat_id: int, *,
                     video_url: Optional[str], local_path: Optional[str],
                     size: int, mime: str, caption: str = "",
                     duration: int = 0, width: int = 0, height: int = 0,
                     downloader: Optional[Callable] = None,
                     dbg: Optional[DebugLogger] = None
                     ) -> Tuple[Optional[str], Optional[Dict[str, Any]],
                                Optional[str]]:
    cap = caption[:CAPTION_MAX] if caption else ""
    attrs: Dict[str, int] = {}
    for k, v in (("duration", duration), ("width", width), ("height", height)):
        if v and int(v) > 0:
            attrs[k] = int(v)
    errs: List[str] = []

    # ---- مسیر ۱: تلگرام خودش از اینترنت برمی‌دارد (صفر بایت روی سرور) ----
    if (CFG["ALLOW_URL_SEND"] and video_url and mime.startswith("video/")
            and (size <= 0 or size <= CFG["URL_LIMIT"])):
        payload: Dict[str, Any] = {"chat_id": str(chat_id), "video": video_url}
        payload.update(attrs)
        if cap:
            payload.update({"caption": cap, "parse_mode": "HTML"})
        try:
            r = await client.post(TG() + "/sendVideo", json=payload, timeout=120)
            if r.status_code == 200 and r.json().get("ok"):
                if dbg:
                    await dbg.note("sendVideo(url) · %.1fMB · صفر بایت روی سرور"
                                   % (size / MB))
                return "url", r.json()["result"], None
            errs.append("url/%s:%s" % (r.status_code, r.text[:150]))
        except httpx.HTTPError as e:
            errs.append("url/exc:%s" % e)
        if dbg:
            await dbg.err("sendVideo(url)", errs[-1])

    # ---- مسیر ۲: یک‌بار دانلود، یک‌بار آپلود multipart ----
    path = local_path
    if not path and downloader is not None:
        if dbg:
            await dbg.step("دانلود ویدیو (مسیر آپلود)")
        path = await downloader()
    if path and os.path.exists(path):
        real = size or os.path.getsize(path)
        if real <= CFG["UPLOAD_LIMIT"]:
            data: Dict[str, str] = {"chat_id": str(chat_id),
                                    "supports_streaming": "true"}
            data.update({k: str(v) for k, v in attrs.items()})
            if cap:
                data.update({"caption": cap, "parse_mode": "HTML"})
            with open(path, "rb") as fh:
                try:
                    r = await client.post(
                        TG() + "/sendVideo", data=data, timeout=900,
                        files={"video": (os.path.basename(path), fh,
                                         mime or "video/mp4")})
                    if r.status_code == 200 and r.json().get("ok"):
                        if dbg:
                            await dbg.note("sendVideo(upload) · %.1fMB"
                                           % (os.path.getsize(path) / MB))
                        return "upload", r.json()["result"], None
                    errs.append("upload/%s:%s" % (r.status_code, r.text[:150]))
                except httpx.HTTPError as e:
                    errs.append("upload/exc:%s" % e)
        else:
            errs.append("upload/skip:%.1fMB>%dMB" % (real / MB,
                                                     CFG["UPLOAD_LIMIT"] / MB))

    return None, None, " | ".join(errs) or "no viable route"

async def send_message(client: httpx.AsyncClient, chat_id: int, text: str,
                       reply_to: Optional[int] = None) -> None:
    payload: Dict[str, Any] = {"chat_id": str(chat_id), "text": text,
                               "parse_mode": "HTML",
                               "link_preview_options": {"is_disabled": True}}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    r = await client.post(TG() + "/sendMessage", json=payload, timeout=30)
    if r.status_code == 400 and "link_preview" in r.text:
        payload.pop("link_preview_options", None)
        await client.post(TG() + "/sendMessage", json=payload, timeout=30)

async def send_chunks(client: httpx.AsyncClient, chat_id: int,
                      chunks: List[str], reply_to: Optional[int] = None) -> None:
    for ch in chunks:
        try:
            await send_message(client, chat_id, ch, reply_to)
        except httpx.HTTPError as e:
            log("sendMessage failed:", e)

# ------------------------------------------------------------------ #
#  5) متن: تشخیص زبان، نرمال‌سازی، تقویم، قالب‌بندی                   #
# ------------------------------------------------------------------ #
def detect_lang(text: str) -> str:
    fa = len(re.findall(r"[\u0600-\u06FF]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    return "fa" if fa > en else "en"

def is_persian(l: str) -> bool:
    return str(l or "").lower() in ("fa", "persian", "farsi")

def lang_name(code: str) -> str:
    c = str(code or "").strip().lower()
    if not c:
        return "the source language"
    if is_persian(c):
        return "Persian (Farsi)"
    return {"en": "English", "ar": "Arabic", "fr": "French", "de": "German",
            "es": "Spanish", "ru": "Russian", "tr": "Turkish",
            "it": "Italian"}.get(c, c)

def normalize_fa(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\u064a", "\u06cc").replace("\u0643", "\u06a9")
    t = re.sub(r"(^|\s)(ن?می) (?=[\u0600-\u06FF])", r"\g<1>\g<2>" + ZWNJ, t)
    t = re.sub(r"([\u0600-\u06FF]) ها(?=$|[\s.,،؟?!؛:])", r"\g<1>" + ZWNJ + "ها", t)
    return t.strip()

def g2j(gy: int, gm: int, gd: int) -> Tuple[int, int, int]:
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    jy = 979 if gy > 1600 else 0
    gy -= 1600 if gy > 1600 else 621
    gy2 = gy + 1 if gm > 2 else gy
    days = (365 * gy) + ((gy2 + 3) // 4) - ((gy2 + 99) // 100) + \
           ((gy2 + 399) // 400) - 80 + gd + g_d_m[gm - 1]
    jy += 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    jm = 1 + (days // 31) if days < 186 else 7 + ((days - 186) // 30)
    jd = 1 + (days % 31 if days < 186 else (days - 186) % 30)
    return jy, jm, jd

def tehran_tz() -> dt.tzinfo:
    try:
        return zoneinfo.ZoneInfo("Asia/Tehran")
    except Exception:
        return dt.timezone(dt.timedelta(hours=3, minutes=30), "IRST")

def fmt_dates(ts: float) -> Dict[str, str]:
    d = dt.datetime.fromtimestamp(ts, dt.timezone.utc)
    t = d.astimezone(tehran_tz())
    jy, jm, jd = g2j(t.year, t.month, t.day)
    return {"iso": d.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "shamsi": "%04d-%02d-%02d %s" % (jy, jm, jd, t.strftime("%H:%M:%S"))}

# ---------------------------- HTML helpers ---------------------------- #
def esc(s: Any) -> str:
    return html.escape(str(s or ""), quote=False)

def esc_attr(s: Any) -> str:
    return html.escape(str(s or ""), quote=True)

def md_to_html(s: str) -> str:
    s = esc(s)
    s = re.sub(r"^[ \t]*#{1,6}[ \t]*(.+)$", r"<b>\1</b>", s, flags=re.MULTILINE)
    s = re.sub(r"^[ \t]*[-*•][ \t]+", "• ", s, flags=re.MULTILINE)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s, flags=re.DOTALL)
    s = s.replace("**", "")
    return re.sub(r"<(b|i|u|s|code)>\s*</\1>", "", s)   # حذف entity خالی

def split_at_word(text: str, max_len: int) -> Tuple[str, str]:
    if len(text) <= max_len:
        return text, ""
    p = text.rfind("\n\n", 0, max_len)
    if p >= max_len * 0.4:
        return text[:p].strip(), text[p + 2:].strip()
    m = list(re.finditer(r"[.!?؟\n]\s+", text[:max_len]))
    if m and m[-1].start() >= max_len * 0.4:
        cut = m[-1].end()
        return text[:cut].strip(), text[cut:].strip()
    sp = text.rfind(" ", 0, max_len)
    if sp > 0:
        return text[:sp].strip(), text[sp + 1:].strip()
    return text[:max_len].strip(), text[max_len:].strip()

def block(label: str, md: str) -> str:
    h = md_to_html(md) if md else ""
    if not label:
        return h
    return "<b>%s</b>\n<blockquote>%s</blockquote>" % (esc(label), h)

def build_containers(sections: List[Dict[str, Any]],
                     first_is_caption: bool) -> List[str]:
    out: List[str] = []
    cur: List[str] = []
    cur_len = 0
    limit = CAPTION_MAX if first_is_caption else MSG_MAX

    def flush() -> None:
        nonlocal cur, cur_len, limit
        if cur:
            out.append("\n\n".join(cur))
            cur, cur_len = [], 0
        limit = MSG_MAX

    for sec in sections:
        text = (sec.get("text") or "").strip()
        if not text:
            continue
        if sec.get("raw"):
            if cur_len + len(text) > limit and cur:
                flush()
            cur.append(text)
            cur_len += len(text)
            continue
        label = sec.get("label", "")
        rest = text
        while rest:
            sep = 2 if cur else 0
            label_len = len(label) + 44 if label else 0
            avail = max(160, limit - cur_len - sep - label_len)
            if len(rest) <= avail:
                b = block(label, rest)
                cur.append(b)
                cur_len += sep + len(b)
                rest = ""
            else:
                chunk, rest = split_at_word(rest, avail)
                b = block(label, chunk)
                cur.append(b)
                cur_len += sep + len(b)
                if label and not label.endswith("(ادامه)"):
                    label += " (ادامه)"
                flush()
    if cur:
        out.append("\n\n".join(cur))
    return out

def build_footer(meta: Dict[str, Any], url: str, lang: str, target: str,
                 translated: bool, total: float) -> str:
    d = fmt_dates(dt.datetime.now(dt.timezone.utc).timestamp())
    dur = int(meta.get("duration") or 0)
    dur_s = " | %02d:%02d" % (dur // 60, dur % 60) if dur else ""
    uid = meta.get("uploader_id") or ""
    name = meta.get("uploader") or uid or "Instagram User"
    user_url = "https://instagram.com/" + uid if uid else url
    tr = " (translated to %s)" % esc(target) if translated else ""
    lines = ["------",
             "<b>Message link:</b> <a href=\"%s\">Open Instagram Link</a>" % esc_attr(url),
             "<b>Type:</b> Video%s | Instagram" % dur_s,
             "<b>Language:</b> %s%s" % (esc(lang), tr),
             "<b>Sender:</b> <a href=\"%s\">%s</a>" % (esc_attr(user_url), esc(name)),
             "<b>TimeStamp Date:</b> %s" % d["iso"],
             "<b>Time Tehran:</b> %s" % d["shamsi"],
             "<b>Processing:</b> %.1fs" % total]
    return "\n" + "\n".join(lines)

# ------------------------------------------------------------------ #
#  6) کارهای متنی Gemini (سه درخواست، تک‌مدل)                        #
# ------------------------------------------------------------------ #
async def ai_transcribe(gl: "GeminiLite", src: str, url: str, meta: Dict[str, Any],
                        dbg: DebugLogger) -> str:
    mime = mime_for(src)
    chunk_min = CFG["AUDIO_CHUNK_MIN"]
    if chunk_min > 0 and int(meta.get("duration") or 0) > chunk_min * 60:
        segs = await fetch_audio_chunks(url, src.rsplit(".", 1)[0],
                                        int(meta["duration"]), chunk_min)
        if segs:
            if dbg:
                await dbg.note("%d قطعه‌ی صوتی × %d دقیقه" % (len(segs), chunk_min))
            sem = asyncio.Semaphore(2)

            async def one(p: str) -> str:
                async with sem:
                    try:
                        return await gl.transcribe(p, mime_for(p))
                    except GeminiError as e:
                        await dbg.err(os.path.basename(p), e)
                        return ""

            pieces = await asyncio.gather(*(one(p) for p in segs))
            for p in segs:
                try:
                    os.remove(p)
                except OSError:
                    pass
            return "\n\n".join(x for x in pieces if x)
    return await gl.transcribe(src, mime)

async def ai_polish(gl: "GeminiLite", text: str, lang: str,
                    dbg: DebugLogger) -> Dict[str, str]:
    # ویرایش + عنوان در یک درخواست (به‌جای سه درخواست جداگانه)
    sysmsg = ("You are an expert editor of speech-to-text transcripts in "
              + lang_name(lang) + ". Fix recognition, phonetic and punctuation "
              "errors from context. Do NOT summarize, expand or paraphrase. "
              "Split into logical paragraphs. Keep the same language.")
    prompt = ("Return ONLY a JSON object, no markdown fence, with exactly two "
              "keys: \"title\" (a 3-8 word title in the same language) and "
              "\"body\" (the corrected transcript).\n\nTranscript:\n" + text)
    try:
        raw = await gl.text(prompt, system=sysmsg, max_tokens=GEM_OUT_MAX,
                            temperature=0.1, label="polish")
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            data = json.loads(m.group(0))
            body = str(data.get("body") or "").strip()
            title = re.sub(r"[\*\"«»“”:]", "", str(data.get("title") or "")).strip()
            if body:
                return {"title": title[:120], "body": body}
    except Exception as e:
        await dbg.err("polish", e)
    words = " ".join(text.split()[:8])
    return {"title": words, "body": text}

async def ai_translate(gl: "GeminiLite", text: str, src: str, dst: str,
                       dbg: DebugLogger) -> str:
    sysmsg = ("You are a master literary translator. Translate the user's text "
              "from " + lang_name(src) + " into " + lang_name(dst) + ". "
              "Rules: formal, polite and standard written tone; no colloquial "
              "or broken wording; 100% faithful; keep proper names, technical "
              "terms and numbers exact; preserve paragraph breaks, numbered "
              "lists and **bold** markings. Output ONLY the translation.")
    if is_persian(dst):
        sysmsg += (" Use standard Persian letters (ی and ک) and correct ZWNJ "
                   "(نیم‌فاصله), e.g. می‌دانم، کتاب‌ها، رفته‌اند.")
    try:
        out = await gl.text(text, system=sysmsg, max_tokens=GEM_OUT_MAX,
                            temperature=0.2, label="translate")
        return normalize_fa(out) if is_persian(dst) else out.strip()
    except GeminiError as e:
        await dbg.err("translate", e)
        return ""

# ------------------------------------------------------------------ #
#  7) خط لوله‌ی کامل                                                  #
# ------------------------------------------------------------------ #
JOBS: set = set()
SEM: Optional[asyncio.Semaphore] = None

IG_RE = re.compile(
    r"https?://(?:www\.)?instagram\.com/(?:reels?|p|tv|share)/[A-Za-z0-9_-]+/?")

async def process(chat_id: int, url: str) -> None:
    async with SEM:                                     # محدودیت همزمانی
        uid = uuid.uuid4().hex[:8]
        base = "%s/%s" % (CFG["TMP"], uid)
        local_video: Optional[str] = None
        gl: Optional[GeminiLite] = None
        status_id: Optional[int] = None
        async with httpx.AsyncClient(
                timeout=httpx.Timeout(900.0, connect=25.0),
                follow_redirects=True) as client:
            t0 = dt.datetime.now(dt.timezone.utc)
            dbg = DebugLogger(client, CFG["DEBUG_CHAT_ID"], "Instagram", chat_id)

            async def status(text: str) -> None:
                nonlocal status_id
                try:
                    if status_id:
                        await client.post(TG() + "/editMessageText", json={
                            "chat_id": chat_id, "message_id": status_id,
                            "text": "⏳ " + text}, timeout=8.0)
                    else:
                        r = await client.post(TG() + "/sendMessage", json={
                            "chat_id": chat_id, "text": "⏳ " + text}, timeout=8.0)
                        if r.status_code == 200:
                            status_id = r.json()["result"]["message_id"]
                except Exception:
                    pass

            async def ensure_gl() -> Optional[GeminiLite]:
                # ساخته‌ی تنبل کلاینت Gemini؛ کلید یک بار گرفته و ثابت می‌ماند
                nonlocal gl
                if gl is None:
                    gl = GeminiLite(client, await pick_key(), dbg)
                return gl

            try:
                await dbg.step("probe (بدون دانلود)")
                await status("در حال خواندن اطلاعات پست…")
                meta = await to_thread(probe, url)
                if not meta.get("video_url"):
                    raise RuntimeError("هیچ فرمت ویدیویی قابل استفاده پیدا نشد")

                ct, rsize, code = await remote_stat(client, meta["video_url"])
                mime = normalize_mime(ct) or "video/mp4"
                size = rsize or meta["video_size"]
                route = decide(size, mime)
                await dbg.note("route=%s · %.1fMB · %s · %ss" % (
                    route, size / MB, mime, meta["duration"]))

                # ---------- ۱) ارسال ویدیو (قبل از هر پردازش متنی) ---------- #
                await dbg.step("ارسال ویدیو به تلگرام")
                await status("در حال ارسال ویدیو…")
                cap_meta = meta["caption"]

                async def downloader() -> Optional[str]:
                    nonlocal local_video
                    local_video = await fetch_media(url, base + "_video", "video")
                    return local_video

                if route == "impossible":
                    await send_message(client, chat_id,
                        "⚠️ این ویدیو %.0f مگابایت است و از سقف ۵۰ مگابایتی "
                        "ربات تلگرام بزرگ‌تر است. فقط متن و کپشن ارسال می‌شود."
                        % (size / MB))
                    sent, msg, err = None, None, "too large"
                else:
                    sent, msg, err = await send_video(
                        client, chat_id, video_url=meta["video_url"],
                        local_path=local_video, size=size, mime=mime,
                        caption="", duration=meta["duration"],
                        width=meta["width"], height=meta["height"],
                        downloader=downloader, dbg=dbg)
                reply_to = (msg or {}).get("message_id")

                # ---------- ۲) صوت برای Gemini (دست‌نخورده) ---------- #
                await dbg.step("آماده‌سازی صوت برای Gemini")
                await status("در حال آماده‌سازی صوت…")
                src = local_video
                if not src:
                    src = await fetch_media(url, base + "_audio", "audio")
                text = ""
                if src and os.path.exists(src) and os.path.getsize(src) > 1024:
                    g = await ensure_gl()
                    await dbg.step("ترنسکریپت با %s" % CFG["GEMINI_MODEL"])
                    await status("در حال تبدیل صوت به متن…")
                    try:
                        text = await ai_transcribe(g, src, url, meta, dbg)
                    except GeminiError as e:
                        await dbg.err("transcribe", e)
                        text = ""
                        await send_message(client, chat_id,
                            "⚠️ تبدیل صوت به متن انجام نشد؛ ویدیو و کپشن ارسال شد.")
                # فایل صوت دیگر لازم نیست
                if src and src != local_video:
                    try:
                        os.remove(src)
                    except OSError:
                        pass

                # ---------- ۳) ویرایش، عنوان، ترجمه ---------- #
                lang = detect_lang(text) if text else "none"
                target = CFG["TRANSLATE_TO"]
                need = bool(target) and bool(text) and lang != target and \
                    not (is_persian(lang) and is_persian(target))
                tr_text = ""
                title = ""
                if text:
                    g = await ensure_gl()
                    if g:
                        await dbg.step("ویرایش و عنوان")
                        await status("در حال ویرایش ترنسکریپت…")
                        pol = await ai_polish(g, text, lang, dbg)
                        title, text = pol["title"], pol["body"]
                        if need:
                            await dbg.step("ترجمه به " + target)
                            await status("در حال ترجمه…")
                            tr_text = await ai_translate(
                                g, "**%s**\n\n%s" % (title, text), lang, target, dbg)

                tr_cap = ""
                if cap_meta and target:
                    cl = detect_lang(cap_meta)
                    if cl != target and not (is_persian(cl) and is_persian(target)):
                        g = await ensure_gl()
                        if g:
                            await dbg.step("ترجمه‌ی کپشن")
                            tr_cap = await ai_translate(g, cap_meta, cl, target, dbg)

                # ---------- ۴) ساخت پیام‌ها ---------- #
                await dbg.step("ساخت پیام")
                sections: List[Dict[str, Any]] = []
                if text:
                    if tr_text:
                        sections.append({"label": "🗣 ترجمه ترنسکریپت:", "text": tr_text})
                        sections.append({"label": "🎙 متن اصلی صوت:", "text": text})
                    else:
                        sections.append({"label": "🎙 متن ترنسکریپت صوت:", "text": text})
                if cap_meta:
                    if tr_cap:
                        sections.append({"label": "📝 ترجمه کپشن:", "text": tr_cap})
                        sections.append({"label": "📄 کپشن اصلی:", "text": cap_meta})
                    else:
                        sections.append({"label": "📄 کپشن اینستاگرام:", "text": cap_meta})
                sections.append({"label": "", "raw": True,
                                 "text": build_footer(
                                     meta, url, lang, target,
                                     bool(tr_text or tr_cap),
                                     (dt.datetime.now(dt.timezone.utc)
                                      - t0).total_seconds())})

                containers = build_containers(sections, first_is_caption=bool(sent))
                caption = containers[0] if (sent and containers) else ""
                rest = containers[1:] if sent else containers

                # اگر ویدیو بدون کپشن رفت، کپشن را جدا بفرست
                if sent and caption:
                    await send_message(client, chat_id, caption, reply_to)
                if rest:
                    await dbg.step("ارسال %d پیام متنی" % len(rest))
                    await send_chunks(client, chat_id, rest, reply_to)

                if status_id:
                    try:
                        await client.post(TG() + "/deleteMessage", json={
                            "chat_id": chat_id, "message_id": status_id})
                    except Exception:
                        pass
                await dbg.end(True)

            except Exception as e:
                log("job failed:\n" + traceback.format_exc())
                await dbg.fatal_end(e)
                if status_id:
                    try:
                        await client.post(TG() + "/editMessageText", json={
                            "chat_id": chat_id, "message_id": status_id,
                            "text": "💥 خطا: %s" % str(e)[:400]})
                    except Exception:
                        pass
                else:
                    try:
                        await send_message(client, chat_id,
                                           "💥 خطا: %s" % str(e)[:400])
                    except Exception:
                        pass
            finally:
                if gl:
                    try:
                        await gl.cleanup()          # فایل‌های Files API
                    except Exception:
                        pass
                for p in glob.glob(base + "*"):     # /tmp
                    try:
                        os.remove(p)
                    except OSError:
                        pass

# ------------------------------------------------------------------ #
#  8) worker و اپلیکیشن                                              #
# ------------------------------------------------------------------ #
async def worker() -> None:
    global SEM
    SEM = asyncio.Semaphore(CFG["JOBS"])
    seen: Dict[int, float] = {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=15.0)) as client:
        try:
            await client.get(TG() + "/deleteWebhook")
        except Exception:
            pass
        offset = 0
        while True:
            try:
                params: Dict[str, Any] = {"timeout": 25, "offset": offset,
                                          "allowed_updates": ["message"]}
                r = await client.get(TG() + "/getUpdates", params=params)
                if r.status_code != 200:
                    log("getUpdates", r.status_code, r.text[:200])
                    await asyncio.sleep(3)
                    continue
                if r.json().get("ok") is False:
                    log("getUpdates not ok:", r.text[:200])
                    await asyncio.sleep(3)
                    continue
                now = dt.datetime.now(dt.timezone.utc).timestamp()
                for upd in r.json().get("result", []):
                    offset = upd["update_id"] + 1
                    m = upd.get("message") or upd.get("channel_post") or {}
                    mid = int(m.get("message_id") or 0)
                    if mid and mid in seen:            # dedupe
                        continue
                    if mid:
                        seen[mid] = now
                    if len(seen) > 5000:
                        seen = {k: v for k, v in seen.items()
                                if now - v < 3600}
                    body = (m.get("text") or m.get("caption") or "").strip()
                    chat = (m.get("chat") or {}).get("id")
                    if not chat:
                        continue
                    low = body.lower()
                    if low.startswith(("/start", "/help")):
                        await send_message(client, chat,
                            "👋 یک لینک اینستاگرام (reel / post / tv / share) "
                            "بفرستید.\nویدیو با آدرس به تلگرام می‌رود (تا ۲۰MB) "
                            "و متن آن با Gemini استخراج و ترجمه می‌شود.")
                        continue
                    match = IG_RE.search(body)
                    if not match:
                        if body and not body.startswith("/"):
                            await send_message(client, chat,
                                "لینک اینستاگرامی پیدا نکردم. نمونه:\n"
                                "https://www.instagram.com/reel/XXXX/")
                        continue
                    task = asyncio.create_task(process(chat, match.group(0)))
                    JOBS.add(task)
                    task.add_done_callback(JOBS.discard)
            except Exception as e:
                log("worker error:", e)
                await asyncio.sleep(2.0)

@asynccontextmanager
async def lifespan(_: FastAPI):
    global SEM
    SEM = asyncio.Semaphore(CFG["JOBS"])
    log("yt-dlp %s | model %s | jobs %d | url<=%dMB upload<=%dMB | ffmpeg not required"
        % (getattr(yt_dlp, "version", {}).get("version", "?"),
           CFG["GEMINI_MODEL"], CFG["JOBS"],
           CFG["URL_LIMIT"] / MB, CFG["UPLOAD_LIMIT"] / MB))
    if not CFG["BOT_TOKEN"]:
        log("FATAL: BOT_TOKEN تنظیم نشده است")
    if not CFG["GEMINI_KEYS"]:
        log("FATAL: GEMINI_KEYS تنظیم نشده است")
    t = asyncio.create_task(worker())
    JOBS.add(t)
    t.add_done_callback(JOBS.discard)
    yield
    for task in list(JOBS):
        task.cancel()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model": CFG["GEMINI_MODEL"],
            "active_jobs": len(JOBS), "revision": 3}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.getenv("PORT") or 8000), workers=1)