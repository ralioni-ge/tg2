import os
import re
import html
import uuid
import asyncio
import datetime
import zoneinfo
from typing import Optional, List, Dict, Any, Tuple, AsyncIterator
import httpx
import yt_dlp
import uvicorn
from fastapi import FastAPI

# ======================= پیکربندی و توکن‌ها از محیط Render =======================
CONFIG = {
    "BOT_TOKEN": os.getenv("BOT_TOKEN", ""),
    "GEMINI_KEYS": [k.strip() for k in os.getenv("GEMINI_KEYS", "").split(",") if k.strip()],
    "DEBUG_CHAT_ID": os.getenv("DEBUG_CHAT_ID", ""),
    "GEMINI_MODEL": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),   # تنها مدل مورد استفاده
    "GEMINI_THINKING_LEVEL": os.getenv("GEMINI_THINKING_LEVEL", "low"),   # minimal / low / medium / high
    "TRANSLATE_TO": os.getenv("TRANSLATE_TO", "fa"),
}
if not CONFIG["BOT_TOKEN"]:
    raise SystemExit("BOT_TOKEN تنظیم نشده است.")

TELEGRAM_API = f"https://api.telegram.org/bot{CONFIG['BOT_TOKEN']}"
GEMINI_BASE = "https://generativelanguage.googleapis.com"
CAPTION_MAX = 850
MSG_MAX = 3800
ZWNJ = "\u200c"
TG_UPLOAD_LIMIT = 49 * 1024 * 1024   # سقف آپلود مستقیم Bot API (fallback)

app = FastAPI()


@app.get("/")
def health_check():
    return {"status": "Bot is running 24/7!"}


# ======================= ابزارهای کمکی لاگ و دیباگ =======================
class DebugLogger:
    def __init__(self, client: httpx.AsyncClient, chat_id: str, kind: str, origin_id: int):
        self.client = client
        self.chat_id = chat_id.strip()
        self.on = bool(self.chat_id)
        self.kind = kind
        self.origin_id = origin_id
        self.t0 = datetime.datetime.now()
        self.lines: List[str] = []
        self.mid: Optional[int] = None
        self.running_idx = -1
        self.run_name = ""
        self.run_start = datetime.datetime.now()

    def _el(self) -> str:
        return f"{(datetime.datetime.now() - self.t0).total_seconds():.1f}s"

    async def _flush(self):
        if not self.on:
            return
        header = f"🔧 {self.kind} · msg {self.origin_id}"
        body = "\n".join(self.lines)
        if len(body) > 3400:
            body = "…\n" + body[-3400:]
        text = f"{header}\n\n{body}"
        try:
            if self.mid:
                await self.client.post(f"{TELEGRAM_API}/editMessageText", json={
                    "chat_id": self.chat_id, "message_id": self.mid, "text": text}, timeout=5.0)
            else:
                r = await self.client.post(f"{TELEGRAM_API}/sendMessage", json={
                    "chat_id": self.chat_id, "text": text}, timeout=5.0)
                if r.status_code == 200:
                    self.mid = r.json().get("result", {}).get("message_id")
        except Exception:
            pass

    def _close_running(self, mark: str = "✓"):
        if self.running_idx >= 0:
            dur = (datetime.datetime.now() - self.run_start).total_seconds()
            self.lines[self.running_idx] = f"{mark} {self.run_name} — {dur:.1f}s (at {self._el()})"
            self.running_idx = -1

    async def step(self, name: str):
        self._close_running()
        self.run_name = name
        self.run_start = datetime.datetime.now()
        self.lines.append(f"▶ {name} … (from {self._el()})")
        self.running_idx = len(self.lines) - 1
        await self._flush()

    async def note(self, text: str):
        self.lines.append(f"• {text}")
        await self._flush()

    async def err(self, label: str, e: Any):
        msg = str(getattr(e, 'message', e)).replace("\n", " ")[:300]
        self.lines.append(f"✗ [{self._el()}] {label}: {msg}")
        await self._flush()

    async def fatal(self, stage: str, e: Any):
        self._close_running("✗")
        self.lines.append(f"✗ [{self._el()}] FATAL: {str(e)[:300]}")
        await self._flush()

    async def end(self, ok: bool):
        self._close_running("✓" if ok else "✗")
        self.lines.append(f"🏁 finished in {self._el()}" if ok else f"🛑 stopped after {self._el()}")
        await self._flush()


# ======================= اجرای غیرمسدودکنندهٔ ffmpeg =======================
async def run_ffmpeg(args: List[str], timeout: float = 900.0) -> Tuple[bool, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        try:
            _, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return False, "ffmpeg timeout"
        return proc.returncode == 0, (err or b"").decode(errors="ignore")[-400:]
    except FileNotFoundError:
        return False, "ffmpeg not installed"
    except Exception as e:
        return False, str(e)[:400]


# ======================= Gemini: فقط gemini-3.5-flash-lite =======================
gemini_key_index = 0


def get_next_gemini_key() -> Optional[str]:
    global gemini_key_index
    keys = [k for k in CONFIG["GEMINI_KEYS"] if k]
    if not keys:
        return None
    key = keys[gemini_key_index % len(keys)]
    gemini_key_index += 1
    return key


async def gemini_generate(
    client: httpx.AsyncClient,
    parts: List[Dict[str, Any]],
    system_instruction: str = "",
    max_tokens: int = 65536,
    temperature: float = 0.2,
    label: str = "gemini",
    dbg: Optional[DebugLogger] = None,
    api_key: Optional[str] = None,   # برای Files API باید همان کلید آپلود باشد
) -> str:
    keys = [api_key] if api_key else [k for k in CONFIG["GEMINI_KEYS"] if k]
    if not keys:
        raise ValueError("هیچ کلید معتبری برای Gemini در متغیرهای محیطی یافت نشد.")

    model = CONFIG["GEMINI_MODEL"]
    body: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "maxOutputTokens": min(max_tokens, 65536),
            "temperature": temperature,
            "thinkingConfig": {"thinkingLevel": CONFIG["GEMINI_THINKING_LEVEL"]},
        },
    }
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    last_err: Optional[Exception] = None
    max_attempts = len(keys) * 2
    for attempt in range(1, max_attempts + 1):
        key = api_key or get_next_gemini_key()
        url = f"{GEMINI_BASE}/v1beta/models/{model}:generateContent?key={key}"
        try:
            t_start = datetime.datetime.now()
            resp = await client.post(url, json=body, timeout=300.0)

            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
                if dbg:
                    await dbg.err(f"{label} (Key ...{key[-6:]})", last_err)
                await asyncio.sleep(1.5 * attempt)
                continue
            if resp.status_code != 200:
                try:
                    emsg = resp.json().get("error", {}).get("message", resp.text[:200])
                except Exception:
                    emsg = resp.text[:200]
                raise RuntimeError(f"HTTP {resp.status_code} [{model}]: {emsg[:250]}")

            data = resp.json()
            cand = (data.get("candidates") or [{}])[0]
            p_list = cand.get("content", {}).get("parts", [])
            out = "".join(p.get("text", "") for p in p_list if not p.get("thought")).strip()
            finish = cand.get("finishReason", "")
            if not out:
                why = data.get("promptFeedback", {}).get("blockReason") or finish or "empty"
                raise ValueError(f"خروجی هوش مصنوعی خالی بود: {why}")
            dur = (datetime.datetime.now() - t_start).total_seconds()
            if dbg:
                extra = " ⚠️ MAX_TOKENS" if finish == "MAX_TOKENS" else ""
                await dbg.note(f"{label}: موفق · {len(out)} کاراکتر · {dur:.1f}s{extra}")
            return out
        except RuntimeError as e:
            # خطاهای 400/403/404 قابل تکرار نیستند
            raise e
        except Exception as e:
            last_err = e
            if dbg:
                await dbg.err(f"{label} attempt {attempt}", e)
            await asyncio.sleep(0.8)

    raise last_err or Exception("ارتباط با جمینای با تمام کلیدها ناموفق بود.")


# ======================= Gemini Files API =======================
async def _file_chunks(path: str, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]:
    loop = asyncio.get_running_loop()
    with open(path, "rb") as f:
        while True:
            chunk = await loop.run_in_executor(None, f.read, chunk_size)
            if not chunk:
                break
            yield chunk


async def gemini_upload_file(client: httpx.AsyncClient, path: str, mime_type: str, api_key: str,
                             dbg: Optional[DebugLogger] = None) -> Dict[str, Any]:
    """آپلود resumable به Files API و انتظار تا ACTIVE → متادیتای فایل (name, uri, mimeType)"""
    size = os.path.getsize(path)
    r = await client.post(
        f"{GEMINI_BASE}/upload/v1beta/files?key={api_key}",
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        },
        json={"file": {"display_name": os.path.basename(path)}},
        timeout=30.0,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Files API start → HTTP {r.status_code}: {r.text[:200]}")
    upload_url = r.headers.get("x-goog-upload-url")
    if not upload_url:
        raise RuntimeError("Files API: هدر X-Goog-Upload-URL برنگشت.")

    r = await client.post(
        upload_url,
        headers={
            "Content-Length": str(size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        },
        content=_file_chunks(path),
        timeout=1800.0,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Files API upload → HTTP {r.status_code}: {r.text[:200]}")
    info = r.json().get("file", {}) or {}
    name = info.get("name")
    if not name:
        raise RuntimeError(f"Files API: پاسخ نامعتبر: {r.text[:200]}")

    waited = 0.0
    while info.get("state") == "PROCESSING" and waited < 600:
        await asyncio.sleep(2.0)
        waited += 2.0
        g = await client.get(f"{GEMINI_BASE}/v1beta/{name}?key={api_key}", timeout=15.0)
        if g.status_code == 200:
            info = g.json()
    if info.get("state") != "ACTIVE":
        raise RuntimeError(f"Files API: فایل ACTIVE نشد (state={info.get('state')})")
    if dbg:
        await dbg.note(f"Files API: آپلود شد · {size / 1024 / 1024:.2f} MB · {name}")
    return info


async def gemini_delete_file(client: httpx.AsyncClient, name: str, api_key: str):
    try:
        await client.delete(f"{GEMINI_BASE}/v1beta/{name}?key={api_key}", timeout=15.0)
    except Exception:
        pass


# ======================= تشخیص زبان، تقویم و نرمال‌سازی =======================
def detect_lang(text: str) -> str:
    arabic = len(re.findall(r'[\u0600-\u06FF]', text))
    latin = len(re.findall(r'[A-Za-z]', text))
    return "fa" if arabic > latin else "en"


def is_persian(l: str) -> bool:
    return str(l or "").lower() in ["fa", "persian", "farsi"]


def lang_name(code: str) -> str:
    c = str(code or "").strip().lower()
    if not c:
        return "the source language"
    if is_persian(c):
        return "Persian (Farsi)"
    mapping = {"en": "English", "ar": "Arabic", "fr": "French", "de": "German",
               "es": "Spanish", "ru": "Russian", "tr": "Turkish", "it": "Italian"}
    return mapping.get(c, c)


def normalize_fa(t: str) -> str:
    if not t:
        return ""
    t = t.replace('\u064A', '\u06CC').replace('\u0643', '\u06A9')
    t = re.sub(r'(^|\s)(ن?می) (?=[\u0600-\u06FF])', r'\g<1>\g<2>' + ZWNJ, t)
    t = re.sub(r'([\u0600-\u06FF]) ها(?=$|[\s.,،؟?!؛:])', r'\g<1>' + ZWNJ + 'ها', t)
    return t.strip()


def g2j(gy: int, gm: int, gd: int) -> Tuple[int, int, int]:
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    jy = 979 if gy > 1600 else 0
    gy -= 1600 if gy > 1600 else 621
    gy2 = gy + 1 if gm > 2 else gy
    days = (365 * gy) + ((gy2 + 3) // 4) - ((gy2 + 99) // 100) + ((gy2 + 399) // 400) - 80 + gd + g_d_m[gm - 1]
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


def fmt_dates(unix_ts: float) -> Dict[str, str]:
    d = datetime.datetime.fromtimestamp(unix_ts, datetime.timezone.utc)
    iso = d.strftime("%Y-%m-%d %H:%M:%S UTC")
    d_teh = d.astimezone(zoneinfo.ZoneInfo("Asia/Tehran"))
    jy, jm, jd = g2j(d_teh.year, d_teh.month, d_teh.day)
    return {"iso": iso, "shamsi": f"{jy:04d}-{jm:02d}-{jd:02d} {d_teh.strftime('%H:%M:%S')}"}


# ======================= موتور ترجمه و اصلاح ترنسکریپت =======================
def translation_system_prompt(src: str, dst: str) -> str:
    lines = [
        "You are a master literary and professional translator.",
        f"Translate the user's text from {lang_name(src)} into {lang_name(dst)}.",
        "",
        "Strict Translation Rules:",
        "- Maintain a FORMAL, POLITE, and STANDARD written tone (فارسی معیار و کتابی).",
        "- STRICTLY FORBIDDEN: Do NOT use colloquial, street slang, or broken words (اصلاً عامیانه، محاوره‌ای و شکسته ترجمه نکن).",
        "- Be 100% faithful and accurate to the original text; do not invent or omit information.",
        "- Preserve the original meaning with natural, elegant grammatical structures.",
        "- Keep proper names, technical concepts, and numbers accurate.",
        "",
        "Formatting:",
        "- Preserve existing paragraph breaks, numbered lists, and **bold** markings.",
        "- Output ONLY the final translation without greetings, notes, or intros."
    ]
    if is_persian(dst):
        lines += ["", "Persian Quality Rules:",
                  "- Use standard Persian letters (ی and ک instead of ي and ك).",
                  "- Apply ZWNJ (نیم‌فاصله) correctly: e.g. می‌دانم، می‌توان، کتاب‌ها، رفته‌اند."]
    return "\n".join(lines)


async def translate_markdown(client, md: str, src: str, dst: str, dbg) -> str:
    out = await gemini_generate(client, [{"text": md}], system_instruction=translation_system_prompt(src, dst),
                                temperature=0.2, label="translation", dbg=dbg)
    return normalize_fa(out).strip() if is_persian(dst) else out.strip()


def gemini_spell_prompt(lang: str) -> str:
    lines = [
        f"You are an expert editor of speech-to-text transcripts in {lang_name(lang)}.",
        "Task: Fix audio recognition, phonetic, and punctuation errors based on context.",
        "Do NOT summarize, expand, or paraphrase.",
        "Output ONLY the corrected transcript."
    ]
    if is_persian(lang):
        lines.append("Use standard Persian letters ی و ک, and apply proper ZWNJ (نیم‌فاصله). Maintain a standard written style.")
    return "\n".join(lines)


async def ai_spell(client, text: str, lang: str, dbg) -> str:
    try:
        out = await gemini_generate(client, [{"text": text}], system_instruction=gemini_spell_prompt(lang),
                                    temperature=0.1, label="spellcheck", dbg=dbg)
        return normalize_fa(out) if is_persian(lang) else out
    except Exception:
        return text


async def ai_generate_title(client, text: str, lang: str, dbg) -> str:
    fallback = " ".join(text.split()[:6]) + "…"
    prompt = "Generate a concise title (3-8 words) in the SAME LANGUAGE. Output ONLY the title."
    try:
        raw = await gemini_generate(client, [{"text": f"{prompt}\n\nText:\n{text[:2000]}"}],
                                    max_tokens=1024, temperature=0.2, label="title", dbg=dbg)
        clean = re.sub(r'[\*\"«»“”]', '', raw).replace("title:", "").replace("Title:", "").strip()
        return clean if (3 <= len(clean) <= 100) else fallback
    except Exception:
        return fallback


async def ai_format(client, text: str, dbg) -> Dict[str, Any]:
    prompt = ("You are a transcript editor.\nReply in EXACTLY this format:\n---\n<the formatted transcript>\n\n"
              "Keep the same language. Split into logical paragraphs. Add punctuation. Output nothing else.")
    try:
        raw = await gemini_generate(client, [{"text": f"{prompt}\n\nTranscript:\n{text}"}],
                                    temperature=0.1, label="formatting", dbg=dbg)
        m = re.search(r'^---\s*\n([\s\S]*)$', raw, re.MULTILINE)
        return {"title": "", "body": (m.group(1) if m else raw).strip(), "formatted": True}
    except Exception:
        return {"title": "", "body": text, "formatted": False}


# ======================= ترنسکریپت صوت با Files API =======================
TRANSCRIBE_PROMPT = (
    "Listen to this audio carefully and transcribe all spoken words verbatim in their original language. "
    "Do not summarize or paraphrase. Output ONLY the exact transcribed text. "
    "If there is completely no speech (only background music, silence, or noise), reply with: NO_SPEECH"
)
TRANSCRIBE_SYSTEM = "You are a professional audio transcriber. Transcribe spoken words accurately."


def _clean_transcript(out: str) -> str:
    clean = out.strip()
    if clean.upper() in ["NO_SPEECH", "[NO_SPEECH]", "NO SPEECH", "NO_SPEECH."]:
        return ""
    if clean.startswith("NO_SPEECH"):
        clean = clean.replace("NO_SPEECH", "").strip()
    return clean


async def ai_transcribe_file(client: httpx.AsyncClient, path: str, mime_type: str, dbg) -> str:
    """آپلود فایل (صوتی یا ویدیویی) به Files API و ترنسکریپت با همان کلید؛ سپس حذف فایل از سرور گوگل."""
    if os.path.getsize(path) < 500:
        return ""
    keys = list(dict.fromkeys(k for k in CONFIG["GEMINI_KEYS"] if k))
    if not keys:
        raise ValueError("هیچ کلید Gemini تنظیم نشده است.")
    last_err: Optional[Exception] = None
    for key in keys:
        file_name = None
        try:
            finfo = await gemini_upload_file(client, path, mime_type, key, dbg)
            file_name = finfo.get("name")
            parts = [{"text": TRANSCRIBE_PROMPT},
                     {"fileData": {"mimeType": finfo.get("mimeType", mime_type), "fileUri": finfo["uri"]}}]
            out = await gemini_generate(client, parts, system_instruction=TRANSCRIBE_SYSTEM, temperature=0.1,
                                        label="audio-transcribe", dbg=dbg, api_key=key)
            return _clean_transcript(out)
        except Exception as e:
            last_err = e
            if dbg:
                await dbg.err(f"Files API با کلید ...{key[-6:]}", e)
        finally:
            if file_name:
                await gemini_delete_file(client, file_name, key)
    raise last_err or RuntimeError("ترنسکریپت با Files API ناموفق بود.")


# ======================= جدا کردن ترک صدا مستقیماً از URL (بدون دانلود ویدیو، بدون فشرده‌سازی) =======================
async def extract_audio_from_url(video_url: str, headers: Dict[str, str], uid: str, dbg) -> Tuple[Optional[str], str]:
    """
    ffmpeg خودش جریان ویدیو را از CDN می‌خواند و فقط ترک صدا را جدا می‌کند.
    اولویت ۱: -c:a copy  (ترک AAC اصلی، بدون re-encode)  → audio/aac
    اولویت ۲: WAV خام PCM (اگر ترک صوتی AAC نبود)        → audio/wav
    """
    hdr = "".join(f"{k}: {v}\r\n" for k, v in headers.items()) if headers else ""
    base = ["-headers", hdr] if hdr else []
    base += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5", "-i", video_url, "-vn"]

    aac_path = f"/tmp/{uid}_audio.aac"
    ok, err = await run_ffmpeg(base + ["-c:a", "copy", "-f", "adts", aac_path])
    if ok and os.path.exists(aac_path) and os.path.getsize(aac_path) > 500:
        if dbg:
            await dbg.note(f"audio (copy/aac): {os.path.getsize(aac_path) / 1024:.0f} KB")
        return aac_path, "audio/aac"
    if dbg:
        await dbg.err("ffmpeg -c:a copy", err or "no output")

    wav_path = f"/tmp/{uid}_audio.wav"
    ok, err = await run_ffmpeg(base + ["-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_path])
    if ok and os.path.exists(wav_path) and os.path.getsize(wav_path) > 500:
        if dbg:
            await dbg.note(f"audio (wav): {os.path.getsize(wav_path) / 1024:.0f} KB")
        return wav_path, "audio/wav"
    if dbg:
        await dbg.err("ffmpeg wav", err or "no output")
    return None, ""


# ======================= Smart Split و قالب‌بندی =======================
def esc(s: str) -> str:
    return html.escape(str(s or ""), quote=False)


def balance_markdown(chunk: str, rest: str) -> Tuple[str, str]:
    if len(re.findall(r'\*\*', chunk)) % 2 != 0:
        chunk += "**"
        rest = "**" + rest
    return chunk, rest


def split_at_word(text: str, max_len: int) -> Tuple[str, str]:
    if len(text) <= max_len:
        return text, ""
    p_index = text.rfind("\n\n", 0, max_len)
    if p_index >= max_len * 0.4:
        return balance_markdown(text[:p_index].strip(), text[p_index + 2:].strip())
    s_matches = list(re.finditer(r'[.!?؟\n]\s+', text[:max_len]))
    if s_matches and s_matches[-1].start() >= max_len * 0.4:
        cut = s_matches[-1].end()
        return balance_markdown(text[:cut].strip(), text[cut:].strip())
    sp_index = text.rfind(" ", 0, max_len)
    if sp_index > 0:
        return balance_markdown(text[:sp_index].strip(), text[sp_index + 1:].strip())
    return balance_markdown(text[:max_len].strip(), text[max_len:].strip())


def md_to_html(s: str) -> str:
    s = esc(s)
    s = re.sub(r'^[ \t]*#{1,6}[ \t]*(.+)$', r'<b>\1</b>', s, flags=re.MULTILINE)
    s = re.sub(r'^[ \t]*[-*•][ \t]+', '• ', s, flags=re.MULTILINE)
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    return s.replace("**", "")


def format_section_block(label: str, md_piece: str) -> str:
    h = md_to_html(md_piece) if md_piece else ""
    return h if not label else f"<b>{esc(label)}</b>\n<blockquote>{h}</blockquote>"


def build_containers(sections: List[Dict[str, Any]], has_media_caption: bool) -> List[str]:
    containers: List[str] = []
    current_blocks: List[str] = []
    current_len = 0
    limit = CAPTION_MAX if has_media_caption else MSG_MAX

    def flush():
        nonlocal current_blocks, current_len, limit
        if current_blocks:
            containers.append("\n\n".join(current_blocks))
            current_blocks = []
            current_len = 0
        limit = MSG_MAX

    for sec in sections:
        if not sec or not sec.get("text"):
            continue
        if sec.get("isRawHtml"):
            item_len = len(sec["text"])
            sep = 2 if current_blocks else 0
            if current_len + sep + item_len > limit and current_blocks:
                flush()
            current_blocks.append(sec["text"])
            current_len += sep + item_len
            continue
        remaining_text = sec["text"].strip()
        current_label = sec.get("label", "")
        while remaining_text:
            sep = 2 if current_blocks else 0
            label_len = len(current_label) + 40 if current_label else 0
            available = limit - current_len - sep - label_len
            if available < 120 and current_blocks:
                flush()
                continue
            if len(remaining_text) <= available:
                block_html = format_section_block(current_label, remaining_text)
                current_blocks.append(block_html)
                current_len += sep + len(block_html)
                remaining_text = ""
            else:
                chunk, rest = split_at_word(remaining_text, available)
                block_html = format_section_block(current_label, chunk)
                current_blocks.append(block_html)
                current_len += sep + len(block_html)
                remaining_text = rest
                if current_label and not current_label.endswith("(cont.)"):
                    current_label += " (cont.)"
                flush()
    if current_blocks:
        containers.append("\n\n".join(current_blocks))
    return containers


def build_footer(info: dict, post_url: str, lang: str, target: str, translated: bool, total_sec: float) -> str:
    dt = fmt_dates(datetime.datetime.now().timestamp())
    dur = info.get("duration")
    dur_str = f" | {int(dur)//60:02d}:{int(dur)%60:02d}" if dur else ""
    cand_id = info.get("uploader_id") or ""
    cand_name = info.get("uploader") or cand_id or "Instagram User"
    user_url = f"https://instagram.com/{cand_id}" if cand_id else post_url
    lines = ["------",
             f"<b>Message link:</b> <a href=\"{esc(post_url)}\">Open Instagram Link</a>",
             f"<b>Type:</b> Video{dur_str} | Instagram",
             f"<b>Language:</b> {esc(lang)}{(' (translated to ' + esc(target) + ')') if translated else ''}",
             f"<b>Sender:</b> <a href=\"{esc(user_url)}\">{esc(cand_name)}</a>",
             f"<b>TimeStamp Date:</b> {dt['iso']}",
             f"<b>Time Tehran:</b> {dt['shamsi']}",
             f"<b>Processing:</b> {total_sec:.1f}s"]
    return "\n" + "\n".join(lines)


# ======================= انتخاب لینک مستقیم ویدیو از yt-dlp (بدون دانلود) =======================
def pick_direct_video(info: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, str], Dict[str, Any]]:
    """بهترین فرمت MP4 که همزمان ویدیو و صدا دارد را برمی‌گرداند: (url, http_headers, format)"""
    fmts = info.get("formats") or []
    good = [f for f in fmts
            if f.get("url") and f.get("vcodec") not in (None, "none") and f.get("acodec") not in (None, "none")
            and (f.get("ext") in (None, "mp4") or "mp4" in str(f.get("ext")))
            and str(f.get("protocol", "https")).startswith("http")]
    if good:
        good.sort(key=lambda f: ((f.get("height") or 0), (f.get("tbr") or 0)))
        best = good[-1]
        return best["url"], best.get("http_headers") or info.get("http_headers") or {}, best
    if info.get("url") and info.get("acodec") not in ("none",):
        return info["url"], info.get("http_headers") or {}, info
    return None, {}, {}


# ======================= ارسال ویدیو: اول با URL، در صورت رد شدن fallback دانلود+آپلود =======================
async def send_video_by_url(client, chat_id: int, video_url: str, caption: str,
                            width, height, duration, thumb_url: Optional[str], dbg) -> Tuple[bool, Optional[str]]:
    data = {"chat_id": str(chat_id), "video": video_url, "supports_streaming": "true"}
    if width:
        data["width"] = str(int(width))
    if height:
        data["height"] = str(int(height))
    if duration:
        data["duration"] = str(int(duration))
    if caption:
        data["caption"] = caption
        data["parse_mode"] = "HTML"
    if thumb_url:
        data["cover"] = thumb_url   # کاور ویدیو (در Bot API 8.x می‌تواند URL باشد)
    resp = await client.post(f"{TELEGRAM_API}/sendVideo", data=data, timeout=120.0)
    if resp.status_code == 200:
        return True, None
    desc = resp.text[:200]
    try:
        desc = resp.json().get("description", desc)
    except Exception:
        pass
    if dbg:
        await dbg.err("sendVideo(URL)", desc)
    # اگر خطا از کپشن/کاور بود، یک بار بدون آن‌ها تلاش کن
    if "caption" in desc.lower() or "cover" in desc.lower() or "parse" in desc.lower():
        data.pop("caption", None); data.pop("parse_mode", None); data.pop("cover", None)
        resp = await client.post(f"{TELEGRAM_API}/sendVideo", data=data, timeout=120.0)
        if resp.status_code == 200:
            return False, None   # ویدیو رفت ولی بدون کپشن
    return False, desc


async def send_video_fallback_upload(client, chat_id: int, video_url: str, headers: Dict[str, str], uid: str,
                                     caption: str, width, height, duration, dbg) -> Tuple[bool, Optional[str]]:
    """فقط وقتی تلگرام URL را نپذیرفت (مثلاً بالای ۲۰MB): دانلود موقت → در صورت نیاز فشرده‌سازی → آپلود."""
    path = f"/tmp/{uid}_video.mp4"
    comp = f"/tmp/{uid}_video_comp.mp4"
    try:
        if dbg:
            await dbg.step("Fallback: download video")
        async with client.stream("GET", video_url, headers=headers, timeout=600.0, follow_redirects=True) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                async for chunk in r.aiter_bytes(1024 * 512):
                    f.write(chunk)
        send_path = path
        if os.path.getsize(path) > TG_UPLOAD_LIMIT:
            if dbg:
                await dbg.step("Fallback: compress >50MB")
            dur_s = max(int(duration or 0), 1)
            v_kbps = max(int(46 * 8192 / dur_s) - 96, 200)
            ok, err = await run_ffmpeg(["-i", path, "-c:v", "libx264", "-preset", "veryfast",
                                        "-b:v", f"{v_kbps}k", "-maxrate", f"{v_kbps}k", "-bufsize", f"{v_kbps*2}k",
                                        "-vf", "scale='min(1280,iw)':-2", "-c:a", "aac", "-b:a", "96k",
                                        "-movflags", "+faststart", comp])
            if ok and os.path.exists(comp):
                send_path = comp
            else:
                return False, f"compress failed: {err}"
        data = {"chat_id": str(chat_id), "supports_streaming": "true"}
        if width: data["width"] = str(int(width))
        if height: data["height"] = str(int(height))
        if duration: data["duration"] = str(int(duration))
        if caption:
            data["caption"] = caption
            data["parse_mode"] = "HTML"
        with open(send_path, "rb") as vf:
            resp = await client.post(f"{TELEGRAM_API}/sendVideo", data=data,
                                     files={"video": ("video.mp4", vf, "video/mp4")}, timeout=600.0)
        if resp.status_code == 200:
            return True, None
        data.pop("caption", None); data.pop("parse_mode", None)
        with open(send_path, "rb") as vf:
            resp = await client.post(f"{TELEGRAM_API}/sendVideo", data=data,
                                     files={"video": ("video.mp4", vf, "video/mp4")}, timeout=600.0)
        if resp.status_code == 200:
            return False, None
        try:
            return False, resp.json().get("description", resp.text[:200])
        except Exception:
            return False, resp.text[:200]
    except Exception as e:
        return False, str(e)[:200]
    finally:
        for p in (path, comp):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


# ======================= چرخه کامل پردازش =======================
async def process_and_send(chat_id: int, url: str):
    async with httpx.AsyncClient(timeout=300.0) as client:
        t0 = datetime.datetime.now()
        dbg = DebugLogger(client, CONFIG["DEBUG_CHAT_ID"], "Instagram Job", chat_id)
        await dbg.step("Start Job")
        status_msg_id = None

        async def update_status(text: str):
            nonlocal status_msg_id
            try:
                if status_msg_id:
                    await client.post(f"{TELEGRAM_API}/editMessageText", json={
                        "chat_id": chat_id, "message_id": status_msg_id, "text": f"⏳ {text}"}, timeout=5.0)
                else:
                    r = await client.post(f"{TELEGRAM_API}/sendMessage", json={
                        "chat_id": chat_id, "text": f"⏳ {text}"}, timeout=5.0)
                    if r.status_code == 200:
                        status_msg_id = r.json().get("result", {}).get("message_id")
            except Exception:
                pass

        async def send_text(txt: str):
            await client.post(f"{TELEGRAM_API}/sendMessage", json={
                "chat_id": chat_id, "text": txt, "parse_mode": "HTML",
                "link_preview_options": {"is_disabled": True}}, timeout=30.0)

        uid = str(uuid.uuid4())[:8]
        audio_path: Optional[str] = None
        try:
            # ۱. فقط متادیتا و لینک مستقیم (بدون دانلود)
            await dbg.step("yt-dlp extract_info (no download)")
            await update_status("در حال دریافت اطلاعات ویدیو از اینستاگرام...")
            ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
            loop = asyncio.get_running_loop()
            info = await loop.run_in_executor(None, lambda: yt_dlp.YoutubeDL(ydl_opts).extract_info(url, download=False))
            if info.get("entries"):
                info = next((e for e in info["entries"] if e and e.get("formats")), info["entries"][0])
            video_url, cdn_headers, fmt = pick_direct_video(info)
            if not video_url:
                raise RuntimeError("لینک مستقیم ویدیو پیدا نشد (شاید پست عکس باشد یا نیاز به لاگین دارد).")
            width = fmt.get("width") or info.get("width")
            height = fmt.get("height") or info.get("height")
            duration = int(info["duration"]) if info.get("duration") else None
            thumb_url = info.get("thumbnail")
            await dbg.note(f"direct url ok · {width}x{height} · {duration}s · {fmt.get('format_id')}")

            # ۲. جدا کردن ترک صدا مستقیماً از URL (بدون re-encode)
            await dbg.step("FFmpeg: audio stream-copy from URL")
            await update_status("در حال جدا کردن صدا از جریان ویدیو...")
            audio_path, audio_mime = await extract_audio_from_url(video_url, cdn_headers, uid, dbg)

            # ۳. ترنسکریپت با Files API
            await dbg.step("Gemini Files API transcription")
            await update_status("در حال آپلود صدا به Gemini و تبدیل صوت به متن...")
            extracted_text = ""
            if audio_path:
                extracted_text = await ai_transcribe_file(client, audio_path, audio_mime, dbg)

            # ۴. پردازش متن
            lang = detect_lang(extracted_text) if extracted_text else "fa"
            target = CONFIG["TRANSLATE_TO"]
            need_translate = target and (lang != target) and not (is_persian(lang) and is_persian(target))
            fm = {"title": "", "body": extracted_text, "formatted": False}
            tr_transcript = ""
            if extracted_text:
                if not need_translate:
                    await dbg.step("AI Spellcheck")
                    await update_status("در حال اصلاح ترنسکریپت...")
                    extracted_text = await ai_spell(client, extracted_text, lang, dbg)
                await dbg.step("AI Formatting & Title")
                await update_status("در حال ساخت عنوان و ساختاربندی ترنسکریپت...")
                title, fm = await asyncio.gather(ai_generate_title(client, extracted_text, lang, dbg),
                                                 ai_format(client, extracted_text, dbg))
                fm["title"] = title
                if need_translate:
                    await dbg.step("AI Translation")
                    await update_status("در حال ترجمه ترنسکریپت به فارسی معیار...")
                    tr_transcript = await translate_markdown(client, f"**{title}**\n\n{fm['body']}", lang, target, dbg)

            # ۵. کپشن اینستاگرام
            raw_caption = (info.get("description") or info.get("title") or "").strip()
            tr_caption = ""
            if raw_caption:
                cap_lang = detect_lang(raw_caption)
                if target and (cap_lang != target) and not (is_persian(cap_lang) and is_persian(target)):
                    await dbg.step("Translate Caption")
                    await update_status("در حال ترجمه کپشن پست...")
                    tr_caption = await translate_markdown(client, raw_caption, cap_lang, target, dbg)

            # ۶. ساخت بخش‌ها
            sections: List[Dict[str, Any]] = []
            if extracted_text:
                body_md = f"**{fm['title']}**\n\n{fm['body']}"
                if tr_transcript:
                    sections.append({"label": "🗣 ترجمه ترنسکریپت (فارسی):", "text": tr_transcript})
                    sections.append({"label": "🎙 متن اصلی صوت (Transcript):", "text": body_md})
                else:
                    sections.append({"label": "🎙 متن ترنسکریپت صوت:", "text": body_md})
            if raw_caption:
                if tr_caption:
                    sections.append({"label": "📝 ترجمه کپشن اینستاگرام:", "text": tr_caption})
                    sections.append({"label": "📄 کپشن اصلی اینستاگرام:", "text": raw_caption})
                else:
                    sections.append({"label": "📄 کپشن اینستاگرام:", "text": raw_caption})
            elapsed = (datetime.datetime.now() - t0).total_seconds()
            sections.append({"label": "", "isRawHtml": True,
                             "text": build_footer(info, url, lang if extracted_text else "none", target,
                                                  bool(tr_transcript or tr_caption), elapsed)})
            containers = build_containers(sections, has_media_caption=True)
            caption_for_media = containers[0] if containers else ""
            remaining = containers[1:]

            # ۷. ارسال ویدیو با URL (بدون دانلود/آپلود)
            await dbg.step("Telegram sendVideo by URL")
            await update_status("در حال ارسال ویدیو به تلگرام...")
            with_caption, err = await send_video_by_url(client, chat_id, video_url, caption_for_media,
                                                        width, height, duration, thumb_url, dbg)
            if err:
                # تلگرام URL را نپذیرفت (معمولاً بالای ۲۰MB یا CDN مسدود) → fallback
                with_caption, err = await send_video_fallback_upload(client, chat_id, video_url, cdn_headers, uid,
                                                                     caption_for_media, width, height, duration, dbg)

            # ۸. متن‌های تکمیلی
            if with_caption:
                for m in remaining:
                    await send_text(m)
            elif err is None:
                for m in containers:
                    await send_text(m)
            else:
                await send_text(f"⚠️ ویدیو به دلیل خطای تلگرام ارسال نشد:\n<code>{html.escape(err)}</code>\n\nمتن استخراج‌شده در ادامه ارسال می‌شود:")
                for m in containers:
                    await send_text(m)

            if status_msg_id:
                await client.post(f"{TELEGRAM_API}/deleteMessage", json={"chat_id": chat_id, "message_id": status_msg_id})
            await dbg.end(True)

        except Exception as e:
            await dbg.fatal("Handler Crash", e)
            if status_msg_id:
                try:
                    await client.post(f"{TELEGRAM_API}/editMessageText", json={
                        "chat_id": chat_id, "message_id": status_msg_id, "text": f"💥 خطا در پردازش:\n{str(e)[:400]}"})
                except Exception:
                    pass
        finally:
            for p in [audio_path, f"/tmp/{uid}_audio.aac", f"/tmp/{uid}_audio.wav"]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass


# ======================= حلقه دریافت پیام‌ها =======================
BACKGROUND_TASKS = set()
JOB_SEMAPHORE = asyncio.Semaphore(3)


async def guarded_job(chat_id: int, url: str):
    async with JOB_SEMAPHORE:
        await process_and_send(chat_id, url)


async def telegram_worker():
    async with httpx.AsyncClient(timeout=40.0) as client:
        await client.get(f"{TELEGRAM_API}/deleteWebhook")
        offset = None
        ig_regex = re.compile(r'https?://(?:www\.|m\.)?instagram\.com/(?:[\w.]+/)?(?:share/)?(?:reels?|p|tv)/[A-Za-z0-9_-]+')
        while True:
            try:
                params = {"timeout": 20}
                if offset:
                    params["offset"] = offset
                resp = await client.get(f"{TELEGRAM_API}/getUpdates", params=params)
                if resp.status_code == 200:
                    for update in resp.json().get("result", []):
                        offset = update["update_id"] + 1
                        msg = update.get("message") or update.get("channel_post", {})
                        text = msg.get("text", "") or msg.get("caption", "")
                        chat_id = msg.get("chat", {}).get("id")
                        if not chat_id or not text:
                            continue
                        match = ig_regex.search(text)
                        if match:
                            t = asyncio.create_task(guarded_job(chat_id, match.group(0)))
                            BACKGROUND_TASKS.add(t)
                            t.add_done_callback(BACKGROUND_TASKS.discard)
            except Exception:
                await asyncio.sleep(2.0)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(telegram_worker())


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
