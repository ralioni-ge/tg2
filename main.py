import os
import re
import html
import uuid
import base64
import asyncio
import datetime
import zoneinfo
import subprocess
from typing import Optional, List, Dict, Any, Tuple

import httpx
import yt_dlp
import uvicorn
from fastapi import FastAPI

# ======================= پیکربندی و توکن‌ها از محیط Render =======================
CONFIG = {
    "BOT_TOKEN": os.getenv("BOT_TOKEN", "9005346823:AAHe2Pqnx5xJgYNZMHyw2_uTeD93glTshXc"),
    "GEMINI_KEYS": [
        k.strip() for k in os.getenv("GEMINI_KEYS", "").split(",") if k.strip()
    ],
    "DEBUG_CHAT_ID": os.getenv("DEBUG_CHAT_ID", ""),
    "GEMINI_MODEL": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    "TRANSLATE_TO": os.getenv("TRANSLATE_TO", "fa"),
}

TELEGRAM_API = f"https://api.telegram.org/bot{CONFIG['BOT_TOKEN']}"
CAPTION_MAX = 850   # حاشیه امن تضمینی همراه با تگ‌های HTML برای سقف ۱۰۲۴ کاراکتری تلگرام
MSG_MAX = 3800      # حاشیه امن برای سقف ۴۰۹۶ کاراکتری تلگرام
ZWNJ = "\u200c"     # نیم‌فاصله استاندارد یونیکد

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
                    "chat_id": self.chat_id, "message_id": self.mid, "text": text
                }, timeout=5.0)
            else:
                r = await self.client.post(f"{TELEGRAM_API}/sendMessage", json={
                    "chat_id": self.chat_id, "text": text
                }, timeout=5.0)
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

# ======================= مدیریت کلیدها و ارتباط با Gemini =======================
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
    max_tokens: int = 8192,
    temperature: float = 0.2,
    label: str = "gemini",
    dbg: Optional[DebugLogger] = None
) -> str:
    keys = [k for k in CONFIG["GEMINI_KEYS"] if k]
    if not keys:
        raise ValueError("هیچ کلید معتبری برای Gemini در متغیرهای محیطی Render یافت نشد.")

    body: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature
        }
    }
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    models_to_try = [CONFIG["GEMINI_MODEL"], "gemini-1.5-flash", "gemini-2.0-flash"]
    last_err = None

    for model in dict.fromkeys(models_to_try):
        max_attempts = len(keys) * 2
        for attempt in range(1, max_attempts + 1):
            api_key = get_next_gemini_key()
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            try:
                t_start = datetime.datetime.now()
                resp = await client.post(url, json=body, timeout=60.0)

                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    if dbg:
                        await dbg.err(f"{label} (Key ...{api_key[-6:]})", last_err)
                    await asyncio.sleep(1.0 * attempt)
                    continue

                if resp.status_code == 404:
                    break

                if resp.status_code != 200:
                    last_err = Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    if dbg:
                        await dbg.err(f"{label} error", last_err)
                    break

                data = resp.json()
                cand = data.get("candidates", [{}])[0]
                p_list = cand.get("content", {}).get("parts", [])
                out = "".join(p.get("text", "") for p in p_list).strip()

                if not out:
                    why = data.get("promptFeedback", {}).get("blockReason") or cand.get("finishReason") or "empty"
                    raise ValueError(f"خروجی هوش مصنوعی خالی بود: {why}")

                dur = (datetime.datetime.now() - t_start).total_seconds()
                if dbg:
                    await dbg.note(f"{label}: موفق · {len(out)} کاراکتر · {dur:.1f}s")
                return out

            except Exception as e:
                last_err = e
                if dbg:
                    await dbg.err(f"{label} attempt {attempt}", e)
                await asyncio.sleep(0.8)

    raise last_err or Exception("ارتباط با جمینای با تمام کلیدها ناموفق بود.")

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
    mapping = {
        "en": "English", "ar": "Arabic", "fr": "French", "de": "German",
        "es": "Spanish", "ru": "Russian", "tr": "Turkish", "it": "Italian"
    }
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
    teh_tz = zoneinfo.ZoneInfo("Asia/Tehran")
    d_teh = d.astimezone(teh_tz)
    jy, jm, jd = g2j(d_teh.year, d_teh.month, d_teh.day)
    shamsi = f"{jy:04d}-{jm:02d}-{jd:02d} {d_teh.strftime('%H:%M:%S')}"
    return {"iso": iso, "shamsi": shamsi}

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
        lines.extend([
            "",
            "Persian Quality Rules:",
            "- Use standard Persian letters (ی and ک instead of ي and ك).",
            "- Apply ZWNJ (نیم‌فاصله) correctly: e.g. می‌دانم، می‌توان، کتاب‌ها، رفته‌اند."
        ])
    return "\n".join(lines)

async def translate_markdown(client: httpx.AsyncClient, md: str, src: str, dst: str, dbg: Optional[DebugLogger]) -> str:
    sys = translation_system_prompt(src, dst)
    parts = [{"text": md}]
    out = await gemini_generate(client, parts, system_instruction=sys, max_tokens=8192, temperature=0.2, label="translation", dbg=dbg)
    if is_persian(dst):
        out = normalize_fa(out)
    return out.strip()

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

async def ai_spell(client: httpx.AsyncClient, text: str, lang: str, dbg: Optional[DebugLogger]) -> str:
    sys = gemini_spell_prompt(lang)
    parts = [{"text": text}]
    try:
        out = await gemini_generate(client, parts, system_instruction=sys, max_tokens=8192, temperature=0.1, label="spellcheck", dbg=dbg)
        return normalize_fa(out) if is_persian(lang) else out
    except Exception:
        return text

async def ai_generate_title(client: httpx.AsyncClient, text: str, lang: str, dbg: Optional[DebugLogger]) -> str:
    fallback = " ".join(text.split()[:6]) + "…"
    prompt = "Generate a concise title (3-8 words) in the SAME LANGUAGE. Output ONLY the title."
    parts = [{"text": f"{prompt}\n\nText:\n{text[:2000]}"}]
    try:
        raw = await gemini_generate(client, parts, max_tokens=100, temperature=0.2, label="title", dbg=dbg)
        clean = re.sub(r'[\*\"«»“”]', '', raw).replace("title:", "").replace("Title:", "").strip()
        return clean if (3 <= len(clean) <= 100) else fallback
    except Exception:
        return fallback

async def ai_format(client: httpx.AsyncClient, text: str, dbg: Optional[DebugLogger]) -> Dict[str, Any]:
    prompt = (
        "You are a transcript editor.\n"
        "Reply in EXACTLY this format:\n---\n<the formatted transcript>\n\n"
        "Keep the same language. Split into logical paragraphs. Add punctuation. Output nothing else."
    )
    parts = [{"text": f"{prompt}\n\nTranscript:\n{text}"}]
    try:
        raw = await gemini_generate(client, parts, max_tokens=4096, temperature=0.1, label="formatting", dbg=dbg)
        m = re.search(r'^---\s*\n([\s\S]*)$', raw, re.MULTILINE)
        if m:
            return {"title": "", "body": m.group(1).strip(), "formatted": True}
        return {"title": "", "body": raw.strip(), "formatted": True}
    except Exception:
        return {"title": "", "body": text, "formatted": False}

# ======================= ترنسکریپت اختصاصی صوت با Gemini =======================
async def ai_transcribe_audio(client: httpx.AsyncClient, audio_path: str, dbg: Optional[DebugLogger]) -> str:
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    if len(audio_bytes) < 500:
        return ""

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    prompt = (
        "Listen to this audio carefully and transcribe all spoken words verbatim in their original language. "
        "Do not summarize or paraphrase. Output ONLY the exact transcribed text. "
        "If there is completely no speech (only background music, silence, or noise), reply with: NO_SPEECH"
    )
    parts = [
        {"text": prompt},
        {"inlineData": {"mimeType": "audio/wav", "data": audio_b64}}
    ]
    out = await gemini_generate(
        client,
        parts,
        system_instruction="You are a professional audio transcriber. Transcribe spoken words accurately.",
        max_tokens=8192,
        temperature=0.1,
        label="audio-transcribe",
        dbg=dbg
    )
    clean = out.strip()
    if clean.upper() in ["NO_SPEECH", "[NO_SPEECH]", "NO SPEECH", "NO_SPEECH."]:
        return ""
    if clean.startswith("NO_SPEECH"):
        clean = clean.replace("NO_SPEECH", "").strip()
    return clean

# ======================= شکستن هوشمند و قالب‌بندی بدون قطعی (Smart Split) =======================
def esc(s: str) -> str:
    return html.escape(str(s or ""), quote=False)

def balance_markdown(chunk: str, rest: str) -> Tuple[str, str]:
    stars = len(re.findall(r'\*\*', chunk))
    if stars % 2 != 0:
        chunk += "**"
        rest = "**" + rest
    return chunk, rest

def split_at_word(text: str, max_len: int) -> Tuple[str, str]:
    if len(text) <= max_len:
        return text, ""

    p_index = text.rfind("\n\n", 0, max_len)
    if p_index >= max_len * 0.4:
        chunk = text[:p_index].strip()
        rest = text[p_index + 2:].strip()
        return balance_markdown(chunk, rest)

    s_matches = list(re.finditer(r'[.!?؟\n]\s+', text[:max_len]))
    if s_matches:
        last_m = s_matches[-1]
        if last_m.start() >= max_len * 0.4:
            cut = last_m.end()
            chunk = text[:cut].strip()
            rest = text[cut:].strip()
            return balance_markdown(chunk, rest)

    sp_index = text.rfind(" ", 0, max_len)
    if sp_index > 0:
        chunk = text[:sp_index].strip()
        rest = text[sp_index + 1:].strip()
        return balance_markdown(chunk, rest)

    return balance_markdown(text[:max_len].strip(), text[max_len:].strip())

def md_to_html(s: str) -> str:
    s = esc(s)
    s = re.sub(r'^[ \t]*#{1,6}[ \t]*(.+)$', r'<b>\1</b>', s, flags=re.MULTILINE)
    s = re.sub(r'^[ \t]*[-*•][ \t]+', '• ', s, flags=re.MULTILINE)
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    return s.replace("**", "")

def format_section_block(label: str, md_piece: str) -> str:
    h = md_to_html(md_piece) if md_piece else ""
    if not label:
        return h
    return f"<b>{esc(label)}</b>\n<blockquote>{h}</blockquote>"

def build_containers(sections: List[Dict[str, Any]], has_media_caption: bool) -> List[str]:
    containers: List[str] = []
    current_blocks: List[str] = []
    current_len = 0
    is_caption = has_media_caption
    limit = CAPTION_MAX if is_caption else MSG_MAX

    def flush():
        nonlocal current_blocks, current_len, is_caption, limit
        if current_blocks:
            containers.append("\n\n".join(current_blocks))
            current_blocks = []
            current_len = 0
        is_caption = False
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

# ======================= ساخت فوتر کامل =======================
def build_footer(info: dict, post_url: str, lang: str, target: str, translated: bool, total_sec: float) -> str:
    dt = fmt_dates(datetime.datetime.now().timestamp())
    dur = info.get("duration")
    dur_str = f" | {dur//60:02d}:{dur%60:02d}" if dur else ""
    cand_id = info.get("uploader_id") or ""
    cand_name = info.get("uploader") or cand_id or "Instagram User"
    user_url = f"https://instagram.com/{cand_id}" if cand_id else post_url

    lines = ["------"]
    lines.append(f"<b>Message link:</b> <a href=\"{esc(post_url)}\">Open Instagram Link</a>")
    lines.append(f"<b>Type:</b> Video{dur_str} | Instagram")
    tr_info = f" (translated to {esc(target)})" if translated else ""
    lines.append(f"<b>Language:</b> {esc(lang)}{tr_info}")
    lines.append(f"<b>Sender:</b> <a href=\"{esc(user_url)}\">{esc(cand_name)}</a>")
    lines.append(f"<b>TimeStamp Date:</b> {dt['iso']}")
    lines.append(f"<b>Time Tehran:</b> {dt['shamsi']}")
    lines.append(f"<b>Processing:</b> {total_sec:.1f}s")
    return "\n" + "\n".join(lines)

# ======================= ارسال ایمن مدیا به تلگرام =======================
async def send_video_robust(
    client: httpx.AsyncClient,
    chat_id: int,
    video_path: str,
    thumb_path: Optional[str],
    caption: str = "",
    width: Optional[int] = None,
    height: Optional[int] = None,
    duration: Optional[int] = None
) -> Tuple[bool, httpx.Response]:
    data = {
        "chat_id": chat_id,
        "supports_streaming": "true"
    }
    if caption:
        data["caption"] = caption
        data["parse_mode"] = "HTML"
    if width:
        data["width"] = width
    if height:
        data["height"] = height
    if duration:
        data["duration"] = duration

    # تلاش اول: ارسال با کپشن و تامبنیل
    with open(video_path, "rb") as vf:
        files = {"video": (os.path.basename(video_path), vf, "video/mp4")}
        if thumb_path and os.path.exists(thumb_path):
            with open(thumb_path, "rb") as tf:
                files["thumbnail"] = ("thumb.jpg", tf, "image/jpeg")
                resp = await client.post(f"{TELEGRAM_API}/sendVideo", data=data, files=files, timeout=300.0)
        else:
            resp = await client.post(f"{TELEGRAM_API}/sendVideo", data=data, files=files, timeout=300.0)

    if resp.status_code == 200:
        return True, resp

    # تلاش دوم (در صورت رد شدن کپشن توسط تلگرام): ارسال فایل بدون کپشن
    fallback_data = {
        "chat_id": chat_id,
        "supports_streaming": "true"
    }
    if width: fallback_data["width"] = width
    if height: fallback_data["height"] = height
    if duration: fallback_data["duration"] = duration

    with open(video_path, "rb") as vf:
        files = {"video": (os.path.basename(video_path), vf, "video/mp4")}
        if thumb_path and os.path.exists(thumb_path):
            with open(thumb_path, "rb") as tf:
                files["thumbnail"] = ("thumb.jpg", tf, "image/jpeg")
                resp_retry = await client.post(f"{TELEGRAM_API}/sendVideo", data=fallback_data, files=files, timeout=300.0)
        else:
            resp_retry = await client.post(f"{TELEGRAM_API}/sendVideo", data=fallback_data, files=files, timeout=300.0)

    return False, resp_retry

# ======================= چرخه کامل دانلود، پردازش و ارسال =======================
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
                        "chat_id": chat_id, "message_id": status_msg_id, "text": f"⏳ {text}"
                    }, timeout=5.0)
                else:
                    r = await client.post(f"{TELEGRAM_API}/sendMessage", json={
                        "chat_id": chat_id, "text": f"⏳ {text}"
                    }, timeout=5.0)
                    if r.status_code == 200:
                        status_msg_id = r.json().get("result", {}).get("message_id")
            except Exception:
                pass

        uid = str(uuid.uuid4())[:8]
        output_template = f"/tmp/{uid}_%(id)s.%(ext)s"
        thumb_path = f"/tmp/{uid}_thumb.jpg"
        audio_path = f"/tmp/{uid}_audio.wav"
        downloaded_file = None

        try:
            # ۱. دانلود ویدیو با yt-dlp
            await dbg.step("Download with yt-dlp")
            await update_status("در حال دریافت اطلاعات و دانلود ویدیو از اینستاگرام...")
            ydl_opts = {
                'outtmpl': output_template,
                'format': 'bestvideo+bestaudio/best',
                'merge_output_format': 'mp4',
                'quiet': True,
                'no_warnings': True,
            }
            loop = asyncio.get_running_loop()
            info = await loop.run_in_executor(None, lambda: yt_dlp.YoutubeDL(ydl_opts).extract_info(url, download=True))
            downloaded_file = yt_dlp.YoutubeDL(ydl_opts).prepare_filename(info)
            if not os.path.exists(downloaded_file):
                downloaded_file = f"{os.path.splitext(downloaded_file)[0]}.mp4"

            # ۲. ساخت تامبنیل استاندارد
            await dbg.step("FFmpeg Thumbnail")
            cmd_thumb = f'ffmpeg -y -ss 00:00:00.500 -i "{downloaded_file}" -vframes 1 -vf "scale=\'if(gt(a,1),320,-2)\':\'if(gt(a,1),-2,320)\'" -q:v 2 "{thumb_path}"'
            subprocess.run(cmd_thumb, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # ۳. استخراج صوت با فرمت خالص WAV (PCM 16-bit 16kHz)
            await dbg.step("FFmpeg Extract Audio WAV")
            cmd_audio = f'ffmpeg -y -i "{downloaded_file}" -vn -c:a pcm_s16le -ar 16000 -ac 1 "{audio_path}"'
            subprocess.run(cmd_audio, shell=True, capture_output=True, text=True)
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 500:
                cmd_alt = f'ffmpeg -y -i "{downloaded_file}" -vn -ar 16000 -ac 1 "{audio_path}"'
                subprocess.run(cmd_alt, shell=True, capture_output=True, text=True)

            # ۴. تبدیل صدا به متن (Speech-to-Text با هوش مصنوعی)
            await dbg.step("AI Audio Transcription")
            await update_status("در حال تبدیل صوت به متن (Speech-to-Text با Gemini)...")
            extracted_text = ""
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                extracted_text = await ai_transcribe_audio(client, audio_path, dbg)

            # ۵. پردازش متن، ویرایش و ترجمه ترنسکریپت
            lang = detect_lang(extracted_text) if extracted_text else "fa"
            target = CONFIG["TRANSLATE_TO"]
            need_translate = target and (lang != target) and not (is_persian(lang) and is_persian(target))

            fm = {"title": "", "body": extracted_text, "formatted": False}
            tr_transcript = ""

            if extracted_text:
                if not need_translate:
                    await dbg.step("AI Spellcheck (Same Language)")
                    await update_status("در حال بهینه‌سازی و اصلاح ترنسکریپت...")
                    extracted_text = await ai_spell(client, extracted_text, lang, dbg)

                await dbg.step("AI Formatting & Title")
                await update_status("در حال ساخت عنوان و ساختاربندی ترنسکریپت...")
                title_task = ai_generate_title(client, extracted_text, lang, dbg)
                format_task = ai_format(client, extracted_text, dbg)
                title, fm = await asyncio.gather(title_task, format_task)
                fm["title"] = title

                if need_translate:
                    await dbg.step("AI Translation of Transcript")
                    await update_status("در حال ترجمه دقیق و روان ترنسکریپت به فارسی معیار...")
                    source_md = f"**{title}**\n\n{fm['body']}"
                    tr_transcript = await translate_markdown(client, source_md, lang, target, dbg)

            # ۶. پردازش و ترجمه کپشن اصلی پست اینستاگرام
            raw_caption = (info.get('description', '') or info.get('title', '') or "").strip()
            tr_caption = ""
            if raw_caption:
                cap_lang = detect_lang(raw_caption)
                if target and (cap_lang != target) and not (is_persian(cap_lang) and is_persian(target)):
                    await dbg.step("Translate Instagram Caption")
                    await update_status("در حال ترجمه کپشن پست اینستاگرام...")
                    tr_caption = await translate_markdown(client, raw_caption, cap_lang, target, dbg)

            # ۷. ساخت بخش‌های پیام
            await dbg.step("Build Message Sections")
            sections: List[Dict[str, Any]] = []

            # صوت ویدیو
            if extracted_text:
                if tr_transcript:
                    sections.append({"label": "🗣 ترجمه ترنسکریپت (فارسی):", "text": tr_transcript})
                    sections.append({"label": "🎙 متن اصلی صوت (Transcript):", "text": f"**{fm['title']}**\n\n{fm['body']}"})
                else:
                    sections.append({"label": "🎙 متن ترنسکریپت صوت:", "text": f"**{fm['title']}**\n\n{fm['body']}"})

            # کپشن اینستاگرام
            if raw_caption:
                if tr_caption:
                    sections.append({"label": "📝 ترجمه کپشن اینستاگرام:", "text": tr_caption})
                    sections.append({"label": "📄 کپشن اصلی اینستاگرام:", "text": raw_caption})
                else:
                    sections.append({"label": "📄 کپشن اینستاگرام:", "text": raw_caption})

            # فوتر نهایی
            elapsed = (datetime.datetime.now() - t0).total_seconds()
            footer_html = build_footer(info, url, lang if extracted_text else "none", target, bool(tr_transcript or tr_caption), elapsed)
            sections.append({"label": "", "text": footer_html, "isRawHtml": True})

            # ۸. بسته‌بندی بر اساس محدودیت کاراکتر تلگرام
            containers = build_containers(sections, has_media_caption=True)
            caption_for_media = containers[0] if containers else ""
            remaining_messages = containers[1:] if len(containers) > 1 else []

            # ۹. آپلود و ارسال ویدیو به صورت ایمن
            await dbg.step("Upload Video to Telegram")
            await update_status("در حال آپلود و ارسال ویدیو به تلگرام...")
            
            success_with_caption, _ = await send_video_robust(
                client=client,
                chat_id=chat_id,
                video_path=downloaded_file,
                thumb_path=thumb_path,
                caption=caption_for_media,
                width=info.get('width'),
                height=info.get('height'),
                duration=int(info['duration']) if info.get('duration') else None
            )

            # اگر ارسال با کپشن ناموفق بود و ویدیو خام ارسال شد، کل متن‌ها به صورت پیام متنی ارسال می‌شوند
            if not success_with_caption:
                remaining_messages = containers

            # ۱۰. ارسال پیام‌های تکمیلی
            for extra_msg in remaining_messages:
                await client.post(f"{TELEGRAM_API}/sendMessage", json={
                    "chat_id": chat_id,
                    "text": extra_msg,
                    "parse_mode": "HTML",
                    "link_preview_options": {"is_disabled": True}
                }, timeout=30.0)

            # ۱۱. حذف پیام وضعیت موقت
            if status_msg_id:
                await client.post(f"{TELEGRAM_API}/deleteMessage", json={"chat_id": chat_id, "message_id": status_msg_id})
            await dbg.end(True)

        except Exception as e:
            await dbg.fatal("Handler Crash", e)
            if status_msg_id:
                try:
                    await client.post(f"{TELEGRAM_API}/editMessageText", json={
                        "chat_id": chat_id, "message_id": status_msg_id, "text": f"💥 خطا در پردازش:\n{str(e)[:400]}"
                    })
                except Exception:
                    pass
        finally:
            for p in [downloaded_file, thumb_path, audio_path]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

# ======================= حلقه دریافت پیام‌ها (Telegram Worker) =======================
async def telegram_worker():
    async with httpx.AsyncClient(timeout=40.0) as client:
        await client.get(f"{TELEGRAM_API}/deleteWebhook")
        offset = None
        ig_regex = re.compile(r'https?://(?:www\.)?instagram\.com/(?:reel|reels|p|tv)/[A-Za-z0-9_-]+')

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
                            url = match.group(0)
                            asyncio.create_task(process_and_send(chat_id, url))
            except Exception:
                await asyncio.sleep(2.0)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(telegram_worker())

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
