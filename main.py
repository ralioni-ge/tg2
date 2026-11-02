import os
import re
import uuid
import html
import asyncio
import datetime
import zoneinfo
import subprocess
import base64
from typing import Optional, List, Dict, Any

import httpx
import yt_dlp
import uvicorn
from fastapi import FastAPI


# ======================= پیکربندی و توکن‌ها از محیط =======================
BOT_TOKEN = os.getenv("BOT_TOKEN", "9005346823:AAHe2Pqnx5xJgYNZMHyw2_uTeD93glTshXc")
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

# خواندن کلیدهای جمینای که با کاما (,) جدا شده‌اند
raw_keys = os.getenv("GEMINI_KEYS", "")
if raw_keys:
    GEMINI_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]
else:
    # اگر در محیط تعریف نشده بود، لیست خالی نباشد تا برنامه کرش نکند
    GEMINI_KEYS = []

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "Bot is running 24/7!"}

# ======================= مدیریت کلیدهای Gemini =======================
_key_index = 0
def get_next_gemini_key() -> str:
    global _key_index
    key = GEMINI_KEYS[_key_index % len(GEMINI_KEYS)]
    _key_index += 1
    return key

async def call_gemini(client: httpx.AsyncClient, parts: List[Dict[str, Any]], system_instruction: str = "") -> str:
    max_attempts = len(GEMINI_KEYS) * 2
    last_err = None

    body: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096}
    }
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    for _ in range(max_attempts):
        api_key = get_next_gemini_key()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
        try:
            resp = await client.post(url, json=body, timeout=60.0)
            if resp.status_code == 429 or resp.status_code >= 500:
                await asyncio.sleep(1.0)
                continue
            
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                cand_parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in cand_parts).strip()
                if text:
                    return text
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.8)

    if last_err:
        raise last_err
    return ""

# ======================= ابزارهای متنی و تاریخ =======================
def normalize_fa(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\u064A', '\u06CC').replace('\u0643', '\u06A9') # تبدیل ي و ك عربی
    text = re.sub(r'(^|\s)(ن?می) (?=[\u0600-\u06FF])', r'\1\2\u200c', text) # نیم‌فاصله‌ی "می"
    text = re.sub(r'([\u0600-\u06FF]) ها(?=$|[\s.,،؟?!؛:])', r'\1\u200cها', text)
    return text.strip()

def detect_lang(text: str) -> str:
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    latin_chars = len(re.findall(r'[A-Za-z]', text))
    return "fa" if arabic_chars > latin_chars else "en"

def gregorian_to_jalali(gy, gm, gd):
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

def get_formatted_timestamps() -> tuple:
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    utc_str = utc_now.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    tehran_tz = zoneinfo.ZoneInfo("Asia/Tehran")
    teh_now = utc_now.astimezone(tehran_tz)
    jy, jm, jd = gregorian_to_jalali(teh_now.year, teh_now.month, teh_now.day)
    shamsi_str = f"{jy:04d}-{jm:02d}-{jd:02d} {teh_now.strftime('%H:%M:%S')}"
    return utc_str, shamsi_str

# ======================= پردازش‌های صوتی و ترجمه با Gemini =======================
async def ai_transcribe_audio(client: httpx.AsyncClient, audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        "Transcribe all speech in this audio verbatim in its original language. "
        "If there is no speech, ambient sounds only, or music only, reply with: [NO_SPEECH]. "
        "Do not explain, provide ONLY the transcription."
    )
    parts = [
        {"inline_data": {"mime_type": "audio/mp3", "data": audio_b64}},
        {"text": prompt}
    ]
    res = await call_gemini(client, parts)
    if "[NO_SPEECH]" in res or len(res.strip()) < 4:
        return ""
    return res

async def ai_translate_to_persian(client: httpx.AsyncClient, text: str, is_speech: bool = False) -> str:
    if not text.strip():
        return ""
    
    style_guide = (
        "شما یک مترجم مسلط و حرفه‌ای هستید. متن زیر را به فارسی روان، دقیق و معیار (کتابی و استاندارد) ترجمه کن. "
        "از اصطلاحات عامیانه بی‌مورد یا ترجمه واژه‌به‌واژه خودداری کن. لحن گوینده را حفظ کن. "
        "تنها خروجی شما باید ترجمه نهایی باشد."
    )
    if is_speech:
        style_guide += " متن مربوط به فایل صوتی است؛ بنابراین جمله‌بندی را منسجم و مرتب کن."

    parts = [{"text": f"Text to translate:\n\n{text}"}]
    translated = await call_gemini(client, parts, system_instruction=style_guide)
    return normalize_fa(translated)

# ======================= مدیریت پیام‌های تلگرام =======================
async def edit_or_send_status(client: httpx.AsyncClient, chat_id: int, text: str, msg_id: Optional[int] = None) -> Optional[int]:
    try:
        if msg_id:
            await client.post(f"{TELEGRAM_API}/editMessageText", json={
                "chat_id": chat_id, "message_id": msg_id, "text": text
            }, timeout=10.0)
            return msg_id
        else:
            r = await client.post(f"{TELEGRAM_API}/sendMessage", json={
                "chat_id": chat_id, "text": text
            }, timeout=10.0)
            if r.status_code == 200:
                return r.json().get("result", {}).get("message_id")
    except Exception:
        pass
    return msg_id

async def delete_msg(client: httpx.AsyncClient, chat_id: int, message_id: int):
    try:
        await client.post(f"{TELEGRAM_API}/deleteMessage", json={
            "chat_id": chat_id, "message_id": message_id
        }, timeout=10.0)
    except Exception:
        pass

def build_post_blocks(info: dict, post_url: str, transcript: str, transcript_fa: str, caption_fa: str) -> List[str]:
    raw_caption = info.get('description', '') or info.get('title', '') or ""
    cand_id = info.get('uploader_id') or ""
    cand_uploader = info.get('uploader') or ""
    username = cand_id if re.match(r'^[a-zA-Z0-9._]+$', cand_id) else (cand_uploader or "Instagram User")
    display_name = cand_uploader or username

    sections = []

    # ۱. ترنسکریپت و ترجمه آن
    if transcript_fa:
        sections.append(f"<b>🗣 ترجمه ترنسکریپت صوت:</b>\n<blockquote>{html.escape(transcript_fa)}</blockquote>")
        if transcript != transcript_fa:
            sections.append(f"<b>🎙 متن اصلی صوت:</b>\n<blockquote>{html.escape(transcript)}</blockquote>")
    elif transcript:
        sections.append(f"<b>🎙 متن صوت:</b>\n<blockquote>{html.escape(transcript)}</blockquote>")

    # ۲. کپشن پست و ترجمه آن
    if caption_fa:
        sections.append(f"<b>📝 ترجمه کپشن پست:</b>\n<blockquote>{html.escape(caption_fa)}</blockquote>")
        if raw_caption != caption_fa:
            sections.append(f"<b>📄 کپشن اصلی پست:</b>\n<blockquote>{html.escape(raw_caption)}</blockquote>")
    elif raw_caption:
        sections.append(f"<b>📄 کپشن پست:</b>\n<blockquote>{html.escape(raw_caption)}</blockquote>")

    # ۳. فوتر کامل
    utc_str, shamsi_str = get_formatted_timestamps()
    user_url = f"https://instagram.com/{username}"
    footer_lines = [
        "──────────────",
        f'🔗 <b>لینک پست:</b> <a href="{post_url}">Instagram Link</a>',
        f'👤 <b>سازنده:</b> <a href="{user_url}">@{html.escape(username)}</a> ({html.escape(display_name)})',
        f'🕒 <b>زمان (تهران):</b> <code>{shamsi_str}</code>',
        f'🌐 <b>زمان (UTC):</b> <code>{utc_str}</code>'
    ]
    sections.append("\n".join(footer_lines))
    return sections

def pack_containers(sections: List[str], max_first_len: int = 1000, max_other_len: int = 3900) -> List[str]:
    containers = []
    current_chunk = []
    current_len = 0
    limit = max_first_len

    for sec in sections:
        sec_len = len(sec)
        sep_len = 2 if current_chunk else 0

        if current_len + sep_len + sec_len <= limit:
            current_chunk.append(sec)
            current_len += sep_len + sec_len
        else:
            if current_chunk:
                containers.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0
                limit = max_other_len

            # اگر خود بلوک بزرگتر از حد مجاز یک پیام باشد
            if len(sec) > limit:
                words = sec.split(" ")
                sub_part = ""
                for w in words:
                    if len(sub_part) + len(w) + 1 <= limit:
                        sub_part += (" " if sub_part else "") + w
                    else:
                        containers.append(sub_part)
                        sub_part = w
                        limit = max_other_len
                if sub_part:
                    current_chunk.append(sub_part)
                    current_len = len(sub_part)
            else:
                current_chunk.append(sec)
                current_len = len(sec)

    if current_chunk:
        containers.append("\n\n".join(current_chunk))

    return containers

# ======================= وظیفه دانلود، پردازش و ارسال =======================
async def process_and_send(chat_id: int, url: str):
    async with httpx.AsyncClient(timeout=300.0) as client:
        status_id = await edit_or_send_status(client, chat_id, "⏳ در حال بررسی و دانلود ویدیو از اینستاگرام...")
        
        uid = str(uuid.uuid4())[:8]
        output_template = f"/tmp/{uid}_%(id)s.%(ext)s"
        thumb_path = f"/tmp/{uid}_thumb.jpg"
        audio_path = f"/tmp/{uid}_audio.mp3"

        ydl_opts = {
            'outtmpl': output_template,
            'format': 'bestvideo+bestaudio/best',
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
        }

        downloaded_file = None
        try:
            # ۱. دانلود ویدیو با yt-dlp
            loop = asyncio.get_running_loop()
            info = await loop.run_in_executor(None, lambda: yt_dlp.YoutubeDL(ydl_opts).extract_info(url, download=True))
            downloaded_file = yt_dlp.YoutubeDL(ydl_opts).prepare_filename(info)
            if not os.path.exists(downloaded_file):
                base, _ = os.path.splitext(downloaded_file)
                downloaded_file = f"{base}.mp4"

            video_width = info.get('width')
            video_height = info.get('height')
            duration = int(info.get('duration') or 0)
            raw_caption = info.get('description', '') or info.get('title', '') or ""

            # ۲. ساخت تصویر پیش‌نمایش (Thumbnail) بدون دفرمه شدن
            cmd_thumb = (
                f'ffmpeg -y -ss 00:00:00.500 -i "{downloaded_file}" -vframes 1 '
                f'-vf "scale=\'if(gt(a,1),320,-2)\':\'if(gt(a,1),-2,320)\'" -q:v 2 "{thumb_path}"'
            )
            subprocess.run(cmd_thumb, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # ۳. استخراج صوت با حجم بسیار پایین (جهت ترنسکریپت فوق سریع)
            status_id = await edit_or_send_status(client, chat_id, "🎧 در حال استخراج صوت و ترنسکریپت با Gemini...", status_id)
            cmd_audio = f'ffmpeg -y -i "{downloaded_file}" -vn -acodec libmp3lame -b:a 48k -ac 1 -ar 16000 "{audio_path}"'
            subprocess.run(cmd_audio, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            transcript = ""
            transcript_fa = ""
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                transcript = await ai_transcribe_audio(client, audio_path)
                if transcript:
                    if detect_lang(transcript) != "fa":
                        status_id = await edit_or_send_status(client, chat_id, "🌐 در حال ترجمه ترنسکریپت به فارسی روان...", status_id)
                        transcript_fa = await ai_translate_to_persian(client, transcript, is_speech=True)
                    else:
                        transcript = normalize_fa(transcript)
                        transcript_fa = transcript

            # ۴. ترجمه کپشن اصلی در صورت نیاز
            caption_fa = ""
            if raw_caption.strip():
                if detect_lang(raw_caption) != "fa":
                    status_id = await edit_or_send_status(client, chat_id, "📝 در حال ترجمه کپشن اینستاگرام...", status_id)
                    caption_fa = await ai_translate_to_persian(client, raw_caption, is_speech=False)
                else:
                    caption_fa = normalize_fa(raw_caption)

            # ۵. ساخت بلاک‌های متنی نهایی
            sections = build_post_blocks(info, url, transcript, transcript_fa, caption_fa)
            messages = pack_containers(sections, max_first_len=1000, max_other_len=3900)

            video_caption = messages[0] if messages else ""
            remaining_messages = messages[1:] if len(messages) > 1 else []

            # ۶. ارسال ویدیو به تلگرام
            status_id = await edit_or_send_status(client, chat_id, "🚀 در حال آپلود ویدیو به تلگرام...", status_id)
            
            with open(downloaded_file, "rb") as vf:
                filename = os.path.basename(downloaded_file)
                files = {"video": (filename, vf, "video/mp4")}
                
                tf = None
                if os.path.exists(thumb_path):
                    tf = open(thumb_path, "rb")
                    files["thumbnail"] = ("thumb.jpg", tf, "image/jpeg")

                data = {
                    "chat_id": chat_id,
                    "caption": video_caption,
                    "parse_mode": "HTML",
                    "supports_streaming": "true"
                }
                if video_width:
                    data["width"] = video_width
                if video_height:
                    data["height"] = video_height
                if duration:
                    data["duration"] = duration

                await client.post(f"{TELEGRAM_API}/sendVideo", data=data, files=files, timeout=300.0)
                if tf:
                    tf.close()

            # ارسال ادامه متن‌ها اگر متن طولانی بود
            for extra_text in remaining_messages:
                await client.post(f"{TELEGRAM_API}/sendMessage", json={
                    "chat_id": chat_id,
                    "text": extra_text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True
                }, timeout=30.0)

            # پاک کردن پیام وضعیت
            if status_id:
                await delete_msg(client, chat_id, status_id)

        except Exception as e:
            await edit_or_send_status(client, chat_id, f"❌ خطا در پردازش: {str(e)}", status_id)
        finally:
            # پاک‌سازی فایل‌های موقت
            for path in [downloaded_file, thumb_path, audio_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

# ======================= حلقه دریافت آپدیت‌ها (Telegram Worker) =======================
async def telegram_worker():
    async with httpx.AsyncClient(timeout=40.0) as client:
        await client.get(f"{TELEGRAM_API}/deleteWebhook")
        offset = None
        ig_pattern = re.compile(r'https?://(?:www\.)?instagram\.com/(?:reel|p|tv)/[A-Za-z0-9_-]+')

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

                        # جستجوی لینک اینستاگرام در پیام
                        match = ig_pattern.search(text)
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
