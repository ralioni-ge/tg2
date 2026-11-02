import asyncio
import datetime as dt
import html
import logging
import os
import re
import tempfile
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx
import uvicorn
import yt_dlp
from fastapi import FastAPI


BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GEMINI_KEYS = [key.strip() for key in os.getenv("GEMINI_KEYS", "").split(",") if key.strip()]
DEBUG_CHAT_ID = os.getenv("DEBUG_CHAT_ID", "").strip()
TRANSLATE_TO = os.getenv("TRANSLATE_TO", "fa").strip() or "fa"
MODEL = "gemini-3.5-flash-lite"
GOOGLE = "https://generativelanguage.googleapis.com"
TELEGRAM = f"https://api.telegram.org/bot{BOT_TOKEN}"
CAPTION_LIMIT = 850
TEXT_LIMIT = 3500
MAX_AUDIO_BYTES = 2 * 1024 ** 3
MAX_AUDIO_SECONDS = 9.5 * 3600
LOG = logging.getLogger("instagram_bot")
IG_LINK = re.compile(r"https?://(?:www\.|m\.)?instagram\.com/(?:reel|reels|p|tv)/[\w-]+", re.I)
JOBS: set[asyncio.Task] = set()
SEMAPHORE = asyncio.Semaphore(2)
key_index = 0


def next_key() -> str:
    global key_index
    if not GEMINI_KEYS:
        raise RuntimeError("GEMINI_KEYS تنظیم نشده است")
    key = GEMINI_KEYS[key_index % len(GEMINI_KEYS)]
    key_index += 1
    return key


def api_error(response: httpx.Response, service: str) -> RuntimeError:
    try:
        message = response.json().get("error", {}).get("message") or response.json().get("description")
    except ValueError:
        message = None
    return RuntimeError(f"{service} HTTP {response.status_code}: {(message or response.text)[:400]}")


async def telegram_call(client: httpx.AsyncClient, method: str, data: dict[str, Any], timeout=30) -> Any:
    response = await client.post(f"{TELEGRAM}/{method}", json=data, timeout=timeout)
    if response.status_code != 200:
        raise api_error(response, "Telegram")
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram: {payload.get('description', 'unknown error')}")
    return payload.get("result")


async def say(client: httpx.AsyncClient, chat_id: int, text: str, *, reply_to: Optional[int] = None) -> None:
    params: dict[str, Any] = {
        "chat_id": chat_id, "text": text, "parse_mode": "HTML",
        "link_preview_options": {"is_disabled": True},
    }
    if reply_to:
        params["reply_parameters"] = {"message_id": reply_to}
    await telegram_call(client, "sendMessage", params)


def source_info(url: str) -> dict[str, Any]:
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True,
            "format": "best[ext=mp4]/best", "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info:
        raise RuntimeError("اطلاعات پست دریافت نشد")
    if info.get("entries"):
        info = next((item for item in info["entries"] if item and item.get("formats")), None)
        if not info:
            raise RuntimeError("در این پست ویدیوی قابل پردازش پیدا نشد")
    return info


def select_video(info: dict[str, Any]) -> tuple[str, dict[str, str], dict[str, Any]]:
    # A single progressive MP4 is required; DASH video-only cannot be sent by URL.
    formats = [f for f in info.get("formats", [])
               if f.get("url", "").startswith("https://")
               and f.get("ext") == "mp4"
               and f.get("vcodec") not in (None, "none")
               and f.get("acodec") not in (None, "none")
               and str(f.get("protocol", "https")).startswith("http")]
    if formats:
        fmt = max(formats, key=lambda f: (f.get("height") or 0, f.get("tbr") or 0))
    elif (info.get("url", "").startswith("https://") and info.get("ext") == "mp4"
          and info.get("vcodec") not in (None, "none")
          and info.get("acodec") not in (None, "none")):
        fmt = info
    else:
        raise RuntimeError("لینک مستقیم MP4 شامل صدا و تصویر پیدا نشد")
    return fmt["url"], fmt.get("http_headers") or info.get("http_headers") or {}, fmt


async def send_video_url(client: httpx.AsyncClient, chat_id: int, video_url: str,
                         info: dict[str, Any], fmt: dict[str, Any], caption: str) -> int:
    # Telegram fetches this URL itself. It cannot use yt-dlp's cookies or custom headers.
    payload: dict[str, Any] = {"chat_id": chat_id, "video": video_url,
                                "supports_streaming": True}
    if caption:
        payload.update(caption=caption, parse_mode="HTML")
    for field in ("width", "height", "duration"):
        value = fmt.get(field) or info.get(field)
        if value:
            payload[field] = int(value)
    result = await telegram_call(client, "sendVideo", payload, timeout=180)
    return result["message_id"]


async def run_ffmpeg(*args: str) -> tuple[bool, str]:
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", *args,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=900)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise RuntimeError("ffmpeg timeout")
    return proc.returncode == 0, stderr.decode(errors="replace")[-400:]


async def extract_audio(url: str, headers: dict[str, str], folder: Path) -> tuple[Path, str]:
    # Only copy the AAC audio track. ffmpeg reads the remote video stream; it
    # never writes the video itself to disk. Cookies are not passed to Telegram.
    safe_headers = {k: v for k, v in headers.items()
                    if k.lower() in ("user-agent", "referer", "cookie") and "\n" not in v and "\r" not in v}
    header_args = ["-headers", "".join(f"{k}: {v}\r\n" for k, v in safe_headers.items())] if safe_headers else []
    source_args = [*header_args, "-reconnect", "1", "-reconnect_streamed", "1",
                   "-reconnect_delay_max", "5", "-i", url, "-map", "0:a:0", "-vn"]
    aac = folder / "audio.aac"
    ok, err = await run_ffmpeg(*source_args, "-c:a", "copy", "-f", "adts", str(aac))
    if ok and aac.exists() and aac.stat().st_size > 500:
        return aac, "audio/aac"
    # No lossy compression: decode an unsupported audio codec to raw PCM WAV.
    wav = folder / "audio.wav"
    ok_wav, err_wav = await run_ffmpeg(*source_args, "-c:a", "pcm_s16le", str(wav))
    if ok_wav and wav.exists() and wav.stat().st_size > 500:
        return wav, "audio/wav"
    raise RuntimeError(f"صدای ویدیو استخراج نشد: {err_wav or err}")


async def stream_file(path: Path, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]:
    with path.open("rb") as file:
        while chunk := await asyncio.to_thread(file.read, chunk_size):
            yield chunk


async def files_upload(client: httpx.AsyncClient, path: Path, mime: str,
                       key: str) -> dict[str, Any]:
    size = path.stat().st_size
    if size > MAX_AUDIO_BYTES:
        raise RuntimeError("حجم صدا از سقف ۲ گیگابایت Files API بیشتر است")
    auth = {"x-goog-api-key": key}
    start = await client.post(f"{GOOGLE}/upload/v1beta/files", headers={
        **auth, "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(size),
        "X-Goog-Upload-Header-Content-Type": mime,
        "Content-Type": "application/json",
    }, json={"file": {"display_name": path.name}}, timeout=30)
    if start.status_code != 200:
        raise api_error(start, "Files API start")
    upload_url = start.headers.get("x-goog-upload-url")
    if not upload_url:
        raise RuntimeError("Files API آدرس آپلود را برنگرداند")
    response = await client.post(upload_url, headers={
        "Content-Length": str(size), "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize",
    }, content=stream_file(path), timeout=900)
    if response.status_code != 200:
        raise api_error(response, "Files API upload")
    file = response.json().get("file") or {}
    if not file.get("name"):
        raise RuntimeError("Files API نام فایل آپلودشده را برنگرداند")
    return file


async def wait_for_file(client: httpx.AsyncClient, file: dict[str, Any], key: str) -> dict[str, Any]:
    # Poll even when initial state is missing, rather than treating it as ACTIVE.
    for attempt in range(150):
        state = file.get("state")
        if state == "ACTIVE":
            return file
        if state == "FAILED":
            raise RuntimeError(f"پردازش فایل در Gemini ناموفق بود: {file.get('error', '')}")
        if attempt:
            await asyncio.sleep(2)
        response = await client.get(f"{GOOGLE}/v1beta/{file['name']}",
                                    headers={"x-goog-api-key": key}, timeout=20)
        if response.status_code != 200:
            raise api_error(response, "Files API get")
        file = response.json()
    raise RuntimeError("زمان انتظار برای آماده شدن فایل Gemini تمام شد")


async def delete_file(client: httpx.AsyncClient, name: str, key: str) -> None:
    try:
        await client.delete(f"{GOOGLE}/v1beta/{name}",
                            headers={"x-goog-api-key": key}, timeout=20)
    except Exception:
        LOG.warning("Could not delete Gemini file %s", name)


async def generate(client: httpx.AsyncClient, text: str, *, instruction: str = "",
                   file_uri: str = "", mime: str = "", key: Optional[str] = None,
                   max_tokens: int = 65536) -> str:
    parts: list[dict[str, Any]] = [{"text": text}]
    if file_uri:
        parts.append({"fileData": {"fileUri": file_uri, "mimeType": mime}})
    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": max_tokens},
    }
    if instruction:
        body["systemInstruction"] = {"parts": [{"text": instruction}]}
    last_error: Optional[Exception] = None
    # Uploaded files must be used with the same key. Text-only requests rotate keys.
    for attempt in range(2 if key else max(2, len(GEMINI_KEYS) * 2)):
        current_key = key or next_key()
        try:
            response = await client.post(f"{GOOGLE}/v1beta/models/{MODEL}:generateContent",
                                         headers={"x-goog-api-key": current_key}, json=body, timeout=300)
            if response.status_code != 200:
                error = api_error(response, "Gemini")
                if response.status_code not in (429, 500, 502, 503, 504):
                    raise error
                last_error = error
            else:
                payload = response.json()
                candidates = payload.get("candidates") or []
                candidate = candidates[0] if candidates else {}
                finish = candidate.get("finishReason")
                if finish == "MAX_TOKENS":
                    raise RuntimeError("پاسخ Gemini به سقف توکن رسید؛ متن ناقص ارسال نمی شود")
                result = "".join(p.get("text", "") for p in candidate.get("content", {}).get("parts", [])
                                 if not p.get("thought")).strip()
                if not result:
                    raise RuntimeError(f"خروجی Gemini خالی است: {finish or payload.get('promptFeedback', '')}")
                return result
        except httpx.TransportError as error:
            last_error = error
        if attempt < (1 if key else max(2, len(GEMINI_KEYS) * 2) - 1):
            await asyncio.sleep(min(2 * (attempt + 1), 8))
    raise RuntimeError(f"Gemini در دسترس نیست: {last_error}")


async def transcribe(client: httpx.AsyncClient, path: Path, mime: str) -> str:
    last_error: Optional[Exception] = None
    for _ in range(len(GEMINI_KEYS)):
        key = next_key()
        uploaded: Optional[dict[str, Any]] = None
        try:
            uploaded = await files_upload(client, path, mime, key)
            active = await wait_for_file(client, uploaded, key)
            result = await generate(
                client,
                "Transcribe every spoken word in its original language. Do not summarize. "
                "Return only the transcript. If there is no speech, return NO_SPEECH.",
                instruction="You are an accurate audio transcriber.",
                file_uri=active["uri"], mime=active.get("mimeType") or mime, key=key,
            )
            return "" if result.strip().upper().strip(".[] ") == "NO_SPEECH" else result
        except Exception as error:
            last_error = error
            LOG.warning("Transcription attempt failed: %s", error)
        finally:
            if uploaded and uploaded.get("name"):
                await delete_file(client, uploaded["name"], key)
    raise RuntimeError(f"تبدیل صوت به متن ناموفق بود: {last_error}")


def detect_language(text: str) -> str:
    return "fa" if len(re.findall(r"[\u0600-\u06ff]", text)) > len(re.findall(r"[A-Za-z]", text)) else "en"


async def make_text(client: httpx.AsyncClient, transcript: str, original_caption: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    if transcript:
        lang = detect_language(transcript)
        try:
            corrected = await generate(client, transcript, instruction=(
                "Correct punctuation and recognition errors only. Keep every detail and original language. "
                "Output only the corrected transcript."))
        except Exception as error:
            LOG.warning("Transcript correction failed: %s", error)
            corrected = transcript
        sections.append(("متن صوت", corrected))
        if lang != TRANSLATE_TO:
            try:
                translation = await generate(client, corrected, instruction=(
                    f"Translate faithfully into {TRANSLATE_TO}, in formal written style. "
                    "Preserve paragraph breaks, names and numbers. Output only the translation."))
                sections.insert(0, ("ترجمه صوت", translation))
            except Exception as error:
                LOG.warning("Transcript translation failed: %s", error)
    if original_caption:
        sections.append(("کپشن اصلی", original_caption))
        if detect_language(original_caption) != TRANSLATE_TO:
            try:
                translation = await generate(client, original_caption, instruction=(
                    f"Translate faithfully into {TRANSLATE_TO} in formal written style. "
                    "Output only the translation."))
                sections.append(("ترجمه کپشن", translation))
            except Exception as error:
                LOG.warning("Caption translation failed: %s", error)
    return sections


def split_plain(text: str, limit: int) -> list[str]:
    chunks = []
    text = text.strip()
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 3:
            cut = text.rfind(" ", 0, limit)
        if cut < limit // 3:
            cut = limit
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    return chunks


async def send_sections(client: httpx.AsyncClient, chat_id: int,
                        sections: list[tuple[str, str]], reply_to: Optional[int], url: str) -> None:
    for label, text in sections:
        for index, piece in enumerate(split_plain(text, TEXT_LIMIT)):
            heading = f"{label} ({index + 1})" if index else label
            await say(client, chat_id, f"<b>{html.escape(heading)}</b>\n{html.escape(piece)}", reply_to=reply_to)
    await say(client, chat_id, f'<a href="{html.escape(url, quote=True)}">لینک پست اینستاگرام</a>', reply_to=reply_to)


async def process(chat_id: int, post_url: str) -> None:
    async with httpx.AsyncClient() as client:
        status: Optional[int] = None
        video_message: Optional[int] = None
        video_error: Optional[str] = None
        try:
            status_result = await telegram_call(client, "sendMessage", {
                "chat_id": chat_id, "text": "در حال دریافت اطلاعات پست..."})
            status = status_result["message_id"]
            info = await asyncio.to_thread(source_info, post_url)
            direct_url, cdn_headers, fmt = select_video(info)
            original_caption = str(info.get("description") or info.get("title") or "").strip()

            # Send the video early. A failed URL must not prevent transcription.
            caption = html.escape(original_caption[:CAPTION_LIMIT]) if original_caption else ""
            try:
                video_message = await send_video_url(client, chat_id, direct_url, info, fmt, caption)
            except Exception as error:
                video_error = str(error)
                LOG.warning("sendVideo URL failed: %s", error)
                await say(client, chat_id,
                          "تلگرام ویدیو را از لینک مستقیم نپذیرفت (سقف URL: ۲۰ مگابایت، "
                          "یا CDN نیازمند دسترسی ویژه است). ویدیو دانلود/آپلود نمی شود.\n"
                          f"<code>{html.escape(video_error[:250])}</code>\n"
                          f'<a href="{html.escape(post_url, quote=True)}">مشاهده پست اصلی</a>')

            await telegram_call(client, "editMessageText", {
                "chat_id": chat_id, "message_id": status, "text": "در حال استخراج و پردازش صدا..."})
            transcript = ""
            try:
                with tempfile.TemporaryDirectory(prefix="ig_audio_") as tmp:
                    audio_path, mime = await extract_audio(direct_url, cdn_headers, Path(tmp))
                    duration = float(info.get("duration") or 0)
                    if duration > MAX_AUDIO_SECONDS:
                        raise RuntimeError("مدت صدا از ۹.۵ ساعت بیشتر است")
                    transcript = await transcribe(client, audio_path, mime)
            except Exception as error:
                LOG.exception("Audio processing failed")
                await say(client, chat_id, f"پردازش صدا ناموفق بود: <code>{html.escape(str(error)[:300])}</code>",
                          reply_to=video_message)

            sections = await make_text(client, transcript, original_caption)
            await send_sections(client, chat_id, sections, video_message, post_url)
            if video_error:
                LOG.info("Text completed despite video URL failure: %s", video_error)
        except Exception as error:
            LOG.exception("Job failed")
            await say(client, chat_id, f"خطا در پردازش پست: <code>{html.escape(str(error)[:350])}</code>")
        finally:
            if status:
                with suppress(Exception):
                    await telegram_call(client, "deleteMessage", {"chat_id": chat_id, "message_id": status})


async def guarded(chat_id: int, url: str) -> None:
    async with SEMAPHORE:
        await process(chat_id, url)


async def worker() -> None:
    async with httpx.AsyncClient(timeout=40) as client:
        await telegram_call(client, "deleteWebhook", {"drop_pending_updates": False})
        offset: Optional[int] = None
        while True:
            try:
                args: dict[str, Any] = {"timeout": 20, "allowed_updates": ["message", "channel_post"]}
                if offset is not None:
                    args["offset"] = offset
                response = await client.get(f"{TELEGRAM}/getUpdates", params=args, timeout=35)
                if response.status_code != 200:
                    raise api_error(response, "getUpdates")
                for update in response.json().get("result", []):
                    offset = update["update_id"] + 1
                    msg = update.get("message") or update.get("channel_post") or {}
                    text = msg.get("text") or msg.get("caption") or ""
                    match = IG_LINK.search(text)
                    chat_id = (msg.get("chat") or {}).get("id")
                    if match and chat_id:
                        task = asyncio.create_task(guarded(chat_id, match.group(0)))
                        JOBS.add(task)
                        task.add_done_callback(JOBS.discard)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                LOG.warning("Polling failed: %s", error)
                await asyncio.sleep(3)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not BOT_TOKEN or not GEMINI_KEYS:
        raise RuntimeError("BOT_TOKEN و GEMINI_KEYS باید تنظیم شوند")
    task = asyncio.create_task(worker())
    try:
        yield
    finally:
        task.cancel()
        for job in JOBS:
            job.cancel()
        await asyncio.gather(task, *JOBS, return_exceptions=True)


app = FastAPI(lifespan=lifespan)


@app.get("/")
def health_check():
    return {"status": "ok", "model": MODEL}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
  