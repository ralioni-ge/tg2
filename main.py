import os
import re
import uuid
import html
import asyncio
import threading
import subprocess
import httpx
import yt_dlp
import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "Bot is running 24/7!"}

BOT_TOKEN = "9005346823:AAHe2Pqnx5xJgYNZMHyw2_uTeD93glTshXc"
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

async def send_msg(client, chat_id, text):
    try:
        await client.post(f"{TELEGRAM_API}/sendMessage", json={"chat_id": chat_id, "text": text}, timeout=15.0)
    except Exception:
        pass

def format_caption(info, post_url):
    raw_caption = info.get('description', '') or info.get('title', '')
    
    # تفکیک دقیق نام نمایشی و آیدی انگلیسی
    cand_id = info.get('uploader_id') or ""
    cand_uploader = info.get('uploader') or ""
    cand_channel = info.get('channel') or ""

    username = ""
    display_name = ""

    # بررسی اینکه کدام مقدار حاوی آیدی استاندارد (بدون ایموجی و فاصله) است
    if re.match(r'^[a-zA-Z0-9._]+$', cand_id):
        username = cand_id
        display_name = cand_uploader or cand_channel or cand_id
    elif re.match(r'^[a-zA-Z0-9._]+$', cand_uploader):
        username = cand_uploader
        display_name = cand_channel or cand_id or cand_uploader
    else:
        username = cand_id or cand_uploader

    if not display_name:
        display_name = username

    # ۱. خط لینک پست اینستاگرام
    link_line = f'🔗 <a href="{post_url}">Instagram Link</a>'
    
    # ۲. خط مشخصات سازنده (هایپرلینک فقط روی آیدی انگلیسی)
    user_line = ""
    if username:
        user_url = f"https://instagram.com/{username}"
        if display_name and display_name.strip() != username.strip():
            user_line = f"👤 {html.escape(display_name)} (<a href=\"{user_url}\">@{html.escape(username)}</a>)"
        else:
            user_line = f"👤 <a href=\"{user_url}\">@{html.escape(username)}</a>"

    footer = f"\n\n{link_line}"
    if user_line:
        footer += f"\n{user_line}"

    max_caption_len = 1000 - len(footer)
    if len(raw_caption) > max_caption_len:
        raw_caption = raw_caption[:max_caption_len] + "..."

    return f"{html.escape(raw_caption)}{footer}"

async def process_and_send(chat_id, url):
    async with httpx.AsyncClient(timeout=300.0) as client:
        await send_msg(client, chat_id, "⏳ در حال دانلود و بهینه‌سازی ویدیو...")
        
        unique_id = str(uuid.uuid4())[:8]
        output_template = f"/tmp/{unique_id}_%(id)s.%(ext)s"
        thumb_path = f"/tmp/{unique_id}_thumb.jpg"

        ydl_opts = {
            'outtmpl': output_template,
            'format': 'bestvideo+bestaudio/b',
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
        }

        downloaded_file = None
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_file = ydl.prepare_filename(info)
                if not os.path.exists(downloaded_file):
                    base, _ = os.path.splitext(downloaded_file)
                    downloaded_file = f"{base}.mp4"
                
                caption = format_caption(info, url)

            # ۱. ساخت خودکار تامبنیل با کیفیت از ثانیه ۰.۵ ویدیو برای حذف تصویر سیاه
            cmd = f'ffmpeg -y -ss 00:00:00.500 -i "{downloaded_file}" -vframes 1 -vf "scale=320:-2" -q:v 2 "{thumb_path}"'
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # ۲. آماده‌سازی فایل‌ها برای ارسال
            with open(downloaded_file, "rb") as vf:
                filename = os.path.basename(downloaded_file)
                files = {"video": (filename, vf, "video/mp4")}
                
                tf = None
                if os.path.exists(thumb_path):
                    tf = open(thumb_path, "rb")
                    files["thumbnail"] = ("thumb.jpg", tf, "image/jpeg")

                data = {
                    "chat_id": chat_id,
                    "caption": caption,
                    "parse_mode": "HTML",
                    "supports_streaming": "true"  # فعال‌سازی استریم و پیش‌نمایش فوری در تلگرام
                }
                
                await client.post(f"{TELEGRAM_API}/sendVideo", data=data, files=files, timeout=300.0)

                if tf:
                    tf.close()

        except Exception as e:
            await send_msg(client, chat_id, f"❌ خطا: {str(e)}")
        finally:
            if downloaded_file and os.path.exists(downloaded_file):
                try:
                    os.remove(downloaded_file)
                except Exception:
                    pass
            if os.path.exists(thumb_path):
                try:
                    os.remove(thumb_path)
                except Exception:
                    pass

async def telegram_worker():
    async with httpx.AsyncClient(timeout=30.0) as client:
        await client.get(f"{TELEGRAM_API}/deleteWebhook")
        offset = None
        while True:
            try:
                params = {"timeout": 20}
                if offset:
                    params["offset"] = offset

                resp = await client.get(f"{TELEGRAM_API}/getUpdates", params=params)
                if resp.status_code == 200:
                    for update in resp.json().get("result", []):
                        offset = update["update_id"] + 1
                        msg = update.get("message", {})
                        text = msg.get("text", "")
                        chat_id = msg.get("chat", {}).get("id")

                        if not chat_id or not text:
                            continue

                        if text == "/start":
                            await send_msg(client, chat_id, "سلام! ربات فعال است. لینک ریلز بفرستید.")
                            continue

                        match = re.search(r"https?://(?:www\.)?instagram\.com/(?:p|reel|tv)/[a-zA-Z0-9_\-]+", text)
                        if match:
                            asyncio.create_task(process_and_send(chat_id, match.group(0)))
            except Exception:
                await asyncio.sleep(2)

def run_bot():
    asyncio.run(telegram_worker())

threading.Thread(target=run_bot, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)