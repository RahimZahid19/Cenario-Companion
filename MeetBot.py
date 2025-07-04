import asyncio
from playwright.async_api import async_playwright
from dotenv import load_dotenv
import os
import re
from datetime import datetime
import threading
import asyncio
import json
import sys
import platform

# Windows-specific asyncio fix
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Load credentials (kept for fallback if needed)
load_dotenv()
EMAIL = os.getenv("GOOGLE_EMAIL")
PASSWORD = os.getenv("GOOGLE_PASSWORD")

# Global variables to track bot state
bot_running = False
bot_page = None
bot_browser = None


class CaptionLogger:
    def __init__(self, filename="chat.txt", known_speakers=None):
        self.filename = filename
        self.current_speaker = None
        self.current_text = ""
        self.last_logged_text = ""
        self.last_container_hash = ""
        self.known_speakers = known_speakers or []

    def process_container_text(self, container_text):
        if not container_text:
            return
        container_hash = hash(container_text)
        if container_hash == self.last_container_hash:
            return
        self.last_container_hash = container_hash

        speaker, text = self.parse_current_speaker_and_text(container_text)
        if speaker and text:
            if self.current_speaker and self.current_speaker != speaker:
                if (
                    self.current_text.strip()
                    and self.current_text.strip() != self.last_logged_text
                ):
                    self.log_sentence(self.current_speaker, self.current_text.strip())
                    self.last_logged_text = self.current_text.strip()
                self.current_speaker = speaker
                self.current_text = text
            elif self.current_speaker == speaker:
                if len(text) > len(self.current_text):
                    self.current_text = text
            else:
                self.current_speaker = speaker
                self.current_text = text

    def parse_current_speaker_and_text(self, container_text):
        text = (
            container_text.replace("arrow_downward", "")
            .replace("Jump to bottom", "")
            .strip()
        )
        last_speaker = None
        last_speaker_pos = -1
        for speaker in self.known_speakers:
            pos = text.rfind(speaker)
            if pos > last_speaker_pos:
                last_speaker = speaker
                last_speaker_pos = pos
        if last_speaker:
            speaker_text = text[last_speaker_pos + len(last_speaker) :].strip()
            return last_speaker, speaker_text
        return None, ""

    def log_sentence(self, speaker, sentence):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {speaker}: {sentence}\n"
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(log_entry)


async def cleanup_playwright_resources():
    """Clean up Playwright resources to prevent asyncio pipe errors (async version)"""
    global bot_page, bot_browser
    try:
        if bot_page:
            try:
                await bot_page.close()
            except Exception as e:
                print(f"Error closing page: {e}")
            bot_page = None
        if bot_browser:
            try:
                await bot_browser.close()
            except Exception as e:
                print(f"Error closing browser: {e}")
            bot_browser = None
        import gc

        gc.collect()
    except Exception as e:
        print(f"Warning: Error during Playwright cleanup: {e}")


def is_bot_actually_running():
    """Check if the bot is actually running by verifying both flag and browser state"""
    global bot_running, bot_page, bot_browser

    if not bot_running:
        return False

    # Simple check - if we have browser and page objects, assume it's running
    # Don't try to actually test the page as it causes asyncio pipe errors
    if bot_page and bot_browser:
        return True
    else:
        # Reset state if objects are missing
        bot_running = False
        bot_page = None
        bot_browser = None
        return False


def start_meeting_bot(meeting_url: str):
    global bot_running
    try:
        # Check if bot is actually running, not just the flag
        if is_bot_actually_running():
            print("Bot is already running")
            return False

        # Clear chat.txt when starting a new meeting
        try:
            with open("chat.txt", "w", encoding="utf-8") as f:
                f.write("")
            print("Cleared chat.txt for new meeting")
        except Exception as e:
            print(f"Warning: Could not clear chat.txt: {e}")

        def run():
            # Windows-specific event loop setup for threads
            if platform.system() == "Windows":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(start_bot_task())
            finally:
                loop.close()

        async def start_bot_task():
            print(f"🔁 Joining meeting: {meeting_url}")
            page, speakers = await join_meeting_with_auth(meeting_url)
            print(f"✅ Joined. Capturing captions for: {speakers}")
            await capture_and_log_captions(page, speakers, save_path="chat.txt")

        # Start bot in background thread
        threading.Thread(target=run, daemon=True).start()
        bot_running = True
        return True

    except Exception as e:
        print(f"❌ Failed to start meeting bot: {e}")
        return False


def leave_meeting():
    """Manually leave the meeting and process the transcript"""
    global bot_running

    try:
        if not bot_running:
            print("Bot is not running")
            return False

        print("🔄 Manually leaving meeting...")

        # Set flag to stop the bot - this will trigger the cleanup in the async function
        bot_running = False

        # Process the meeting end
        print("🔄 Processing meeting transcript...")
        process_meeting_end()

        print("✅ Bot state reset complete. Ready for next meeting.")
        return True

    except Exception as e:
        print(f"Error leaving meeting: {e}")
        # Ensure bot_running is reset even if there's an error
        bot_running = False
        return False


def process_meeting_end():
    """Process the meeting end and generate all documents"""
    try:
        # Check if chat.txt exists and has content
        if not os.path.exists("chat.txt"):
            print("❌ chat.txt file not found")
            return None

        with open("chat.txt", "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            print("⚠️ chat.txt file is empty. No transcript to process.")
            print("💡 This could be because:")
            print("   - Captions were not enabled in the meeting")
            print("   - No one was speaking during the meeting")
            print("   - The meeting ended too quickly")
            print("   - The bot couldn't detect the captions container")
            return None

        print(
            f"✅ Found {len(content)} characters in chat.txt, processing transcript..."
        )

        from cleaner import convert_chat_to_transcript

        result = convert_chat_to_transcript()

        if result:
            print(f"✅ Transcript processed successfully: {result}")
        else:
            print("⚠️ Transcript processing failed")

        return result

    except Exception as e:
        print(f"❌ Error processing meeting end: {e}")
        return None


def check_auth_file():
    """Check if auth.json exists and is valid"""
    if not os.path.exists("auth.json"):
        print(
            "❌ auth.json not found. Please run login.py first to create authentication file."
        )
        return False
    return True


# Function 1: Join Google Meet using auth.json
async def join_meeting_with_auth(meet_url: str):
    global bot_browser, bot_page

    # Check if auth.json exists
    if not check_auth_file():
        raise Exception("Authentication file not found. Run login.py first.")

    playwright = await async_playwright().start()
    bot_browser = await playwright.chromium.launch(
        headless=False,
        args=[
            "--use-fake-ui-for-media-stream",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-features=TranslateUI",
            "--disable-ipc-flooding-protection",
        ],
    )

    print("🔐 Loading authentication from auth.json...")

    try:
        # Load the authentication state from auth.json
        context = await bot_browser.new_context(storage_state="auth.json")
        bot_page = await context.new_page()

        # Test if the auth state is still valid by visiting Google
        print("🔍 Validating authentication state...")
        await bot_page.goto("https://accounts.google.com/")
        await bot_page.wait_for_timeout(2000)

        # Check if we're logged in by looking for the account avatar or email
        logged_in = await bot_page.evaluate(
            """() => {
            return document.querySelector('div[data-email]') !== null || 
                   document.querySelector('img[alt*="profile"]') !== null ||
                   document.querySelector('div[aria-label*="Google Account"]') !== null ||
                   document.querySelector('div[aria-label*="Account"]') !== null;
        }"""
        )

        if logged_in:
            print("✅ Authentication state is valid")
        else:
            print("❌ Authentication state expired. Please run login.py again.")
            await bot_browser.close()
            raise Exception("Authentication expired. Run login.py to refresh.")

    except Exception as e:
        print(f"❌ Error loading authentication: {e}")
        await bot_browser.close()
        raise Exception(f"Failed to load authentication: {e}")

    # Navigate directly to the meeting
    print(f"🚀 Joining meeting: {meet_url}")
    await bot_page.goto(meet_url)
    await bot_page.wait_for_timeout(5000)

    # --- Mute mic and turn off camera before joining ---
    try:
        # Mute microphone
        mic_button = await bot_page.query_selector(
            'button[aria-label*="Turn off microphone"], button[aria-label*="Mute microphone"]'
        )
        if mic_button:
            mic_aria_pressed = await mic_button.get_attribute("aria-pressed")
            if mic_aria_pressed == "false":
                await mic_button.click()
                print("🔇 Microphone muted")
    except Exception as e:
        print(f"Could not mute microphone: {e}")

    try:
        # Turn off camera
        cam_button = await bot_page.query_selector(
            'button[aria-label*="Turn off camera"], button[aria-label*="Turn camera off"]'
        )
        if cam_button:
            cam_aria_pressed = await cam_button.get_attribute("aria-pressed")
            if cam_aria_pressed == "false":
                await cam_button.click()
                print("📷 Camera turned off")
    except Exception as e:
        print(f"Could not turn off camera: {e}")

    try:
        join_btn = await bot_page.wait_for_selector(
            "button:has-text('Join now')", timeout=10000
        )
    except:
        try:
            join_btn = await bot_page.wait_for_selector(
                "button:has-text('Ask to join')", timeout=10000
            )
        except:
            await bot_browser.close()
            raise Exception("Join button not found")

    await join_btn.click()
    await bot_page.wait_for_timeout(5000)

    # Get participants for speaker detection
    await bot_page.click('button[aria-label="People"]')
    await asyncio.sleep(1.5)

    listitems = bot_page.locator('div[role="listitem"]')
    names = []
    count = await listitems.count()
    for i in range(count):
        aria_label = await listitems.nth(i).get_attribute("aria-label")
        if aria_label:
            name = aria_label.strip()
            if name and name not in names:
                names.append(name)

    print("👥 Participants detected:")
    for name in names:
        print("  -", name)

    return bot_page, names


# Function 2: Capture and log captions
async def capture_and_log_captions(page, known_speakers: list, save_path="chat.txt"):
    global bot_running, bot_page, bot_browser

    logger = CaptionLogger(filename=save_path, known_speakers=known_speakers)

    print("🎤 Attempting to enable captions...")

    # Try multiple methods to enable captions
    captions_enabled = False

    # Method 1: Keyboard shortcut
    try:
        await page.keyboard.down("Shift")
        await page.keyboard.press("KeyC")
        await page.keyboard.up("Shift")
        await page.wait_for_timeout(2000)
        print("✅ Captions enabled via keyboard shortcut")
        captions_enabled = True
    except Exception as e:
        print(f"⚠️ Keyboard shortcut failed: {e}")

    # Method 2: Try clicking the captions button if keyboard shortcut didn't work
    if not captions_enabled:
        try:
            # Look for captions button
            captions_button = await page.wait_for_selector(
                'button[aria-label*="Captions"], button[aria-label*="Live captions"], button[aria-label*="CC"]',
                timeout=5000,
            )
            if captions_button:
                await captions_button.click()
                await page.wait_for_timeout(2000)
                print("✅ Captions enabled via button click")
                captions_enabled = True
        except Exception as e:
            print(f"⚠️ Button click failed: {e}")

    # Method 3: Try the three dots menu
    if not captions_enabled:
        try:
            # Click three dots menu
            three_dots = await page.wait_for_selector(
                'button[aria-label="More options"], button[aria-label="Settings"]',
                timeout=5000,
            )
            if three_dots:
                await three_dots.click()
                await page.wait_for_timeout(1000)

                # Look for captions option in menu
                captions_option = await page.wait_for_selector(
                    'div[role="menuitem"]:has-text("Captions"), div[role="menuitem"]:has-text("Live captions")',
                    timeout=3000,
                )
                if captions_option:
                    await captions_option.click()
                    await page.wait_for_timeout(2000)
                    print("✅ Captions enabled via menu")
                    captions_enabled = True
        except Exception as e:
            print(f"⚠️ Menu method failed: {e}")

    if not captions_enabled:
        print(
            "⚠️ Could not enable captions automatically. Please enable them manually in the meeting."
        )
        print(
            "💡 Tip: Look for the 'CC' or 'Live captions' button in the meeting controls."
        )

    # Wait a bit for captions to appear
    await page.wait_for_timeout(3000)

    last_caption_time = datetime.now()
    no_caption_threshold = 30  # seconds without captions to consider meeting ended
    caption_check_count = 0

    try:
        while bot_running:
            try:
                # Check if meeting has ended (leave button disappeared)
                leave_button = page.locator("div.VYBDae-Bz112c-RLmnJb")
                leave_count = await leave_button.count()
                if leave_count == 0:
                    print("\n📞 Meeting ended (leave button disappeared)")
                    break

                # Try multiple selectors for captions container
                container_info = await page.evaluate(
                    """() => {
                    // Try multiple selectors for captions
                    const selectors = [
                        'div[jscontroller="KPn5nb"][aria-label="Captions"]',
                        'div[aria-label="Captions"]',
                        'div[aria-label="Live captions"]',
                        'div[data-captions]',
                        'div[class*="captions"]',
                        'div[class*="caption"]'
                    ];
                    
                    for (const selector of selectors) {
                        const container = document.querySelector(selector);
                        if (container && container.textContent && container.textContent.trim()) {
                            return { exists: true, text: container.textContent, selector: selector };
                        }
                    }
                    
                    return { exists: false, text: "", selector: "none" };
                }"""
                )

                if container_info["exists"] and container_info["text"].strip():
                    logger.process_container_text(container_info["text"])
                    last_caption_time = datetime.now()
                    caption_check_count = 0

                    # Print first few captions for debugging
                    if caption_check_count < 3:
                        # print(f"🎤 Captions detected: {container_info['text'][:100]}...")
                        caption_check_count += 1
                else:
                    # Check if we've been without captions for too long
                    time_since_last_caption = (
                        datetime.now() - last_caption_time
                    ).total_seconds()
                    if time_since_last_caption > no_caption_threshold:
                        print(
                            f"\n⏰ No captions for {no_caption_threshold} seconds. Meeting may have ended."
                        )
                        print(
                            "💡 If you're still in the meeting, please check if captions are enabled."
                        )
                        break

                    # Print debug info every 10 seconds
                    if (
                        int(time_since_last_caption) % 10 == 0
                        and time_since_last_caption > 0
                    ):
                        print(
                            f"🔍 No captions detected. Time since last caption: {int(time_since_last_caption)}s"
                        )
                        print(
                            f"🔍 Tried selectors: {container_info.get('selector', 'unknown')}"
                        )

            except Exception as e:
                print(f"Caption error: {e}")
                # If there's a persistent error, the meeting might have ended
                await asyncio.sleep(5)
                continue

            await asyncio.sleep(1)

    finally:
        # Always clean up when the function ends (whether naturally or due to error)
        print("\n🔄 Processing meeting transcript...")

        # Check if we captured any content
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    print(f"✅ Captured {len(content)} characters of transcript")
                else:
                    print("⚠️ No transcript content captured. This could be because:")
                    print("   - Captions were not enabled in the meeting")
                    print("   - No one was speaking during the meeting")
                    print("   - The meeting ended too quickly")
        except Exception as e:
            print(f"⚠️ Could not check transcript file: {e}")

        process_meeting_end()

        # Reset bot state
        print("🔄 Resetting bot state...")
        bot_running = False

        # Close browser resources
        try:
            if bot_page:
                await bot_page.close()
                bot_page = None
        except Exception as e:
            print(f"Warning: Error closing page: {e}")

        try:
            if bot_browser:
                await bot_browser.close()
                bot_browser = None
        except Exception as e:
            print(f"Warning: Error closing browser: {e}")

        # Use the dedicated cleanup function
        await cleanup_playwright_resources()

        print("✅ Bot state reset complete. Ready for next meeting.")
