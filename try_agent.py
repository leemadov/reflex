"""Open a browser with the agent extension loaded, on the demo shop. Needs `python model.py serve` running.

Launched through Playwright: a plain chrome.exe start of this build from here shows a blank page that never loads.
Usage: python try_agent.py [url]
"""
import asyncio
import json
import os
import sys
import urllib.request

from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "extension")
PROFILE = os.path.join(HERE, ".agent-browser-profile")
SHOP = "http://127.0.0.1:5055"
URL = sys.argv[1] if len(sys.argv) > 1 else f"{SHOP}/shop?sid=try"
# Playwright's browser was installed from inside the Claude app, so it lives in that app's redirected LocalAppData.
CHROME = os.path.expandvars(r"%LOCALAPPDATA%\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ms-playwright"
                            r"\chromium-1243\chrome-win64\chrome.exe")


async def main():
    if URL.startswith(SHOP):  # the demo shop is empty until its session is seeded
        req = urllib.request.Request(f"{SHOP}/api/reset", json.dumps({"sid": "try", "seed": 0}).encode(),
                                     {"Content-Type": "application/json"})
        urllib.request.urlopen(req).read()
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            PROFILE, headless=False, no_viewport=True,
            executable_path=CHROME if os.path.exists(CHROME) else None,
            args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}", "--start-maximized"])
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto(URL)
        closed = asyncio.Event()
        ctx.on("close", lambda _: closed.set())
        await closed.wait()  # keep running until the user closes the window


asyncio.run(main())
