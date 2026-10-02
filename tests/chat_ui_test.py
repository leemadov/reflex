"""Chat panel end to end: a task, then a follow-up that needs the context; screenshots of the panel (as a tab).

    python chat_ui_test.py      (needs model.py serve + computer-use-agent apps/server.py)
"""
import asyncio
import json
import os
import sys
import tempfile

from e2e_extension import CHROME, EXT, SHOP, api
from playwright.async_api import async_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
TURNS = ["Add Wool Socks to the cart", "now add a backpack too"]


async def main():
    sid = "chat-ui"
    api("/api/reset", {"sid": sid, "seed": 0})
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(tempfile.mkdtemp(), executable_path=CHROME, headless=True,
            args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}"], viewport={"width": 1280, "height": 800})
        sw = ctx.service_workers[0] if ctx.service_workers else await ctx.wait_for_event("serviceworker")
        shop = await ctx.new_page()
        await shop.goto(f"{SHOP}/shop?sid={sid}")
        await shop.wait_for_load_state("networkidle")
        tab = await sw.evaluate(f"chrome.tabs.query({{url: '{SHOP}/*'}}).then(t => t[0].id)")
        panel = await ctx.new_page()
        await panel.set_viewport_size({"width": 420, "height": 760})
        await panel.goto(sw.url.replace("background.js", "popup.html"))
        await panel.wait_for_timeout(600)
        await panel.screenshot(path=f"{OUT}/chat_0_empty.png")
        settings = {"server": "http://127.0.0.1:8765", "maxSteps": 8, "thinkBelow": 0.2, "plan": True}
        for i, task in enumerate(TURNS, 1):
            await sw.evaluate("([tabId, task, settings]) => { start({ tabId, task, settings }); }", [tab, task, settings])
            shot = False
            for _ in range(600):
                await asyncio.sleep(0.5)
                live = await sw.evaluate("chrome.storage.session.get('agent').then(r => r.agent || null)")
                if live and live.get("log") and not shot:
                    await panel.screenshot(path=f"{OUT}/chat_{i}_running.png")
                    shot = True
                chat = await sw.evaluate("chrome.storage.local.get({chat: []}).then(r => r.chat)")
                if sum(m["role"] == "agent" for m in chat) >= i:
                    break
            await panel.wait_for_timeout(500)
            await panel.screenshot(path=f"{OUT}/chat_{i}_done.png")
            last = chat[-1]
            print(f"turn {i}: said {task!r} -> understood {last.get('task')!r}, status {last.get('status')}, "
                  f"steps {[(e['op'], e['label']) for e in last.get('log', [])]}", flush=True)
        state = api(f"/api/state?sid={sid}")["shop"]
        names = {q["id"]: q["name"] for q in state["products"]}
        print("cart:", [f"{names.get(r['product_id'], '?')} x{r['qty']}" for r in state["cart"]])
        await ctx.close()

sys.stdout.reconfigure(encoding="utf-8")
asyncio.run(main())
