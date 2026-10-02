"""Run tasks on the demo shop and print the agent's plan, steps (fast vs thought), and final status.

    python probe.py ["task" ...]                    # System 1 only
    PROBE_SETTINGS='{"thinkBelow": 0.5}' python probe.py "task"   # with System 2 escalation (+ planning, on by default)
"""
import asyncio
import json
import os
import sys
import tempfile

from e2e_extension import CHROME, EXT, FINAL, SHOP, api, wait_status
from playwright.async_api import async_playwright

TASKS = sys.argv[1:] or ["Add a Wool Socks to the cart", "Add Wool Socks and a Backpack to the cart",
                         "Search for socks", "Clear the cart", "Show only Home products, then add the Desk Lamp to the cart"]
SETTINGS = {"server": "http://127.0.0.1:8765", "maxSteps": 10, "plan": False, **json.loads(os.environ.get("PROBE_SETTINGS", "{}"))}


async def main():
    async with async_playwright() as p:
        for i, task in enumerate(TASKS):
            sid = f"probe-{i}"
            api("/api/reset", {"sid": sid, "seed": 0})
            ctx = await p.chromium.launch_persistent_context(tempfile.mkdtemp(), executable_path=CHROME, headless=True,
                args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}"], viewport={"width": 1280, "height": 800})
            sw = ctx.service_workers[0] if ctx.service_workers else await ctx.wait_for_event("serviceworker")
            page = await ctx.new_page()
            await page.goto(f"{SHOP}/shop?sid={sid}")
            await page.wait_for_load_state("networkidle")
            tab = await sw.evaluate(f"chrome.tabs.query({{url: '{SHOP}/*'}}).then(t => t[0].id)")
            await sw.evaluate("([tabId, task, settings]) => { start({ tabId, task, settings }); }", [tab, task, SETTINGS])
            v = await wait_status(sw, FINAL, timeout=600)
            print(f"\n{task!r} -> {v['status']} {v.get('error') or ''}")
            print("   plan:", v.get("plan"))
            for e in v["log"]:
                print(f"   {e['step']}. [{e.get('by')}] {e['op']} {e['label']} {e['text']!r} p={e['p']} part={e.get('part')}")
                if e.get("thought"):
                    print(f"      thought: {e['thought'][:200]!r}")
            cart = api(f"/api/state?sid={sid}")["shop"]
            names = {q["id"]: q["name"] for q in cart["products"]}
            print("   cart:", [f"{names.get(r['product_id'], '?')} x{r['qty']}" for r in cart["cart"]],
                  "| filter:", cart.get("category_filter"))
            await ctx.close()

sys.stdout.reconfigure(encoding="utf-8")
asyncio.run(main())
