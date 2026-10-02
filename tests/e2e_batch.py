"""Fully autonomous success rate of the extension on the demo shop (needs model.py serve + computer-use-agent apps/server.py)."""
import asyncio
import json
import os
import sys
import tempfile

from e2e_extension import CHROME, EXT, FINAL, SHOP, api, wait_status
from playwright.async_api import async_playwright

TASKS = [  # (task, success check on the shop state)
    ('Add "Wireless Mouse" to your cart.', lambda s, n: "Wireless Mouse" in n(s["cart"])),
    ('Add "Desk Lamp" to your cart.', lambda s, n: "Desk Lamp" in n(s["cart"])),
    ('Add "Backpack" to your cart.', lambda s, n: "Backpack" in n(s["cart"])),
    ('Search for "Coffee Mug" and add it to your cart.', lambda s, n: "Coffee Mug" in n(s["cart"]) and s["cart"][-1]["qty"] >= 1),
    ('Search for "Notebook" and add it to your cart.', lambda s, n: "Notebook" in n(s["cart"])),
    ("Filter the catalog to show only the Electronics category.", lambda s, n: s["category_filter"] == "Electronics"),
    ("Filter the catalog to show only the Apparel category.", lambda s, n: s["category_filter"] == "Apparel"),
    ('Add "Running Shoes" to your wishlist.', lambda s, n: "Running Shoes" in [p["name"] for p in s["products"] if p["id"] in s["wishlist"]]),
]


async def one(p, task, check, i):
    sid = f"batch-{i}"
    api("/api/reset", {"sid": sid, "seed": 0})
    before = api(f"/api/state?sid={sid}")["shop"]
    ctx = await p.chromium.launch_persistent_context(tempfile.mkdtemp(), executable_path=CHROME, headless=True,
                                                     args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}"],
                                                     viewport={"width": 1280, "height": 800})
    try:
        sw = ctx.service_workers[0] if ctx.service_workers else await ctx.wait_for_event("serviceworker")
        page = await ctx.new_page()
        await page.goto(f"{SHOP}/shop?sid={sid}")
        await page.wait_for_load_state("networkidle")
        tab = await sw.evaluate(f"chrome.tabs.query({{url: '{SHOP}/*'}}).then(t => t[0].id)")
        settings = {"server": "http://127.0.0.1:8765", "maxSteps": 8, **json.loads(os.environ.get("PROBE_SETTINGS", "{}"))}
        await sw.evaluate("([tabId, task, settings]) => { start({ tabId, task, settings }); }", [tab, task, settings])
        v = await wait_status(sw, FINAL)
        after = api(f"/api/state?sid={sid}")["shop"]
        names = lambda rows: [next((q["name"] for q in after["products"] if q["id"] == r["product_id"]), "?") for r in rows]  # noqa: E731
        added = [r for r in after["cart"] if r not in before["cart"]]
        ok = check(after, names)
        steps = " > ".join(f"{e['op']} {e['label']}" + (f" '{e['text']}'" if e["text"] else "") for e in v["log"])
        print(f"{'PASS' if ok else 'FAIL'} | {task} | {v['status']} | {steps} | cart +{names(added)}", flush=True)
        return ok
    finally:
        await ctx.close()


async def main():
    async with async_playwright() as p:
        results = [await one(p, t, c, i) for i, (t, c) in enumerate(TASKS)]
    print(f"\n{sum(results)}/{len(results)} tasks succeeded fully automatically")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
