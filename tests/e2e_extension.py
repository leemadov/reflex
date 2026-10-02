"""End-to-end test of the extension against the local demo shop (computer-use-agent/apps, port 5055)."""
import asyncio
import json
import sys
import tempfile
import urllib.request

from playwright.async_api import async_playwright

EXT = r"C:\Users\Lee Madover\Desktop\Computer-Desicion\extension"
# Chrome for Testing 153. Real path: this process runs inside the Claude MSIX container, which redirects AppData
# writes to LocalCache; the Windows loader resolves chrome's version manifest outside that redirection.
CHROME = r"C:\Users\Lee Madover\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ms-playwright\chromium-1243\chrome-win64\chrome.exe"
SHOP = "http://127.0.0.1:5055"
OUT = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
FINAL = ("done", "blocked", "error", "stopped", "out of steps")


def api(path, body=None):
    req = urllib.request.Request(SHOP + path, json.dumps(body).encode() if body else None, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


async def wait_status(sw, want, timeout=240, after_step=0):
    for _ in range(timeout * 4):
        v = await sw.evaluate("chrome.storage.session.get('agent').then(r => r.agent || null)")
        if v and v["status"] in want and (v["status"] not in ("waiting", "thinking") or v["step"] > after_step):
            return v
        await asyncio.sleep(0.25)
    raise TimeoutError(f"no status in {want}")


async def main(task, mode):
    sid = "e2e-" + mode
    st = api("/api/reset", {"sid": sid, "seed": 0})
    print("products:", [p["name"] for p in st["shop"]["products"]][:12])
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(tempfile.mkdtemp(), executable_path=CHROME, headless=True,
                                                         args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}"],
                                                         viewport={"width": 1280, "height": 800})
        sw = ctx.service_workers[0] if ctx.service_workers else await ctx.wait_for_event("serviceworker")
        page = await ctx.new_page()
        await page.goto(f"{SHOP}/shop?sid={sid}")
        await page.wait_for_load_state("networkidle")
        tab = await sw.evaluate(f"chrome.tabs.query({{url: '{SHOP}/*'}}).then(t => t[0].id)")

        snap = await sw.evaluate("""async (tabId) => { run = { tabId }; await chrome.debugger.attach({ tabId }, '1.3');
            try { return (await snapshot()).page; } finally { await chrome.debugger.detach({ tabId }); run = null; } }""", tab)
        print(f"--- snapshot: {len(snap.splitlines())} lines; first 40 ---")
        print("\n".join(snap.splitlines()[:40]))

        auto = 0 if mode == "auto" else 1.01
        await sw.evaluate("([tabId, task, autoAbove]) => { start({ tabId, task, settings: { server: 'http://127.0.0.1:8765', maxSteps: 8, autoAbove } }); }",
                          [tab, task, auto])
        popup = await ctx.new_page()  # the real popup UI, opened as a page
        await popup.set_viewport_size({"width": 380, "height": 640})
        await popup.goto(sw.url.replace("background.js", "popup.html"))
        await page.bring_to_front()
        shots = 0
        last = 0  # last step answered: its "waiting" stays in storage until the worker moves on, so skip it
        while (v := await wait_status(sw, ("waiting",) + FINAL, after_step=last))["status"] == "waiting":
            last = v["step"]
            pend = v["pending"]
            print(f"  waiting: model says {pend['op']} p={pend['p']:.3f} note={pend.get('note')!r}; top targets "
                  f"{[(t['label'], round(t['q'], 2)) for t in pend['targets']]}")
            if shots < 2:  # popup + highlighted page, as the user would see them
                await popup.wait_for_timeout(400)
                await popup.screenshot(path=f"{OUT}\\popup_pending{shots}.png", full_page=True)
                await page.screenshot(path=f"{OUT}\\page_highlight{shots}.png")
                shots += 1
            if pend.get("note"):  # a repeat: answer like a user who sees the item is already in the cart
                await popup.check("input[name=op][value=DONE]")
            await popup.click("#pending button.primary")
        await popup.wait_for_timeout(400)
        await popup.screenshot(path=f"{OUT}\\popup_final_{mode}.png", full_page=True)
        print("--- run ---"); print(json.dumps({k: v.get(k) for k in ("status", "answer", "error")}, indent=1))
        for e in v["log"]:
            print(f"  {e['step']}. {e['op']} {e.get('label', '')} {e.get('text', '')!r} p={e['p']} auto={e['auto']}")
        await page.screenshot(path=f"{OUT}\\page_after_{mode}.png")
        shop = api(f"/api/state?sid={sid}")["shop"]
        names = {p["id"]: p["name"] for p in shop["products"]}
        print("cart:", [(names[r["product_id"]], r["qty"]) for r in shop["cart"]], "| wishlist:", shop.get("wishlist"))
        await ctx.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "auto"))
