"""Docked side window (the no-side-panel fallback): opens beside the browser, targets its tab, restores the width."""
import asyncio, sys, tempfile
from e2e_extension import CHROME, EXT, SHOP, api
from playwright.async_api import async_playwright

async def main():
    api("/api/reset", {"sid": "dock", "seed": 0})
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(tempfile.mkdtemp(), executable_path=CHROME, headless=True, no_viewport=True,
            args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}"])
        sw = ctx.service_workers[0] if ctx.service_workers else await ctx.wait_for_event("serviceworker")
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto(f"{SHOP}/shop?sid=dock")
        before = await sw.evaluate("chrome.windows.getAll({populate: true}).then(ws => ws.map(w => [w.id, w.type, w.left, w.width, w.height]))")
        tab = await sw.evaluate(f"chrome.tabs.query({{url: '{SHOP}/*'}}).then(t => t[0])")
        await sw.evaluate("(tab) => openDocked(tab)", tab)
        await asyncio.sleep(1.5)
        after = await sw.evaluate("chrome.windows.getAll().then(ws => ws.map(w => [w.id, w.type, w.left, w.width, w.height]))")
        print("before:", before); print("after: ", after)
        panel = next(pg for pg in ctx.pages if "popup.html?window=1" in pg.url)
        target = await panel.evaluate("targetTab().then(t => t && t.url)")
        theme = await panel.evaluate("document.documentElement.dataset.theme")
        print("panel targets:", target, "| theme:", theme)
        await panel.wait_for_timeout(500)
        await panel.screenshot(path="dock_panel.png")
        await sw.evaluate("chrome.storage.session.get('dock').then(({dock}) => chrome.windows.remove(dock.id))")
        await asyncio.sleep(1)
        print("closed:", await sw.evaluate("chrome.windows.getAll().then(ws => ws.map(w => [w.id, w.type, w.left, w.width]))"))
        await ctx.close()

sys.stdout.reconfigure(encoding="utf-8"); asyncio.run(main())
