"""Run one task on a live site in a visible browser and print the agent's status every second."""
import asyncio, json, sys, tempfile
from e2e_extension import CHROME, EXT
from playwright.async_api import async_playwright

URL, TASK = sys.argv[1], sys.argv[2]

async def main():
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(tempfile.mkdtemp(), executable_path=CHROME, headless=False,
            args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}"], viewport={"width": 1280, "height": 800})
        sw = ctx.service_workers[0] if ctx.service_workers else await ctx.wait_for_event("serviceworker")
        sw.on("console", lambda m: print("  [sw]", m.text, flush=True))
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto(URL); await page.wait_for_load_state("load")
        try:  # a consent wall is the user's call; in this throwaway test profile take the privacy-preserving option
            await page.get_by_role("button", name="Reject all").click(timeout=4000); await page.wait_for_timeout(1500)
        except Exception:
            pass
        tab = await sw.evaluate("chrome.tabs.query({active: true}).then(t => t[0].id)")
        await sw.evaluate("([tabId, task]) => { start({ tabId, task, settings: { server: 'http://127.0.0.1:8765', maxSteps: 8 } }).catch(e => console.log('start failed', e.message)); }", [tab, TASK])
        last = None
        for _ in range(180):
            v = await sw.evaluate("chrome.storage.session.get('agent').then(r => r.agent || null)")
            s = json.dumps({k: v.get(k) for k in ("status", "step", "error")} | {"log": [(e["op"], e["label"], e["text"]) for e in v.get("log", [])]}, ensure_ascii=False) if v else None
            if s != last: print(s, "| url:", page.url[:80], flush=True); last = s
            if v and v["status"] not in ("starting", "thinking"): break
            await asyncio.sleep(1)
        await page.screenshot(path="probe_live.png")
        await ctx.close()

sys.stdout.reconfigure(encoding="utf-8"); asyncio.run(main())
