import asyncio, sys, tempfile
from e2e_extension import CHROME, EXT
from playwright.async_api import async_playwright
OUT = sys.argv[1] if len(sys.argv) > 1 else "caps_before.png"
async def main():
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(tempfile.mkdtemp(), executable_path=CHROME, headless=True,
            args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}"], device_scale_factor=2)
        sw = ctx.service_workers[0] if ctx.service_workers else await ctx.wait_for_event("serviceworker")
        pg = await ctx.new_page(); await pg.set_viewport_size({"width": 420, "height": 700})
        await pg.goto(sw.url.replace("background.js", "popup.html")); await pg.wait_for_timeout(900)
        box = await pg.locator(".empty").bounding_box()
        await pg.screenshot(path=OUT, clip={"x": box["x"], "y": box["y"], "width": box["width"], "height": 200})
        await ctx.close()
asyncio.run(main())
