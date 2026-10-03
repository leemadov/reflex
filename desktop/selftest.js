// REFLEX_SELFTEST=<folder>: a real end-to-end run inside the app. Waits for the models, runs chat turns on the local
// demo shop, records timings/results to <folder>/selftest.jsonl and window screenshots (start, mid-run, done), quits.
const fs = require("fs");
const path = require("path");
const { app, desktopCapturer } = require("electron");

module.exports = async function selftest({ win, browser, agent, server, dir }) {
  fs.mkdirSync(dir, { recursive: true });
  const out = path.join(dir, "selftest.jsonl");
  fs.writeFileSync(out, "");
  const log = (o) => fs.appendFileSync(out, JSON.stringify({ t: Date.now(), ...o }) + "\n");
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const shot = async (name) => { // the composited window (shell + browser + chat views), as the user sees it
    const b = win.getBounds();
    const sources = await desktopCapturer.getSources({ types: ["window"], thumbnailSize: { width: b.width, height: b.height } });
    const s = sources.find((x) => x.id === win.getMediaSourceId()) || sources.find((x) => x.name === "Reflex");
    if (s) fs.writeFileSync(path.join(dir, `${name}.png`), s.thumbnail.toPNG());
    log({ event: "shot", name, found: !!s });
  };
  try {
    agent.dispatch({ cmd: "clear" }); // start from an empty chat
    const t0 = Date.now();
    while (!["ready", "error"].includes(server.state.phase)) await wait(300);
    log({ event: "server", ms: Date.now() - t0, state: server.state });
    await wait(1500);
    await shot("1-start");

    await fetch("http://127.0.0.1:5055/api/reset", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sid: "selftest", seed: 0 }) });
    await browser.webContents.loadURL("http://127.0.0.1:5055/shop?sid=selftest");
    const stored = await agent.store.local.get({ server: "http://127.0.0.1:8765", theme: "pink" });
    const settings = { ...stored, maxSteps: 15, thinkBelow: 0.2, plan: true, motion: "fast" }; // Balanced, whatever the user picked
    const tasks = (process.env.REFLEX_SELFTEST_TASKS || "Add Wool Socks to the cart|now add a backpack too").split("|");
    for (const [i, task] of tasks.entries()) {
      const start = Date.now();
      const before = (await agent.store.local.get({ chat: [] })).chat.length;
      agent.dispatch({ cmd: "start", tabId: 1, task, settings });
      let mid = false;
      for (;;) {
        await wait(150);
        const { chat } = await agent.store.local.get({ chat: [] });
        if (chat.length >= before + 2) {
          const m = chat[chat.length - 1];
          log({ event: "turn", task, ms: Date.now() - start, understood: m.task, status: m.status, answer: m.answer, verified: !!m.verified,
            steps: (m.log || []).map((e) => [e.op, e.label, e.by, e.p]) });
          break;
        }
        if (!mid && i === 0) {
          const { agent: v } = await agent.store.session.get("agent");
          if (v?.log?.length) { mid = true; await shot("2-working"); }
        }
      }
    }
    const state = await (await fetch("http://127.0.0.1:5055/api/state?sid=selftest")).json();
    const names = Object.fromEntries(state.shop.products.map((p) => [p.id, p.name]));
    log({ event: "cart", cart: state.shop.cart.map((r) => `${names[r.product_id]} x${r.qty}`) });
    await wait(1200);
    await shot("3-done");
    // saved chats: the chat gets a model-written title shortly after its first run
    for (let i = 0; i < 40; i++) {
      const { chats } = await agent.store.local.get({ chats: [] });
      if (chats[0] && chats[0].messages.length >= tasks.length * 2) { log({ event: "chats", chats: chats.slice(0, 3).map((c) => [c.title, c.messages.length]) }); break; }
      await wait(250);
    }
    const chatView = win.contentView.children.find((v) => v !== browser).webContents;
    await chatView.executeJavaScript("document.querySelector('.actions summary')?.click()");
    await wait(600);
    await shot("4-actions-open");
    await chatView.executeJavaScript("document.getElementById('history')?.click()");
    await wait(600);
    await shot("5-saved-chats");
  } catch (e) {
    log({ event: "error", error: String(e?.stack || e) });
  }
  app.quit();
};
