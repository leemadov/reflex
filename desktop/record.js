// REFLEX_RECORD=<dir>: record the app doing a real task, for the ad. Sizes the window so the area under the top bar is
// 1600x857 (the ad's demo-window ratio), keeps it on top, records that region with ffmpeg, runs the task on the local
// demo shop, and logs every agent action (cursor glide, typing, click) with its time in the recording: events.json.
const fs = require("fs");
const path = require("path");
const { spawn, spawnSync } = require("child_process");
const { app, screen } = require("electron");

const W = 1600, H = 857, TOP = 52;
const TASK = process.env.REFLEX_RECORD_TASK || 'Search for "Coffee Mug" and add it to the cart';

module.exports = async function record({ win, browser, agent, server, dir, setTrace }) {
  fs.mkdirSync(dir, { recursive: true });
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const raw = path.join(dir, "raw.mp4");
  try {
    win.setContentSize(W, H + TOP);
    win.center();
    win.setAlwaysOnTop(true, "screen-saver"); // nothing may cover the region being recorded
    while (!["ready", "error"].includes(server.state.phase)) await wait(300);
    agent.dispatch({ cmd: "clear" });
    await fetch("http://127.0.0.1:5055/api/reset", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sid: "record", seed: 0 }) });
    await browser.webContents.loadURL("http://127.0.0.1:5055/shop?sid=record");
    await wait(2500);

    const cb = win.getContentBounds();
    const sf = screen.getDisplayMatching(cb).scaleFactor;
    const even = (v) => Math.round(v) & ~1;
    const args = ["-y", "-f", "gdigrab", "-framerate", "30", "-offset_x", String(Math.round(cb.x * sf)), "-offset_y", String(Math.round((cb.y + TOP) * sf)),
      "-video_size", `${even(W * sf)}x${even(H * sf)}`, "-draw_mouse", "0", "-i", "desktop",
      "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p", raw];
    const ff = spawn("ffmpeg", args, { stdio: ["pipe", "ignore", "pipe"] });
    let ffErr = "";
    ff.stderr.on("data", (d) => { ffErr = (ffErr + d).slice(-4000); });
    const events = [];
    setTrace((m, p) => {
      const t = Date.now();
      if (m === "Input.insertText") events.push({ t, kind: "type", text: p.text });
      else if (m === "Input.dispatchMouseEvent" && p.type === "mousePressed") events.push({ t, kind: "click" });
      else if (m === "Runtime.evaluate" && /__cdCursor\.move/.test(p.expression || "")) events.push({ t, kind: "glide" });
    });
    await wait(1800); // lead-in: the page at rest

    const settings = { ...(await agent.store.local.get({ server: "http://127.0.0.1:8765", theme: "pink" })), maxSteps: 12, thinkBelow: 0.2, plan: true, motion: "smooth" };
    const before = (await agent.store.local.get({ chat: [] })).chat.length;
    const tTask = Date.now();
    agent.dispatch({ cmd: "start", tabId: 1, task: TASK, settings });
    let reply;
    for (;;) {
      await wait(150);
      const { chat } = await agent.store.local.get({ chat: [] });
      if (chat.length >= before + 2) { reply = chat[chat.length - 1]; break; }
    }
    const tDone = Date.now();
    await wait(2800); // tail: the summary in the chat

    const tStop = Date.now();
    ff.stdin.write("q");
    await new Promise((r) => ff.on("close", r));
    setTrace(null);
    const dur = Number(spawnSync("ffprobe", ["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", raw]).stdout.toString().trim());
    const t0 = tStop - dur * 1000; // the recording's first frame, from its length (robust to ffmpeg's start-up delay)
    const rel = (t) => Math.round(t - t0) / 1000;
    fs.writeFileSync(path.join(dir, "events.json"), JSON.stringify({
      task: TASK, duration: dur, scale: sf, taskAt: rel(tTask), doneAt: rel(tDone),
      events: events.map((e) => ({ ...e, t: rel(e.t) })),
      reply: { status: reply.status, task: reply.task, plan: reply.plan, answer: reply.answer, steps: (reply.log || []).map((e) => [e.op, e.label, e.by, e.p]) },
      ffmpeg: dur ? undefined : ffErr,
    }, null, 1));
  } catch (e) {
    fs.writeFileSync(path.join(dir, "error.txt"), String(e?.stack || e));
  }
  app.quit();
};
