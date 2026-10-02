// The local model server: reuse one that's already running, else start `python model.py serve --preload` (System 2 and
// the writer load in the background so no request waits for them), watch /v1/status, and stop it on quit if we
// started it.
const { spawn } = require("child_process");
const fs = require("fs");

function modelServer({ project, base = "http://127.0.0.1:8765", logFile, onStatus }) {
  let child = null;
  let state = { phase: "starting", system2: false, warm: 0, error: null, external: false };
  const emit = (patch) => { state = { ...state, ...patch }; onStatus(state); };
  const status = async () => {
    try {
      const r = await fetch(`${base}/v1/status`, { signal: AbortSignal.timeout(1500) });
      return r.ok ? r.json() : null;
    } catch { return null; }
  };

  async function start() {
    emit({ phase: "starting" });
    if (await status()) emit({ external: true });
    else {
      const log = fs.createWriteStream(logFile, { flags: "a" });
      child = spawn("python", ["model.py", "serve", "--preload"], { cwd: project, windowsHide: true });
      child.stdout.pipe(log);
      child.stderr.pipe(log);
      child.on("error", (e) => emit({ phase: "error", error: e.code === "ENOENT" ? "Python not found on PATH" : e.message }));
      child.on("exit", (code) => { child = null; if (state.phase !== "stopping") emit({ phase: "error", error: `Model server exited (${code}); see ${logFile}` }); });
    }
    for (;;) { // poll: quick until everything is loaded, then a slow heartbeat
      const s = await status();
      if (s) emit({ phase: s.system2 && s.writer ? "ready" : "system1", system2: !!s.system2, warm: s.warm ?? 1, error: null });
      else if (state.phase === "ready" || state.phase === "system1") emit({ phase: "error", error: "Model server not responding" });
      await new Promise((r) => setTimeout(r, state.phase === "ready" ? 8000 : 1000));
    }
  }

  function stop() {
    state.phase = "stopping";
    if (child) child.kill();
  }

  return { start, stop, get state() { return state; } };
}

module.exports = { modelServer };
