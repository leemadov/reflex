// Reflex desktop: one window with a built-in browser (the page Reflex works on), the chat panel and a status bar.
// The agent is the extension's own background.js, run here through agent-host.js; the chat panel is the extension's
// own popup.html, given a chrome.* bridge by preload-chat.js. The model server is started and watched by server.js.
const { app, BrowserWindow, WebContentsView, ipcMain, Menu, shell } = require("electron");
const path = require("path");
const { startAgent } = require("./agent-host");
const { modelServer } = require("./server");

const PROJECT = path.resolve(__dirname, "..");
const EXT = path.join(PROJECT, "extension");
const START = path.join(__dirname, "start.html");
const MODES = { // the top bar's speed/quality switch -> the agent's settings (same keys as the extension)
  fast: { thinkBelow: 0, plan: true, motion: "fast", maxSteps: 15 },          // System 1 only: every step ~0.2 s
  balanced: { thinkBelow: 0.2, plan: true, motion: "fast", maxSteps: 15 },    // thinks it through when unsure
  best: { thinkBelow: 0.3, plan: true, motion: "smooth", maxSteps: 25 },      // more thinking + steps for long tasks
  // (held-out routing eval: 0.2 and 0.4 tie for best accuracy; higher cutoffs only add System 2 calls, so "best" is
  // the careful middle with a longer step budget, not a claim of higher accuracy)
};

app.setName("Reflex");
// Keep rendering when the window is covered or in the background: Chromium otherwise stops painting "occluded" windows,
// and a page that isn't producing frames can drop the agent's dispatched clicks (hit-testing needs a current frame).
app.commandLine.appendSwitch("disable-features", "CalculateNativeWinOcclusion");
app.commandLine.appendSwitch("disable-renderer-backgrounding");
app.commandLine.appendSwitch("disable-backgrounding-occluded-windows");
// self-tests and recordings get their own profile, so they run beside the user's open Reflex instead of quitting
if (process.env.REFLEX_SELFTEST || process.env.REFLEX_RECORD) app.setPath("userData", path.join(app.getPath("temp"), "reflex-test-profile"));
if (!app.requestSingleInstanceLock()) app.quit();
let win, browser, chat, agent, server, traceHook = null;

const send = (channel, ...args) => {
  for (const wc of [win?.webContents, chat?.webContents]) if (wc && !wc.isDestroyed()) wc.send(channel, ...args);
};

function navState() {
  const wc = browser.webContents;
  const url = wc.getURL();
  return { url: url.startsWith("file:") ? "" : url, title: url.startsWith("file:") ? "Start" : wc.getTitle(), loading: wc.isLoading(),
    canBack: wc.navigationHistory.canGoBack(), canForward: wc.navigationHistory.canGoForward() };
}

function toUrl(text) {
  const t = text.trim();
  if (!t) return null;
  if (/^[a-z]+:\/\//i.test(t)) return t;
  if (/^[\w-]+(\.[\w-]+)+(:\d+)?(\/\S*)?$/i.test(t) || /^localhost(:\d+)?/i.test(t) || /^127\.0\.0\.1/.test(t)) return `${/^(localhost|127\.)/.test(t) ? "http" : "https"}://${t}`;
  return `https://www.google.com/search?hl=en&q=${encodeURIComponent(t)}`;
}

function shortcuts(wc) {
  wc.on("before-input-event", (e, input) => {
    if (input.type !== "keyDown") return;
    const k = input.key.toLowerCase(), ctrl = input.control || input.meta;
    const bw = browser.webContents;
    if (ctrl && k === "l") { win.webContents.focus(); win.webContents.send("focus-address"); }
    else if (k === "f5" || (ctrl && k === "r")) bw.reload();
    else if (input.alt && k === "arrowleft") bw.navigationHistory.goBack();
    else if (input.alt && k === "arrowright") bw.navigationHistory.goForward();
    else if (k === "f12") bw.openDevTools({ mode: "detach" });
    else if (ctrl && k === "k") { chat.webContents.focus(); chat.webContents.executeJavaScript("document.getElementById('task')?.focus()"); }
    else return;
    e.preventDefault();
  });
}

function createWindow() {
  win = new BrowserWindow({
    width: 1480, height: 940, minWidth: 1100, minHeight: 680, show: false, backgroundColor: "#F5F5F7",
    title: "Reflex", icon: path.join(__dirname, "icon.ico"),
    titleBarStyle: "hidden", titleBarOverlay: { color: "#F5F5F7", symbolColor: "#1D1D1F", height: 52 },
    webPreferences: { preload: path.join(__dirname, "preload-shell.js"), contextIsolation: true, sandbox: true, backgroundThrottling: false },
  });
  Menu.setApplicationMenu(null);

  // the page Reflex works on: sandboxed, no preload, persistent cookies (sign-ins and cookie choices stick)
  browser = new WebContentsView({ webPreferences: { partition: "persist:reflex", sandbox: true, contextIsolation: true, backgroundThrottling: false } });
  browser.setBorderRadius(16);
  browser.setBackgroundColor("#FFFFFF");
  chat = new WebContentsView({ webPreferences: { preload: path.join(__dirname, "preload-chat.js"), contextIsolation: true, sandbox: true, backgroundThrottling: false } });
  chat.setBorderRadius(18);
  chat.setBackgroundColor("#F7F7FA");
  win.contentView.addChildView(browser);
  win.contentView.addChildView(chat);

  const bw = browser.webContents;
  bw.setWindowOpenHandler(({ url }) => { bw.loadURL(url); return { action: "deny" }; }); // keep it in one view
  for (const ev of ["did-navigate", "did-navigate-in-page", "page-title-updated", "did-start-loading", "did-stop-loading"]) {
    bw.on(ev, () => send("nav", navState()));
  }
  bw.on("page-favicon-updated", (_e, icons) => send("favicon", icons[0] || ""));
  chat.webContents.setWindowOpenHandler(({ url }) => { shell.openExternal(url); return { action: "deny" }; });
  for (const wc of [win.webContents, bw, chat.webContents]) shortcuts(wc);

  agent = startAgent({ extDir: EXT, browser, storeFile: path.join(app.getPath("userData"), "reflex-store.json"),
    onStorageChange: (changes, area) => send("storage-changed", changes, area), trace: (m, p) => traceHook?.(m, p) });
  // re-apply the chosen mode's current definition on every start (definitions can change between versions)
  const mode = MODES[agent.store.raw.local.mode] ? agent.store.raw.local.mode : "balanced";
  agent.store.local.set({ mode, theme: agent.store.raw.local.theme || "pink", ...MODES[mode] });

  win.loadFile(path.join(__dirname, "shell.html"));
  chat.webContents.loadFile(path.join(EXT, "popup.html"), { query: { desktop: "1" } });
  bw.loadFile(START);
  win.once("ready-to-show", () => win.show());

  server = modelServer({ project: PROJECT, logFile: path.join(app.getPath("userData"), "model-server.log"),
    onStatus: (s) => send("server", s) });
  server.start();
  if (process.env.REFLEX_SELFTEST) win.once("ready-to-show", () => require("./selftest")({ win, browser, agent, server, dir: process.env.REFLEX_SELFTEST }));
  if (process.env.REFLEX_RECORD) win.once("ready-to-show", () => require("./record")({ win, browser, agent, server, dir: process.env.REFLEX_RECORD,
    setTrace: (fn) => { traceHook = fn; } }));
}

// ---------- IPC: shell (top bar) ----------
ipcMain.on("layout", (_e, r) => { // the shell measures its stage + chat slots; the native views follow
  const b = (x) => ({ x: Math.round(x.x), y: Math.round(x.y), width: Math.max(1, Math.round(x.width)), height: Math.max(1, Math.round(x.height)) });
  browser.setBounds(b(r.stage));
  chat.setBounds(b(r.chat));
});
ipcMain.on("navigate", (_e, text) => { const u = toUrl(text); if (u) browser.webContents.loadURL(u).catch(() => {}); });
ipcMain.on("nav-action", (_e, a) => {
  const wc = browser.webContents;
  if (a === "back") wc.navigationHistory.goBack();
  else if (a === "forward") wc.navigationHistory.goForward();
  else if (a === "reload") wc.reload();
  else if (a === "stop") wc.stop();
  else if (a === "home") wc.loadFile(START);
});
ipcMain.handle("state", () => ({ nav: navState(), server: server?.state, modes: Object.keys(MODES) }));
ipcMain.handle("set-mode", (_e, mode) => MODES[mode] && agent.store.local.set({ mode, ...MODES[mode] }));

// ---------- IPC: the chrome.* bridge for the chat panel (and the shell's storage reads) ----------
ipcMain.handle("storage-get", (_e, area, keys) => agent.store[area].get(keys));
ipcMain.handle("storage-set", (_e, area, items) => agent.store[area].set(items));
ipcMain.handle("storage-remove", (_e, area, keys) => agent.store[area].remove(keys));
ipcMain.handle("tabs-query", () => [agent.tab()]);
ipcMain.on("runtime-message", (_e, msg) => agent.dispatch(msg));

app.on("second-instance", () => { if (win) { if (win.isMinimized()) win.restore(); win.focus(); } });
app.whenReady().then(createWindow);
app.on("window-all-closed", () => app.quit());
app.on("before-quit", () => server?.stop());
