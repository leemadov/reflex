// In the desktop app this page gets its chrome.* calls through the app's bridge (desktop/preload-chat.js).
if (globalThis.reflexHost) globalThis.chrome = globalThis.reflexHost;

// Reflex chat panel. The conversation lives in chrome.storage.local ("chat", written by background.js); the run in
// progress lives in chrome.storage.session ("agent"). This file only renders them and sends messages.
const DEFAULTS = { server: "http://127.0.0.1:8765", maxSteps: 15, thinkBelow: 0.2, plan: true, theme: "pink", motion: "smooth" }; // thinkBelow 0.2: best routed accuracy (eval_route.py)
const ACTIVE = ["starting", "planning", "thinking", "thinking hard"];
const LIVE_TEXT = { starting: "Starting…", planning: "Reading your message and planning…", thinking: "Deciding the next step…",
  "thinking hard": "Not sure, so taking a screenshot and thinking it through…" };
const END = { done: "Done", stopped: "Stopped", blocked: "Blocked", error: "Error", "out of steps": "Out of steps" };
const VERB = { CLICK: "Click", TYPE_TEXT: "Type", HOVER: "Hover", SELECT: "Choose", SCROLL_DOWN: "Scroll down", SCROLL_UP: "Scroll up",
  GO_BACK: "Go back", GO_FORWARD: "Go forward", GOTO_URL: "Open", PRESS_KEY: "Press", DONE: "Finished", BLOCKED: "Can't continue" };
const SUGGEST = ["Add a Coffee Mug to the cart", "Search for the weather in Athens", "Show only Electronics, then sort by price"];
const ICON = { fast: '<svg viewBox="0 0 24 24"><path d="M13 2 4 14h7l-1 8 9-12h-7z"/></svg>',
  thought: '<svg viewBox="0 0 24 24"><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M18 6l-2.5 2.5M8.5 15.5 6 18"/></svg>' };

const $ = (id) => document.getElementById(id);
const pct = (p) => `${Math.round(p * 100)}%`;
// Page-derived strings (labels, answers, reasoning) only ever become text nodes, never HTML
function el(tag, props = {}, ...kids) {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...kids.filter((k) => k != null && k !== false));
  return e;
}
const icon = (name) => Object.assign(document.createElement("span"), { className: "ic", innerHTML: ICON[name] }); // static markup only

// ?window=1: the docked window (no side panel in this browser), so "this page" is the active tab of the browser window
const docked = new URLSearchParams(location.search).has("window");
if (globalThis.reflexHost?.desktop) document.body.classList.add("desktop");
async function targetTab() {
  if (!docked) return (await chrome.tabs.query({ active: true, currentWindow: true }))[0];
  const { dock } = await chrome.storage.session.get("dock");
  return (await chrome.tabs.query({ active: true, windowId: dock?.browser ?? -2 }).catch(() => []))[0]
    ?? (await chrome.tabs.query({ active: true, windowType: "normal" }))[0];
}

// ---------- settings ----------
let settings = { ...DEFAULTS };
const fields = ["server", "maxSteps", "thinkBelow", "plan"];
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  for (const b of $("swatches").children) b.setAttribute("aria-pressed", String(b.dataset.t === t));
}
chrome.storage.local.get(DEFAULTS).then((s) => {
  settings = s;
  for (const k of fields) $(k)[k === "plan" ? "checked" : "value"] = s[k];
  applyTheme(s.theme);
  checkServer();
});
for (const k of fields) {
  $(k).addEventListener("change", () => {
    const v = k === "server" ? $(k).value.trim().replace(/\/$/, "") : k === "plan" ? $(k).checked : Number($(k).value);
    settings[k] = v;
    chrome.storage.local.set({ [k]: v });
    if (k === "server") checkServer();
  });
}
$("swatches").addEventListener("click", (e) => {
  const t = e.target.closest(".swatch")?.dataset.t;
  if (t) { settings.theme = t; applyTheme(t); chrome.storage.local.set({ theme: t }); }
});
const sheet = (id, open) => { $(id).classList.toggle("open", open); $(id).setAttribute("aria-hidden", String(!open)); };
$("gear").onclick = () => sheet("sheet", true);
$("close").onclick = () => sheet("sheet", false);
$("history").onclick = () => sheet("chats-sheet", true);
$("chats-close").onclick = () => sheet("chats-sheet", false);
for (const id of ["sheet", "chats-sheet"]) $(id).onclick = (e) => { if (e.target === $(id)) sheet(id, false); };
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { sheet("sheet", false); sheet("chats-sheet", false); } });

async function checkServer() {
  try {
    await fetch(settings.server + "/v1/status", { mode: "no-cors" });
    $("server-state").textContent = "";
    $("status").classList.remove("off");
  } catch {
    $("server-state").textContent = "Model server offline";
    $("status").classList.add("off");
    $("status-text").textContent = "Offline";
    clearTimeout(checkServer.retry);
    checkServer.retry = setTimeout(checkServer, 3000); // keep looking: it may still be starting
  }
  render();
}

// ---------- sending ----------
let state = { chat: [], live: null, chats: [], chatId: null };
const running = () => Boolean(state.live && ACTIVE.includes(state.live.status));
function autosize() { const t = $("task"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 120) + "px"; }
$("task").addEventListener("input", autosize);
$("task").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("form").requestSubmit(); }
});
async function send(text) {
  const task = text.trim();
  if (!task || running()) return;
  const tab = await targetTab();
  if (!tab?.id) return;
  $("task").value = "";
  autosize();
  chrome.runtime.sendMessage({ cmd: "start", tabId: tab.id, task, settings });
}
$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  if (running()) chrome.runtime.sendMessage({ cmd: "stop" });
  else send($("task").value);
});
$("new").onclick = () => { if (!running()) chrome.runtime.sendMessage({ cmd: "clear" }); };

// ---------- rendering ----------
function stepRow(e) {
  const m = /^(\w+) '(.*)'(.*)$/.exec(e.label || "");
  // "(would repeat an earlier action)" and similar are internal reasons, not something to show
  const target = m ? el("span", {}, ` “${m[2]}”`, el("span", {}, ` ${m[1]}${m[3]}`)) : e.label && !e.label.startsWith("(") ? ` ${e.label}` : "";
  const typed = e.text ? (e.op === "TYPE_TEXT" ? ` “${e.text}” into` : ` ${e.text}`) : "";
  const what = el("div", { className: "what" }, VERB[e.op] || e.op, typed, target);
  const meta = el("div", { className: "meta" }, e.by === "thought" ? "thought it through" : `fast · ${pct(e.p)}`);
  const why = e.thought ? el("details", {}, el("summary", {}, "Reasoning"), el("p", {}, e.thought)) : null;
  return el("li", { className: `step ${e.by === "thought" ? "thought" : "fast"}` }, icon(e.by === "thought" ? "thought" : "fast"), what, meta, why);
}

const openActions = new Set(); // ids of agent messages whose "Actions" the user opened
function agentCard(v, live) {
  const card = el("div", { className: `msg agent${live ? " running" : ""}` });
  if (v.said && v.task && v.task !== v.said) card.append(el("p", { className: "understood" }, "Understood as ", el("b", {}, v.task)));
  if (v.note) card.append(el("p", { className: "understood" }, v.note));
  if (v.plan?.length > 1) {
    const done = v.status === "done";
    card.append(el("ol", { className: "plan" }, ...v.plan.map((s, i) => {
      const li = el("li", { className: done || i < v.part ? "past" : i === v.part ? "now" : "" }, s);
      li.dataset.n = String(i + 1);
      return li;
    })));
  }
  if (v.log?.length) { // every action folds into one "Actions" row; stays open across re-renders once opened
    const box = el("details", { className: "actions", open: openActions.has(v.id) },
      el("summary", {}, "Actions", el("span", { className: "count" }, String(v.log.length))),
      el("ul", { className: "steps" }, ...v.log.map(stepRow)));
    box.addEventListener("toggle", () => (box.open ? openActions.add(v.id) : openActions.delete(v.id)));
    card.append(box);
  }
  if (live) card.append(el("div", { className: "live" }, el("span", { className: "dots" }, el("i"), el("i"), el("i")),
    (LIVE_TEXT[v.status] || "Working…") + (v.step ? ` · step ${v.step}` : "")));
  else {
    const status = v.status || "stopped";
    const res = el("div", { className: "result" }, el("span", { className: `badge ${status.replace(/ /g, "-")}` }, END[status] || status));
    if (v.answer) res.append(el("p", { className: "answer" }, v.answer,
      status === "done" ? el("small", {}, v.verified ? "Read off a screenshot of the final page." : "Summary written by a small model without re-checking the page.") : null));
    if (v.error) res.append(el("p", { className: "err" }, v.error));
    card.append(res);
  }
  return card;
}

function emptyState() {
  const caps = el("div", { className: "caps" }, el("i"), el("i"), el("i"), el("i"), el("i"), el("i"));
  caps.setAttribute("aria-hidden", "true");
  return el("div", { className: "empty" }, caps,
    el("h2", {}, "What should I ", el("span", { className: "gt" }, "do?")),
    el("p", {}, "I work on the page you're looking at. Ask for a task, then just follow up: \"now add two more\", \"make it the 6th\"."),
    el("div", { className: "chips" }, ...SUGGEST.map((s) => Object.assign(el("button", { className: "chip", type: "button" }, s), { onclick: () => send(s) }))));
}

function render() {
  const box = $("chat");
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
  const nodes = [];
  for (const m of state.chat) nodes.push(m.role === "user" ? el("div", { className: "msg user" }, m.text) : agentCard(m, false));
  const live = state.live && !state.chat.some((m) => m.role === "agent" && m.id === state.live.id) ? state.live : null;
  if (live) nodes.push(agentCard(live, ACTIVE.includes(live.status)));
  box.replaceChildren(...(nodes.length ? nodes : [emptyState()]));
  if (atBottom || live) box.scrollTop = box.scrollHeight;

  const busy = running();
  $("send").classList.toggle("stop", busy);
  $("send").setAttribute("aria-label", busy ? "Stop" : "Send");
  $("send").innerHTML = busy ? '<svg viewBox="0 0 24 24"><rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor"/></svg>'
    : '<svg viewBox="0 0 24 24"><path d="M12 19V5M5 12l7-7 7 7"/></svg>';
  $("new").disabled = busy || !state.chat.length;
  const current = state.chats.find((c) => c.id === state.chatId);
  $("chat-title").textContent = current?.title ?? "New chat";
  renderChats(busy);
  if (!$("status").classList.contains("off")) {
    $("status").classList.toggle("live", busy);
    $("status-text").textContent = busy ? (state.live.status === "thinking hard" ? "Thinking" : "Working") : "Ready";
  }
}

// ---------- saved chats ----------
function ago(t) {
  const m = Math.round((Date.now() - t) / 60000);
  return m < 1 ? "just now" : m < 60 ? `${m} min ago` : m < 1440 ? `${Math.round(m / 60)} h ago`
    : new Date(t).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
function renderChats(busy) {
  const rows = state.chats.map((c) => {
    const n = c.messages.filter((m) => m.role === "user").length;
    const b = el("button", { type: "button", className: `chat-item${c.id === state.chatId ? " current" : ""}`, disabled: busy },
      el("b", {}, c.title), el("small", {}, `${n} message${n === 1 ? "" : "s"} · ${ago(c.updated)}`));
    b.onclick = () => { chrome.runtime.sendMessage({ cmd: "open", id: c.id }); sheet("chats-sheet", false); };
    return el("li", {}, b);
  });
  $("chat-list").replaceChildren(...(rows.length ? rows : [el("li", { className: "no-chats" }, "No saved chats yet. Every chat is saved here.")]));
}

Promise.all([chrome.storage.local.get({ chat: [], chats: [], chatId: null }), chrome.storage.session.get("agent")]).then(([l, s]) => {
  state = { chat: l.chat, live: s.agent ?? null, chats: l.chats, chatId: l.chatId };
  render();
  $("task").focus();
});
chrome.storage.onChanged.addListener((c, area) => {
  if (area === "local") {
    for (const k in DEFAULTS) if (c[k] && k !== "theme") { settings[k] = c[k].newValue; if (fields.includes(k)) $(k)[k === "plan" ? "checked" : "value"] = c[k].newValue; }
    if (c.theme) { settings.theme = c.theme.newValue; applyTheme(c.theme.newValue); }
  }
  if (area === "local" && (c.chat || c.chats || "chatId" in c)) {
    if (c.chat) state.chat = c.chat.newValue ?? [];
    if (c.chats) state.chats = c.chats.newValue ?? [];
    if ("chatId" in c && (c.chatId.newValue ?? null) !== state.chatId) { state.chatId = c.chatId.newValue ?? null; openActions.clear(); }
  } else if (area === "session" && "agent" in c) state.live = c.agent.newValue ?? null;
  else return;
  render();
});
