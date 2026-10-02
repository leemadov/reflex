// Computer Decisions agent. Each step: read the page's accessibility tree over CDP, printed exactly like BrowserGym
// (the format cd-0.6b was trained on) -> ask the model for operation + element -> act with trusted CDP input.
// Fully autonomous. A multi-step task is first split into a short plan (System 2's base model); each step works on
// the current plan item. System 1 decides in one pass; when its calibrated confidence is low, System 2 looks at a
// screenshot, reasons, and picks the action instead.

const OPS = {
  CLICK: "click an element on the page",
  TYPE_TEXT: "type text into an input field",
  HOVER: "hover the mouse over an element",
  SCROLL_DOWN: "scroll the page down",
  SCROLL_UP: "scroll the page up",
  GO_BACK: "go back to the previous page",
  DONE: "the task is complete; stop and give the answer",
  BLOCKED: "the task cannot be completed; give up",
};
const TARGETED = new Set(["CLICK", "TYPE_TEXT", "HOVER", "SELECT"]);
// what System 2 may answer and act() can carry out (it can also say SELECT / GOTO_URL / PRESS_KEY, System 1 cannot)
const ACTIONS = new Set([...TARGETED, "SCROLL_DOWN", "SCROLL_UP", "GO_BACK", "GO_FORWARD", "GOTO_URL", "PRESS_KEY", "DONE", "BLOCKED"]);
const KEYS = { enter: ["Enter", 13, "\r"], tab: ["Tab", 9], escape: ["Escape", 27], backspace: ["Backspace", 8],
  arrowdown: ["ArrowDown", 40], arrowup: ["ArrowUp", 38], pagedown: ["PageDown", 34], pageup: ["PageUp", 33] };
const QUESTIONS = {
  operation: { type: "choice", instructions: "What is the next operation to perform to accomplish the `task`?", criteria: OPS },
  target: { type: "choice", instructions: "Which element should the next operation act on?", criteria: "elements" },
};
const IGNORED_PROPS = new Set(["editable", "readonly", "level", "settable", "multiline", "invalid", "focusable"]);

let run = null; // the one active run

const cdp = (method, params = {}) => chrome.debugger.sendCommand({ tabId: run.tabId }, method, params);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Python repr(): BrowserGym prints names and property values with it
function py(v) {
  if (typeof v === "boolean") return v ? "True" : "False";
  if (typeof v !== "string") return String(v);
  const q = v.includes("'") && !v.includes('"') ? '"' : "'";
  const body = v.replace(/\\/g, "\\\\").replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t");
  return q + body.replaceAll(q, "\\" + q) + q;
}

// Port of BrowserGym's flatten_axtree_to_str(with_clickable=True, with_visible=True). Element ids are backend
// node ids; clickable and visible come from one DOM snapshot instead of BrowserGym's injected marking script.
async function snapshot() {
  const [{ nodes }, { documents: [d] }, { cssLayoutViewport: v }] = await Promise.all([
    cdp("Accessibility.getFullAXTree"),
    cdp("DOMSnapshot.captureSnapshot", { computedStyles: [] }),
    cdp("Page.getLayoutMetrics"),
  ]);
  const meta = new Map();
  const clickable = new Set(d.nodes.isClickable?.index ?? []);
  d.nodes.backendNodeId.forEach((b, i) => meta.set(b, { el: d.nodes.nodeType[i] === 1, click: clickable.has(i), vis: false }));
  d.layout.nodeIndex.forEach((i, j) => { // visible = at least half of its box inside the viewport
    const [x, y, w, h] = d.layout.bounds[j];
    const ix = Math.min(x + w, v.pageX + v.clientWidth) - Math.max(x, v.pageX);
    const iy = Math.min(y + h, v.pageY + v.clientHeight) - Math.max(y, v.pageY);
    if (w * h > 0 && ix > 0 && iy > 0 && (ix * iy) / (w * h) >= 0.5) meta.get(d.nodes.backendNodeId[i]).vis = true;
  });

  const byId = new Map(nodes.map((n) => [n.nodeId, n]));
  const lines = [], elems = {}, seen = new Set();
  const walk = (n, depth, parentName) => {
    if (seen.has(n.nodeId)) return;
    seen.add(n.nodeId);
    const role = n.role?.value ?? "", name = n.name?.value ?? "";
    let skip = !n.name || role === "LineBreak" || role === "InlineTextBox";
    if (!skip) {
      const m = meta.get(n.backendDOMNodeId), bid = m?.el ? String(n.backendDOMNodeId) : null;
      const attrs = [];
      for (const p of n.properties ?? []) {
        if (p.value?.value === undefined || IGNORED_PROPS.has(p.name)) continue;
        if (["required", "focused", "atomic"].includes(p.name)) { if (p.value.value) attrs.push(p.name); }
        else attrs.push(`${p.name}=${py(p.value.value)}`);
      }
      if (role === "generic" && !attrs.length) skip = true;
      if (role === "StaticText") skip ||= parentName.includes(name); // text already in the parent's name
      else if (bid) attrs.unshift(...[m.click && "clickable", m.vis && "visible"].filter(Boolean));
      if (!skip) {
        const label = role === "generic" && !name ? role : `${role} ${py(name.trim())}`;
        let s = bid ? `[${bid}] ${label}` : label;
        if (n.value?.value !== undefined) s += ` value=${py(n.value.value)}`;
        if (attrs.length) s += ", " + attrs.join(", ");
        lines.push("\t".repeat(depth) + s);
        if (bid) elems[bid] = label;
      }
    }
    for (const c of n.childIds ?? []) if (byId.has(c)) walk(byId.get(c), skip ? depth : depth + 1, name);
  };
  walk(nodes[0], 0, "");
  // For people: "button 'Add to cart' (5 of 19)" tells identical labels apart (the model reads the whole tree)
  const count = {}, nth = {}, shown = {};
  for (const l of Object.values(elems)) count[l] = (count[l] ?? 0) + 1;
  for (const [b, l] of Object.entries(elems)) {
    nth[l] = (nth[l] ?? 0) + 1;
    shown[b] = count[l] > 1 ? `${l} (${nth[l]} of ${count[l]})` : l;
  }
  return { page: lines.join("\n"), elems, shown };
}

async function post(path, body) {
  const r = await fetch(run.settings.server + path, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

// The decision model never writes text; the server's small LLM does (what to type, the final answer).
// Few-shot completion prompts: a 0.6B model copies the pattern, where an instruction gets a chatty sentence back.
const write = async (prompt, max_tokens = 32) => (await post("/v1/text", { prompt, max_tokens })).text.trim();
// Search boxes get their own examples: with form-field ones mixed in, the model types "Athens" for "weather in Athens".
const typeText = async (label) => (await write(/search/i.test(label) ? "Turn a task into the query to type into a search box.\n\n" +
  "Task: Search for the weather in Paris\nQuery: weather in Paris\n\n" +
  "Task: Find cheap flights to Madrid next week\nQuery: cheap flights to Madrid\n\n" +
  "Task: Search for \"Desk Lamp\" and add it to your cart.\nQuery: Desk Lamp\n\n" +
  "Task: Look up the definition of the word \"serendipity\".\nQuery: serendipity definition\n\n" +
  "Task: who won the 2018 world cup\nQuery: 2018 world cup winner\n\n" +
  `Task: ${run.goal}\nQuery:` : "Decide what a web agent should type into a field.\n\n" +
  "Task: Find the cheapest flight from Boston to Denver on May 3.\nField: combobox 'From'\nText: Boston\n\n" +
  "Task: Look up the definition of the word \"serendipity\".\nField: searchbox 'Search dictionary'\nText: serendipity\n\n" +
  "Task: Email Dana that the meeting moved to Friday.\nField: textbox 'Subject'\nText: Meeting moved to Friday\n\n" +
  `Task: ${run.goal}\nField: ${label}\nText:`)).replace(/^["'“]+|["'”]+$/g, "");
const answerText = (page) => write("Report the result of a finished web task in one sentence.\n\n" +
  "Steps:\nclick [12] where [12] is link 'Lyon'\nPage:\nheading 'Lyon'\nStaticText 'Population: 522,250'\n" +
  "Task: What is the population of Lyon?\nResult: Lyon has a population of 522,250.\n\n" +
  "Steps:\ntype [7] [ana@mail.com] where [7] is textbox 'Email'\nclick [9] where [9] is button 'Subscribe'\n" +
  "Page:\nStaticText 'Thanks for subscribing!'\nTask: Subscribe to the newsletter with ana@mail.com.\n" +
  "Result: Subscribed ana@mail.com to the newsletter.\n\n" +
  `Steps:\n${run.history.slice(-6).join("\n") || "none"}\nPage:\n${page.slice(0, 3000)}\nTask: ${run.task}\nResult:`, 48);

// One sentence on the result, written by the vision model from a screenshot of the final page.
async function summarizeShot() {
  const { data } = await cdp("Page.captureScreenshot", { format: "jpeg", quality: 75 });
  const r = await post("/v1/summarize", { task: run.task, screenshot: data });
  if (r.text) run.view.verified = true;
  return r.text;
}

async function act(op, bid, label, text) {
  if (op === "GO_BACK") return cdp("Runtime.evaluate", { expression: "history.back()" });
  if (op === "GO_FORWARD") return cdp("Runtime.evaluate", { expression: "history.forward()" });
  if (op === "GOTO_URL") {
    if (!/^https?:\/\//i.test(text)) throw new Error(`Won't open "${text}": only http(s) addresses.`);
    return cdp("Page.navigate", { url: text });
  }
  if (op === "PRESS_KEY") {
    const k = KEYS[text.toLowerCase().replace(/\s+/g, "")];
    if (!k) throw new Error(`Key "${text}" isn't supported yet.`);
    const ev = { key: k[0], code: k[0], windowsVirtualKeyCode: k[1] };
    await cdp("Input.dispatchKeyEvent", { type: "keyDown", ...ev, ...(k[2] ? { text: k[2] } : {}) });
    return cdp("Input.dispatchKeyEvent", { type: "keyUp", ...ev });
  }
  if (op === "SELECT") { // pick the <select> option whose text matches, as a user choosing it would
    await point(bid);
    const { object } = await cdp("DOM.resolveNode", { backendNodeId: Number(bid) });
    const { result } = await cdp("Runtime.callFunctionOn", { objectId: object.objectId, returnByValue: true, arguments: [{ value: text }],
      functionDeclaration: "function (want) { const all = [...(this.options || [])], w = want.toLowerCase(); " +
        "const o = all.find((o) => o.text.trim().toLowerCase() === w) || all.find((o) => o.text.toLowerCase().includes(w)); " +
        "if (!o) return false; this.value = o.value; this.dispatchEvent(new Event('input', { bubbles: true })); " +
        "this.dispatchEvent(new Event('change', { bubbles: true })); return true; }" });
    if (!result.value) throw new Error(`No option "${text}" in that list.`);
    return;
  }
  if (op.startsWith("SCROLL")) {
    const { cssLayoutViewport: v } = await cdp("Page.getLayoutMetrics");
    await glide(v.clientWidth / 2, v.clientHeight / 2);
    cursor(`scroll(${op === "SCROLL_DOWN" ? 1 : -1})`);
    return cdp("Input.dispatchMouseEvent", { type: "mouseWheel", x: v.clientWidth / 2, y: v.clientHeight / 2,
      deltaX: 0, deltaY: (op === "SCROLL_DOWN" ? 0.8 : -0.8) * v.clientHeight });
  }
  const [x, y] = await point(bid); // the cursor is usually already there
  await cdp("Input.dispatchMouseEvent", { type: "mouseMoved", x, y });
  if (op === "HOVER") return cursor("hover()");
  cursor("click()");
  for (const type of ["mousePressed", "mouseReleased"]) {
    await cdp("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: 1 });
  }
  if (op !== "TYPE_TEXT") return;
  const { result: focused } = await cdp("Runtime.evaluate", { returnByValue: true, expression:
    "(() => { const a = document.activeElement; if (a && a.select) a.select(); return a ? a.type || a.tagName : ''; })()" });
  if (focused.value === "password") throw new Error("Won't type into a password field; do that part yourself.");
  await cdp("Input.insertText", { text });
  if (/search|combobox/i.test(label)) { // WebArena's type action presses Enter; only do it where it can't submit a half-filled form
    const enter = { key: "Enter", code: "Enter", windowsVirtualKeyCode: 13 };
    await cdp("Input.dispatchKeyEvent", { type: "keyDown", text: "\r", ...enter });
    await cdp("Input.dispatchKeyEvent", { type: "keyUp", ...enter });
  }
}

// The on-page cursor (cursor.js) is decoration: a page that breaks it must never break the run
const CURSOR = fetch(chrome.runtime.getURL("cursor.js")).then((r) => r.text());
const CURSOR_COLORS = { pink: ["#FFD6F3", "#FF4FD8", "#E60ACF"], violet: ["#E4DBFF", "#8B5CF6", "#4B1FE6"],
  blue: ["#E0F2FE", "#38BDF8", "#2563EB"], lime: ["#F4FBD0", "#7FCC1E", "#2E8F0C"] };
async function cursor(call) {
  try {
    const theme = JSON.stringify({ colors: CURSOR_COLORS[run?.settings?.theme] ?? CURSOR_COLORS.blue, speed: run?.settings?.motion === "fast" ? 0.35 : 1 });
    const { result } = await cdp("Runtime.evaluate", { expression: `window.__cdTheme = ${theme};\n${await CURSOR}\nwindow.__cdCursor.${call}`, returnByValue: true });
    return result.value ?? 0;
  } catch { return 0; }
}
async function glide(x, y) { // cursor.js returns the glide's length; wait here, page animation frames stall in background tabs
  await sleep(await cursor(`move(${x}, ${y}, ${JSON.stringify(run.at ?? null)})`));
  run.at = [x, y];
}

// Point at an element: scroll it into view, give it a blue glow, glide the cursor onto it. -> its centre
async function point(bid) {
  if (!run) return null;
  await cdp("Runtime.evaluate", { expression: "for (const e of document.querySelectorAll('[data-cd-glow]')) " +
    "{ e.style.outline = e.dataset.cdOutline; e.style.boxShadow = e.dataset.cdShadow; delete e.dataset.cdGlow; }" });
  if (!bid) return null;
  const { object } = await cdp("DOM.resolveNode", { backendNodeId: Number(bid) });
  const { result } = await cdp("Runtime.callFunctionOn", { objectId: object.objectId, returnByValue: true, functionDeclaration:
    "function () { const b = this.getBoundingClientRect(); " + // scroll only if not fully on screen, then centre it
    "if (b.top < 0 || b.left < 0 || b.bottom > innerHeight || b.right > innerWidth) this.scrollIntoView({ block: 'center', inline: 'center' }); " +
    "Object.assign(this.dataset, { cdGlow: '', cdOutline: this.style.outline, cdShadow: this.style.boxShadow }); " +
    "this.style.outline = '2px solid #38bdf8'; " +
    "this.style.boxShadow = '0 0 0 4px rgba(56,189,248,.25), 0 0 22px rgba(56,189,248,.6)'; " +
    "const r = this.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; }" });
  await glide(...result.value);
  return result.value;
}

// The model can't tell 19 "Add to cart" buttons apart and picks one by position. Each twin (same op + label) gets the
// title of its own card (largest ancestor holding no other twin; its first line of text). Prefer a twin whose title the
// task names and that this run hasn't acted on yet; if every named one is done, the task is done (-> null).
// ponytail: title/word matching, not the model; when the task names no item the model's guess stands.
// keep=true (System 2 chose, having seen the screenshot): its element stands; only the card's key is wanted.
// Labels often carry state ("Available. Select as check-in date" -> "Selected check-in date"): key on the part
// before the first sentence break, so the same element keeps one identity while its state text changes.
const core = (label) => (label ?? "").replace(/\.\s.*$/, "").replace(/\.?'?$/, "");

async function disambiguate(op, bid, elems, keep = false) {
  const twins = Object.keys(elems).filter((b) => elems[b] === elems[bid]);
  if (twins.length < 2) return { bid, key: `${op}|${core(elems[bid])}` };
  const ids = await Promise.all(twins.map(async (b) => (await cdp("DOM.resolveNode", { backendNodeId: Number(b) })).object.objectId));
  const { result } = await cdp("Runtime.callFunctionOn", { objectId: ids[0], returnByValue: true,
    arguments: ids.map((objectId) => ({ objectId })), functionDeclaration: "function (...els) { return els.map((e) => { " +
      "let c = e; while (c.parentElement && !els.some((o) => o !== e && c.parentElement.contains(o))) c = c.parentElement; " +
      "return c.innerText.toLowerCase(); }); }" });
  const cards = result.value.map((text, i) => {
    const title = text.split("\n").map((l) => l.trim()).find(Boolean) ?? twins[i];
    return { bid: twins[i], text, title, key: `${op}|${core(elems[bid])}|${title}` };
  });
  if (keep) return cards.find((c) => c.bid === bid);
  const task = run.goal.toLowerCase();
  const fresh = (c) => !run.acted.has(c.key);
  const pick = (list) => list.find((c) => c.bid === bid) ?? list[0];
  const named = cards.filter((c) => task.includes(c.title));
  if (named.length) { // in the order the task names them
    const left = named.filter(fresh).sort((x, y) => task.indexOf(x.title) - task.indexOf(y.title));
    return left[0] ?? null;
  }
  // No title named: fall back to word overlap, ignoring words every card shares ("add", "cart")
  const words = (task.match(/[\p{L}\p{N}]{3,}/gu) ?? []).filter((w) => !cards.every((c) => c.text.includes(w)));
  const score = (c) => words.filter((w) => c.text.includes(w)).length;
  const best = Math.max(...cards.filter(fresh).map(score), 0);
  if (best > 0) return pick(cards.filter((c) => fresh(c) && score(c) === best));
  const own = cards.find((c) => c.bid === bid);
  return fresh(own) ? own : null;
}

// System 2: screenshot + page -> reasoning -> one action. null if it wrote nothing usable on this page.
async function deliberate(url, page, elems) {
  const { data } = await cdp("Page.captureScreenshot", { format: "jpeg", quality: 70 });
  const task = run.plan.length > 1 ? `${run.task} (current step: ${run.goal})` : run.task;
  const r = await post("/v1/systemtwo", { state: { task, url, history: run.history, page }, screenshot: data });
  const a = r.action;
  if (!a || !ACTIONS.has(a.op) || (TARGETED.has(a.op) && !elems[a.target])) return null;
  return { op: a.op, bid: a.target, given: a.arg ?? "", thought: r.thinking };
}

// Let navigations and re-renders finish before the next snapshot: wait until the document is loaded and its DOM has
// stopped changing for 200 ms (capped at 1.5 s for pages that never sit still), instead of a fixed 0.8 s. A short head
// start first, so a click that triggers a navigation has begun it before we look.
const QUIET = "new Promise((ok) => { if (document.readyState !== 'complete') return ok(false); let t; const mo = new MutationObserver(" +
  "() => { clearTimeout(t); t = setTimeout(done, 200); }); const done = () => { mo.disconnect(); ok(true); }; t = setTimeout(done, 200); " +
  "mo.observe(document, { subtree: true, childList: true, characterData: true }); setTimeout(done, 1500); })";
async function settle() {
  await sleep(250);
  for (let i = 0; i < 60; i++) {
    try {
      const { result } = await cdp("Runtime.evaluate", { expression: QUIET, awaitPromise: true, returnByValue: true });
      if (result.value) return;
    } catch { /* mid-navigation */ }
    await sleep(150);
  }
}

const historyLine = (op, bid, label, text) => ({ // WebArena action strings, as in the training histories
  CLICK: `click [${bid}] where [${bid}] is ${label}`,
  TYPE_TEXT: `type [${bid}] [${text}] where [${bid}] is ${label}`,
  HOVER: `hover [${bid}] where [${bid}] is ${label}`,
  SELECT: `select [${bid}] [${text}] where [${bid}] is ${label}`,
  PRESS_KEY: `press [${text}]`,
  GOTO_URL: `goto [${text}]`,
  GO_FORWARD: "go_forward",
  SCROLL_DOWN: "scroll [down]",
  SCROLL_UP: "scroll [up]",
  GO_BACK: "go_back",
})[op];

async function publish(patch) {
  Object.assign(run.view, patch);
  const s = run.view.status;
  const badge = { done: "✓", error: "!", blocked: "!", planning: "…", thinking: String(run.view.step),
    "thinking hard": String(run.view.step) }[s] ?? "";
  chrome.action.setBadgeText({ text: badge });
  await chrome.storage.session.set({ agent: run.view });
}

// The chat: every message and every finished run, kept in local storage so it survives closing the panel.
// Each new message gets the last few turns as context, so "now add a backpack too" can be resolved into a full task.
const CHAT_TURNS = 6;
async function chat() { return (await chrome.storage.local.get({ chat: [] })).chat; }
async function addToChat(msg) { await chrome.storage.local.set({ chat: [...await chat(), msg] }); }
const turnText = (m) => m.role === "user" ? m.text
  : `Did "${m.task}" (${m.status})${m.answer ? ". " + m.answer : ""}${m.error ? ". Error: " + m.error : ""}`;

// Chrome won't let extensions drive its own pages (new tab, chrome://, the Web Store). Started there, open Google in that
// tab and carry on, and say so in the chat.
const START_PAGE = "https://www.google.com/?hl=en";
function loaded(tabId) {
  return new Promise((resolve) => {
    const done = (id, info) => { if (id === tabId && info.status === "complete") { chrome.tabs.onUpdated.removeListener(done); resolve(); } };
    chrome.tabs.onUpdated.addListener(done);
    setTimeout(() => { chrome.tabs.onUpdated.removeListener(done); resolve(); }, 15000);
  });
}
async function attach(tabId) {
  try {
    await chrome.debugger.attach({ tabId }, "1.3");
  } catch (e) {
    if (!/chrome:\/\/|chrome-extension:|Cannot access|webstore|edge:\/\//i.test(String(e?.message))) throw e;
    const ready = loaded(tabId);
    await chrome.tabs.update(tabId, { url: START_PAGE });
    await ready;
    await chrome.debugger.attach({ tabId }, "1.3");
    await publish({ note: "That tab was a browser page Chrome doesn't let extensions control, so I opened Google first." });
  }
}

async function start({ tabId, task, settings }) {
  if (run) throw new Error("A run is already active.");
  const context = (await chat()).slice(-CHAT_TURNS).map((m) => ({ role: m.role, text: turnText(m) }));
  const id = `run-${(await chat()).length}-${task.length}`;
  await addToChat({ role: "user", text: task, id });
  run = { tabId, task, settings, history: [], acted: new Set(), plan: [task], part: 0,
    view: { id, status: "starting", task, said: task, step: 0, log: [], plan: [task], part: 0 } };
  const keepAlive = setInterval(() => chrome.runtime.getPlatformInfo(), 20e3); // MV3 stops a worker idle for 30 s, even mid-run
  try {
    await attach(tabId);
    if (settings.plan !== false || context.length) {
      await publish({ status: "planning" });
      const r = await post("/v1/plan", { task, context, plan: settings.plan !== false }).catch(() => null);
      if (r?.task) run.task = r.task; // the follow-up rewritten as a standalone task
      if (r?.steps?.length) run.plan = r.steps;
      await publish({ task: run.task, plan: run.plan });
    }
    for (let step = 1; step <= settings.maxSteps && !run.stopped; step++) {
      run.goal = run.plan[run.part];
      await publish({ status: "thinking", step, part: run.part });
      const { page, elems, shown } = await snapshot();
      const url = (await cdp("Runtime.evaluate", { expression: "location.href", returnByValue: true })).result.value;
      const { answers: a } = await post("/v1/systemone", { state: { task: run.goal, url, history: run.history, page }, questions: QUESTIONS });
      const prob = (o, b) => a.operation.probabilities[o] * (TARGETED.has(o) ? a.target.probabilities[b] ?? 0 : 1);
      let op = a.operation.choice, bid = a.target.choice, by = "fast", thought = "", given = "";
      const p = prob(op, bid); // calibrated: how often System 1 is right at this confidence
      if (p < settings.thinkBelow) {
        await publish({ status: "thinking hard" });
        const d = await deliberate(url, page, elems).catch((e) => { console.warn("System 2:", e); return null; });
        if (d) ({ op, bid, given, thought } = d), by = "thought";
      }
      // The model is weak at noticing it is finished and will happily repeat "Add to cart": redoing an action this run
      // already did (same op on the same element or card, matched by label: re-renders change ids) means it is done.
      let key = null, why = "";
      if (TARGETED.has(op) && elems[bid]) {
        const t = await disambiguate(op, bid, elems, by === "thought");
        // stop on a repeat (A, A) or a back-and-forth (A, B, A): the step is done, or the model is going in circles
        if (t && run.lastKey !== t.key && run.prevKey !== t.key) ({ bid, key } = t);
        else { op = "DONE"; why = "(would repeat an earlier action)"; }
      }
      const text = op === "TYPE_TEXT" ? given || await typeText(elems[bid]) : given;
      await point(TARGETED.has(op) ? bid : null); // glide to what is about to be clicked or typed into, auto steps too
      if (TARGETED.has(op)) await sleep(settings.motion === "fast" ? 40 : 250); // a beat on target before the click
      const label = TARGETED.has(op) ? elems[bid] : "";
      run.view.log.push({ step, op, label: TARGETED.has(op) ? shown[bid] : why, text, p: Math.round(p * 1000) / 1000, by, thought, part: run.part });
      if (op === "DONE" && run.part < run.plan.length - 1) { // this plan item is done: on to the next
        run.part++;
        run.lastKey = run.prevKey = null;
        continue;
      }
      if (op === "DONE" || op === "BLOCKED") {
        const answer = op === "BLOCKED" ? "The model judged the task blocked or impossible from here. Over to you."
          : await summarizeShot().catch(() => "") || await answerText(page);
        await publish({ status: op === "DONE" ? "done" : "blocked", answer });
        return;
      }
      await act(op, bid, label, text);
      run.history.push(historyLine(op, bid, label, text));
      if (key) run.acted.add(key);
      run.prevKey = run.lastKey;
      run.lastKey = key;
      await settle();
    }
    await publish({ status: run.stopped ? "stopped" : "out of steps" });
  } catch (e) {
    await publish({ status: "error", error: String(e?.message ?? e) });
  } finally {
    clearInterval(keepAlive);
    await addToChat({ role: "agent", ...run.view }).catch(() => {});
    await point(null).catch(() => {}); // remove the glow
    await cursor("hide()");
    await chrome.debugger.detach({ tabId }).catch(() => {});
    run = null;
  }
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.cmd === "start") start(msg).catch((e) => addToChat({ role: "agent", task: msg.task, status: "error", error: e.message, log: [] }));
  else if (msg.cmd === "stop" && run) run.stopped = true;
  else if (msg.cmd === "clear" && !run) chrome.storage.local.set({ chat: [] });
});

// Chrome/Edge: the icon opens the side panel. Browsers without one (Opera): a full-height window docked to the right of
// the browser window, which narrows to make room and gets its width back when the panel closes.
const DOCK_W = 440;
async function openDocked(tab) {
  const { dock } = await chrome.storage.session.get("dock");
  if (dock) {
    try { return await chrome.windows.update(dock.id, { focused: true }); } catch { /* it was closed */ }
  }
  const w = await chrome.windows.get(tab.windowId); // maximized windows report the full screen bounds
  const width = Math.max(640, w.width - DOCK_W);
  let panel;
  try {
    await chrome.windows.update(w.id, { state: "normal", left: w.left, top: w.top, width, height: w.height });
    panel = await chrome.windows.create({ url: "popup.html?window=1", type: "popup", focused: true,
      left: w.left + width, top: w.top, width: DOCK_W, height: w.height });
  } catch { // odd screen layouts can reject the bounds: still open the panel, just not docked
    panel = await chrome.windows.create({ url: "popup.html?window=1", type: "popup", focused: true, width: DOCK_W, height: w.height });
  }
  await chrome.storage.session.set({ dock: { id: panel.id, browser: w.id, width: w.width, state: w.state } });
}
chrome.windows.onRemoved.addListener(async (id) => {
  const { dock } = await chrome.storage.session.get("dock");
  if (dock?.id !== id) return;
  await chrome.storage.session.remove("dock");
  chrome.windows.update(dock.browser, dock.state === "maximized" ? { state: "maximized" } : { width: dock.width }).catch(() => {});
});
// Open the panel ourselves on the icon click (a user gesture): if Chrome's side panel refuses for any reason, the docked
// window opens instead, so a click always shows Reflex. Chrome's automatic open-on-click is switched off so this fires.
chrome.sidePanel?.setPanelBehavior?.({ openPanelOnActionClick: false }).catch(() => {});
chrome.action.setPopup({ popup: "" }); // clear any popup an older version left behind
chrome.action.onClicked.addListener((tab) => {
  const docked = () => openDocked(tab).catch((e) => console.warn("dock:", e));
  if (!chrome.sidePanel?.open) return docked();
  chrome.sidePanel.open({ windowId: tab.windowId }).catch((e) => { console.warn("side panel:", e); docked(); });
});

// The user picked pink: make it the colour after an install or update, even if another one was chosen before.
chrome.runtime.onInstalled.addListener(() => chrome.storage.local.set({ theme: "pink" }));
chrome.debugger.onDetach.addListener(({ tabId }) => { // e.g. the user closed the "is debugging this browser" bar
  if (run?.tabId === tabId) run.stopped = true;
});
