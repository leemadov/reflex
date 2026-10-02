// Runs the extension's own agent (extension/background.js) inside the desktop app, unchanged, against a small chrome.*
// shim: chrome.debugger -> the built-in browser's webContents.debugger, chrome.storage -> a JSON file + memory,
// chrome.tabs -> that one browser view (tab id 1). One agent implementation for both the extension and the app.
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { fileURLToPath, pathToFileURL } = require("url");

const TAB = 1;
const clone = (v) => (v === undefined ? undefined : JSON.parse(JSON.stringify(v))); // plain data across realms/IPC

function makeStore(file, onChange) {
  const areas = { local: {}, session: {} };
  try { areas.local = JSON.parse(fs.readFileSync(file, "utf8")); } catch { /* first run */ }
  let timer = null;
  const persist = () => {
    clearTimeout(timer);
    timer = setTimeout(() => fs.writeFile(file, JSON.stringify(areas.local), () => {}), 150);
  };
  const area = (name) => ({
    async get(keys) {
      const data = areas[name];
      if (keys == null) return clone(data);
      if (typeof keys === "string") keys = [keys];
      const out = {};
      if (Array.isArray(keys)) { for (const k of keys) if (k in data) out[k] = clone(data[k]); }
      else for (const k in keys) out[k] = k in data ? clone(data[k]) : clone(keys[k]);
      return out;
    },
    async set(items) {
      const changes = {};
      for (const k in items) {
        changes[k] = { oldValue: areas[name][k], newValue: clone(items[k]) };
        areas[name][k] = clone(items[k]);
      }
      if (name === "local") persist();
      onChange(changes, name);
    },
    async remove(keys) {
      const changes = {};
      for (const k of [].concat(keys)) if (k in areas[name]) { changes[k] = { oldValue: areas[name][k] }; delete areas[name][k]; }
      if (name === "local") persist();
      if (Object.keys(changes).length) onChange(changes, name);
    },
  });
  return { local: area("local"), session: area("session"), raw: areas };
}

function startAgent({ extDir, browser, storeFile, onStorageChange }) {
  const wc = browser.webContents;
  const listeners = { message: [], detach: [], updated: [], storage: [] };
  const store = makeStore(storeFile, (changes, name) => {
    for (const cb of listeners.storage) cb(changes, name);
    onStorageChange(changes, name);
  });
  wc.on("did-start-loading", () => listeners.updated.forEach((cb) => cb(TAB, { status: "loading" }, { id: TAB })));
  wc.on("did-stop-loading", () => listeners.updated.forEach((cb) => cb(TAB, { status: "complete" }, { id: TAB })));
  wc.debugger.on("detach", () => listeners.detach.forEach((cb) => cb({ tabId: TAB })));
  const tab = () => ({ id: TAB, windowId: 1, active: true, url: wc.getURL(), title: wc.getTitle() });
  const noop = () => {};
  const event = (list) => ({ addListener: (cb) => list.push(cb), removeListener: (cb) => { const i = list.indexOf(cb); if (i >= 0) list.splice(i, 1); }, hasListeners: () => list.length > 0 });

  const chrome = {
    debugger: {
      // attached once and kept: no reattach per run (faster), and the app has no "is debugging" bar to clear
      async attach() { if (!wc.debugger.isAttached()) wc.debugger.attach("1.3"); },
      async detach() {},
      sendCommand: (_target, method, params) => wc.debugger.sendCommand(method, params || {}),
      onDetach: event(listeners.detach),
    },
    storage: { local: store.local, session: store.session, onChanged: event(listeners.storage) },
    tabs: {
      async update(_id, { url }) { wc.loadURL(url).catch(noop); return tab(); },
      async get() { return tab(); },
      async query() { return [tab()]; },
      onUpdated: event(listeners.updated),
    },
    runtime: {
      getURL: (p) => pathToFileURL(path.join(extDir, p)).href,
      getPlatformInfo: async () => ({ os: "win" }),
      onMessage: event(listeners.message),
      onInstalled: { addListener: noop }, // the app sets its own defaults
    },
    action: { setBadgeText: noop, setPopup: noop, onClicked: { addListener: noop } },
    windows: { get: noop, update: noop, create: noop, remove: noop, onRemoved: { addListener: noop } },
  };
  // background.js fetches its cursor.js with fetch(chrome.runtime.getURL(...)): serve file:// from disk
  const hostFetch = (url, opts) => String(url).startsWith("file:")
    ? Promise.resolve({ ok: true, text: async () => fs.readFileSync(fileURLToPath(url), "utf8") })
    : fetch(url, opts);

  const sandbox = { chrome, fetch: hostFetch, console, setTimeout, clearTimeout, setInterval, clearInterval, URL, URLSearchParams,
    TextEncoder, TextDecoder, AbortController, structuredClone };
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.join(extDir, "background.js"), "utf8"), sandbox, { filename: "extension/background.js" });

  return {
    store,
    // a message from the chat panel, exactly as chrome.runtime.sendMessage would deliver it
    dispatch: (msg) => { for (const cb of listeners.message) cb(clone(msg), { id: "desktop" }, noop); },
    tab,
  };
}

module.exports = { startAgent };
