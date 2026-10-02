// The chat panel is the extension's popup.html. It talks to chrome.storage / chrome.runtime / chrome.tabs; this bridge
// gives it the same calls over IPC to the agent running in the main process. popup.js adopts window.reflexHost as
// `chrome` (Electron already defines a window.chrome, so it can't be exposed under that name).
const { contextBridge, ipcRenderer } = require("electron");

const listeners = [];
ipcRenderer.on("storage-changed", (_e, changes, area) => { for (const cb of listeners) cb(changes, area); });
const area = (name) => ({
  get: (keys) => ipcRenderer.invoke("storage-get", name, keys),
  set: (items) => ipcRenderer.invoke("storage-set", name, items),
  remove: (keys) => ipcRenderer.invoke("storage-remove", name, keys),
});

contextBridge.exposeInMainWorld("reflexHost", {
  desktop: true,
  storage: { local: area("local"), session: area("session"), onChanged: { addListener: (cb) => listeners.push(cb) } },
  runtime: { sendMessage: (msg) => { ipcRenderer.send("runtime-message", msg); return Promise.resolve(); } },
  tabs: { query: () => ipcRenderer.invoke("tabs-query") },
});
