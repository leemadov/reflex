// The top bar's bridge to the main process: layout of the native views, navigation, model status, the speed/quality
// mode, and the agent's live state (for the glow around the browser while Reflex works).
const { contextBridge, ipcRenderer } = require("electron");

const on = (channel) => (cb) => ipcRenderer.on(channel, (_e, ...args) => cb(...args));
contextBridge.exposeInMainWorld("reflex", {
  layout: (rects) => ipcRenderer.send("layout", rects),
  navigate: (text) => ipcRenderer.send("navigate", text),
  nav: (action) => ipcRenderer.send("nav-action", action),
  state: () => ipcRenderer.invoke("state"),
  setMode: (mode) => ipcRenderer.invoke("set-mode", mode),
  get: (area, keys) => ipcRenderer.invoke("storage-get", area, keys),
  stop: () => ipcRenderer.send("runtime-message", { cmd: "stop" }),
  onNav: on("nav"),
  onFavicon: on("favicon"),
  onServer: on("server"),
  onStorage: on("storage-changed"),
  onFocusAddress: on("focus-address"),
});
