// The agent's on-page cursor: a glassy blue pointer that glides along a soft arc with a short light trail and
// ripples on click. Purely visual: aria-hidden keeps it out of the accessibility tree the model reads, and
// pointer-events: none lets the real (CDP) clicks land on the page underneath. background.js evaluates this file
// before every cursor call; it builds itself once per page. DOM is built with createElement and styled through an
// adopted stylesheet, because strict CSP / Trusted Types pages reject innerHTML and inline <style>.
(() => {
  if (window.__cdCursor) return;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  // colours + glide speed come from the host (theme setting); default: the original glassy blue
  const theme = window.__cdTheme || {};
  const [hi, mid, deep] = theme.colors || ["#e0f2fe", "#38bdf8", "#2563eb"];
  const rgba = (hex, a) => { const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; };
  const SVG = "http://www.w3.org/2000/svg";
  const make = (tag, attrs = {}, ns) => {
    const e = ns ? document.createElementNS(ns, tag) : document.createElement(tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  };

  const host = make("div", { "aria-hidden": "true" });
  host.style.cssText = "position:fixed;inset:0;pointer-events:none;z-index:2147483647";
  const root = host.attachShadow({ mode: "closed" });
  const sheet = new CSSStyleSheet();
  sheet.replaceSync(`
    * { pointer-events: none; }
    .ptr { position: fixed; left: 0; top: 0; will-change: transform; opacity: 0; transition: opacity .3s ease; }
    .ptr svg { display: block; overflow: visible; transform-origin: 2px 2px;
               filter: drop-shadow(0 0 3px ${rgba(hi, 0.95)}) drop-shadow(0 0 12px ${rgba(mid, 0.75)})
                       drop-shadow(0 0 26px ${rgba(deep, 0.45)}); }
    .halo { position: absolute; left: -18px; top: -18px; width: 36px; height: 36px; border-radius: 50%;
            background: radial-gradient(circle, ${rgba(mid, 0.55)} 0, ${rgba(mid, 0.18)} 45%, ${rgba(mid, 0)} 70%);
            transition: transform .35s ease; }
    .ptr.moving .halo { transform: scale(1.35); }
    .dot { position: fixed; left: -3.5px; top: -3.5px; width: 7px; height: 7px; border-radius: 50%; opacity: 0;
           background: ${hi}; box-shadow: 0 0 8px 3px ${rgba(mid, 0.7)}; transition: opacity .35s ease; }
    .ring { position: fixed; left: -16px; top: -16px; width: 32px; height: 32px; border-radius: 50%;
            border: 2px solid ${rgba(hi, 0.95)};
            box-shadow: 0 0 14px ${rgba(mid, 0.85)}, inset 0 0 10px ${rgba(mid, 0.55)}; }
  `);
  root.adoptedStyleSheets = [sheet];

  const ptr = make("div", { class: "ptr" });
  const svg = make("svg", { width: "22", height: "28", viewBox: "0 0 22 28" }, SVG);
  const grad = make("linearGradient", { id: "cd-glass", x1: "0", y1: "0", x2: "0.7", y2: "1" }, SVG);
  for (const [offset, color] of [["0", hi], ["0.45", mid], ["1", deep]]) {
    grad.append(make("stop", { offset, "stop-color": color }, SVG));
  }
  const defs = make("defs", {}, SVG);
  defs.append(grad);
  svg.append(defs, make("path", { d: "M2 2 L2 22 L7.6 16.9 L11.3 25 L14.8 23.5 L11.2 15.6 L18.6 15.4 Z",
    fill: "url(#cd-glass)", "fill-opacity": "0.94", stroke: hi, "stroke-opacity": "0.9",
    "stroke-width": "1.4", "stroke-linejoin": "round" }, SVG));
  ptr.append(make("div", { class: "halo" }), svg);
  const dots = Array.from({ length: 6 }, () => make("div", { class: "dot" }));
  root.append(...dots, ptr);
  document.documentElement.append(host);

  let pos = null;
  const place = (x, y, tilt = 0) => { ptr.style.transform = `translate(${x}px, ${y}px) rotate(${tilt}deg)`; };
  const ring = (scale, ms, peak) => {
    if (reduce || !pos) return;
    const r = make("div", { class: "ring" });
    root.append(r);
    const at = `translate(${pos[0]}px, ${pos[1]}px)`;
    r.animate([{ transform: `${at} scale(.3)`, opacity: peak }, { transform: `${at} scale(${scale})`, opacity: 0 }],
      { duration: ms, easing: "cubic-bezier(.2,.7,.3,1)" }).onfinish = () => r.remove();
  };

  window.__cdCursor = {
    // Glide to (x, y) in viewport px; `from` places a fresh cursor (after a navigation) where the last one was.
    // Returns the duration in ms: the caller waits it out, because requestAnimationFrame stalls in background tabs.
    move(x, y, from) {
      if (!pos) {
        pos = from || [innerWidth / 2, innerHeight - 48];
        place(...pos);
        ptr.style.opacity = "1";
      }
      const [x0, y0] = pos, d = Math.hypot(x - x0, y - y0);
      const ms = reduce || d < 2 ? 0 : Math.round(Math.min(950, 320 + d * 0.5) * (theme.speed || 1));
      pos = [x, y];
      if (!ms) { place(x, y); return 0; }
      const cx = (x0 + x) / 2 - (y - y0) * 0.18, cy = (y0 + y) / 2 + (x - x0) * 0.18; // bend the path a little
      const t0 = performance.now(), trail = [];
      let last = [x0, y0];
      ptr.classList.add("moving");
      const frame = (now) => {
        const t = Math.min(1, (now - t0) / ms);
        const e = t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2, u = 1 - e; // ease in-out cubic
        const px = u * u * x0 + 2 * u * e * cx + e * e * x, py = u * u * y0 + 2 * u * e * cy + e * e * y;
        place(px, py, Math.max(-12, Math.min(12, (px - last[0]) * 0.8))); // lean into the motion
        last = [px, py];
        trail.push(last);
        dots.forEach((dot, i) => {
          const p = trail[trail.length - 1 - (i + 1) * 2];
          if (!p) return;
          dot.style.transform = `translate(${p[0]}px, ${p[1]}px) scale(${1 - i * 0.14})`;
          dot.style.opacity = t < 1 ? String(0.6 * (1 - i / dots.length)) : "0";
        });
        if (t < 1) return requestAnimationFrame(frame);
        place(x, y);
        ptr.classList.remove("moving");
        for (const dot of dots) dot.style.opacity = "0";
      };
      requestAnimationFrame(frame);
      return ms;
    },
    click() {
      if (!reduce) svg.animate([{ transform: "scale(1)" }, { transform: "scale(.8)" }, { transform: "scale(1)" }], { duration: 240, easing: "ease-out" });
      ring(2.1, 520, 1);
    },
    hover() { ring(1.6, 700, 0.6); },
    scroll(dir) {
      if (!reduce) svg.animate([{ transform: "translateY(0)" }, { transform: `translateY(${dir * 12}px)` }, { transform: "translateY(0)" }], { duration: 460, easing: "ease-in-out" });
    },
    hide() {
      ptr.style.opacity = "0";
      setTimeout(() => host.remove(), 320);
      delete window.__cdCursor;
    },
  };
})();
