# Atlas AI · Reflex design system

The look shared by everything Reflex: the Chrome extension, the desktop app, the ad and the short, the thumbnails and
the wallpapers. Light ground, near-black type, one gradient, glossy capsules.

**In one line:** a calm light canvas, bold near-black headlines, and a single key word in the theme gradient, framed
by glossy capsule columns.

---

## 1. Brand

| Asset | File | Use |
|---|---|---|
| Atlas mark ("A" with arch), vector | [`docs/brand/atlas-mark.svg`](docs/brand/atlas-mark.svg) | Any size; the main symbol |
| "ATLAS AI" wordmark, vector | [`docs/brand/atlas-ai-wordmark.svg`](docs/brand/atlas-ai-wordmark.svg) | "By ATLAS AI" lines |
| Atlas logo, dark and light | [`docs/atlas-logo-dark.png`](docs/atlas-logo-dark.png), [`docs/atlas-logo-white.png`](docs/atlas-logo-white.png) | README header (801×510) |
| Atlas logo, original | [`docs/brand/atlas-logo-original.png`](docs/brand/atlas-logo-original.png) | The source everything is cut from: white mark + ATLAS on black, 1254×1254 |
| Mark and wordmark, PNG | `videos/clicky-ad/assets/atlas-mark.png`, `atlas-mark-dark.png`, `atlas-wordmark.png`, `atlas-wordmark-dark.png` (local only) | What the ad and short use (mark 416×339, wordmark 801×91); prefer the SVGs elsewhere |

- **Names:** the company is **Atlas AI**; the product is **Reflex**; the models are **Reflex Instinct 0.6B** (fast)
  and **Reflex Reason 2B** (thinks it through).
- **Lockup:** the Atlas mark centred over the capsule cluster (section 5), "Reflex" in the gradient below it, then
  "By" + the ATLAS AI wordmark.
- **The Atlas lettering** exists only as these capitals (A, T, L, S, I), traced from the logo. Set "By" in Inter 500
  at the wordmark's cap height. Don't fake other letters in that style.
- **Mark colour:** ink `#1D1D1F` on light grounds; white on dark. Never put the gradient on the mark.
- **Name clash:** "reflex 4B" and "reflex-27b" on the JevBench board are someone else's. Always say "Reflex by Atlas
  AI" or the full model name where they could be confused.

## 2. Colour

### Fixed neutrals (every theme)

| Token | Hex | Use |
|---|---|---|
| `--ground` | `#F5F5F7` | Full-bleed background |
| `--card` | `#FFFFFF` | Cards, chat bubbles, panels |
| `--ink` | `#1D1D1F` | Headlines, body, the mark |
| `--muted` | `#6E6E73` | Sub-lines, secondary text |
| `--light` | `#86868B` | Fine print, hints, timestamps |
| `--line` | `rgba(29,29,31,0.08)` | Hairline borders |
| `--ok` / `--bad` | `#1F9D55` / `#D92D20` | Done / error badges |

### Themes

Pink is the default and the brand colour. The other three are user-selectable in the app and extension, and were
the ad's A/B variants. `c1`–`c5` paint capsules and glows; `g1`–`g4` are the text gradient, darkened so it reads on
the light ground.

| Token | Role | Pink | Violet | Blue | Lime |
|---|---|---|---|---|---|
| `--c1` | deep | `#E60ACF` | `#4B1FE6` | `#0030FF` | `#3DB511` |
| `--c2` | mid | `#FF4FD8` | `#8B5CF6` | `#0A84FF` | `#9ED80F` |
| `--c3` | bright | `#FF9BDD` | `#CDBBFF` | `#19D3FF` | `#F2E600` |
| `--c4` | accent | `#6EC8F0` | `#A6E3C4` | `#F2C3A0` | `#5FE3D2` |
| `--c5` | pale tint | `#F9D7EE` | `#ECE6FF` | `#DDF1FF` | `#F4F9D6` |
| `--g1` | gradient 1 | `#E60ACF` | `#3D1FE0` | `#0030FF` | `#2E8F0C` |
| `--g2` | gradient 2 | `#B83CF0` | `#6D3BF5` | `#0A6CFF` | `#3DA60F` |
| `--g3` | gradient 3 | `#6A5CFF` | `#8B5CF6` | `#0096D6` | `#129E86` |
| `--g4` | gradient 4 | `#1E9BD7` | `#1F9E78` | `#0A6CFF` | `#0E86A8` |

Agent cursor colours (glow, core, ring), per theme:

| Pink | Violet | Blue | Lime |
|---|---|---|---|
| `#FFD6F3` `#FF4FD8` `#E60ACF` | `#E4DBFF` `#8B5CF6` `#4B1FE6` | `#E0F2FE` `#38BDF8` `#2563EB` | `#F4FBD0` `#7FCC1E` `#2E8F0C` |

**Rules**
- Never hard-code a theme colour: use `var(--cN, <pink value>)`. For a transparent version, use
  `color-mix(in srgb, var(--c2) 35%, transparent)`.
- **One gradient word per line, at headline size only.** Everything else is ink.

## 3. Type

**Inter** for everything (`Inter-latin.woff2`, weights 100–900, shipped with each project).

| Role | Size | Weight | Tracking | Example |
|---|---|---|---|---|
| Hero (video, wallpaper) | 120–200 px · 11 vh | 700–800 | -0.045 to -0.06em | "Meet **Reflex.**" |
| Headline | 48–96 px | 600–700 | -0.03em | "Not sure? **It thinks.**" |
| Big number | 330–400 px | 800 | -0.06em | "**0.2**s" (the "s" at half size) |
| Sub-line | 40–64 px | 600 | -0.03em | "per decision", in `--muted` |
| UI title | 15–18 px | 650 | -0.02 to -0.03em | "Reflex", chat titles |
| UI body | 13–14 px | 400–500 | 0 | Chat text |
| Fine print / eyebrow | 11–12 px | 600 | 0.08em | Hints, in `--light` |

- Headlines are sentence case, never all caps. End hero lines with a full stop ("Done.", "Meet Reflex.").
- Copy is short, plain and concrete: the action itself ("It types.", "It clicks."), then the proof ("0.2 s per decision").

## 4. Gradient text

```css
.gt {
  background: linear-gradient(95deg, var(--g1) 5%, var(--g2) 38%, var(--g3) 66%, var(--g4) 95%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  padding: 0 0.04em 0.08em 0;   /* stops glyph edges and descenders from clipping */
}
```

Put the gradient on the span that wraps the word, not on a full-width block. Otherwise the gradient stretches across
the whole line and the word only gets part of it.

## 5. Capsules (the signature shape)

Tall glossy pills: a three-stop vertical gradient from the theme, a white rim, a highlight, and a soft coloured glow.

```css
.cap {
  border-radius: 999px;
  background: linear-gradient(180deg, var(--a) 0%, var(--b) 50%, var(--c) 100%);
  box-shadow: inset 0 0 0 2px rgba(255,255,255,.35),
              inset 0 20px 30px rgba(255,255,255,.45),
              0 0 24px color-mix(in srgb, var(--c2) 28%, transparent);
}
```

Stop palettes (top / middle / bottom), picked at random per pill with a seeded random:
`[c3, c1, c2]` `[c5, c3, c4]` `[c2, #B83CF0, c4]` `[c4, c5, c3]` `[c1, c2, c5]`.

Three uses:
- **Wall:** columns of pills across the frame. Width about 4.4 vh, step 5.4 vh, pill height 13–37 vh, gaps 1.4–6 vh,
  opacity 0.72–1. The hook and thumbnail fill the whole frame with it.
- **Edge frame:** the same wall, masked so it shows only at the outer edges and fades out before the content:
  `mask-image: linear-gradient(90deg, #000 0%, rgba(0,0,0,.9) 11%, transparent 29%, transparent 71%, rgba(0,0,0,.9) 89%, #000 100%)`.
  Used by the wallpapers, the demo and the thumbnail.
- **Cluster:** nine pills behind the Atlas mark, tallest in the middle, heights `9 13 17.5 21.5 24.5 21.5 17.5 13 9`
  vh, width 3.2 vh, step 4.3 vh. In motion they rise from the centre outwards. This is the end card and wallpaper
  lockup, and the three-pill version is the app icon and header mark.

Text never sits directly on a busy wall. Put it on a calm band (`rgba(245,245,247,.95)` veil) or in the masked-out
centre.

## 6. Glow and light

- **Blobs:** two or three soft circles behind the hero: pink `rgba(230,10,207,.22–.30)`, blue `rgba(30,155,215,.18–.22)`
  and violet `rgba(184,60,240,.16–.20)`, with `filter: blur(70–80px)` (14 vh on the wallpapers).
- **Edge glow:** a running agent, a focused input and the demo window all get a rotating conic-gradient halo:
  `conic-gradient(from var(--a), var(--c1), var(--c2), var(--c4), var(--c3), var(--c1))`, blur 8–24 px, opacity
  0.5–0.8.
- **Cards:** white, radius 18–28 px, shadow `0 12px 32px rgba(29,29,31,.08), 0 1px 3px rgba(29,29,31,.05)`.
  Hero cards also get a pink ring, `0 0 0 8px rgba(230,10,207,.28)`, and a pink glow.

## 7. Components

- **Chat:**
  - **User bubble:** `linear-gradient(135deg, g1, g3)`, white text, radius `18 18 6 18`.
  - **Agent card:** white, radius `18 18 18 6`, glowing edge while it's working.
  - **Plan list:** numbered dots; the current step's dot gets the gradient, and done steps get a green ✓ and are
    struck through.
  - **"Actions" row:** a collapsed pill with a count.
- **Chips:** white pills, height about 66 px in video and 28 px in the UI, a 12 px gradient dot, and a bold value
  ("type **0.2 s**").
- **Badges:** small pills: Done `--ok` on a 14 % tint, Error/Blocked `--bad` on a 12 % tint, everything else `--light`
  on ground.
- **Composer:** a white rounded input (radius 22) with a gradient send button (circle, `135deg g1 → g3`); it gets the
  edge glow on focus.
- **Window chrome** (demos, mockups): a light bar `#F5F5F7` with traffic lights `#FF5F57` `#FEBC2E` `#28C840`, radius 28.
- **Cursor:** the agent's cursor is a glassy arrow in the theme colours with a soft halo. A trail of six dots follows
  it while it moves, and a ring ripples out on each click. In videos the user's own pointer is a plain white arrow with
  a dark outline.

## 8. Motion

- **Eases:** `expo.out` / `power3.out` for entrances, `power2.in` for exits, `back.out(1.6)` only for chips and pops.
  No bounce on headlines.
- **Entrances:** rise 30–40 px plus fade, 0.45–0.7 s. Hero numbers come in with a blur (16 px → 0) and a scale (0.7 → 1).
- **Exits:** every scene leaves in its last 0.35–0.45 s, moving up or zooming through (scale 1.08–1.12, blur
  10 px, fade). The next scene enters in the same direction.
- **Headline swaps:** the old line goes up and out in 0.22 s, and the new one comes up from below 0.12 s later.
- **Holds:** every headline sits still for at least 0.6 s.
- **Camera:** on footage, punch in to the action (1.7–2.2×) as the cursor arrives, then pull back out to show the result.
- **Sound:** a soft click on clicks, typing under typing, a chime on "Done", whooshes on cuts, and music with the beat
  drop on the first cut.
- **In the product:** respect `prefers-reduced-motion`. The cursor has Smooth and Fast speeds.

## 9. Formats

| Piece | Size | Where it lives |
|---|---|---|
| Ad | 1920×1080, 30 s, 4 themes | `videos/clicky-ad/` (local only) |
| Short | 1080×1920, 15.5 s | `videos/reflex-short/` (local only) |
| Thumbnail | 1280×720 at 1.5× = 1920×1080 | `videos/clicky-ad/thumbnail/` |
| Wallpapers | 7680 wide: 16:9, 16:10, 21:9, 32:9 | [`wallpapers/`](wallpapers/) |
| Extension | side panel, min 340 px wide | `extension/popup.html` |
| Desktop app | 1480×940 window, 52 px top bar | `desktop/shell.html` |

Vertical video: keep key content between y 200 and 1500 and away from the right edge, so the Shorts, Reels and
TikTok overlays don't cover it.

## 10. Don'ts

- No dark or black backgrounds. The original black-stage look was retired in favour of this light one.
- No full-frame "AI" gradients, bokeh, particles or grids. Colour lives in the capsules, the glows and one word.
- Don't stretch, recolour or add a gradient to the Atlas mark, and don't use low-res bitmaps of it: use the SVG.
- Don't put more than one gradient word per line, and never put gradient on body text.
- Don't use other fonts, or all-caps headlines.
