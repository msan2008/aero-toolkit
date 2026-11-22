// Wing Lab scene: world drawing, day/night cycle and weather.
//
// Everything here is scenery. Weather and time of day are NOT inputs to the
// model and never change predictions, distances or points.
//
// Drawing is plain Canvas 2D. Static art (island, mainland, sea texture, cloud
// sprites) is pre-rendered to off-screen canvases on resize; the world is drawn
// to an off-screen "world" canvas each frame, tinted for time of day/weather,
// then composited over the sky. Lights (windows, lighthouse, campfire, moon
// glitter), rain, fog and lightning are drawn on top, untinted.

/* global window, document */
(function () {
  // ---------------------------------------------------------------- helpers
  const hex = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
  const mixc = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
  const rgb = (c, a = 1) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const sstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
  function rng(seed) {
    return () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }

  // ---------------------------------------------------------------- world geometry
  const X0 = -150, X1 = 1180, YMAX = 240;
  const LAUNCH_H = 22, LAND_H = 14, DOCK_M = 955, MAINLAND_M = 1000;
  let W = 800, H = 460, DPR = 1, HZ = 300, S = 1;            // S = art scale
  const sx = (m) => (m - X0) / (X1 - X0) * W;
  const sy = (m) => HZ - m / YMAX * (HZ - 46);

  const cv = document.getElementById("c"), ctx = cv.getContext("2d");
  const wc = document.createElement("canvas"), wx = wc.getContext("2d");
  const L = {};                                              // pre-rendered layers
  const geo = {};                                            // island / mainland anchors (px)

  function canvasLayer(w, h, draw) {
    const c = document.createElement("canvas");
    c.width = Math.ceil(w * DPR); c.height = Math.ceil(h * DPR);
    const g = c.getContext("2d"); g.setTransform(DPR, 0, 0, DPR, 0, 0); draw(g); return c;
  }

  // ---------------------------------------------------------------- time of day
  const CYCLE_MS = 6 * 60 * 1000;              // one full day = 6 minutes
  const START_PHASE = 0.29;                    // ~07:00 when a student first opens the game
  let cycleStart = Date.now();
  try {                                        // keep the same clock across Streamlit reruns
    const saved = +sessionStorage.getItem("wl_cycle_start");
    if (saved) cycleStart = saved; else sessionStorage.setItem("wl_cycle_start", String(cycleStart));
  } catch (e) { /* storage blocked: clock restarts with the page */ }
  let phaseOverride = null;                    // only used by automated screenshot tests
  const dayPhase = () => phaseOverride !== null ? phaseOverride : (START_PHASE + (Date.now() - cycleStart) / CYCLE_MS) % 1;

  // ---------------------------------------------------------------- weather
  const WEATHER = {
    sunny:    { icon: "☀️", night: "🌙", name: "Sunny", nightName: "Clear", cloud: .12, rain: 0, storm: 0, fog: 0, wind: .2, w: .30 },
    partly:   { icon: "⛅", night: "☁️", name: "Partly cloudy", cloud: .42, rain: 0, storm: 0, fog: 0, wind: .35, w: .25 },
    overcast: { icon: "☁️", name: "Overcast", cloud: .82, rain: 0, storm: 0, fog: .1, wind: .45, w: .14 },
    rain:     { icon: "🌧️", name: "Rain", cloud: .92, rain: .55, storm: 0, fog: .15, wind: .55, w: .12 },
    storm:    { icon: "⛈️", name: "Thunderstorm", cloud: 1, rain: 1, storm: 1, fog: .1, wind: 1, w: .08 },
    fog:      { icon: "🌫️", name: "Fog", cloud: .5, rain: 0, storm: 0, fog: .85, wind: .1, w: .11 },
  };
  const KEYS = ["cloud", "rain", "storm", "fog", "wind"];
  const wx_now = Object.assign({}, WEATHER.sunny);
  let wxTarget = "sunny", wxNext = performance.now() + 60000, forced = null;
  function pickWeather() {
    const options = Object.keys(WEATHER).filter((k) => k !== wxTarget);
    let r = Math.random() * options.reduce((s, k) => s + WEATHER[k].w, 0);
    for (const k of options) { r -= WEATHER[k].w; if (r <= 0) return k; }
    return options[0];
  }
  function setWeather(name, instant) {
    wxTarget = name; wxNext = performance.now() + (45 + Math.random() * 45) * 1000;
    if (instant) KEYS.forEach((k) => { wx_now[k] = WEATHER[name][k]; });
  }

  // ---------------------------------------------------------------- environment state per frame
  const env = { phase: .3, sunElev: 1, dark: 0, sun: [0, 0], moon: [0, 0], moonElev: -1 };
  let lightning = null, onLightning = null;

  function update(dt, now) {
    if (forced) { if (wxTarget !== forced) setWeather(forced, true); }
    else if (now > wxNext) setWeather(pickWeather(), false);
    const target = WEATHER[wxTarget], k = 1 - Math.exp(-dt / 6);   // ~6 s blend
    KEYS.forEach((key) => { wx_now[key] += (target[key] - wx_now[key]) * k; });

    env.phase = dayPhase();
    env.sunElev = Math.sin(2 * Math.PI * (env.phase - 0.25));
    env.moonElev = -env.sunElev;
    env.dark = 1 - sstep(-0.22, 0.1, env.sunElev);
    const dayProg = ((env.phase - 0.25) % 1 + 1) % 1 / 0.5;          // 0..1 while the sun is up
    env.sun = [lerp(-0.05 * W, 1.05 * W, dayProg), HZ - env.sunElev * (HZ - 60)];
    const nightProg = ((env.phase - 0.75) % 1 + 1) % 1 / 0.5;
    env.moon = [lerp(-0.05 * W, 1.05 * W, nightProg), HZ - env.moonElev * (HZ - 70)];

    if (lightning && now - lightning.t0 > 900) lightning = null;
    if (!lightning && wx_now.storm > 0.6 && Math.random() < dt * 0.14) {
      const bx = W * (0.25 + Math.random() * 0.7), pts = [[bx, 30]];
      let y = 30, x = bx;
      while (y < HZ) { y += 14 + Math.random() * 22; x += (Math.random() - .5) * 34; pts.push([x, Math.min(y, HZ)]); }
      lightning = { t0: now, pts };
      if (onLightning) onLightning();
    }
  }

  function mood() {                              // used by the music: morning / day / night
    if (env.dark > 0.45) return "night";
    return env.phase < 0.42 ? "morning" : "day";
  }

  // where the light comes from (sun by day, moon by night) and how strong it is
  function light() {
    if (env.sunElev > 0) return { x: env.sun[0], k: Math.min(1, env.sunElev * 3) * (1 - wx_now.cloud * .75), warm: true };
    if (env.moonElev > 0) return { x: env.moon[0], k: Math.min(1, env.moonElev * 3) * .45 * (1 - wx_now.cloud * .8), warm: false };
    return { x: W / 2, k: 0, warm: false };
  }
  function shadow(g, x, y, w, h) {              // soft shadow cast away from the light
    const L = light(); if (L.k < .03) return;
    const dx = clamp((x - L.x) / W, -1, 1) * 26 * Math.max(.5, S);
    g.fillStyle = `rgba(30,20,10,${.28 * L.k})`;
    g.beginPath(); g.ellipse(x + dx, y, w + Math.abs(dx) * .6, h, 0, 0, 7); g.fill();
  }

  function clockLabel() {
    const mins = Math.floor(env.phase * 24 * 60), h = Math.floor(mins / 60), m = mins % 60;
    const w = WEATHER[wxTarget], night = env.dark > 0.5;
    const icon = night && w.night ? w.night : w.icon, name = night && w.nightName ? w.nightName : w.name;
    return `🕐 ${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")} · ${icon} ${name}`;
  }

  // ---------------------------------------------------------------- pre-rendered layers
  function buildLayers() {
    const R = rng(11);
    // sea texture (tileable horizontally), wave crests get smaller toward the horizon
    const one = canvasLayer(W, H, (g) => {
      for (let i = 0; i < 1300; i++) {
        const t = R() ** 1.4, y = lerp(HZ + 3, H, t), len = lerp(4, 44, t) * Math.max(.5, S), x0 = R() * W;
        for (const x of [x0, x0 - W]) {
          g.strokeStyle = `rgba(255,255,255,${lerp(.16, .36, t) * R()})`; g.lineWidth = lerp(.5, 2, t);
          g.beginPath(); g.moveTo(x, y); g.quadraticCurveTo(x + len / 2, y - lerp(.5, 3, t), x + len, y); g.stroke();
          g.strokeStyle = `rgba(5,40,70,${lerp(.05, .16, t)})`;
          g.beginPath(); g.moveTo(x, y + lerp(1, 4, t)); g.lineTo(x + len, y + lerp(1, 4, t)); g.stroke();
        }
      }
    });
    L.sea = one;

    // cloud sprites: white puffs and a darker storm version of each
    L.clouds = [];
    for (let i = 0; i < 6; i++) {
      const cw = (170 + R() * 170) * Math.max(.55, S), ch = cw * (0.22 + R() * 0.08), n = 26 + (R() * 16 | 0);
      const puffs = [];
      for (let j = 0; j < n; j++) {
        const px = cw / 2 + (R() - .5) * cw * .85, edge = 1 - Math.abs(px - cw / 2) / (cw / 2);
        puffs.push([px, ch * .55 + (R() - .5) * ch * .8 * edge, ch * (0.35 + R() * 0.45) * (0.6 + edge * .5)]);
      }
      const pad = ch * .9;                     // room so puffs are never clipped into boxes
      const draw = (light, mid, base) => canvasLayer(cw + pad * 2, ch * 1.6 + pad * 2, (g) => {
        g.translate(pad, pad);
        puffs.forEach(([px, py, r]) => {
          const gg = g.createRadialGradient(px - r * .3, py - r * .4, r * .1, px, py, r);
          gg.addColorStop(0, rgb(light, .95)); gg.addColorStop(.6, rgb(mid, .8)); gg.addColorStop(1, rgb(mid, 0));
          g.fillStyle = gg; g.beginPath(); g.arc(px, py, r, 0, 7); g.fill();
        });
        const gb = g.createLinearGradient(0, ch * .5, 0, ch * 1.1);
        gb.addColorStop(0, rgb(base, 0)); gb.addColorStop(1, rgb(base, .35));
        g.fillStyle = gb; g.beginPath(); g.ellipse(cw / 2, ch * .8, cw * .42, ch * .32, 0, 0, 7); g.fill();
      });
      L.clouds.push({ w: cw + pad * 2, h: ch * 1.6 + pad * 2, white: draw([255, 255, 255], [238, 244, 250], [170, 188, 208]),
                      dark: draw([132, 142, 156], [92, 102, 116], [52, 60, 72]) });
    }
    // cloud slots: fixed sky positions, more of them appear as cloud cover rises
    L.slots = Array.from({ length: 18 }, (_, i) => ({
      x: R(), y: 0.06 + R() * 0.42, sprite: i % 6, speed: 0.5 + R(), order: R(),
    })).sort((a, b) => a.order - b.order);
    L.stars = Array.from({ length: 140 }, () => [R(), R() ** 1.3 * 0.9, 0.5 + R() * 1.3, R() * 6]);

    // ---- island (left). Link launches from the ramp at x = 0 m.
    const rx = Math.max(64, 150 * S), cx = sx(0) - rx * 0.35, base = HZ + rx * 0.06, ry = rx * 0.3;
    Object.assign(geo, { cx, rx, ry, base });
    geo.islandY = (x) => base - ry * Math.sqrt(Math.max(0, 1 - ((x - cx) / rx) ** 2));
    geo.linkX = sx(0) - 26 * S;
    geo.palms = [[cx - rx * .45, 118 * S, .26], [cx - rx * .2, 96 * S, -.2]];
    geo.fire = [cx + rx * .08, geo.islandY(cx + rx * .08)];
    geo.boatIsland = cx + rx + 36 * S;
    L.island = canvasLayer(W, H, (g) => {
      let gr = g.createRadialGradient(cx, base, rx * .3, cx, base, rx * 1.45);
      gr.addColorStop(0, "rgba(80,240,220,.85)"); gr.addColorStop(1, "rgba(80,240,220,0)");
      g.fillStyle = gr; g.beginPath(); g.ellipse(cx, base + 6 * S, rx * 1.35, ry * .9, 0, 0, 7); g.fill();
      g.strokeStyle = "rgba(255,255,255,.85)"; g.lineWidth = 2.5;
      g.beginPath(); g.ellipse(cx, base + 1, rx * 1.02, ry * .35, 0, .02 * Math.PI, .98 * Math.PI); g.stroke();
      g.strokeStyle = "rgba(255,255,255,.4)"; g.lineWidth = 1.6;
      g.beginPath(); g.ellipse(cx, base + 5 * S, rx * 1.1, ry * .55, 0, .05 * Math.PI, .95 * Math.PI); g.stroke();
      gr = g.createLinearGradient(0, base - ry, 0, base + 6);
      gr.addColorStop(0, "#fff0bf"); gr.addColorStop(.7, "#f5cf7f"); gr.addColorStop(1, "#d6a656");
      g.fillStyle = gr; g.beginPath(); g.ellipse(cx, base, rx, ry, 0, Math.PI, 0); g.fill();
      g.fillStyle = "#b9955a"; g.beginPath(); g.ellipse(cx, base, rx, 4 * S + 2, 0, 0, Math.PI); g.fill();
      [[cx - rx * .85, 20, 12], [cx + rx * .78, 24, 14], [cx + rx * .9, 13, 8]].forEach(([x, w, h]) => {
        const rg = g.createLinearGradient(x, base - h * S, x, base); rg.addColorStop(0, "#8d939a"); rg.addColorStop(1, "#4b5157");
        g.fillStyle = rg; g.beginPath(); g.ellipse(x, base, w * S, h * S, 0, Math.PI, 0); g.fill();
      });
      // driftwood launch ramp ending at x = 0 m, launch height
      const x1 = sx(0), y1 = sy(LAUNCH_H), x0 = geo.linkX + 6 * S, y0 = geo.islandY(x0);
      g.strokeStyle = "#8b6b45"; g.lineWidth = 7 * S + 2; g.lineCap = "round";
      g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
      g.strokeStyle = "#6e5233"; g.lineWidth = 3 * S + 1;
      g.beginPath(); g.moveTo(x1 - 2, y1); g.lineTo(x1 - 2, geo.islandY(x1 - 2)); g.stroke();
      // supply crate
      const kx = cx - rx * .7, ky = geo.islandY(kx);
      g.fillStyle = "#9c7a4e"; g.fillRect(kx, ky - 15 * S, 24 * S, 15 * S);
      g.strokeStyle = "#6b5232"; g.lineWidth = 1.5; g.strokeRect(kx, ky - 15 * S, 24 * S, 15 * S);
      // campfire stones and logs
      const [fx, fy] = geo.fire;
      g.fillStyle = "#6b6f75"; for (let i = 0; i < 6; i++) { g.beginPath(); g.arc(fx - 10 * S + i * 4 * S, fy + 1, 2.6 * S + .6, 0, 7); g.fill(); }
      g.strokeStyle = "#5a3d22"; g.lineWidth = 3 * S + 1;
      g.beginPath(); g.moveTo(fx - 8 * S, fy); g.lineTo(fx + 6 * S, fy - 5 * S); g.moveTo(fx + 8 * S, fy); g.lineTo(fx - 6 * S, fy - 5 * S); g.stroke();
    });

    // ---- mainland (right): distant ridge, hills, beach, town, lighthouse, dock
    const beachX = sx(MAINLAND_M - 15), beachY = sy(LAND_H);
    geo.windows = []; geo.lamp = null; geo.dock = sx(DOCK_M);
    L.mainland = canvasLayer(W, H, (g) => {
      const ridge = (x0, amp, col, off) => {
        g.fillStyle = col; g.beginPath(); g.moveTo(x0, HZ);
        for (let x = x0; x <= W + 8; x += 6) {
          const k = (x - x0) / (W - x0);
          g.lineTo(x, HZ - Math.sin(Math.min(1, k * 1.3) * Math.PI * .5) * amp * (0.75 + .25 * Math.sin(x * .013 + off) + .1 * Math.sin(x * .05)));
        }
        g.lineTo(W, HZ); g.fill();
      };
      ridge(sx(860), 70 * S, "rgba(122,152,176,.55)", 1);
      ridge(sx(930), 50 * S, "rgba(96,132,150,.75)", 3);
      const hill = (pts, top, bottom) => {
        const gg = g.createLinearGradient(0, Math.min(...pts.map((p) => p[1])), 0, HZ + 30);
        gg.addColorStop(0, top); gg.addColorStop(1, bottom); g.fillStyle = gg;
        g.beginPath(); g.moveTo(pts[0][0], HZ + 30);
        pts.forEach(([x, y], i) => i ? g.quadraticCurveTo((pts[i - 1][0] + x) / 2, Math.min(pts[i - 1][1], y) - 12 * S, x, y) : g.lineTo(x, y));
        g.lineTo(W, HZ + 30); g.fill();
      };
      hill([[beachX + 30 * S, beachY], [sx(1070), HZ - 70 * S], [sx(1120), HZ - 88 * S], [W, HZ - 96 * S]], "#3fa34d", "#1f6b34");
      hill([[beachX + 10 * S, beachY + 2], [sx(1050), HZ - 40 * S], [sx(1110), HZ - 52 * S], [W, HZ - 58 * S]], "#7fd45a", "#2f8a3c");
      g.fillStyle = "#e9d39c"; g.beginPath(); g.moveTo(beachX - 20 * S, HZ + 3); g.quadraticCurveTo(beachX, beachY, beachX + 40 * S, beachY);
      g.lineTo(W, beachY - 2); g.lineTo(W, HZ + 30); g.lineTo(beachX - 20 * S, HZ + 30); g.fill();
      g.strokeStyle = "rgba(255,255,255,.8)"; g.lineWidth = 2;
      g.beginPath(); g.moveTo(beachX - 26 * S, HZ + 5); g.quadraticCurveTo(beachX, beachY + 10, beachX + 50 * S, beachY + 9); g.lineTo(W, beachY + 7); g.stroke();
      // town
      const houses = [[sx(1080), HZ - 46 * S, 24, 20, "#f4efe6", "#b5523b"], [sx(1102), HZ - 50 * S, 18, 24, "#eadfca", "#8e3b2e"],
                      [sx(1125), HZ - 46 * S, 28, 19, "#f7f3ea", "#a64836"], [sx(1155), HZ - 54 * S, 20, 27, "#e3d6bd", "#7d3427"]];
      houses.forEach(([hx, hy, w, h, wall, roof]) => {
        w *= S; h *= S;
        g.fillStyle = "rgba(0,0,0,.18)"; g.fillRect(hx + 3, hy - h + 3, w, h);
        g.fillStyle = wall; g.fillRect(hx, hy - h, w, h);
        g.fillStyle = roof; g.beginPath(); g.moveTo(hx - 3 * S, hy - h); g.lineTo(hx + w / 2, hy - h - 11 * S); g.lineTo(hx + w + 3 * S, hy - h); g.fill();
        g.fillStyle = "#5b7e9a"; const ww = 5 * S;
        [[hx + 4 * S, hy - h + 5 * S], [hx + w - 9 * S, hy - h + 5 * S]].forEach(([wx2, wy]) => { g.fillRect(wx2, wy, ww, ww); geo.windows.push([wx2 + ww / 2, wy + ww / 2, ww]); });
      });
      // lighthouse
      const lx = sx(1022), ly = beachY, lh = 64 * S;
      g.fillStyle = "#f5f5f5"; g.beginPath(); g.moveTo(lx - 8 * S, ly); g.lineTo(lx - 5 * S, ly - lh); g.lineTo(lx + 5 * S, ly - lh); g.lineTo(lx + 8 * S, ly); g.fill();
      g.fillStyle = "#c0392b"; g.fillRect(lx - 7.3 * S, ly - lh * .35, 14.6 * S, 8 * S); g.fillRect(lx - 6.3 * S, ly - lh * .7, 12.6 * S, 8 * S);
      g.fillStyle = "#2c3e50"; g.fillRect(lx - 7 * S, ly - lh - 6 * S, 14 * S, 6 * S);
      g.fillStyle = "#d9c27a"; g.fillRect(lx - 4.5 * S, ly - lh - 13 * S, 9 * S, 7 * S);
      g.fillStyle = "#c0392b"; g.beginPath(); g.moveTo(lx - 7 * S, ly - lh - 13 * S); g.lineTo(lx, ly - lh - 20 * S); g.lineTo(lx + 7 * S, ly - lh - 13 * S); g.fill();
      geo.lamp = [lx, ly - lh - 9.5 * S];
      // dock
      g.fillStyle = "#7a5a3a"; g.fillRect(geo.dock - 4 * S, HZ - 5 * S, beachX - geo.dock + 8 * S, 4 * S);
      for (let x = geo.dock; x < beachX; x += 14 * S) g.fillRect(x, HZ - 5 * S, 2.5 * S, 12 * S);
      // finish sign
      g.fillStyle = "rgba(10,25,40,.72)"; g.beginPath(); g.roundRect(beachX + 8 * S, beachY - 40 * S, 74 * S, 16 * S, 4); g.fill();
      g.fillStyle = "#ffd166"; g.font = `bold ${Math.max(8, 10 * S)}px system-ui`; g.textAlign = "center";
      g.fillText("🏁 MAINLAND", beachX + 45 * S, beachY - 28.5 * S);
    });
  }

  function resize() {
    DPR = window.devicePixelRatio || 1;
    W = cv.clientWidth || 800; H = W < 640 ? 400 : 460; HZ = Math.round(H * 0.64);
    S = clamp(W / 1400, 0.42, 1.15);
    cv.style.height = H + "px";
    for (const c of [cv, wc]) { c.width = W * DPR; c.height = H * DPR; }
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0); wx.setTransform(DPR, 0, 0, DPR, 0, 0);
    buildLayers();
  }

  // ---------------------------------------------------------------- dynamic art
  const SKY = {
    day: ["#1565d8", "#3fa9f5", "#ffe7bd"].map(hex), dusk: ["#2e3f8f", "#d9607a", "#ffa94d"].map(hex),
    night: ["#050a1c", "#0c1734", "#1e3156"].map(hex), storm: ["#3d4752", "#58636f", "#7d8a96"].map(hex),
  };
  function drawSky(g, t) {
    const e = env.sunElev;
    let cols = SKY.night.map((c, i) => mixc(c, SKY.dusk[i], sstep(-0.25, 0.02, e)));
    cols = cols.map((c, i) => mixc(c, SKY.day[i], sstep(0.02, 0.35, e)));
    const stormCol = SKY.storm.map((c) => mixc(c, [14, 18, 26], env.dark * .85));
    cols = cols.map((c, i) => mixc(c, stormCol[i], wx_now.cloud * .78));
    const gr = g.createLinearGradient(0, 0, 0, HZ);
    gr.addColorStop(0, rgb(cols[0])); gr.addColorStop(.55, rgb(cols[1])); gr.addColorStop(1, rgb(cols[2]));
    g.fillStyle = gr; g.fillRect(0, 0, W, HZ);
    // stars
    const starA = env.dark * (1 - wx_now.cloud) * (1 - wx_now.fog);
    if (starA > 0.02) L.stars.forEach(([x, y, r, ph]) => {
      g.fillStyle = `rgba(255,255,240,${starA * (0.55 + 0.45 * Math.sin(t * 2 + ph))})`;
      g.beginPath(); g.arc(x * W, y * HZ, r * Math.max(.6, S), 0, 7); g.fill();
    });
    const clear = 1 - wx_now.cloud * .85;
    // moon
    if (env.moonElev > -0.1) {
      const [mx, my] = env.moon, r = 13 * Math.max(.6, S);
      const gg = g.createRadialGradient(mx, my, 0, mx, my, r * 7);
      gg.addColorStop(0, `rgba(220,230,255,${.35 * clear})`); gg.addColorStop(1, "rgba(220,230,255,0)");
      g.fillStyle = gg; g.beginPath(); g.arc(mx, my, r * 7, 0, 7); g.fill();
      g.fillStyle = `rgba(240,240,228,${.4 + .6 * clear})`; g.beginPath(); g.arc(mx, my, r, 0, 7); g.fill();
      g.fillStyle = `rgba(190,192,185,${.5 * clear})`;
      [[-.3, -.2, .25], [.25, .15, .18], [-.05, .35, .12]].forEach(([dx, dy, rr]) => { g.beginPath(); g.arc(mx + dx * r, my + dy * r, rr * r, 0, 7); g.fill(); });
    }
    // sun
    if (env.sunElev > -0.12) {
      const [sx2, sy2] = env.sun, warm = 1 - sstep(0, 0.35, env.sunElev);
      const core = mixc([255, 250, 225], [255, 176, 92], warm);
      const gg = g.createRadialGradient(sx2, sy2, 0, sx2, sy2, 260 * Math.max(.5, S));
      gg.addColorStop(0, rgb(core, clear)); gg.addColorStop(.08, rgb(core, .95 * clear));
      gg.addColorStop(.25, rgb(core, .35 * clear)); gg.addColorStop(1, rgb(core, 0));
      g.fillStyle = gg; g.fillRect(0, 0, W, HZ);
    }
  }

  function drawClouds(g, t) {
    const n = Math.round(3 + wx_now.cloud * 15), drift = 4 + wx_now.wind * 38;
    for (let i = 0; i < n && i < L.slots.length; i++) {
      const s = L.slots[i], c = L.clouds[s.sprite], span = W + c.w * 2;
      const x = ((s.x * span + t * drift * s.speed) % span) - c.w;
      const y = s.y * HZ * (1 - wx_now.cloud * .35);
      g.globalAlpha = .55 + .35 * wx_now.cloud; g.drawImage(c.white, x, y, c.w, c.h);
      g.globalAlpha = clamp((wx_now.cloud - .35) * 1.2 + wx_now.storm * .4, 0, 1); g.drawImage(c.dark, x, y, c.w, c.h);
    }
    g.globalAlpha = 1;
  }

  function drawSea(g, t) {
    const gr = g.createLinearGradient(0, HZ, 0, H);
    const calm = [["#8ef0ee", "#1fc0d8", "#0a84c6", "#05468f"], ["#8aa4ad", "#4f7486", "#2c4e63", "#1a3446"]];
    const st = wx_now.cloud * .8;
    [0, .15, .55, 1].forEach((p, i) => gr.addColorStop(p, rgb(mixc(hex(calm[0][i]), hex(calm[1][i]), st))));
    g.fillStyle = gr; g.fillRect(0, HZ, W, H - HZ);
    const off = (t * (8 + wx_now.wind * 40)) % W;
    g.globalAlpha = .75 + wx_now.wind * .25;
    g.drawImage(L.sea, -off, 0, W, H); g.drawImage(L.sea, W - off, 0, W, H);
    g.globalAlpha = 1;
    // sun glitter
    const glit = sstep(-0.05, 0.25, env.sunElev) * (1 - wx_now.cloud * .9);
    if (glit > 0.03) drawGlitter(g, env.sun[0], [255, 248, 220], glit, 170);
  }
  function drawGlitter(g, cx, col, a, n) {
    for (let i = 0; i < n; i++) {
      const t = Math.random(), y = lerp(HZ + 2, H, t ** 1.6), k = (y - HZ) / (H - HZ);
      const spread = lerp(30, 230, k) * Math.max(.5, S), x = cx + (Math.random() - .5) * 2 * spread * (0.4 + Math.random());
      const len = lerp(2, 13, k);
      g.strokeStyle = rgb(col, a * (0.25 + Math.random() * .55)); g.lineWidth = lerp(.6, 1.8, k);
      g.beginPath(); g.moveTo(x - len / 2, y); g.lineTo(x + len / 2, y); g.stroke();
    }
  }

  function drawPalm(g, bx, h, lean, t) {
    shadow(g, bx + 10 * S, geo.islandY(bx) + 2, 34 * S, 5 * S);
    const by = geo.islandY(bx), sway = Math.sin(t * (0.8 + wx_now.wind * 2.5)) * (0.02 + wx_now.wind * .06);
    const segs = 11, top = [bx + (lean + sway) * h, by - h];
    for (let i = 0; i < segs; i++) {
      const t0 = i / segs, t1 = (i + 1) / segs, p = (u) => [bx + (lean + sway * u) * u * u * h, by - u * h];
      const [x0, y0] = p(t0), [x1, y1] = p(t1), w = lerp(8, 4.5, t0) * Math.max(.6, S);
      g.fillStyle = i % 2 ? "#8a6236" : "#7a5430";
      g.beginPath(); g.moveTo(x0 - w, y0); g.lineTo(x1 - w * .9, y1); g.lineTo(x1 + w * .9, y1); g.lineTo(x0 + w, y0); g.fill();
    }
    g.fillStyle = "#5a3d22"; [[-4, 5], [4, 7], [0, 10]].forEach(([dx, dy]) => { g.beginPath(); g.arc(top[0] + dx * S, top[1] + dy * S, 4.5 * S + 1, 0, 7); g.fill(); });
    [-2.9, -2.45, -1.95, -1.25, -.65, -.2, .25].forEach((a0, k) => {
      const a = a0 + sway * 3 * (k % 2 ? 1 : -.6), len = h * (0.5 + (k % 3) * .06);
      const ex = top[0] + Math.cos(a) * len, ey = top[1] + Math.sin(a) * len * .55 + len * .28;
      const mx = top[0] + Math.cos(a) * len * .5, my = top[1] + Math.sin(a) * len * .5 - 10 * S;
      const lg = g.createLinearGradient(top[0], top[1], ex, ey); lg.addColorStop(0, "#2e7d32"); lg.addColorStop(1, "#7cc56a");
      g.strokeStyle = lg; g.lineWidth = 2.5 * S + .5; g.beginPath(); g.moveTo(...top); g.quadraticCurveTo(mx, my, ex, ey); g.stroke();
      for (let j = 1; j < 10; j++) {
        const u = j / 10, qx = (1 - u) ** 2 * top[0] + 2 * (1 - u) * u * mx + u * u * ex, qy = (1 - u) ** 2 * top[1] + 2 * (1 - u) * u * my + u * u * ey;
        const ll = lerp(20, 7, u) * S;
        [1, -1].forEach((side) => {
          const tx = qx + Math.cos(a + side * 1.3) * ll * .5, ty = qy + ll * .9;
          g.fillStyle = side > 0 ? "#3d8f43" : "#5aae55";
          g.beginPath(); g.moveTo(qx, qy); g.quadraticCurveTo(tx + side * 4 * S, qy + ll * .3, tx, ty);
          g.quadraticCurveTo(tx - side * 2 * S, qy + ll * .4, qx, qy); g.fill();
        });
      }
    });
  }

  // Link: castaway in a green hoodie (hood up), tan shorts, waving
  function drawLink(g, px, py, t, mood, s0) {
    if (mood === true) mood = "wave"; if (!mood) mood = "idle";
    const s = (s0 || 1) * Math.max(.55, S) * 1.05;
    shadow(g, px, py, 11 * s, 2.6 * s);
    const hop = mood === "excited" ? -Math.abs(Math.sin(t * 9)) * 8 : 0, slump = mood === "sad" ? 2.5 : 0;
    g.save(); g.translate(px, py + (hop + slump) * s); g.scale(s, s);
    g.fillStyle = "#c68e5e"; g.beginPath(); g.roundRect(-6, -17, 5, 17, 2); g.fill(); g.beginPath(); g.roundRect(1, -17, 5, 17, 2); g.fill();
    g.fillStyle = "#c8a46a"; g.beginPath(); g.roundRect(-7, -25, 14, 10, 3); g.fill();                       // shorts
    const sg = g.createLinearGradient(-8, -46, 8, -22); sg.addColorStop(0, "#43a047"); sg.addColorStop(1, "#1b5e20");
    g.fillStyle = sg; g.beginPath(); g.roundRect(-8.5, -45, 17, 22, 5); g.fill();                              // hoodie body
    g.fillStyle = "#2e7d32"; g.fillRect(-5, -31, 10, 5);                                                       // front pocket
    g.strokeStyle = "#2e7d32"; g.lineWidth = 4.2; g.lineCap = "round";
    const arm = (sx0, ang) => {                                                                                 // sleeve + hand
      const ex = sx0 + Math.cos(ang) * 11, ey = -41 + Math.sin(ang) * 11;
      g.beginPath(); g.moveTo(sx0, -41); g.lineTo(ex, ey); g.stroke();
      g.fillStyle = "#d9a06e"; g.beginPath(); g.arc(sx0 + Math.cos(ang) * 12.5, -41 + Math.sin(ang) * 12.5, 2.4, 0, 7); g.fill();
    };
    if (mood === "excited") { const w = Math.sin(t * 14) * .25; arm(-7, -2.0 + w); arm(7, -1.15 - w); }      // both arms up, cheering
    else if (mood === "sad") { arm(-6, 1.75); arm(6, 1.4); }                                                    // arms hanging
    else if (mood === "wave") { arm(-7, 2.0); arm(7, -1.25 + Math.sin(t * 7) * .45); }
    else { arm(-7, 2.0); arm(7, 1.1); }
    g.save();
    if (mood === "sad") { g.translate(0, -44); g.rotate(.32); g.translate(0, 47); }                             // head hangs down
    const hg = g.createRadialGradient(-2, -58, 2, 0, -53, 12); hg.addColorStop(0, "#5cc760"); hg.addColorStop(1, "#1b5e20");
    g.fillStyle = hg; g.beginPath(); g.ellipse(0, -53, 10.5, 11.5, 0, 0, 7); g.fill();                         // rounded hood
    g.fillStyle = "#e3aa76"; g.beginPath(); g.ellipse(1.5, -52, 6.5, 7.5, 0, 0, 7); g.fill();                  // face
    g.fillStyle = "#3b2a1a"; g.beginPath(); g.ellipse(1.5, -58, 6.5, 2.6, 0, 0, 7); g.fill();                   // fringe
    g.strokeStyle = "#222"; g.fillStyle = "#222"; g.lineWidth = 1.1;
    if (mood === "sad") {
      g.beginPath(); g.moveTo(3, -52.5); g.lineTo(6, -51.5); g.stroke();                                       // closed eye
      g.beginPath(); g.arc(4.5, -46.5, 2.2, 1.15 * Math.PI, 1.85 * Math.PI); g.stroke();                       // frown
      g.fillStyle = "rgba(120,190,255,.9)"; g.beginPath(); g.ellipse(6.2, -49 + (t * 6 % 4), 1, 1.6, 0, 0, 7); g.fill(); // tear
    } else {
      g.beginPath(); g.arc(4.5, -52, 1.1, 0, 7); g.fill();
      if (mood === "excited") { g.fillStyle = "#7a2b2b"; g.beginPath(); g.ellipse(4.5, -47.5, 2, 1.6, 0, 0, 7); g.fill(); }
      else { g.beginPath(); g.arc(4.2, -49, 2, .15 * Math.PI, .85 * Math.PI); g.stroke(); }
    }
    g.restore();
    g.restore();
  }

  function drawBubble(g, x, y, text) {
    const s = Math.max(.7, S);
    g.font = `bold ${Math.round(13 * s)}px system-ui`; g.textAlign = "center";
    const w = g.measureText(text).width + 18 * s, h = 24 * s, bx = clamp(x, w / 2 + 4, W - w / 2 - 4);
    g.fillStyle = "rgba(255,255,255,.92)"; g.strokeStyle = "rgba(20,40,60,.5)"; g.lineWidth = 1;
    g.beginPath(); g.roundRect(bx - w / 2, y - h, w, h, 10 * s); g.fill(); g.stroke();
    g.beginPath(); g.moveTo(x - 6 * s, y - 1); g.lineTo(x - 10 * s, y + 9 * s); g.lineTo(x + 3 * s, y - 1); g.fill();
    g.fillStyle = "#13324a"; g.fillText(text, bx, y - 7.5 * s);
  }

  function drawBottle(g, x, y, s0) {
    const s = (s0 || 1) * Math.max(.6, S);
    g.fillStyle = "rgba(110,200,160,.95)"; g.beginPath(); g.roundRect(x - 3 * s, y - 9 * s, 7 * s, 10 * s, 2); g.fill();
    g.fillRect(x - 1 * s, y - 13 * s, 3 * s, 4 * s); g.fillStyle = "#c9a36a"; g.fillRect(x - 1 * s, y - 14.5 * s, 3 * s, 2 * s);
    g.fillStyle = "rgba(255,255,255,.8)"; g.fillRect(x - 1.5 * s, y - 6 * s, 3.5 * s, 4 * s);
  }

  function drawGlider(g, x, y, ang, colour, dark) {
    const s = Math.max(.5, S) * 1.1;
    g.save(); g.translate(x, y); g.rotate(ang); g.scale(s, s);
    let fg = g.createLinearGradient(0, -6, 0, 6); fg.addColorStop(0, "#ffffff"); fg.addColorStop(1, dark);
    g.fillStyle = fg; g.beginPath(); g.moveTo(30, 0); g.quadraticCurveTo(26, -6, 10, -5); g.lineTo(-26, -2); g.lineTo(-26, 2);
    g.lineTo(10, 5); g.quadraticCurveTo(26, 6, 30, 0); g.fill();
    g.fillStyle = colour; g.beginPath(); g.moveTo(-18, -2); g.lineTo(-28, -16); g.lineTo(-31, -16); g.lineTo(-27, -1); g.fill();
    g.fillRect(-30, -1, 12, 3);
    const wg = g.createLinearGradient(0, -4, 0, 4); wg.addColorStop(0, "#fff"); wg.addColorStop(.4, colour); wg.addColorStop(1, dark);
    g.fillStyle = wg; g.beginPath(); g.ellipse(4, 1, 26, 4.5, -0.05, 0, 7); g.fill();
    g.fillStyle = "rgba(60,120,170,.85)"; g.beginPath(); g.ellipse(18, -4, 7, 3, -0.1, Math.PI, 0); g.fill();
    g.strokeStyle = "#5a4630"; g.lineWidth = 1; g.beginPath(); g.moveTo(4, 4); g.lineTo(4, 10); g.stroke();
    g.fillStyle = "rgba(110,200,160,.95)"; g.beginPath(); g.roundRect(-3, 10, 14, 6, 2); g.fill();
    g.fillStyle = "#c9a36a"; g.fillRect(11, 11.5, 4, 3);
    g.restore();
  }

  function drawBoat(g, bx, t, withLink) {
    const s = Math.max(.55, S) * .9, by = HZ + Math.sin(t * 2.4) * 1.5 * s;
    g.save(); g.translate(bx, by); g.scale(s, s);
    g.fillStyle = "rgba(0,30,50,.25)"; g.beginPath(); g.ellipse(0, 9, 40, 4, 0, 0, 7); g.fill();
    const hg = g.createLinearGradient(0, -6, 0, 9); hg.addColorStop(0, "#e74c3c"); hg.addColorStop(1, "#962d22");
    g.fillStyle = hg; g.beginPath(); g.moveTo(-36, -6); g.lineTo(38, -6); g.quadraticCurveTo(32, 6, 24, 9); g.lineTo(-28, 9); g.closePath(); g.fill();
    g.fillStyle = "#fff"; g.fillRect(-36, -8, 74, 3);
    g.fillStyle = "#f8f8f8"; g.fillRect(-12, -24, 26, 16); g.fillStyle = "#8fc8ec"; g.fillRect(-8, -20, 7, 6); g.fillRect(3, -20, 7, 6);
    g.fillStyle = "#e67e22"; g.beginPath(); g.arc(-24, -11, 4, 0, 7); g.fill();
    g.strokeStyle = "#333"; g.lineWidth = 1.5; g.beginPath(); g.moveTo(6, -24); g.lineTo(6, -42); g.stroke();
    g.fillStyle = "#f1c40f"; g.beginPath(); g.moveTo(6, -42); g.lineTo(18, -38); g.lineTo(6, -34); g.fill();
    g.restore();
    if (withLink) drawLink(g, bx - 24 * s, by - 8 * s, t, true, .8);
    return [bx + 6 * s, by - 42 * s];                                   // mast-top lamp position
  }

  function drawBuoys(g, t) {
    [200, 400, 600, 800].forEach((m, i) => {
      const bx = sx(m), by = HZ + 2 + Math.sin(t * 2 + i) * 1.4, s = Math.max(.55, S);
      g.fillStyle = "rgba(0,25,45,.25)"; g.beginPath(); g.ellipse(bx, by + 5 * s, 9 * s, 2.2 * s, 0, 0, 7); g.fill();
      g.fillStyle = "#f1c40f"; g.beginPath(); g.moveTo(bx - 6 * s, by + 3 * s); g.lineTo(bx - 3.5 * s, by - 9 * s); g.lineTo(bx + 3.5 * s, by - 9 * s); g.lineTo(bx + 6 * s, by + 3 * s); g.fill();
      g.fillStyle = "#e67e22"; g.fillRect(bx - 4.5 * s, by - 4 * s, 9 * s, 2.6 * s);
      g.strokeStyle = "#333"; g.lineWidth = 1; g.beginPath(); g.moveTo(bx, by - 9 * s); g.lineTo(bx, by - 20 * s); g.stroke();
      g.fillStyle = "rgba(10,25,40,.72)"; g.beginPath(); g.roundRect(bx - 18 * s, by - 35 * s, 36 * s, 14 * s, 4); g.fill();
      g.fillStyle = "#fff"; g.font = `bold ${Math.max(8, 9.5 * s)}px system-ui`; g.textAlign = "center"; g.fillText(m + " m", bx, by - 24.5 * s);
    });
  }

  // seagulls: three loose flocks drifting with the wind, flapping
  const FLOCKS = [{ x: .1, y: .2, n: 4, sp: 1 }, { x: .55, y: .12, n: 3, sp: .8 }, { x: .8, y: .3, n: 5, sp: 1.2 }];
  function drawBirds(g, t) {
    const vis = (1 - env.dark) * (1 - wx_now.storm * .8) * (1 - wx_now.fog * .6); if (vis < .05) return;
    const s = Math.max(.6, S);
    g.strokeStyle = `rgba(35,45,60,${.85 * vis})`; g.lineCap = "round"; g.lineJoin = "round";
    FLOCKS.forEach((f, fi) => {
      const span = W + 200, fx = ((f.x * span + t * (14 + wx_now.wind * 30) * f.sp) % span) - 100, fy = f.y * HZ + Math.sin(t * .5 + fi) * 10;
      for (let i = 0; i < f.n; i++) {
        const x = fx - i * 22 * s - (i % 2) * 8 * s, y = fy + i * 7 * s * (i % 2 ? 1 : -.6) + Math.sin(t * 1.3 + i) * 3;
        const flap = Math.sin(t * 7 + i * 1.7 + fi), w = 8 * s, up = flap * 4.5 * s;
        g.lineWidth = 1.8 * s;
        g.beginPath(); g.moveTo(x - w, y - up); g.quadraticCurveTo(x - w / 2, y - up * .2 - 3 * s, x, y);
        g.quadraticCurveTo(x + w / 2, y - up * .2 - 3 * s, x + w, y - up); g.stroke();
      }
    });
  }

  // humpback whales: surface, blow a spout, show the back, then the tail fluke as they dive
  let whale = null, nextWhale = performance.now() + 9000;
  function drawWhales(g, t, now) {
    if (!whale && now > nextWhale) {
      whale = { t0: now, x: sx(180 + Math.random() * 600), y: HZ + (H - HZ) * (.32 + Math.random() * .25), dir: Math.random() < .5 ? -1 : 1 };
    }
    if (!whale) return;
    const e = (now - whale.t0) / 1000, k = (whale.y - HZ) / (H - HZ), s = Math.max(.6, S) * (.8 + k * .8);
    const x = whale.x + whale.dir * e * 6 * s, y = whale.y;
    if (e > 10) { whale = null; nextWhale = now + 16000 + Math.random() * 16000; return; }
    const body = "#2b4a63", belly = "#5d7d96";
    g.save(); g.beginPath(); g.rect(0, 0, W, y); g.clip();                     // only the part above water shows
    if (e < 6) {                                                               // back and dorsal hump
      const rise = Math.sin(Math.min(1, e / 6) * Math.PI) * 14 * s;
      const bg = g.createLinearGradient(0, y - rise, 0, y); bg.addColorStop(0, belly); bg.addColorStop(1, body);
      g.fillStyle = bg; g.beginPath(); g.ellipse(x, y + 16 * s - rise, 62 * s, 18 * s, 0, 0, 7); g.fill();
      g.fillStyle = body; g.beginPath(); g.moveTo(x + whale.dir * 18 * s, y - rise + 1);
      g.lineTo(x + whale.dir * 28 * s, y - rise - 6 * s); g.lineTo(x + whale.dir * 34 * s, y - rise + 2); g.fill();
    }
    if (e > 5.5 && e < 9.5) {                                                  // tail fluke rises, then slips under
      const u = (e - 5.5) / 4, lift = Math.sin(u * Math.PI) * 34 * s, tx = x - whale.dir * 40 * s;
      g.fillStyle = body;
      g.beginPath(); g.moveTo(tx - 5 * s, y + 4); g.quadraticCurveTo(tx - 3 * s, y - lift * .6, tx - 24 * s, y - lift);
      g.quadraticCurveTo(tx, y - lift + 7 * s, tx + 24 * s, y - lift); g.quadraticCurveTo(tx + 3 * s, y - lift * .6, tx + 5 * s, y + 4); g.fill();
    }
    g.restore();
    if (e > .8 && e < 3.2) {                                                   // spout
      const u = (e - .8) / 2.4;
      for (let i = 0; i < 26; i++) {
        const a = -Math.PI / 2 + (i / 25 - .5) * .7, v = (40 + (i * 13 % 17)) * s;
        const px = x + whale.dir * 20 * s + Math.cos(a) * v * u, py = y - 8 * s + Math.sin(a) * v * u * 1.2 + 30 * s * u * u;
        g.fillStyle = `rgba(240,250,255,${.75 * (1 - u)})`; g.beginPath(); g.arc(px, py, (1.5 + u * 2.5) * s, 0, 7); g.fill();
      }
    }
    if (e < 9.8) {                                                             // foam ring at the waterline
      g.strokeStyle = `rgba(255,255,255,${.65 * Math.min(1, e) * Math.min(1, (9.8 - e))})`; g.lineWidth = 1.6;
      g.beginPath(); g.ellipse(x, y + 1, 60 * s, 4 * s, 0, 0, 7); g.stroke();
    }
  }

  function drawFireFlames(g, t) {
    const [fx, fy] = geo.fire, s = Math.max(.6, S);
    for (let i = 0; i < 3; i++) {
      const h = (9 + 4 * Math.sin(t * 9 + i * 2)) * s, x = fx + (i - 1) * 3 * s;
      const fg = g.createLinearGradient(x, fy, x, fy - h); fg.addColorStop(0, "#ff7b00"); fg.addColorStop(1, "rgba(255,220,90,.1)");
      g.fillStyle = fg; g.beginPath(); g.moveTo(x - 3 * s, fy - 2); g.quadraticCurveTo(x, fy - h * 1.2, x + 3 * s, fy - 2); g.fill();
    }
  }

  function drawLights(g, t, extra) {
    const d = env.dark, gloom = Math.max(d, wx_now.storm * .55, wx_now.cloud * .3);
    if (gloom > 0.15) {
      geo.windows.forEach(([x, y, w]) => {
        g.fillStyle = `rgba(255,214,120,${gloom})`; g.fillRect(x - w / 2, y - w / 2, w, w);
        const gg = g.createRadialGradient(x, y, 0, x, y, w * 2.5); gg.addColorStop(0, `rgba(255,200,110,${gloom * .4})`); gg.addColorStop(1, "rgba(255,200,110,0)");
        g.fillStyle = gg; g.beginPath(); g.arc(x, y, w * 2.5, 0, 7); g.fill();
      });
      if (geo.lamp) {
        const [lx, ly] = geo.lamp, sweep = Math.sin(t * .9), len = W * .45;
        const a1 = Math.PI + sweep * .35 - .06, a2 = Math.PI + sweep * .35 + .06;
        const bg = g.createRadialGradient(lx, ly, 0, lx, ly, len); bg.addColorStop(0, `rgba(255,248,200,${.45 * gloom})`); bg.addColorStop(1, "rgba(255,248,200,0)");
        g.fillStyle = bg; g.beginPath(); g.moveTo(lx, ly); g.lineTo(lx + Math.cos(a1) * len, ly + Math.sin(a1) * len);
        g.lineTo(lx + Math.cos(a2) * len, ly + Math.sin(a2) * len); g.fill();
        g.fillStyle = `rgba(255,240,170,${.6 + .4 * gloom})`; g.beginPath(); g.arc(lx, ly, 4 * S + 1, 0, 7); g.fill();
      }
      (extra || []).forEach(([x, y]) => { g.fillStyle = `rgba(255,230,140,${gloom})`; g.beginPath(); g.arc(x, y, 2.5 * S + 1, 0, 7); g.fill(); });
    }
    // campfire glow (always a little, strong at night)
    const [fx, fy] = geo.fire, fr = (40 + 6 * Math.sin(t * 11)) * Math.max(.6, S) * (1 + d);
    const fg = g.createRadialGradient(fx, fy - 6, 0, fx, fy - 6, fr);
    fg.addColorStop(0, `rgba(255,160,60,${.18 + .45 * d})`); fg.addColorStop(1, "rgba(255,160,60,0)");
    g.fillStyle = fg; g.beginPath(); g.arc(fx, fy - 6, fr, 0, 7); g.fill();
    // moonlight on the water
    if (env.moonElev > 0 && d > .3) drawGlitter(g, env.moon[0], [210, 225, 255], d * (1 - wx_now.cloud * .9) * .8, 90);
  }

  function drawWeatherFx(g, t, now) {
    // rain
    const n = wx_now.rain > .08 ? Math.round(wx_now.rain * 340) : 0;
    if (n) {
      const slant = 4 + wx_now.wind * 10, a = (.25 + .2 * (1 - env.dark)) * Math.min(1, wx_now.rain * 2);
      g.strokeStyle = `rgba(205,218,235,${a})`; g.lineWidth = 1;
      g.beginPath();
      for (let i = 0; i < n; i++) {
        const x = ((i * 97.13 + t * (300 + wx_now.wind * 220)) % (W + 60)) - 30, y = ((i * 53.7 + t * 720) % (H + 20)) - 10;
        g.moveTo(x, y); g.lineTo(x - slant, y + 14);
      }
      g.stroke();
      g.strokeStyle = `rgba(230,240,250,${a * .8})`;
      for (let i = 0; i < n / 10; i++) {
        const x = Math.random() * W, y = HZ + Math.random() ** 1.5 * (H - HZ), r = 1.5 + Math.random() * 4;
        g.beginPath(); g.ellipse(x, y, r, r * .35, 0, 0, 7); g.stroke();
      }
    }
    // fog
    if (wx_now.fog > .02) {
      const fc = mixc([232, 238, 242], [40, 50, 64], env.dark);
      const fg = g.createLinearGradient(0, HZ - 160, 0, H);
      fg.addColorStop(0, rgb(fc, 0)); fg.addColorStop(.45, rgb(fc, wx_now.fog * .78)); fg.addColorStop(1, rgb(fc, wx_now.fog * .4));
      g.fillStyle = fg; g.fillRect(0, 0, W, H);
    }
    // lightning
    if (lightning) {
      const age = (now - lightning.t0) / 1000;
      g.fillStyle = `rgba(235,240,255,${.55 * Math.exp(-age * 7)})`; g.fillRect(0, 0, W, H);
      if (age < .3) {
        g.strokeStyle = `rgba(255,255,255,${1 - age / .3})`; g.lineWidth = 2.5; g.shadowColor = "#bcd4ff"; g.shadowBlur = 12;
        g.beginPath(); lightning.pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.stroke();
        g.shadowBlur = 0;
      }
    }
  }

  function applyTint(g) {
    const e = env.sunElev;
    const golden = .22 * Math.max(0, 1 - Math.abs(e - .08) / .2);
    const stormA = .38 * clamp((wx_now.cloud - .35) / .65, 0, 1) * (1 - env.dark * .5);
    const nightA = .64 * env.dark;
    g.globalCompositeOperation = "source-atop";
    if (golden > .01) { g.fillStyle = `rgba(255,150,80,${golden})`; g.fillRect(0, 0, W, H); }
    if (stormA > .01) { g.fillStyle = `rgba(40,48,60,${stormA})`; g.fillRect(0, 0, W, H); }
    if (nightA > .01) {
      const mx = env.moonElev > 0 ? env.moon[0] : W / 2, my = env.moonElev > 0 ? env.moon[1] : -H;
      const ng = g.createRadialGradient(mx, my, 0, mx, my, W * .9);
      ng.addColorStop(0, `rgba(10,22,58,${nightA * .9})`); ng.addColorStop(.45, `rgba(6,14,42,${Math.min(.86, nightA * 1.2)})`);
      ng.addColorStop(1, `rgba(4,9,30,${Math.min(.9, nightA * 1.32)})`);
      g.fillStyle = ng; g.fillRect(0, 0, W, H);
    }
    g.globalCompositeOperation = "source-over";
  }

  // Light from the sun or moon, drawn over the composited scene.
  function drawLighting(g, t) {
    const L = light();
    if (env.sunElev > -0.05) {
      const [x, y] = env.sun, up = sstep(-0.05, .25, env.sunElev), clear = 1 - wx_now.cloud * .8;
      // warm sunlight falling across the scene, cooler shade on the far side
      g.globalCompositeOperation = "soft-light";
      const lg = g.createRadialGradient(x, y, 0, x, y, W * 1.1);
      lg.addColorStop(0, `rgba(255,236,190,${.55 * up * clear})`); lg.addColorStop(.5, `rgba(255,210,150,${.18 * up * clear})`);
      lg.addColorStop(1, `rgba(20,50,110,${.3 * up * clear})`);
      g.fillStyle = lg; g.fillRect(0, 0, W, H);
      // god rays through the clouds + bloom
      g.globalCompositeOperation = "lighter";
      const rayA = .028 * up * (.35 + wx_now.cloud) * (1 - wx_now.storm) * (1 - wx_now.fog * .6) * (1 - sstep(.5, .9, env.sunElev) * .5);
      if (rayA > .004) {
        for (let i = 0; i < 14; i++) {
          const a = .22 * Math.PI + i / 13 * .56 * Math.PI + Math.sin(t * .07 + i) * .05, wdt = .02 + (i * 7 % 5) * .012, len = W * .95;
          const rg = g.createRadialGradient(x, y, 0, x, y, len);
          rg.addColorStop(0, `rgba(255,240,200,${rayA * (1 + (i % 3) * .4)})`); rg.addColorStop(.6, `rgba(255,240,200,${rayA * .3})`); rg.addColorStop(1, "rgba(255,240,200,0)");
          g.fillStyle = rg; g.beginPath(); g.moveTo(x, y);
          g.lineTo(x + Math.cos(a - wdt) * len, y + Math.sin(a - wdt) * len); g.lineTo(x + Math.cos(a + wdt) * len, y + Math.sin(a + wdt) * len); g.fill();
        }
      }
      const bl = g.createRadialGradient(x, y, 0, x, y, 120 * Math.max(.6, S));
      bl.addColorStop(0, `rgba(255,245,210,${.35 * up * clear})`); bl.addColorStop(1, "rgba(255,245,210,0)");
      g.fillStyle = bl; g.beginPath(); g.arc(x, y, 120 * Math.max(.6, S), 0, 7); g.fill();
      g.globalCompositeOperation = "source-over";
    }
    if (env.moonElev > 0 && env.dark > .3) {                                    // cool moonlight
      const [x, y] = env.moon, k = env.dark * (1 - wx_now.cloud * .8);
      g.globalCompositeOperation = "lighter";
      const mg = g.createRadialGradient(x, y, 0, x, y, W * .55);
      mg.addColorStop(0, `rgba(120,150,220,${.16 * k})`); mg.addColorStop(1, "rgba(120,150,220,0)");
      g.fillStyle = mg; g.fillRect(0, 0, W, H);
      g.globalCompositeOperation = "source-over";
    }
    return L;
  }

  // One full frame. `actors(g)` draws the moving game objects into the world layer
  // and may return extra light positions (e.g. the boat's lamp).
  function render(t, now, actors) {
    drawSky(ctx, t);
    wx.clearRect(0, 0, W, H);
    drawClouds(wx, t);
    drawSea(wx, t);
    wx.drawImage(L.mainland, 0, 0, W, H);
    wx.drawImage(L.island, 0, 0, W, H);
    geo.palms.forEach(([x, h, lean]) => drawPalm(wx, x, h, lean, t));
    drawBuoys(wx, t);
    drawWhales(wx, t, now);
    drawBirds(wx, t);
    const extraLights = actors ? actors(wx) : null;
    applyTint(wx);
    ctx.drawImage(wc, 0, 0, W, H);
    drawLighting(ctx, t);
    drawFireFlames(ctx, t);
    drawLights(ctx, t, extraLights);
    drawWeatherFx(ctx, t, now);
  }

  window.Scene = {
    resize, update, render, clockLabel, setWeather, WEATHER, mood, drawBubble,
    force(name) { forced = name; if (name) setWeather(name, true); },
    release(next) { forced = null; setWeather(next || "sunny", false); },
    set onLightning(fn) { onLightning = fn; },
    _testSetPhase(p) { phaseOverride = p; },
    sx, sy, get W() { return W; }, get H() { return H; }, get HZ() { return HZ; }, get S() { return S; },
    geo, LAUNCH_H, LAND_H, ctx,
    drawLink, drawGlider, drawBoat, drawBottle,
  };
})();
