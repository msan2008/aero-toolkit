// Wing Lab teaching figures (inline SVG strings).
//
// Cross-sections use the standard NACA 4-digit thickness equation (12% thick).
// "symmetric" has no camber; "cambered" and the whale wing use 4% camber at 40%
// chord. These are representative textbook shapes for explaining the ideas, not
// the exact CFD geometries.
//
// The flow sketch is CONCEPTUAL: it only marks where the model predicts the flow
// separates (x/c). It is not a simulated flow field, and it says so.

/* global window */
(function () {
  const TH = 0.12;
  const yt = (x) => 5 * TH * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4);
  function yc(x, fam) {
    if (fam === "symmetric") return 0;
    const m = 0.04, p = 0.4;
    return x < p ? m / (p * p) * (2 * p * x - x * x) : m / ((1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x - x * x);
  }
  const upper = (x, fam) => yc(x, fam) + yt(x), lower = (x, fam) => yc(x, fam) - yt(x);
  const xs = Array.from({ length: 61 }, (_, i) => (1 - Math.cos(Math.PI * i / 60)) / 2);   // cosine spacing

  // map airfoil coords (x right, y up, chord = 1, nose-up angle in degrees) to SVG
  function frame(x0, y0, c, angDeg) {
    const a = angDeg * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a);
    return (x, y) => [x0 + c * (x * ca + y * sa), y0 - c * (-x * sa + y * ca)];
  }
  const P = (pts) => pts.map(([x, y], i) => (i ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1)).join(" ");
  function airfoilPath(fam, pt) {
    const up = xs.slice().reverse().map((x) => pt(x, upper(x, fam)));
    const lo = xs.slice(1).map((x) => pt(x, lower(x, fam)));
    return P(up.concat(lo)) + " Z";
  }
  const svg = (w, h, body, cls) => `<svg class="${cls || "fig"}" viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px" xmlns="http://www.w3.org/2000/svg">${body}</svg>`;
  const TXT = 'font-family="system-ui,sans-serif"';
  const FILL = { symmetric: "#9ec9ee", cambered: "#ffc56e", biomimetic: "#8fe0b0" };

  // small icon of a cross-section (wing cards)
  function icon(fam) {
    const pt = frame(4, fam === "symmetric" ? 13 : 15, 62, 0);
    let extra = "";
    if (fam === "biomimetic") {
      extra = `<text x="70" y="17" font-size="12" ${TXT}>🐋</text>`;
    }
    return svg(86, 24, `<path d="${airfoilPath(fam, pt)}" fill="${FILL[fam]}" stroke="#fff" stroke-width="1"/>${extra}`, "icon");
  }

  // labelled cross-sections side by side (intro + "Wing Workshop" upgrade)
  function shapes(fams) {
    const W = 330, rowH = 64;
    let body = "";
    fams.forEach((fam, r) => {
      const pt = frame(14, 34 + r * rowH, 190, 0);
      body += `<path d="${airfoilPath(fam, pt)}" fill="${FILL[fam]}" stroke="#fff" stroke-width="1.4"/>`;
      const [lx, ly] = pt(0, 0), [tx, ty] = pt(1, 0);
      body += `<line x1="${lx}" y1="${ly}" x2="${tx}" y2="${ty}" stroke="rgba(255,255,255,.45)" stroke-dasharray="3 3"/>`;
      body += `<text x="218" y="${38 + r * rowH}" font-size="13" font-weight="700" fill="#fff" ${TXT}>${fam === "symmetric" ? "Symmetric" : fam === "cambered" ? "Cambered" : "Whale"}</text>`;
      body += `<text x="218" y="${53 + r * rowH}" font-size="10" fill="#cfe3f2" ${TXT}>${fam === "symmetric" ? "same top & bottom" : "arched on top"}</text>`;
    });
    body += `<text x="14" y="${fams.length * rowH + 10}" font-size="10" fill="#cfe3f2" ${TXT}>Side view (cross-section). Front edge on the left.</text>`;
    return svg(W, fams.length * rowH + 16, body);
  }

  // top view of a whale wing with tubercles along the front edge
  function planform(amps) {
    const W = 300, rowH = 78; let body = "";
    amps.forEach((amp, r) => {
      const y0 = 18 + r * rowH, span = 200, x0 = 20, chord = 44, a = amp / 32.7 * 7, waves = 5;
      const pts = [];
      for (let i = 0; i <= 100; i++) { const u = i / 100; pts.push([x0 + u * span, y0 + 6 - a * Math.sin(u * waves * 2 * Math.PI)]); }
      const d = P(pts) + ` L${x0 + span} ${y0 + chord} L${x0} ${y0 + chord} Z`;
      body += `<path d="${d}" fill="${FILL.biomimetic}" stroke="#fff" stroke-width="1.4"/>`;
      body += `<text x="${x0 + span + 10}" y="${y0 + 20}" font-size="12" font-weight="700" fill="#fff" ${TXT}>${amp.toFixed(1)} mm</text>`;
      body += `<text x="${x0 + span + 10}" y="${y0 + 34}" font-size="10" fill="#cfe3f2" ${TXT}>bumps</text>`;
    });
    body += `<text x="20" y="${amps.length * rowH + 6}" font-size="10" fill="#cfe3f2" ${TXT}>Top view: air arrives from the top of the picture.</text>`;
    return svg(W, amps.length * rowH + 12, body);
  }

  // angle of attack diagram
  function angle(deg) {
    const a = deg * Math.PI / 180, R = 205, pt = frame(30, 40, 170, deg);
    let body = `<line x1="10" y1="40" x2="${30 + R + 10}" y2="40" stroke="rgba(255,255,255,.6)" stroke-dasharray="4 4"/>`;
    body += `<path d="${airfoilPath("cambered", pt)}" fill="${FILL.cambered}" stroke="#fff" stroke-width="1.4"/>`;
    body += `<line x1="30" y1="40" x2="${30 + R * Math.cos(a)}" y2="${40 + R * Math.sin(a)}" stroke="#fff" stroke-width="1" stroke-dasharray="2 3"/>`;
    body += `<path d="M ${30 + R} 40 A ${R} ${R} 0 0 1 ${30 + R * Math.cos(a)} ${40 + R * Math.sin(a)}" fill="none" stroke="#ffd166" stroke-width="2.5"/>`;
    body += `<text x="${30 + R + 8}" y="${40 + R * Math.sin(a) / 2 + 5}" font-size="12" font-weight="700" fill="#ffd166" ${TXT}>${deg}° angle</text>`;
    body += `<text x="${30 + R + 8}" y="${40 + R * Math.sin(a) / 2 + 19}" font-size="10" fill="#cfe3f2" ${TXT}>of attack</text>`;
    body += `<text x="6" y="16" font-size="12" fill="#cfe3f2" ${TXT}>wind →</text>`;
    return svg(320, 40 + R * Math.sin(a) + 26, body);
  }

  // conceptual flow sketch: smooth flow up to x/c, swirling separated air after it
  function flow(fam, angDeg, x) {
    const steep = Math.min(angDeg, 16), W = 320, H = 118 + steep * 3.2, c = 190, pt = frame(62, 66, c, steep);
    const xs_ = Math.max(0.02, Math.min(1, x)), fill = FILL[fam] || FILL.cambered;
    let body = `<defs><linearGradient id="wk" x1="0" x2="1"><stop offset="0" stop-color="rgba(255,120,90,.45)"/><stop offset="1" stop-color="rgba(255,120,90,0)"/></linearGradient></defs>`;
    // separated region (wake) above the surface behind the separation point
    if (xs_ < .99) {
      const top = [], surf = [];
      for (let i = 0; i <= 24; i++) {
        const u = xs_ + (1.35 - xs_) * i / 24, s = Math.min(u, 1);
        top.push(pt(u, upper(s, fam) + .02 + (u - xs_) * .32));
        surf.push(pt(u, u <= 1 ? upper(u, fam) : lower(1, fam) - (u - 1) * .05));
      }
      body += `<path d="${P(top.concat(surf.reverse()))} Z" fill="url(#wk)"/>`;
      for (let k = 0; k < 4; k++) {                                       // swirls
        const u = xs_ + (1.25 - xs_) * (k + .6) / 4.3, [cx, cy] = pt(u, Math.min(upper(Math.min(u, 1), fam), .2) + .05 + (u - xs_) * .14);
        const r = 5 + k * 1.5;
        body += `<path d="M ${cx + r} ${cy} A ${r} ${r} 0 1 0 ${cx} ${cy - r} A ${r * .6} ${r * .6} 0 1 0 ${cx + r * .3} ${cy}" fill="none" stroke="#ffb199" stroke-width="1.5"/>`;
      }
    }
    // streamlines over the top: follow the surface, then peel away at x/c
    [.05, .11, .19].forEach((d, k) => {
      const pts = [];
      for (let i = 0; i <= 40; i++) {
        const u = -0.35 + 1.75 * i / 40;
        let y;
        if (u < 0) y = upper(0, fam) + d + u * u * .15;
        else if (u <= xs_) y = upper(u, fam) + d;
        else y = upper(Math.min(xs_, 1), fam) + d + (u - xs_) * (.34 - k * .09);
        pts.push(pt(u, y));
      }
      body += `<path d="${P(pts)}" fill="none" stroke="#e8f6ff" stroke-width="1.6" opacity="${.95 - k * .2}"/>`;
    });
    // streamline underneath stays smooth
    const low = []; for (let i = 0; i <= 40; i++) { const u = -0.35 + 1.75 * i / 40; low.push(pt(u, (u < 0 ? lower(0, fam) : lower(Math.min(u, 1), fam)) - .07 - (u < 0 ? u * u * .15 : 0))); }
    body += `<path d="${P(low)}" fill="none" stroke="#e8f6ff" stroke-width="1.4" opacity=".7"/>`;
    body += `<path d="${airfoilPath(fam, pt)}" fill="${fill}" stroke="#fff" stroke-width="1.4"/>`;
    // separation point marker
    const [sx, sy] = pt(xs_, upper(xs_, fam));
    body += `<circle cx="${sx}" cy="${sy}" r="5" fill="#ff5a5f" stroke="#fff" stroke-width="1.5"/>`;
    const lx2 = Math.max(70, Math.min(sx, W - 70));
    body += `<rect x="${lx2 - 66}" y="${sy - 30}" width="132" height="18" rx="9" fill="rgba(8,24,40,.85)"/>`;
    body += `<text x="${lx2}" y="${sy - 17}" font-size="11.5" font-weight="700" fill="#ffd166" text-anchor="middle" ${TXT}>separates at x/c ${x.toFixed(2)}</text>`;
    const [lx, ly] = pt(0, 0), [tx, ty] = pt(1, 0);
    body += `<text x="${lx - 6}" y="${ly + 22}" font-size="10" fill="#cfe3f2" text-anchor="middle" ${TXT}>front 0</text>`;
    body += `<text x="${tx + 4}" y="${ty + 22}" font-size="10" fill="#cfe3f2" text-anchor="middle" ${TXT}>back 1</text>`;
    body += `<text x="6" y="16" font-size="11" fill="#cfe3f2" ${TXT}>wind →</text>`;
    body += `<text x="${W - 4}" y="14" font-size="9" fill="#a9c3d6" text-anchor="end" ${TXT}>conceptual sketch, not a simulation</text>`;
    return svg(W, H, body);
  }

  window.Figures = { icon, shapes, planform, angle, flow };
})();
