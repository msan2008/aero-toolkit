// Wing Lab background music.
//
// Five ORIGINAL pieces written for this project (no audio files, nothing to
// license). They are synthesised live with the Web Audio API using modelled
// instruments: plucked-string physics (Karplus-Strong) for harp and piano, a
// breathy flute, layered string sections, horn-style brass, cello, timpani,
// toms and bells, all through a concert-hall reverb.
//
// The music follows the in-game clock:
//   morning -> calm and light        ("Driftwood Dawn")
//   day     -> adventurous orchestra ("Wind Over the Reef", "Tradewinds")
//   night   -> solemn and stormy     ("Marooned", "Lantern on the Shore")
// Each piece plays for about a minute, then the next one for the current time
// of day fades in. If morning/day/night changes mid-piece, the music follows.
//
// Browsers only allow sound after the player clicks, so music starts on the
// first click inside the game.
//
// Optional: to use your own music instead, add app/wing_lab_game/music/playlist.json
//   [{"title": "My Song", "file": "my_song.mp3"}, ...]
// with the files next to it. Only add tracks you have the rights to share
// (your own, CC0, or properly licensed and credited): the repository and the
// deployed app are public.

/* global window, localStorage, fetch, Audio, OfflineAudioContext */
(function () {
  const TRACK_SECONDS = 60, VOL = 0.9;

  // chords: [root midi, "M" | "m"] per bar. melody: per bar, [midi | null, length in eighths]
  const TRACKS = [
    { title: "Driftwood Dawn", mood: "morning", bpm: 84, beat: 6, lead: "flute", leadVol: .13,
      accomp: "harp", pad: "strings", padVol: .035, bass: "cello", perc: "none", reverb: .38, bells: false,
      chords: [[53, "M"], [48, "M"], [50, "m"], [46, "M"], [53, "M"], [55, "m"], [48, "M"], [53, "M"],
               [46, "M"], [53, "M"], [55, "m"], [50, "m"], [46, "M"], [53, "M"], [48, "M"], [48, "M"]],
      melody: [
        [[72, 2], [69, 1], [72, 1], [77, 2]], [[76, 3], [74, 1], [72, 2]], [[74, 2], [77, 2], [74, 2]], [[72, 4], [70, 2]],
        [[69, 2], [72, 1], [74, 1], [72, 2]], [[70, 2], [67, 2], [70, 2]], [[72, 2], [76, 2], [74, 1], [72, 1]], [[77, 4], [null, 2]],
        [[74, 3], [72, 1], [70, 2]], [[69, 2], [72, 2], [69, 2]], [[67, 2], [70, 2], [74, 2]], [[72, 4], [69, 2]],
        [[70, 2], [74, 2], [77, 2]], [[76, 3], [74, 1], [72, 2]], [[74, 2], [72, 2], [67, 2]], [[72, 4], [null, 2]]] },

    { title: "Wind Over the Reef", mood: "day", bpm: 138, beat: 8, lead: "brass", leadVol: .085,
      accomp: "ostinato", pad: "strings", padVol: .03, bass: "pizz", perc: "march", reverb: .28, stabs: false,
      chords: [[50, "M"], [50, "M"], [55, "M"], [57, "M"], [59, "m"], [55, "M"], [52, "m"], [57, "M"],
               [55, "M"], [57, "M"], [54, "m"], [59, "m"], [55, "M"], [57, "M"], [50, "M"], [50, "M"]],
      melody: [
        [[62, 2], [69, 2], [74, 3], [73, 1]], [[74, 1], [76, 1], [78, 4], [74, 2]], [[79, 3], [78, 1], [76, 2], [74, 2]], [[76, 6], [null, 2]],
        [[71, 2], [74, 2], [78, 3], [76, 1]], [[74, 1], [76, 1], [79, 4], [78, 2]], [[76, 2], [74, 2], [71, 2], [73, 2]], [[69, 6], [null, 2]],
        [[67, 2], [71, 2], [74, 2], [79, 2]], [[81, 4], [79, 2], [76, 2]], [[78, 3], [76, 1], [73, 2], [69, 2]], [[74, 4], [71, 2], [74, 2]],
        [[79, 2], [78, 2], [76, 2], [79, 2]], [[81, 3], [79, 1], [76, 2], [73, 2]], [[74, 2], [78, 2], [81, 2], [86, 2]], [[86, 6], [null, 2]]] },

    { title: "Tradewinds", mood: "day", bpm: 146, beat: 8, lead: "violins", leadVol: .075,
      accomp: "harp", pad: "horns", padVol: .03, bass: "pizz", perc: "toms", reverb: .3, stabs: true,
      chords: [[57, "M"], [55, "M"], [50, "M"], [57, "M"], [54, "m"], [50, "M"], [52, "M"], [52, "M"],
               [50, "M"], [52, "M"], [49, "m"], [54, "m"], [50, "M"], [52, "M"], [57, "M"], [57, "M"]],
      melody: [
        [[76, 1], [78, 1], [81, 2], [76, 2], [73, 2]], [[74, 2], [79, 2], [83, 3], [81, 1]], [[78, 2], [74, 2], [78, 2], [81, 2]], [[76, 4], [73, 2], [76, 2]],
        [[78, 1], [81, 1], [85, 2], [83, 2], [81, 2]], [[78, 2], [81, 2], [86, 2], [83, 2]], [[80, 2], [83, 2], [88, 2], [86, 2]], [[83, 6], [null, 2]],
        [[74, 2], [78, 2], [81, 4]], [[80, 2], [83, 2], [85, 2], [83, 2]], [[80, 3], [76, 1], [73, 2], [76, 2]], [[78, 4], [81, 2], [78, 2]],
        [[81, 2], [86, 2], [85, 2], [83, 2]], [[83, 2], [80, 2], [76, 2], [83, 2]], [[81, 2], [85, 2], [88, 2], [85, 2]], [[81, 6], [null, 2]]] },

    { title: "Marooned", mood: "night", bpm: 66, beat: 8, lead: "cello", leadVol: .16,
      accomp: "none", pad: "lowstrings", padVol: .04, bass: "contrabass", perc: "rolls", reverb: .5, bells: true, wind: true,
      chords: [[50, "m"], [50, "m"], [46, "M"], [45, "M"], [43, "m"], [50, "m"], [46, "M"], [45, "M"],
               [43, "m"], [45, "M"], [50, "m"], [46, "M"], [43, "m"], [45, "M"], [50, "m"], [50, "m"]],
      melody: [
        [[62, 4], [60, 2], [57, 2]], [[58, 3], [57, 1], [55, 4]], [[53, 4], [55, 2], [58, 2]], [[57, 8]],
        [[55, 4], [58, 2], [62, 2]], [[65, 3], [64, 1], [62, 4]], [[60, 2], [58, 2], [57, 2], [55, 2]], [[57, 6], [null, 2]],
        [[67, 4], [65, 2], [62, 2]], [[64, 3], [61, 1], [57, 4]], [[62, 2], [65, 2], [69, 4]], [[70, 3], [69, 1], [65, 4]],
        [[67, 2], [65, 2], [62, 2], [58, 2]], [[61, 4], [64, 2], [61, 2]], [[62, 8]], [[null, 8]]] },

    { title: "Lantern on the Shore", mood: "night", bpm: 76, beat: 6, lead: "piano", leadVol: .27,
      accomp: "harpLow", pad: "strings", padVol: .04, bass: "cello", perc: "rumble", reverb: .48, bells: true, wind: true,
      chords: [[57, "m"], [53, "M"], [48, "M"], [55, "M"], [57, "m"], [50, "m"], [52, "M"], [52, "M"],
               [53, "M"], [55, "M"], [48, "M"], [57, "m"], [53, "M"], [55, "M"], [52, "M"], [57, "m"]],
      melody: [
        [[69, 2], [72, 2], [76, 2]], [[77, 3], [76, 1], [72, 2]], [[76, 2], [72, 2], [67, 2]], [[71, 4], [67, 2]],
        [[69, 2], [72, 2], [69, 2]], [[65, 2], [69, 2], [74, 2]], [[71, 3], [68, 1], [64, 2]], [[64, 4], [null, 2]],
        [[72, 2], [69, 2], [65, 2]], [[71, 2], [74, 2], [79, 2]], [[76, 4], [72, 2]], [[69, 4], [72, 2]],
        [[77, 2], [76, 2], [72, 2]], [[74, 3], [71, 1], [67, 2]], [[68, 2], [71, 2], [76, 2]], [[69, 4], [null, 2]]] },
  ];
  TRACKS.forEach((tr) => {                       // flatten melodies into step -> note maps
    tr.mel = {}; let step = 0;
    tr.melody.forEach((bar) => bar.forEach(([n, len]) => { if (n !== null) tr.mel[step] = [n, len]; step += len; }));
    tr.steps = tr.chords.length * tr.beat;
  });

  const hz = (m) => 440 * Math.pow(2, (m - 69) / 12);
  let ac = null, master = null, bus = null, noise = null, ksCache = {};
  let files = null, audioEl = null, fileTimer = null;
  let idx = 0, step = 0, nextT = 0, trackStart = 0, switching = false, started = false, amb = null;
  let muted = false;
  try { muted = localStorage.getItem("wl_muted") === "1"; } catch (e) { /* ignore */ }
  let onNowPlaying = () => {}, onMuteChange = () => {}, moodFn = () => "day";

  // ---------------------------------------------------------------- audio graph
  function buildGraph(context) {
    ac = context; ksCache = {};
    master = ac.createGain(); master.gain.value = 0.0001;
    const comp = ac.createDynamicsCompressor(); comp.threshold.value = -16; comp.ratio.value = 3;
    master.connect(comp); comp.connect(ac.destination);
    bus = ac.createGain(); bus.connect(master);
    const len = ac.sampleRate * 3, ir = ac.createBuffer(2, len, ac.sampleRate);    // hall reverb
    for (let c = 0; c < 2; c++) { const d = ir.getChannelData(c); for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3.2); }
    const conv = ac.createConvolver(); conv.buffer = ir;
    bus.send = ac.createGain(); bus.send.gain.value = 0.3; bus.connect(bus.send); bus.send.connect(conv); conv.connect(master);
    noise = ac.createBuffer(1, ac.sampleRate * 3, ac.sampleRate);
    const ch = noise.getChannelData(0); for (let i = 0; i < ch.length; i++) ch[i] = Math.random() * 2 - 1;
  }
  function out(pan) {
    if (!ac.createStereoPanner) return bus;
    const p = ac.createStereoPanner(); p.pan.value = pan || 0; p.connect(bus); return p;
  }
  function adsr(g, t, v, d, a, r) {             // attack, hold, release
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(v, t + a);
    g.gain.setValueAtTime(v, t + Math.max(a, d - r * .3)); g.gain.linearRampToValueAtTime(0, t + d + r);
  }
  function vibrato(params, t, rate, depth) {
    const l = ac.createOscillator(), g = ac.createGain(); l.frequency.value = rate;
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(depth, t + .35);
    l.connect(g); params.forEach((p) => g.connect(p)); return l;   // caller starts/stops it
  }

  // ---------------------------------------------------------------- instruments
  function ksBuffer(midi, bright) {               // Karplus-Strong plucked string
    const key = midi + (bright ? "b" : "h");
    if (ksCache[key]) return ksCache[key];
    const sr = ac.sampleRate, N = Math.max(2, Math.round(sr / hz(midi))), len = Math.floor(sr * (bright ? 2.6 : 2));
    const buf = ac.createBuffer(1, len, sr), d = buf.getChannelData(0);
    let prev = 0;
    for (let i = 0; i < N; i++) { const r = Math.random() * 2 - 1; d[i] = bright ? r : (r + prev) * .5; prev = r; }
    const decay = bright ? .9992 : .9975;
    for (let i = N; i < len; i++) d[i] = decay * .5 * (d[i - N] + d[i - N - 1 < 0 ? 0 : i - N - 1]);
    let peak = 0; for (let i = 0; i < len; i++) peak = Math.max(peak, Math.abs(d[i]));
    for (let i = 0; i < len; i++) d[i] /= peak || 1;
    return (ksCache[key] = buf);
  }
  function pluck(midi, t, v, pan, bright) {
    const s = ac.createBufferSource(), g = ac.createGain(); s.buffer = ksBuffer(midi, bright);
    g.gain.value = v; s.connect(g); g.connect(out(pan)); s.start(t);
  }
  function piano(midi, t, d, v) {
    pluck(midi, t, v * .8, -.1, true);
    const o = ac.createOscillator(), g = ac.createGain(); o.type = "sine"; o.frequency.value = hz(midi);
    g.gain.setValueAtTime(v * .5, t); g.gain.exponentialRampToValueAtTime(.0001, t + Math.max(.8, d * 1.6));
    o.connect(g); g.connect(out(-.1)); o.start(t); o.stop(t + Math.max(.8, d * 1.6) + .05);
  }
  function flute(midi, t, d, v) {
    const f = hz(midi), g = ac.createGain(), o = ac.createOscillator(), o2 = ac.createOscillator(), g2 = ac.createGain();
    o.type = "sine"; o.frequency.value = f; o2.type = "sine"; o2.frequency.value = f * 2; g2.gain.value = .12;
    o.connect(g); o2.connect(g2); g2.connect(g); g.connect(out(.15)); adsr(g, t, v, d, .07, .18);
    const l = vibrato([o.frequency, o2.frequency], t + .12, 5.2, f * .006);
    const b = ac.createBufferSource(), bf = ac.createBiquadFilter(), bg = ac.createGain();       // breath
    b.buffer = noise; bf.type = "bandpass"; bf.frequency.value = f * 1.5; bf.Q.value = 1.2;
    b.connect(bf); bf.connect(bg); bg.connect(out(.15)); adsr(bg, t, v * .1, d * .6, .03, .2);
    [o, o2, l, b].forEach((n) => { n.start(t); n.stop(t + d + .3); });
  }
  function section(midi, t, d, v, opt) {           // string / horn sections: detuned saws, filtered
    const f = hz(midi), g = ac.createGain(), lp = ac.createBiquadFilter();
    lp.type = "lowpass"; lp.frequency.value = opt.cut; lp.Q.value = .6; lp.connect(g); g.connect(out(opt.pan));
    const oscs = opt.detune.map((c) => { const o = ac.createOscillator(); o.type = opt.wave || "sawtooth"; o.frequency.value = f; o.detune.value = c; o.connect(lp); return o; });
    if (opt.brassy) {
      lp.frequency.setValueAtTime(opt.cut * .25, t); lp.frequency.linearRampToValueAtTime(opt.cut, t + .09);
      lp.frequency.linearRampToValueAtTime(opt.cut * .65, t + .35);
    }
    adsr(g, t, v, d, opt.attack, opt.release);
    const nodes = oscs.slice();
    if (opt.vib) nodes.push(vibrato(oscs.map((o) => o.frequency), t + .15, 5.4, f * opt.vib));
    nodes.forEach((n) => { n.start(t); n.stop(t + d + opt.release + .1); });
  }
  const INST = {
    violins:   (m, t, d, v) => section(m, t, d, v, { detune: [-9, 0, 8], cut: 3400, attack: .07, release: .25, vib: .006, pan: .2 }),
    strings:   (m, t, d, v) => section(m, t, d, v, { detune: [-8, 0, 9], cut: 2200, attack: .35, release: .5, vib: .004, pan: .25 }),
    lowstrings:(m, t, d, v) => section(m, t, d, v, { detune: [-10, 0, 10], cut: 900, attack: .6, release: .9, vib: .003, pan: -.15 }),
    cello:     (m, t, d, v) => section(m, t, d, v, { detune: [-6, 5], cut: 1400, attack: .2, release: .5, vib: .007, pan: -.2 }),
    contrabass:(m, t, d, v) => section(m, t, d, v, { detune: [-5, 5], cut: 500, attack: .25, release: .6, vib: .003, pan: 0 }),
    brass:     (m, t, d, v) => section(m, t, d, v, { detune: [-5, 6], cut: 2600, attack: .045, release: .14, vib: .003, pan: -.1, brassy: true }),
    horns:     (m, t, d, v) => section(m, t, d, v, { detune: [-4, 4], cut: 1300, attack: .25, release: .4, vib: .002, pan: -.3, wave: "triangle" }),
    flute, piano,
  };
  function timpani(midi, t, v) {
    const o = ac.createOscillator(), g = ac.createGain(), f = hz(midi);
    o.type = "sine"; o.frequency.setValueAtTime(f * 1.02, t); o.frequency.exponentialRampToValueAtTime(f * .97, t + .5);
    g.gain.setValueAtTime(v, t); g.gain.exponentialRampToValueAtTime(.0001, t + 1.5);
    o.connect(g); g.connect(out(-.05)); o.start(t); o.stop(t + 1.6);
    noiseHit(t, .12, v * .35, "lowpass", 380, 0);
  }
  function tom(midi, t, v) {
    const o = ac.createOscillator(), g = ac.createGain(), f = hz(midi);
    o.frequency.setValueAtTime(f, t); o.frequency.exponentialRampToValueAtTime(f * .6, t + .3);
    g.gain.setValueAtTime(v, t); g.gain.exponentialRampToValueAtTime(.0001, t + .4);
    o.connect(g); g.connect(out(.1)); o.start(t); o.stop(t + .45);
    noiseHit(t, .06, v * .3, "bandpass", 900, .1);
  }
  function noiseHit(t, dur, v, type, freq, pan) {
    const s = ac.createBufferSource(), f = ac.createBiquadFilter(), g = ac.createGain();
    s.buffer = noise; f.type = type; f.frequency.value = freq; s.connect(f); f.connect(g); g.connect(out(pan));
    g.gain.setValueAtTime(v, t); g.gain.exponentialRampToValueAtTime(.0001, t + dur);
    s.start(t, Math.random() * 1.5); s.stop(t + dur + .05);
  }
  function bell(midi, t, v) {
    [[1, 1], [2.76, .35], [5.4, .15]].forEach(([k, a]) => {
      const o = ac.createOscillator(), g = ac.createGain(); o.frequency.value = hz(midi) * k;
      g.gain.setValueAtTime(v * a, t); g.gain.exponentialRampToValueAtTime(.0001, t + 3.5 / k);
      o.connect(g); g.connect(out(.35)); o.start(t); o.stop(t + 3.6);
    });
  }
  function startAmbience(t) {                       // night wind
    const s = ac.createBufferSource(), f = ac.createBiquadFilter(), g = ac.createGain(), l = ac.createOscillator(), lg = ac.createGain();
    s.buffer = noise; s.loop = true; f.type = "bandpass"; f.frequency.value = 500; f.Q.value = .8;
    l.frequency.value = .09; lg.gain.value = 320; l.connect(lg); lg.connect(f.frequency);
    s.connect(f); f.connect(g); g.connect(master);
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(.05, t + 4);
    s.start(t); l.start(t); return { s, l, g };
  }
  function stopAmbience(t) {
    if (!amb) return; amb.g.gain.setTargetAtTime(0, t, .6); amb.s.stop(t + 3); amb.l.stop(t + 3); amb = null;
  }

  function playStep(tr, s, t, sd) {
    const bar = Math.floor(s / tr.beat), pos = s % tr.beat, [root, q] = tr.chords[bar];
    const tones = q === "M" ? [0, 4, 7, 12] : [0, 3, 7, 12], barDur = tr.beat * sd;
    const note = tr.mel[s];
    if (note) INST[tr.lead](note[0], t, note[1] * sd * .95, tr.leadVol);
    const arp = tr.beat === 6 ? [0, 1, 2, 3, 2, 1] : [0, 1, 2, 3, 2, 1, 2, 3];
    if (tr.accomp === "harp") pluck(root + 12 + tones[arp[pos]], t, .17, -.35, false);
    if (tr.accomp === "harpLow") pluck(root + tones[arp[pos]], t, .16, -.3, false);
    if (tr.accomp === "ostinato") INST.violins(root + 12 + (pos % 2 ? 7 : 0), t, sd * .45, pos === 0 ? .045 : .03);
    if (pos === 0 && tr.pad) [0, 1, 2].forEach((k) => INST[tr.pad](root + 12 + tones[k], t, barDur * 1.02, tr.padVol));
    if (tr.stabs && (pos === 0 || pos === 3)) [1, 2].forEach((k) => INST.brass(root + 12 + tones[k], t, sd * .7, .04));
    if (tr.bass === "pizz" && pos % 4 === 0) pluck(root - 12, t, .3, 0, false);
    if (tr.bass === "cello" && pos === 0) INST.cello(root - 12, t, barDur * .98, .07);
    if (tr.bass === "contrabass" && pos === 0) INST.contrabass(root - 12, t, barDur, .12);
    if (tr.bells && pos === 0 && bar % 4 === 0) bell(root + 24 + tones[2], t, .05);
    if (tr.perc === "march") {
      if (pos === 0 || pos === 4) timpani(root - 12, t, pos ? .22 : .3);
      if (pos === 2 || pos === 6) noiseHit(t, .12, .07, "bandpass", 2000, .1);
      if (bar % 4 === 3 && pos >= 6) [0, .25, .5, .75].forEach((k) => noiseHit(t + k * sd / 2, .08, .045 + k * .03, "bandpass", 2000, .1));
    }
    if (tr.perc === "toms") {
      if (pos === 0 || pos === 3) tom(root - 7, t, .25);
      if (pos === 6) tom(root - 2, t, .2);
      noiseHit(t, .04, pos % 2 ? .02 : .032, "highpass", 6500, .3);                   // shaker
    }
    if (tr.perc === "rolls" && bar % 4 === 3) {           // distant timpani roll, like thunder over the sea
      for (let k = 0; k < 6; k++) timpani(root - 24, t + k * sd / 3, .05 + (pos / tr.beat) * .2 * (k / 6));
    }
    if (tr.perc === "rumble" && pos === 0 && bar % 4 === 2) noiseHit(t, 3, .12, "lowpass", 140, 0);
  }

  // ---------------------------------------------------------------- playback
  function pickTrack(mood, notIdx) {
    const pool = TRACKS.map((t, i) => i).filter((i) => TRACKS[i].mood === mood);
    const options = pool.filter((i) => i !== notIdx);
    return (options.length ? options : pool)[Math.floor(Math.random() * (options.length || pool.length))];
  }
  function startTrack(i) {
    idx = i; step = 0; trackStart = ac.currentTime; nextT = ac.currentTime + 0.08;
    bus.send.gain.value = TRACKS[i].reverb;
    master.gain.cancelScheduledValues(ac.currentTime);
    master.gain.setValueAtTime(0.0001, ac.currentTime);
    master.gain.exponentialRampToValueAtTime(muted ? 0.0001 : VOL, ac.currentTime + 2);
    if (TRACKS[i].wind) amb = startAmbience(ac.currentTime);
    onNowPlaying(TRACKS[i].title);
  }
  function switchTo(i) {
    switching = true;
    master.gain.setTargetAtTime(0.0001, ac.currentTime, 0.6);
    stopAmbience(ac.currentTime);
    setTimeout(() => { switching = false; startTrack(i); }, 2200);
  }
  let moodCheck = 0;
  function tick() {
    if (!ac || muted || ac.state !== "running" || switching) return;
    const tr = TRACKS[idx], sd = 60 / tr.bpm / 2, playing = ac.currentTime - trackStart;
    if (ac.currentTime > moodCheck) {
      moodCheck = ac.currentTime + 1;
      const mood = moodFn();
      if ((tr.mood !== mood && playing > 8) || playing > TRACK_SECONDS - 2) { switchTo(pickTrack(mood, idx)); return; }
    }
    while (nextT < ac.currentTime + 0.15) { playStep(tr, step, nextT, sd); step = (step + 1) % tr.steps; nextT += sd; }
  }

  function startFiles(i) {                        // optional user playlist (see top of file)
    idx = i; const item = files[i];
    if (audioEl) audioEl.pause();
    audioEl = new Audio("music/" + item.file); audioEl.loop = true; audioEl.volume = muted ? 0 : 0.6;
    if (!muted) audioEl.play().catch(() => {});
    onNowPlaying(item.title || item.file);
    clearTimeout(fileTimer); fileTimer = setTimeout(() => startFiles((i + 1) % files.length), TRACK_SECONDS * 1000);
  }

  async function begin() {
    if (started) return; started = true;
    try {
      const r = await fetch("music/playlist.json", { cache: "no-store" });
      if (r.ok) { const j = await r.json(); if (Array.isArray(j) && j.length) files = j; }
    } catch (e) { /* no custom playlist: use the built-in pieces */ }
    if (files) { startFiles(0); return; }
    const AC = window.AudioContext || window.webkitAudioContext; if (!AC) return;
    buildGraph(new AC());
    startTrack(pickTrack(moodFn(), -1));
    setInterval(tick, 25);
    if (muted) ac.suspend();
  }

  function setMuted(m) {
    muted = m;
    try { localStorage.setItem("wl_muted", m ? "1" : "0"); } catch (e) { /* ignore */ }
    if (audioEl) { audioEl.volume = m ? 0 : 0.6; if (m) audioEl.pause(); else audioEl.play().catch(() => {}); }
    if (ac) {
      if (m) { master.gain.setTargetAtTime(0.0001, ac.currentTime, 0.1); setTimeout(() => muted && ac.suspend(), 300); }
      else { ac.resume().then(() => { nextT = ac.currentTime + 0.05; master.gain.setTargetAtTime(VOL, ac.currentTime, 0.3); }); }
    }
    if (!m && !started) begin();
    onMuteChange(m);
  }

  function thunder() {
    if (!ac || muted || ac.state !== "running") return;
    const t = ac.currentTime + 0.4 + Math.random() * 0.8;
    noiseHit(t, 2.6, 0.6, "lowpass", 200, 0);
    noiseHit(t + 0.15, 1.8, 0.3, "lowpass", 420, 0);
  }

  // Offline render of one piece, used only by automated tests to check levels
  // and to export short previews. Returns mono samples.
  async function renderOffline(i, seconds) {
    const saved = [ac, master, bus, noise, ksCache, amb];
    const off = new OfflineAudioContext(2, 44100 * seconds, 44100);
    buildGraph(off); master.gain.value = VOL;
    const tr = TRACKS[i], sd = 60 / tr.bpm / 2; bus.send.gain.value = tr.reverb;
    if (tr.wind) startAmbience(0);
    for (let s = 0, t = 0.05; t < seconds - 0.5; s++, t += sd) playStep(tr, s % tr.steps, t, sd);
    const buf = await off.startRendering();
    [ac, master, bus, noise, ksCache, amb] = saved;
    return buf;
  }

  window.Music = {
    begin, thunder, setMuted, toggle() { setMuted(!muted); }, renderOffline,
    get muted() { return muted; }, tracks: TRACKS.map((t) => ({ title: t.title, mood: t.mood })),
    set onNowPlaying(fn) { onNowPlaying = fn; }, set onMuteChange(fn) { onMuteChange = fn; },
    set mood(fn) { moodFn = fn; },
  };
})();
