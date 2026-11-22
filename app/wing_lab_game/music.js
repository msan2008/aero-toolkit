// Wing Lab background music.
//
// Three ORIGINAL tunes written for this project, synthesised live with the Web
// Audio API (no audio files, nothing to license). Each plays for one minute,
// then fades into the next. Browsers only allow sound after the player clicks,
// so music starts on the first click inside the game.
//
// Optional: to use your own music instead, add app/wing_lab_game/music/playlist.json
//   [{"title": "My Song", "file": "my_song.mp3"}, ...]
// with the files next to it. Only add tracks you have the rights to share
// (your own, CC0, or properly licensed and credited), because the repository
// and the deployed app are public.

/* global window, document, localStorage, fetch, Audio */
(function () {
  const TRACK_SECONDS = 60, VOL = 0.32;

  // chords: [root midi, "M" | "m"] per bar. melody: per bar, [midi | null, length in eighths]
  const TRACKS = [
    { title: "Island Morning", bpm: 96, lead: "triangle", leadVol: .16, arp: "sine", drums: "soft",
      chords: [[53, "M"], [57, "m"], [58, "M"], [48, "M"], [53, "M"], [50, "m"], [58, "M"], [48, "M"]],
      melody: [
        [[69, 2], [72, 2], [77, 3], [76, 1]], [[74, 2], [72, 2], [69, 4]], [[70, 2], [74, 2], [77, 2], [74, 2]], [[72, 6], [null, 2]],
        [[69, 2], [72, 2], [69, 2], [67, 2]], [[65, 2], [69, 2], [74, 4]], [[74, 2], [72, 2], [70, 2], [67, 2]], [[65, 6], [null, 2]]] },
    { title: "Open Sky", bpm: 126, lead: "square", leadVol: .07, arp: "triangle", drums: "march",
      chords: [[50, "M"], [55, "M"], [59, "m"], [57, "M"], [50, "M"], [55, "M"], [57, "M"], [50, "M"]],
      melody: [
        [[74, 1], [78, 1], [81, 2], [78, 2], [81, 2]], [[83, 3], [81, 1], [79, 4]], [[78, 2], [74, 2], [71, 2], [74, 2]], [[76, 3], [78, 1], [76, 2], [73, 2]],
        [[74, 1], [78, 1], [81, 2], [86, 2], [81, 2]], [[83, 2], [79, 2], [83, 2], [86, 2]], [[85, 3], [83, 1], [81, 2], [76, 2]], [[74, 6], [null, 2]]] },
    { title: "Tradewind Dash", bpm: 144, lead: "square", leadVol: .065, arp: "triangle", drums: "latin",
      chords: [[57, "m"], [50, "M"], [57, "m"], [52, "M"], [57, "m"], [50, "M"], [55, "M"], [52, "M"]],
      melody: [
        [[76, 1], [null, 1], [76, 1], [74, 1], [72, 2], [69, 2]], [[74, 1], [null, 1], [78, 2], [81, 2], [78, 2]],
        [[76, 2], [72, 1], [74, 1], [76, 2], [81, 2]], [[68, 2], [71, 2], [76, 4]],
        [[81, 1], [null, 1], [81, 1], [79, 1], [76, 2], [72, 2]], [[74, 2], [78, 2], [81, 2], [83, 2]],
        [[79, 2], [74, 2], [71, 2], [79, 2]], [[76, 2], [68, 2], [69, 4]]] },
  ];
  const DRUMS = {
    soft:  { kick: [0, 4], snare: [], hat: [2, 6] },
    march: { kick: [0, 3, 4], snare: [2, 6], hat: [0, 1, 2, 3, 4, 5, 6, 7] },
    latin: { kick: [0, 3, 6], snare: [4], hat: [0, 1, 2, 3, 4, 5, 6, 7] },
  };
  TRACKS.forEach((tr) => {                       // flatten the melody into a step -> note map
    tr.mel = {}; let step = 0;
    tr.melody.forEach((bar) => bar.forEach(([n, len]) => { if (n !== null) tr.mel[step] = [n, len]; step += len; }));
  });

  const hz = (m) => 440 * Math.pow(2, (m - 69) / 12);
  let ac = null, master = null, noise = null, timer = null, files = null, audioEl = null;
  let idx = 0, step = 0, nextT = 0, trackStart = 0, switching = false, started = false;
  let muted = false;
  try { muted = localStorage.getItem("wl_muted") === "1"; } catch (e) { /* ignore */ }
  let onNowPlaying = () => {}, onMuteChange = () => {};

  // ---------------------------------------------------------------- instruments
  function env(gainNode, t, peak, dur, attack) {
    gainNode.gain.setValueAtTime(0.0001, t);
    gainNode.gain.exponentialRampToValueAtTime(peak, t + (attack || 0.012));
    gainNode.gain.exponentialRampToValueAtTime(0.0001, t + Math.max(0.06, dur));
  }
  function voice(type, f, t, dur, vol, cutoff) {
    const o = ac.createOscillator(), g = ac.createGain(); o.type = type; o.frequency.value = f;
    let out = o;
    if (cutoff) { const b = ac.createBiquadFilter(); b.type = "lowpass"; b.frequency.value = cutoff; o.connect(b); out = b; }
    out.connect(g); g.connect(master); env(g, t, vol, dur); o.start(t); o.stop(t + dur + 0.05);
  }
  function noiseHit(t, dur, vol, type, freq) {
    const s = ac.createBufferSource(), f = ac.createBiquadFilter(), g = ac.createGain();
    s.buffer = noise; f.type = type; f.frequency.value = freq; s.connect(f); f.connect(g); g.connect(master);
    env(g, t, vol, dur, 0.003); s.start(t, Math.random() * 1.5); s.stop(t + dur + 0.05);
  }
  function kick(t) {
    const o = ac.createOscillator(), g = ac.createGain(); o.type = "sine";
    o.frequency.setValueAtTime(140, t); o.frequency.exponentialRampToValueAtTime(45, t + 0.12);
    o.connect(g); g.connect(master); env(g, t, 0.5, 0.18, 0.004); o.start(t); o.stop(t + 0.25);
  }

  function playStep(tr, s, t, sd) {
    const bar = Math.floor(s / 8), pos = s % 8, [root, q] = tr.chords[bar];
    const tones = q === "M" ? [0, 4, 7, 12] : [0, 3, 7, 12];
    const note = tr.mel[s];
    if (note) voice(tr.lead, hz(note[0]), t, note[1] * sd * 0.92, tr.leadVol, tr.lead === "square" ? 2400 : 0);
    voice(tr.arp, hz(root + 12 + tones[pos % 4]), t, sd * 0.85, 0.035);
    if (pos === 0 || pos === 4) voice("triangle", hz(root - 12), t, sd * 3.5, 0.16);
    const d = DRUMS[tr.drums];
    if (d.kick.includes(pos)) kick(t);
    if (d.snare.includes(pos)) noiseHit(t, 0.13, 0.12, "bandpass", 1800);
    if (d.hat.includes(pos)) noiseHit(t, 0.035, pos % 2 ? 0.025 : 0.04, "highpass", 7000);
  }

  // ---------------------------------------------------------------- playback
  function startTrack(i) {
    idx = i; step = 0; trackStart = ac.currentTime; nextT = ac.currentTime + 0.08;
    master.gain.cancelScheduledValues(ac.currentTime);
    master.gain.setValueAtTime(0.0001, ac.currentTime);
    master.gain.exponentialRampToValueAtTime(muted ? 0.0001 : VOL, ac.currentTime + 1.5);
    onNowPlaying(TRACKS[i].title);
  }
  function tick() {
    if (!ac || muted || ac.state !== "running") return;
    const tr = TRACKS[idx], sd = 60 / tr.bpm / 2;
    if (!switching && ac.currentTime - trackStart > TRACK_SECONDS - 2) {
      switching = true;
      master.gain.setTargetAtTime(0.0001, ac.currentTime, 0.5);
      setTimeout(() => { switching = false; startTrack((idx + 1) % TRACKS.length); }, 2000);
    }
    while (nextT < ac.currentTime + 0.15) { playStep(tr, step, nextT, sd); step = (step + 1) % 64; nextT += sd; }
  }

  // optional user playlist (see top of file)
  function startFiles(i) {
    idx = i; const item = files[i];
    if (audioEl) audioEl.pause();
    audioEl = new Audio("music/" + item.file); audioEl.loop = true; audioEl.volume = muted ? 0 : 0.6;
    if (!muted) audioEl.play().catch(() => {});
    onNowPlaying(item.title || item.file);
    clearTimeout(timer); timer = setTimeout(() => startFiles((i + 1) % files.length), TRACK_SECONDS * 1000);
  }

  async function begin() {
    if (started) return; started = true;
    try {
      const r = await fetch("music/playlist.json", { cache: "no-store" });
      if (r.ok) { const j = await r.json(); if (Array.isArray(j) && j.length) files = j; }
    } catch (e) { /* no custom playlist: use the built-in tunes */ }
    if (files) { startFiles(0); return; }
    const AC = window.AudioContext || window.webkitAudioContext; if (!AC) return;
    ac = new AC(); master = ac.createGain(); master.gain.value = 0.0001;
    const comp = ac.createDynamicsCompressor(); master.connect(comp); comp.connect(ac.destination);
    noise = ac.createBuffer(1, ac.sampleRate * 3, ac.sampleRate);
    const ch = noise.getChannelData(0); for (let i = 0; i < ch.length; i++) ch[i] = Math.random() * 2 - 1;
    startTrack(0);
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
    noiseHit(t, 2.4, 0.5, "lowpass", 220);
    noiseHit(t + 0.15, 1.6, 0.25, "lowpass", 400);
  }

  window.Music = {
    begin, thunder, setMuted, toggle() { setMuted(!muted); },
    get muted() { return muted; }, titles: TRACKS.map((t) => t.title),
    set onNowPlaying(fn) { onNowPlaying = fn; }, set onMuteChange(fn) { onMuteChange = fn; },
  };
})();
