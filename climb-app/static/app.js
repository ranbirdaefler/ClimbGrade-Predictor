/* V2 In My Gym — frontend logic (no framework, no build step). */
(() => {
  "use strict";

  // ── Constants ───────────────────────────────────────────────────
  const ROLES = ["start", "hand", "finish", "foot", "volume"];
  const ROLE_LABEL = { start: "Start", hand: "Hand", finish: "Finish", foot: "Foot", volume: "Vol" };
  const ROLE_COLOR = {
    start: "#3ddc84",
    hand: "#4da3ff",
    finish: "#ff4b4b",
    foot: "#ffd23f",
    volume: "#c77dff",
  };
  const MIN_HOLDS = 3;
  const MAX_HOLDS = 40;
  const GRADES = Array.from({ length: 14 }, (_, i) => `V${i}`);

  const DEMOS = {
    red: {
      file: "/static/samples/red.jpg",
      angle: 0,
      holds: [
        [63, 38, "finish"], [318, 202, "hand"], [42, 219, "hand"],
        [214, 358, "hand"], [130, 383, "volume"], [375, 427, "foot"],
      ],
    },
    steep: {
      file: "/static/samples/steep.jpg",
      angle: 45,
      holds: [
        [153, 54, "finish"], [209, 107, "hand"], [251, 193, "hand"], [190, 242, "hand"],
        [206, 278, "hand"], [232, 333, "start"], [265, 342, "hand"],
      ],
    },
  };

  // ── State ───────────────────────────────────────────────────────
  const state = {
    session: null,
    img: null,
    w: 0,
    h: 0,
    holds: [],
    angle: 30,
    model: "loading",
    busy: false,
    result: null,
    actual: null,
    hover: null,
    highlight: null,
    animating: false,
  };
  let view = { s: 1, cw: 0, ch: 0, dpr: 1 };
  let uid = 0;

  // ── DOM ─────────────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);
  const el = {
    stage: $("stage"),
    dropzone: $("dropzone"),
    canvasWrap: $("canvas-wrap"),
    canvas: $("wall"),
    hint: $("hint"),
    stageLoading: $("stage-loading"),
    file: $("file"),
    fileCam: $("file-cam"),
    count: $("count"),
    undo: $("btn-undo"),
    clear: $("btn-clear"),
    change: $("btn-change"),
    holds: $("holds"),
    holdsMeta: $("holds-meta"),
    legend: $("legend"),
    angle: $("angle"),
    angleNum: $("angle-num"),
    angleDesc: $("angle-desc"),
    wallLine: $("wall-line"),
    grade: $("btn-grade"),
    gradeLabel: $("grade-label"),
    gradeNote: $("grade-note"),
    status: $("status"),
    result: $("result"),
    rGrade: $("r-grade"),
    rLow: $("r-low"),
    rHigh: $("r-high"),
    rBand: $("r-band"),
    rPin: $("r-pin"),
    rFacts: $("r-facts"),
    gradeGrid: $("grade-grid"),
    verdict: $("verdict"),
    copy: $("btn-copy"),
    again: $("btn-again"),
    toast: $("toast"),
  };
  const ctx = el.canvas.getContext("2d");

  // ── Utilities ───────────────────────────────────────────────────
  let toastTimer = null;
  function toast(msg, isErr = false) {
    el.toast.textContent = msg;
    el.toast.classList.toggle("err", isErr);
    el.toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.toast.classList.remove("show"), 3200);
  }

  async function api(path, body) {
    const opts = body instanceof FormData
      ? { method: "POST", body }
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    let res;
    try {
      res = await fetch(path, opts);
    } catch {
      throw new Error("Network error. Check your connection and try again.");
    }
    let data = null;
    try {
      data = await res.json();
    } catch {
      /* non-JSON error body */
    }
    if (!res.ok) {
      const detail = data && data.detail;
      const msg = typeof detail === "string" ? detail
        : Array.isArray(detail) && detail[0] && detail[0].msg ? detail[0].msg
        : `Request failed (${res.status})`;
      throw new Error(msg);
    }
    return data;
  }

  const gradeIdx = (g) => parseInt(String(g).replace(/\D/g, ""), 10) || 0;

  function orderedHolds() {
    // Climbing order: lowest hold on the wall is #1.
    return [...state.holds].sort((a, b) => b.y - a.y);
  }

  function angleDesc(a) {
    if (a === 0) return "vertical";
    if (a <= 15) return "slightly overhung";
    if (a <= 40) return "moderate overhang";
    if (a <= 55) return "steep";
    return "roof";
  }

  // ── Model status ────────────────────────────────────────────────
  async function pollHealth() {
    try {
      const r = await fetch("/api/health", { cache: "no-store" });
      const d = await r.json();
      state.model = d.status;
    } catch {
      state.model = "loading";
    }
    const span = el.status.querySelector("span");
    el.status.classList.toggle("ready", state.model === "ready");
    el.status.classList.toggle("error", state.model === "error");
    span.textContent = state.model === "ready" ? "model ready"
      : state.model === "error" ? "model offline" : "warming up model";
    renderGradeButton();
    if (state.model !== "ready") setTimeout(pollHealth, state.model === "error" ? 15000 : 2500);
  }

  // ── Photo loading ───────────────────────────────────────────────
  async function loadPhoto(file) {
    if (!file) return;
    if (!/^image\//.test(file.type) && !/\.(heic|heif)$/i.test(file.name)) {
      toast("That doesn't look like an image.", true);
      return;
    }
    el.dropzone.classList.add("hidden");
    el.canvasWrap.classList.remove("hidden");
    el.stageLoading.classList.remove("hidden");
    try {
      const fd = new FormData();
      fd.append("file", file, file.name || "photo.jpg");
      const [session, img] = await Promise.all([api("/api/session", fd), decodeImage(file)]);
      state.session = session.session_id;
      state.w = session.width;
      state.h = session.height;
      state.img = img;
      state.holds = [];
      state.result = null;
      state.actual = null;
      el.hint.classList.remove("hidden");
      layout();
      render();
      return true;
    } catch (e) {
      toast(e.message || "Couldn't read that photo.", true);
      resetToEmpty();
      return false;
    } finally {
      el.stageLoading.classList.add("hidden");
    }
  }

  function decodeImage(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("Your browser can't display this image format. Try a JPG or PNG.")); };
      img.src = url;
    });
  }

  function resetToEmpty() {
    state.session = null;
    state.img = null;
    state.holds = [];
    state.result = null;
    state.actual = null;
    el.canvasWrap.classList.add("hidden");
    el.dropzone.classList.remove("hidden");
    render();
  }

  async function loadDemo(key) {
    const demo = DEMOS[key];
    if (!demo) return;
    document.getElementById("studio").scrollIntoView({ behavior: "smooth", block: "start" });
    let blob;
    try {
      blob = await (await fetch(demo.file)).blob();
    } catch {
      toast("Couldn't fetch the demo photo.", true);
      return;
    }
    const ok = await loadPhoto(new File([blob], `${key}.jpg`, { type: "image/jpeg" }));
    if (!ok) return;
    setAngle(demo.angle);
    el.hint.classList.add("hidden");
    const holds = demo.holds.map(([x, y, role]) => ({
      id: ++uid, x, y, role, contour: null, thumb: null, pending: true, detected: false, addedAt: performance.now(),
    }));
    // Stagger the reveal so the taps read as a sequence.
    for (let i = 0; i < holds.length; i++) {
      holds[i].addedAt = performance.now() + i * 140;
    }
    state.holds = holds;
    render();
    kick();
    await Promise.all(holds.map(fetchHoldInfo));
    render();
  }

  // ── Holds ───────────────────────────────────────────────────────
  async function fetchHoldInfo(hold) {
    try {
      const r = await api("/api/hold", { session_id: state.session, x: hold.x, y: hold.y });
      if (!state.holds.includes(hold)) return;
      hold.contour = r.contour;
      hold.thumb = r.thumb;
      hold.detected = r.detected;
    } catch (e) {
      if (state.holds.includes(hold)) toast(e.message, true);
    } finally {
      hold.pending = false;
      render();
    }
  }

  function addHold(x, y) {
    if (state.holds.length >= MAX_HOLDS) {
      toast(`That's plenty — ${MAX_HOLDS} holds max.`, true);
      return;
    }
    const hold = {
      id: ++uid, x, y, role: "hand", contour: null, thumb: null,
      pending: true, detected: false, addedAt: performance.now(),
    };
    state.holds.push(hold);
    state.result = null;
    el.hint.classList.add("hidden");
    render();
    kick();
    fetchHoldInfo(hold);
  }

  function removeHold(hold) {
    state.holds = state.holds.filter((h) => h !== hold);
    state.result = null;
    if (state.highlight === hold.id) state.highlight = null;
    render();
  }

  function setRole(hold, role) {
    if (hold.role === role) return;
    hold.role = role;
    state.result = null;
    render();
  }

  function hitTest(sx, sy) {
    const r = 18;
    let best = null;
    let bestD = r * r;
    for (const h of state.holds) {
      const dx = h.x * view.s - sx;
      const dy = h.y * view.s - sy;
      const d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = h; }
    }
    return best;
  }

  // ── Canvas ──────────────────────────────────────────────────────
  function layout() {
    if (!state.img) return;
    const maxW = el.stage.clientWidth;
    const maxH = Math.min(window.innerHeight * 0.76, 860);
    const s = Math.min(maxW / state.w, maxH / state.h);
    const cw = Math.max(1, Math.round(state.w * s));
    const ch = Math.max(1, Math.round(state.h * s));
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    view = { s, cw, ch, dpr };
    el.canvas.style.width = `${cw}px`;
    el.canvas.style.height = `${ch}px`;
    el.canvas.width = Math.round(cw * dpr);
    el.canvas.height = Math.round(ch * dpr);
    draw();
  }

  function draw() {
    if (!state.img) return;
    const { s, cw, ch, dpr } = view;
    const now = performance.now();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    ctx.drawImage(state.img, 0, 0, cw, ch);

    const ordered = orderedHolds();
    const numberOf = new Map(ordered.map((h, i) => [h.id, i + 1]));

    // Contours first so markers sit on top.
    for (const h of state.holds) {
      if (!h.contour) continue;
      const color = ROLE_COLOR[h.role];
      const born = Math.min(1, Math.max(0, (now - h.addedAt) / 450));
      ctx.beginPath();
      h.contour.forEach(([x, y], i) => (i ? ctx.lineTo(x * s, y * s) : ctx.moveTo(x * s, y * s)));
      ctx.closePath();
      ctx.fillStyle = hexA(color, 0.2 * born);
      ctx.fill();
      ctx.lineWidth = state.highlight === h.id ? 3 : 2;
      ctx.strokeStyle = hexA(color, 0.95 * born);
      ctx.stroke();
    }

    for (const h of state.holds) {
      const x = h.x * s;
      const y = h.y * s;
      const color = ROLE_COLOR[h.role];
      const age = now - h.addedAt;
      const hi = state.highlight === h.id;

      // Landing pulse.
      if (age >= 0 && age < 700) {
        const t = age / 700;
        ctx.beginPath();
        ctx.arc(x, y, 12 + t * 26, 0, Math.PI * 2);
        ctx.strokeStyle = hexA(color, (1 - t) * 0.8);
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      if (age < 0) continue; // staggered demo reveal

      // Marker ring.
      ctx.beginPath();
      ctx.arc(x, y, hi ? 14 : 12, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(8,8,10,0.55)";
      ctx.fill();
      ctx.lineWidth = hi ? 4 : 3;
      ctx.strokeStyle = color;
      if (h.pending) {
        ctx.setLineDash([5, 5]);
        ctx.lineDashOffset = -(now / 40) % 10;
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = "#fff";
      ctx.fill();

      // Number pill.
      const n = String(numberOf.get(h.id));
      ctx.font = "600 12px 'JetBrains Mono', ui-monospace, monospace";
      const tw = ctx.measureText(n).width + 12;
      const px = x + 15;
      const py = y - 20;
      roundRect(ctx, px, py, tw, 20, 6);
      ctx.fillStyle = "rgba(8,8,10,0.82)";
      ctx.fill();
      ctx.strokeStyle = hexA(color, 0.7);
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = "#fff";
      ctx.textBaseline = "middle";
      ctx.fillText(n, px + 6, py + 10.5);
    }

    // Hover crosshair.
    if (state.hover && !state.busy) {
      const { x, y } = state.hover;
      const over = hitTest(x, y);
      ctx.beginPath();
      ctx.arc(x, y, 14, 0, Math.PI * 2);
      ctx.strokeStyle = over ? "rgba(255,75,75,0.9)" : "rgba(255,255,255,0.75)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash(over ? [] : [3, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
      if (!over) {
        ctx.beginPath();
        ctx.moveTo(x - 20, y); ctx.lineTo(x - 8, y);
        ctx.moveTo(x + 8, y); ctx.lineTo(x + 20, y);
        ctx.moveTo(x, y - 20); ctx.lineTo(x, y - 8);
        ctx.moveTo(x, y + 8); ctx.lineTo(x, y + 20);
        ctx.strokeStyle = "rgba(255,255,255,0.55)";
        ctx.stroke();
      }
    }
  }

  function needsAnim() {
    const now = performance.now();
    return state.holds.some((h) => h.pending || now - h.addedAt < 720);
  }

  function kick() {
    if (state.animating) return;
    state.animating = true;
    const loop = () => {
      draw();
      if (needsAnim()) requestAnimationFrame(loop);
      else { state.animating = false; draw(); }
    };
    requestAnimationFrame(loop);
  }

  function hexA(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }

  function roundRect(c, x, y, w, h, r) {
    c.beginPath();
    c.moveTo(x + r, y);
    c.arcTo(x + w, y, x + w, y + h, r);
    c.arcTo(x + w, y + h, x, y + h, r);
    c.arcTo(x, y + h, x, y, r);
    c.arcTo(x, y, x + w, y, r);
    c.closePath();
  }

  // Pointer handling: distinguish taps from scrolls/drags.
  let down = null;
  function canvasPoint(ev) {
    const r = el.canvas.getBoundingClientRect();
    return { x: ev.clientX - r.left, y: ev.clientY - r.top };
  }
  el.canvas.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0 && ev.pointerType === "mouse") return;
    down = { ...canvasPoint(ev), id: ev.pointerId, t: performance.now() };
  });
  el.canvas.addEventListener("pointermove", (ev) => {
    if (ev.pointerType === "mouse") {
      state.hover = canvasPoint(ev);
      const over = hitTest(state.hover.x, state.hover.y);
      state.highlight = over ? over.id : null;
      syncHighlightRows();
      if (!state.animating) draw();
    }
  });
  el.canvas.addEventListener("pointerleave", () => {
    state.hover = null;
    state.highlight = null;
    syncHighlightRows();
    if (!state.animating) draw();
  });
  el.canvas.addEventListener("pointerup", (ev) => {
    if (!down || down.id !== ev.pointerId) return;
    const p = canvasPoint(ev);
    const moved = Math.hypot(p.x - down.x, p.y - down.y);
    down = null;
    if (moved > 8 || state.busy) return;
    const hit = hitTest(p.x, p.y);
    if (hit) { removeHold(hit); return; }
    const ix = Math.round(p.x / view.s);
    const iy = Math.round(p.y / view.s);
    if (ix < 0 || iy < 0 || ix >= state.w || iy >= state.h) return;
    addHold(ix, iy);
  });
  el.canvas.addEventListener("pointercancel", () => { down = null; });
  el.canvas.addEventListener("contextmenu", (ev) => ev.preventDefault());

  // ── Panel rendering ─────────────────────────────────────────────
  function render() {
    const n = state.holds.length;
    el.count.textContent = n;
    el.undo.disabled = n === 0;
    el.clear.disabled = n === 0;
    el.holdsMeta.textContent = n === 0 ? `min ${MIN_HOLDS}` : n < MIN_HOLDS ? `${MIN_HOLDS - n} more` : `${n} marked`;
    renderHoldRows();
    renderGradeButton();
    renderResult();
    if (!state.animating) draw();
  }

  function renderHoldRows() {
    const ordered = orderedHolds();
    if (!ordered.length) {
      el.holds.innerHTML = `<div class="empty-holds">${state.img
        ? "Tap holds on the photo to list them here."
        : "Add a photo first, then tap the holds you used."}</div>`;
      el.legend.classList.remove("hidden");
      return;
    }
    el.legend.classList.add("hidden");
    const frag = document.createDocumentFragment();
    ordered.forEach((h, i) => {
      const row = document.createElement("div");
      row.className = "hold-row" + (state.highlight === h.id ? " hi" : "");
      row.dataset.id = h.id;
      row.style.setProperty("--role", ROLE_COLOR[h.role]);
      const thumb = h.thumb
        ? `<img class="thumb" src="${h.thumb}" alt="">`
        : `<div class="thumb pending"></div>`;
      const roles = ROLES.map((r) =>
        `<button type="button" data-role="${r}" class="${h.role === r ? "on" : ""}" style="--c:${ROLE_COLOR[r]}" title="${ROLE_LABEL[r]}">${ROLE_LABEL[r]}</button>`
      ).join("");
      row.innerHTML = `${thumb}
        <div><div class="num"><b>#${i + 1}</b> · ${h.x}, ${h.y}</div><div class="roles">${roles}</div></div>
        <button type="button" class="rm" title="Remove hold" aria-label="Remove hold ${i + 1}">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M6 6l12 12M18 6 6 18"/></svg>
        </button>`;
      row.querySelector(".roles").addEventListener("click", (ev) => {
        const b = ev.target.closest("button[data-role]");
        if (b) setRole(h, b.dataset.role);
      });
      row.querySelector(".rm").addEventListener("click", () => removeHold(h));
      row.addEventListener("pointerenter", () => { state.highlight = h.id; syncHighlightRows(); if (!state.animating) draw(); });
      row.addEventListener("pointerleave", () => { state.highlight = null; syncHighlightRows(); if (!state.animating) draw(); });
      frag.appendChild(row);
    });
    el.holds.replaceChildren(frag);
  }

  function syncHighlightRows() {
    el.holds.querySelectorAll(".hold-row").forEach((row) => {
      row.classList.toggle("hi", Number(row.dataset.id) === state.highlight);
    });
  }

  function renderGradeButton() {
    const n = state.holds.length;
    const pending = state.holds.some((h) => h.pending);
    let label = "Grade this route";
    let disabled = false;
    if (state.busy) { label = "Reading the holds…"; disabled = true; }
    else if (!state.img) { label = "Add a photo to begin"; disabled = true; }
    else if (n < MIN_HOLDS) { label = `Tap ${MIN_HOLDS - n} more hold${MIN_HOLDS - n === 1 ? "" : "s"}`; disabled = true; }
    else if (state.model === "error") { label = "Model offline"; disabled = true; }
    else if (state.model !== "ready") { label = "Warming up the model…"; disabled = true; }
    else if (pending) { label = "Locating holds…"; disabled = true; }
    else if (state.result) { label = "Grade again"; }
    el.gradeLabel.textContent = label;
    el.grade.disabled = disabled;
    el.grade.classList.toggle("busy", state.busy);
    el.grade.querySelector(".spinner")?.remove();
    if (state.busy) {
      const sp = document.createElement("span");
      sp.className = "spinner";
      el.grade.prepend(sp);
    }
  }

  // ── Wall angle ──────────────────────────────────────────────────
  function setAngle(a) {
    state.angle = a;
    el.angle.value = a;
    el.angleNum.textContent = `${a}°`;
    el.angleDesc.textContent = angleDesc(a);
    el.angle.style.setProperty("--pct", `${(a / 70) * 100}%`);
    el.wallLine.style.transform = `rotate(${-a}deg)`;
  }
  el.angle.addEventListener("input", () => {
    setAngle(Number(el.angle.value));
    if (state.result) { state.result = null; renderGradeButton(); renderResult(); }
  });

  // ── Prediction ──────────────────────────────────────────────────
  async function grade() {
    if (el.grade.disabled || state.busy) return;
    state.busy = true;
    state.actual = null;
    renderGradeButton();
    const slow = setTimeout(() => { el.gradeLabel.textContent = "Thinking…"; }, 1600);
    try {
      const r = await api("/api/predict", {
        session_id: state.session,
        holds: state.holds.map((h) => ({ x: h.x, y: h.y, role: h.role })),
        wall_angle: state.angle,
      });
      state.result = r;
      renderResult(true);
      el.result.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (e) {
      toast(e.message, true);
    } finally {
      clearTimeout(slow);
      state.busy = false;
      renderGradeButton();
    }
  }
  el.grade.addEventListener("click", grade);

  function renderResult(fresh = false) {
    const r = state.result;
    if (!r) {
      el.result.classList.add("hidden");
      return;
    }
    el.result.classList.remove("hidden");
    el.rLow.textContent = r.low;
    el.rHigh.textContent = r.high;
    const pos = Math.max(0, Math.min(1, (r.difficulty - 10) / 20));
    const lo = gradeIdx(r.low) / 13;
    const hi = gradeIdx(r.high) / 13;
    el.rBand.style.left = `${lo * 100}%`;
    el.rBand.style.width = `${Math.max(3, (hi - lo) * 100)}%`;
    el.rPin.style.left = `${pos * 100}%`;
    el.rFacts.innerHTML = [
      `${r.n_holds} holds`, `${r.wall_angle}° wall`,
      `difficulty ${r.difficulty.toFixed(1)}`, `${(r.elapsed_ms / 1000).toFixed(1)}s`,
    ].map((t) => `<span>${t}</span>`).join("<span>·</span>");

    // Grade chips.
    el.gradeGrid.replaceChildren(...GRADES.map((g) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = g;
      b.className = (g === r.v_grade ? "pred " : "") + (g === state.actual ? "sel" : "");
      b.addEventListener("click", () => submitFeedback(g));
      return b;
    }));
    renderVerdict();
    if (fresh) spinGrade(r.v_grade);
    else el.rGrade.textContent = r.v_grade;
  }

  function spinGrade(target) {
    // Slot-machine reveal: cycle grades, decelerate, land on the answer.
    const t = gradeIdx(target);
    const seq = [];
    let i = 0;
    while (seq.length < 14) { seq.push(GRADES[i % 14]); i += 1; }
    seq.push(target);
    let k = 0;
    const step = () => {
      el.rGrade.textContent = seq[k];
      k += 1;
      if (k < seq.length) setTimeout(step, 30 + (k / seq.length) ** 2 * 140);
      else el.rGrade.textContent = target;
    };
    if (matchMedia("(prefers-reduced-motion: reduce)").matches || t < 0) el.rGrade.textContent = target;
    else step();
  }

  async function submitFeedback(g) {
    if (!state.result) return;
    state.actual = g;
    renderResult();
    if (!state.result.submission_id) return;
    try {
      await api("/api/feedback", { submission_id: state.result.submission_id, actual_grade: g });
    } catch (e) {
      toast(e.message, true);
    }
  }

  function verdictFor() {
    const r = state.result;
    if (!r || !state.actual) return null;
    const d = gradeIdx(r.v_grade) - gradeIdx(state.actual);
    const n = Math.abs(d);
    const grades = `${n} grade${n === 1 ? "" : "s"}`;
    if (d > 0) return { cls: "soft", head: n >= 3 ? "Soft. Very soft." : "Soft.", body: `Your gym calls it ${state.actual}. The model sees ${r.v_grade} — ${grades} harder.` };
    if (d < 0) return { cls: "stiff", head: n >= 3 ? "Sandbag alert." : "Stiff.", body: `Your gym calls it ${state.actual}. The model sees ${r.v_grade} — ${grades} easier.` };
    return { cls: "fair", head: "Honest gym.", body: `Setter and model agree on ${r.v_grade}.` };
  }

  function renderVerdict() {
    const v = verdictFor();
    el.verdict.classList.toggle("hidden", !v);
    el.verdict.className = "verdict" + (v ? ` ${v.cls}` : " hidden");
    if (v) el.verdict.innerHTML = `<b>${v.head}</b>${v.body}`;
  }

  el.copy.addEventListener("click", async () => {
    const r = state.result;
    if (!r) return;
    const v = verdictFor();
    const text = v
      ? `${v.head} ${v.body} — graded by v2inmygym.net`
      : `The model says this route is ${r.v_grade} (somewhere ${r.low}–${r.high}). — v2inmygym.net`;
    try {
      await navigator.clipboard.writeText(text);
      toast("Copied to clipboard.");
    } catch {
      toast("Couldn't access the clipboard.", true);
    }
  });
  el.again.addEventListener("click", () => {
    resetToEmpty();
    document.getElementById("studio").scrollIntoView({ behavior: "smooth", block: "start" });
  });

  // ── Stage controls ──────────────────────────────────────────────
  el.undo.addEventListener("click", () => {
    const last = state.holds[state.holds.length - 1];
    if (last) removeHold(last);
  });
  el.clear.addEventListener("click", () => {
    state.holds = [];
    state.result = null;
    el.hint.classList.remove("hidden");
    render();
  });
  el.change.addEventListener("click", () => el.file.click());
  $("btn-upload").addEventListener("click", () => el.file.click());
  $("btn-camera").addEventListener("click", () => el.fileCam.click());
  el.file.addEventListener("change", () => { loadPhoto(el.file.files[0]); el.file.value = ""; });
  el.fileCam.addEventListener("change", () => { loadPhoto(el.fileCam.files[0]); el.fileCam.value = ""; });
  document.querySelectorAll("[data-demo]").forEach((b) => b.addEventListener("click", () => loadDemo(b.dataset.demo)));
  $("cta-demo").addEventListener("click", () => loadDemo("steep"));

  // Drag & drop anywhere on the stage.
  ["dragenter", "dragover"].forEach((t) => el.stage.addEventListener(t, (ev) => {
    ev.preventDefault();
    el.stage.classList.add("drag");
  }));
  ["dragleave", "drop"].forEach((t) => el.stage.addEventListener(t, (ev) => {
    ev.preventDefault();
    el.stage.classList.remove("drag");
  }));
  el.stage.addEventListener("drop", (ev) => {
    const f = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0];
    if (f) loadPhoto(f);
  });
  // Paste an image from the clipboard.
  document.addEventListener("paste", (ev) => {
    const item = [...(ev.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) loadPhoto(item.getAsFile());
  });
  // Keyboard: undo.
  document.addEventListener("keydown", (ev) => {
    if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "z" && state.holds.length && !/input|textarea/i.test(ev.target.tagName)) {
      ev.preventDefault();
      removeHold(state.holds[state.holds.length - 1]);
    }
  });

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(layout, 80);
  });

  // ── Init ────────────────────────────────────────────────────────
  setAngle(30);
  render();
  pollHealth();
})();
