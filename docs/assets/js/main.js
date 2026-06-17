/* ============================================================
   MoDA project page interactions
   ============================================================ */
(function () {
  "use strict";
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $  = (s, c = document) => c.querySelector(s);
  const $$ = (s, c = document) => Array.from(c.querySelectorAll(s));

  /* ---------- 1. KaTeX equations ---------- */
  function renderMath() {
    if (!window.katex) return;
    $$(".eq[data-tex]").forEach((el) => {
      try {
        window.katex.render(el.dataset.tex, el, {
          displayMode: true,
          throwOnError: false,
        });
      } catch (e) { /* keep raw text on failure */ }
    });
  }

  /* ---------- 2. The Modulator (hero signature) ----------
     The mask sigma(W . F(T,V)) has the shape of the visual features
     V_aligned in (B, N, E): a per-token, per-channel modulation.
     We render it as an N x E matrix (rows = visual tokens, cols =
     embedding channels); the batch B is the stacked sheets behind. */
  const N_ROWS = 7;    // visual tokens (N)
  const N_COLS = 28;   // embedding channels (E)

  // The mask is a single channel-wise vector of size E (mask ∈ [0,1]^E),
  // produced by F(T, V) and broadcast across all N tokens. It is conditioned
  // on the instruction, so different instructions give different masks.
  // CLIP/SigLIP channels are entangled (no clean per-concept grouping), so the
  // selected channels are DISTRIBUTED across E rather than contiguous bands.
  // Exact values are illustrative; the structure (size-E, [0,1], instruction-
  // conditioned, distributed, token-broadcast) is faithful to the paper.
  // The mask M = σ(W·F(T,V)) has shape (N, E): each visual token cross-attends
  // the instruction and gets its OWN channel-wise gate, so rows (tokens) differ.
  // 2D gaussian blobs over (token, channel) give focused, per-token selections.
  function mask2D(blobs) {
    const m = new Array(N_ROWS * N_COLS);
    for (let r = 0; r < N_ROWS; r++) {
      for (let c = 0; c < N_COLS; c++) {
        let v = 0.06;
        blobs.forEach((b) => {
          const dr = (r - b.r) / b.sr, dc = (c - b.c) / b.sc;
          v += b.w * Math.exp(-0.5 * (dr * dr + dc * dc));
        });
        m[r * N_COLS + c] = Math.min(1, v);
      }
    }
    return m;
  }

  // deterministic base visual features V (N x E magnitudes)
  const baseV = [];
  for (let r = 0; r < N_ROWS; r++) {
    const row = [];
    for (let c = 0; c < N_COLS; c++) {
      const v = 0.62 + 0.3 * Math.sin(r * 1.3 + c * 0.7) + 0.12 * Math.sin(c * 0.31 + 2);
      row.push(Math.min(1, Math.max(0.45, v)));
    }
    baseV.push(row);
  }

  // each instruction -> a focused, per-token selection over (N tokens × E channels).
  // Blob centers stay within bounds (rows 1..5 of 0..6, cols 4..23 of 0..27)
  // so no focal region clips the edge; each instruction is at a distinct spot.
  const PROMPTS = [
    { t: "What color is the dog’s ear?",
      mask: mask2D([{ r: 1.5, c: 6, sr: 1.5, sc: 2.1, w: 1 }, { r: 4.5, c: 18, sr: 1.6, sc: 2, w: 0.68 }]) },
    { t: "Is the toy on the bed or the floor?",
      mask: mask2D([{ r: 3, c: 13, sr: 2.1, sc: 2.5, w: 1 }, { r: 1, c: 22, sr: 1.3, sc: 1.7, w: 0.6 }]) },
    { t: "How many cats are in the photo?",
      mask: mask2D([{ r: 2, c: 5, sr: 1.5, sc: 1.9, w: 0.78 }, { r: 5, c: 20, sr: 1.6, sc: 2, w: 1 }]) },
    { t: "Which key is highlighted on the keyboard?",
      mask: mask2D([{ r: 1.3, c: 11, sr: 1.5, sc: 2, w: 1 }, { r: 5, c: 23, sr: 1.4, sc: 1.8, w: 0.7 }]) },
  ];

  // diverging colormap for mask values in [0,1]:
  // 0 -> teal (suppress), 0.5 -> light neutral, 1 -> amber (boost).
  // Routing through a neutral midpoint avoids the muddy olive of a direct lerp.
  const C_LOW = [38, 132, 142], C_MID = [244, 242, 236], C_HIGH = [221, 122, 12];
  function cmap(v) {
    v = v < 0 ? 0 : v > 1 ? 1 : v;
    let a, b, t;
    if (v < 0.5) { a = C_LOW; b = C_MID; t = v / 0.5; }
    else { a = C_MID; b = C_HIGH; t = (v - 0.5) / 0.5; }
    const c = a.map((x, i) => Math.round(x + (b[i] - x) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  function buildModulator() {
    const gridEl = $("#modGrid");
    const promptEl = $("#modPrompt");
    if (!gridEl || !promptEl) return;

    gridEl.style.setProperty("--cols", N_COLS);

    // the per-token mask map M (N x E): one channel-wise gate per visual token
    const cells = [];
    for (let r = 0; r < N_ROWS; r++) {
      for (let c = 0; c < N_COLS; c++) {
        const el = document.createElement("div");
        el.className = "cell";
        el.style.transitionDelay = (c * 6 + r * 10) + "ms";
        gridEl.appendChild(el);
        cells.push(el);
      }
    }
    const GLOW = "0 0 7px -1px rgba(221,122,12,.65)";
    function applyMask(mask) {
      // each cell (token r, channel c) is its OWN gate value in [0,1]:
      // suppressed cells fade back, the focused regions light up.
      for (let r = 0; r < N_ROWS; r++) {
        for (let c = 0; c < N_COLS; c++) {
          const i = r * N_COLS + c;
          const m = mask[i];
          cells[i].style.backgroundColor = cmap(m);
          cells[i].style.opacity = (0.24 + 0.7 * m + 0.06 * baseV[r][c] * m).toFixed(2);
          cells[i].style.boxShadow = m > 0.62 ? GLOW : "none";
        }
      }
    }

    let idx = 0, typeTimer = null;

    function typeOut(text) {
      promptEl.textContent = "";
      let i = 0;
      clearInterval(typeTimer);
      typeTimer = setInterval(() => {
        promptEl.textContent = text.slice(0, ++i);
        if (i >= text.length) clearInterval(typeTimer);
      }, 26);
    }

    function cycle() {
      const p = PROMPTS[idx];
      applyMask(p.mask);
      typeOut(p.t);
      idx = (idx + 1) % PROMPTS.length;
    }

    if (reduceMotion) {
      promptEl.textContent = PROMPTS[0].t;
      applyMask(PROMPTS[0].mask);
      return;
    }
    cycle();
    setInterval(cycle, 3800);
  }

  /* ---------- 3. Interactive patch grid ---------- */
  const PATCHES = [
    { id: 1, d: "Cushion edge + bedding fabric" },
    { id: 2, d: "The dog’s back + folds of the blanket" },
    { id: 3, d: "Hardwood floor + the border of the bed" },
    { id: 4, d: "The plush toy’s limbs + dark cushion" },
    { id: 5, d: "The dog’s torso + the plush toy + the cushioned bed", flag: true },
    { id: 6, d: "The dog’s head + ear + hardwood floor", flag: true },
    { id: 7, d: "The plush toy’s legs + floor grain" },
    { id: 8, d: "Plush toy body + the dog’s paw + bedding" },
    { id: 9, d: "Hardwood floor planks" },
  ];
  function buildPatches() {
    const overlay = $("#patchOverlay");
    const info = $("#patchInfo");
    if (!overlay || !info) return;
    const idEl = $(".patchinfo__id", info);
    const descEl = $(".patchinfo__desc", info);
    const defaultDesc = descEl.textContent;

    function show(p) {
      idEl.textContent = "Patch " + p.id;
      descEl.innerHTML = p.d + (p.flag ? ' <span class="patchinfo__tag">cited in paper</span>' : "");
      info.classList.toggle("is-flagged", !!p.flag);
    }
    function reset() {
      idEl.textContent = "Patch";
      descEl.textContent = defaultDesc;
      info.classList.remove("is-flagged");
    }

    PATCHES.forEach((p) => {
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "patchcell" + (p.flag ? " is-flagged" : "");
      cell.setAttribute("aria-label", "Patch " + p.id + ": " + p.d);
      cell.innerHTML = `<span class="patchcell__id">${p.id}</span>`;
      cell.addEventListener("mouseenter", () => show(p));
      cell.addEventListener("focus", () => show(p));
      cell.addEventListener("click", () => show(p));
      overlay.appendChild(cell);
    });
    overlay.addEventListener("mouseleave", reset);
  }

  /* ---------- 4. Results tables ---------- */
  const TABLES = {
    vqa: {
      cols: ["GQA", "ScienceQA", "MMBench", "RealWorldQA", "ChartQA"],
      groups: [
        { name: "LLaVA-1.5", llm: "Vicuna-7B", base: [62.4, 69.0, 64.3, 44.3, 17.0], moda: [62.5, 71.0, 64.8, 53.4, 13.2] },
        { name: "LLaVA-MoRE · OpenAI CLIP", llm: "LLaMA 3.1-8B", base: [63.6, 76.3, 72.3, 57.1, 15.5], moda: [64.4, 77.8, 72.0, 58.0, 15.6] },
        { name: "LLaVA-MoRE · SigLIP-S2", llm: "LLaMA 3.1-8B", base: [64.9, 77.1, 71.8, 57.2, 17.3], moda: [65.4, 81.9, 72.4, 58.2, 18.1] },
        { name: "Qwen3-VL-2B-Instruct", llm: "Qwen3-2B", base: [59.4, 79.3, 86.5, 64.7, 80.0], moda: [63.2, 84.2, 87.4, 68.8, 79.0] },
      ],
    },
    vision: {
      cols: ["LLaVA-Wild", "MM-Vet", "MMStar", "V*Bench", "CV-Bench"],
      groups: [
        { name: "LLaVA-1.5", llm: "Vicuna-7B", base: [65.4, 28.1, 27.6, 42.9, 59.0], moda: [68.0, 29.9, 32.9, 44.5, 58.2] },
        { name: "LLaVA-MoRE · OpenAI CLIP", llm: "LLaMA 3.1-8B", base: [71.2, 25.2, 35.7, 42.8, 59.9], moda: [73.9, 26.6, 36.7, 44.0, 61.0] },
        { name: "LLaVA-MoRE · SigLIP-S2", llm: "LLaMA 3.1-8B", base: [72.0, 27.7, 35.8, 44.4, 61.2], moda: [67.6, 28.3, 38.5, 44.8, 62.2] },
        { name: "Qwen3-VL-2B-Instruct", llm: "Qwen3-2B", base: [null, 51.9, 53.9, 77.0, 80.9], moda: [null, 52.0, 55.3, 74.9, 81.0] },
      ],
    },
    hallu: {
      cols: ["POPE", "MMVP"],
      groups: [
        { name: "LLaVA-1.5", llm: "Vicuna-7B", base: [85.6, 24.0], moda: [87.1, 36.0] },
        { name: "LLaVA-MoRE · OpenAI CLIP", llm: "LLaMA-8B", base: [85.1, 27.3], moda: [86.3, 28.7] },
        { name: "LLaVA-MoRE · SigLIP-S2", llm: "LLaMA-8B", base: [86.0, 39.3], moda: [87.7, 42.7] },
        { name: "Qwen3-VL-2B-Instruct", llm: "Qwen3-2B", base: [89.4, null], moda: [89.9, null] },
      ],
    },
  };

  function colMaxes(t) {
    return t.cols.map((_, c) => {
      let mx = -Infinity;
      t.groups.forEach((g) => {
        [g.base[c], g.moda[c]].forEach((v) => { if (v != null && v > mx) mx = v; });
      });
      return mx;
    });
  }
  function fmt(v) { return v == null ? null : v.toFixed(1); }

  function renderTable(key) {
    const t = TABLES[key];
    const maxes = colMaxes(t);
    let head = `<tr><th>Method</th><th class="llm">LLM</th>${t.cols.map((c) => `<th>${c}</th>`).join("")}</tr>`;
    let body = "";
    t.groups.forEach((g) => {
      // baseline row
      body += `<tr class="grp-top"><td class="model">${g.name}</td><td class="llm">${g.llm}</td>` +
        g.base.map((v, c) => {
          if (v == null) return `<td><span class="dash">-</span></td>`;
          const best = v === maxes[c] ? " best" : "";
          return `<td><span class="cellval"><span class="${best.trim()}">${fmt(v)}</span></span></td>`;
        }).join("") + `</tr>`;
      // moda row with deltas
      body += `<tr class="is-moda"><td class="model">+ <span class="moda-tag">MoDA</span></td><td class="llm">${g.llm}</td>` +
        g.moda.map((v, c) => {
          if (v == null) return `<td><span class="dash">-</span></td>`;
          const bv = g.base[c];
          const best = v === maxes[c] ? " best" : "";
          let badge = "";
          if (bv != null) {
            const d = +(v - bv).toFixed(1);
            if (d > 0) badge = `<span class="delta up">+${d.toFixed(1)}</span>`;
            else if (d < 0) badge = `<span class="delta down">${d.toFixed(1)}</span>`;
          }
          return `<td><span class="cellval">${badge}<span class="${best.trim()}">${fmt(v)}</span></span></td>`;
        }).join("") + `</tr>`;
    });
    return `<table class="res"><thead>${head}</thead><tbody>${body}</tbody></table>`;
  }

  function buildResults() {
    const wrap = $("#tablewrap");
    if (!wrap) return;
    wrap.innerHTML = renderTable("vqa");
    $$(".tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        $$(".tab").forEach((x) => { x.classList.remove("is-active"); x.setAttribute("aria-selected", "false"); });
        tab.classList.add("is-active"); tab.setAttribute("aria-selected", "true");
        wrap.innerHTML = renderTable(tab.dataset.tab);
      });
    });
  }

  /* ---------- 5. Scroll reveal ---------- */
  function buildReveal() {
    const els = $$(".reveal");
    if (reduceMotion || !("IntersectionObserver" in window)) {
      els.forEach((e) => e.classList.add("is-in")); return;
    }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (en.isIntersecting) { en.target.classList.add("is-in"); io.unobserve(en.target); }
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -8% 0px" });
    els.forEach((e) => io.observe(e));
  }

  /* ---------- 6. Count-up stats ---------- */
  function buildCounters() {
    const els = $$(".stat__num[data-count]");
    if (reduceMotion || !("IntersectionObserver" in window)) return;
    const io = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (!en.isIntersecting) return;
        const el = en.target, target = +el.dataset.count;
        let cur = 0; const step = Math.max(1, Math.round(target / 24));
        const tick = () => { cur = Math.min(target, cur + step); el.textContent = cur; if (cur < target) requestAnimationFrame(tick); };
        tick(); io.unobserve(el);
      });
    }, { threshold: 0.6 });
    els.forEach((e) => io.observe(e));
  }

  /* ---------- 7. Nav scrolled state ---------- */
  function buildNav() {
    const nav = $(".nav");
    if (!nav) return;
    const onScroll = () => nav.classList.toggle("is-scrolled", window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
  }

  /* ---------- 8. Copy BibTeX ---------- */
  function buildCopy() {
    const btn = $("#copyBib"), pre = $("#bibtex");
    if (!btn || !pre) return;
    btn.addEventListener("click", async () => {
      const txt = pre.innerText;
      try { await navigator.clipboard.writeText(txt); }
      catch (e) {
        const r = document.createRange(); r.selectNode(pre);
        const s = getSelection(); s.removeAllRanges(); s.addRange(r);
        try { document.execCommand("copy"); } catch (_) {} s.removeAllRanges();
      }
      const label = $("span", btn);
      btn.classList.add("is-copied"); if (label) label.textContent = "Copied";
      setTimeout(() => { btn.classList.remove("is-copied"); if (label) label.textContent = "Copy"; }, 1800);
    });
  }

  /* ---------- init ---------- */
  function init() {
    renderMath();
    buildModulator();
    buildPatches();
    buildResults();
    buildReveal();
    buildCounters();
    buildNav();
    buildCopy();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
