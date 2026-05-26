// Prudentia RAG demo UI controller.
// Vanilla DOM + fetch; no framework. The [doc-N] markers in the answer are
// turned into click targets that load the cited source PDF into the right
// pane at the cited page.
//
// PDF rendering: by default uses Mozilla's pdf.js viewer (consistent across
// Chrome, Safari, Firefox, Edge). The viewer is loaded from a CDN; set
// `window.PRUDENTIA_PDF_VIEWER = "native"` BEFORE this script loads to
// fall back to the browser's built-in iframe rendering (e.g. for air-gapped
// demos that cannot reach a CDN).
//
// XSS posture: we never assemble HTML by string concatenation. Every piece
// of server-returned text reaches the DOM through textContent. The only
// element types created are <a>, <iframe>, <option>, <p>, and <code>; all
// attribute values that flow from user/server data (href, src, data-*) are
// passed through encodeURIComponent or set via setAttribute on whitelisted
// keys.

const REFUSAL_PREFIX = "I cannot answer this question from the provided context.";
const DOC_TAG_RE = /\[doc-(\d+)\]/g;
// Models whose names start with "claude" are the production path. Anything
// else — gpt-oss-*, llama*, qwen*, mistral*, phi* — triggers the yellow
// "running on local model" banner.
const PRODUCTION_MODEL_PREFIX = "claude";
const BANNER_DISMISS_KEY = "prudentia-rag.banner-dismissed";

// PDF rendering strategy:
//   "native" (default): use the browser's built-in PDF viewer via #page=N.
//                       Works on Chrome and Safari out of the box; Firefox
//                       and Edge mobile are inconsistent on the page anchor.
//   "pdfjs":            use a Mozilla pdf.js viewer (consistent everywhere).
//                       Requires the viewer to be SAME-ORIGIN (mozilla.github.io
//                       enforces CORS on the file param); vendor pdf.js under
//                       /static/pdfjs/ and set PRUDENTIA_PDF_VIEWER_BASE.
//
// To opt into pdf.js, set BEFORE app.js loads:
//     window.PRUDENTIA_PDF_VIEWER = "pdfjs";
//     window.PRUDENTIA_PDF_VIEWER_BASE = "/static/pdfjs/web/viewer.html";
const PDF_VIEWER_MODE =
  (typeof window !== "undefined" && window.PRUDENTIA_PDF_VIEWER) || "native";
const PDF_VIEWER_BASE =
  (typeof window !== "undefined" && window.PRUDENTIA_PDF_VIEWER_BASE) || null;

function buildPdfViewerUrl(collection, docid, page) {
  const fileParam = `/document/${encodeURIComponent(collection)}/${encodeURIComponent(docid)}`;
  if (PDF_VIEWER_MODE !== "pdfjs" || !PDF_VIEWER_BASE) {
    // Native browser PDF viewer with the standard #page anchor.
    return `${fileParam}#page=${page}`;
  }
  const absoluteFile = new URL(fileParam, window.location.origin).toString();
  return `${PDF_VIEWER_BASE}?file=${encodeURIComponent(absoluteFile)}#page=${page}`;
}

const $ = (id) => document.getElementById(id);

const els = {
  form: $("ask"),
  collection: $("collection"),
  query: $("query"),
  k: $("k"),
  submit: $("submit"),
  status: $("status"),
  answer: $("answer"),
  meta: $("meta"),
  latency: $("latency"),
  model: $("model"),
  tokens: $("tokens"),
  request: $("request"),
  panel: $("panel"),
  panelTitle: $("panelTitle"),
  closePanel: $("closePanel"),
  themeToggle: $("themeToggle"),
  themeIcon: $("themeIcon"),
  providerBanner: $("providerBanner"),
  providerBannerText: $("providerBannerText"),
  providerBannerDismiss: $("providerBannerDismiss"),
};

// ─── Provider banner ──────────────────────────────────────────────────────
// Shown when the resolved model is NOT a Claude variant. Dismissed-per-session
// so the operator isn't nagged after they've acknowledged the local-model
// quality drop. Returns true if the banner ended up visible.
function maybeShowProviderBanner(model) {
  if (!els.providerBanner) return false;
  const isLocal = typeof model === "string" &&
    !model.toLowerCase().startsWith(PRODUCTION_MODEL_PREFIX);
  if (!isLocal) {
    els.providerBanner.classList.add("hidden");
    return false;
  }
  if (sessionStorage.getItem(BANNER_DISMISS_KEY) === "1") {
    els.providerBanner.classList.add("hidden");
    return false;
  }
  // textContent only — never inject the model name as HTML.
  els.providerBannerText.textContent =
    `Running on local model (${model}) — quality and latency lower than production.`;
  els.providerBanner.classList.remove("hidden");
  return true;
}

function dismissProviderBanner() {
  sessionStorage.setItem(BANNER_DISMISS_KEY, "1");
  if (els.providerBanner) els.providerBanner.classList.add("hidden");
}

// ─── Theme handling ─────────────────────────────────────────────────────────
// Persist the operator's choice in localStorage. If no choice has been made,
// fall back to the prefers-color-scheme system preference (handled by CSS).

const THEME_KEY = "prudentia-rag.theme";

function applyTheme(theme) {
  // theme is "light", "dark", or null (system default)
  if (theme === "light" || theme === "dark") {
    document.body.setAttribute("data-theme", theme);
  } else {
    document.body.removeAttribute("data-theme");
  }
  if (els.themeIcon) {
    // Show the icon for the OPPOSITE theme (i.e. what you'll get on click).
    const isLight = (theme === "light") ||
      (theme === null && window.matchMedia("(prefers-color-scheme: light)").matches);
    els.themeIcon.textContent = isLight ? "◑" : "◐";
  }
}

function toggleTheme() {
  const current = document.body.getAttribute("data-theme");
  let next;
  if (current === "light") next = "dark";
  else if (current === "dark") next = "light";
  else {
    // No explicit choice yet — flip relative to the system preference.
    next = window.matchMedia("(prefers-color-scheme: light)").matches ? "dark" : "light";
  }
  localStorage.setItem(THEME_KEY, next);
  applyTheme(next);
}

applyTheme(localStorage.getItem(THEME_KEY));

let lastResponse = null;

async function loadCollections() {
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error(`/health returned ${res.status}`);
    const body = await res.json();
    const cols = body.collections || [];

    clearChildren(els.collection);
    if (cols.length === 0) {
      const opt = document.createElement("option");
      opt.disabled = true;
      opt.selected = true;
      opt.textContent = "(none on disk)";
      els.collection.appendChild(opt);
    } else {
      cols.forEach((c) => {
        const opt = document.createElement("option");
        opt.value = c;
        opt.textContent = c;
        els.collection.appendChild(opt);
      });
    }
    els.status.textContent = `OK · ${cols.length} collection${cols.length === 1 ? "" : "s"}`;
  } catch (err) {
    els.status.textContent = `Unreachable: ${err.message}`;
    els.status.style.color = "var(--refusal)";
  }
}

function setBusy(isBusy) {
  els.submit.disabled = isBusy;
  els.submit.textContent = isBusy ? "Asking…" : "Ask";
}

function clearChildren(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

// Build the answer DOM: walk the raw answer string, emit text nodes for the
// prose and <a class="cite"> elements for each [doc-N] match. Nothing reaches
// the DOM as HTML; only textContent and setAttribute.
function renderAnswerWithCitations(parent, answer, citations) {
  clearChildren(parent);
  let cursor = 0;
  for (const match of answer.matchAll(DOC_TAG_RE)) {
    if (match.index > cursor) {
      parent.appendChild(document.createTextNode(answer.slice(cursor, match.index)));
    }
    const n = parseInt(match[1], 10);
    if (n >= 1 && n <= citations.length) {
      const c = citations[n - 1];
      const a = document.createElement("a");
      a.className = "cite";
      a.href = "#";
      a.dataset.n = String(n);
      a.dataset.docid = c.docid;
      a.textContent = `[doc-${n}]`;
      a.addEventListener("click", (e) => {
        e.preventDefault();
        openCitation(n, a);
      });
      parent.appendChild(a);
    } else {
      // Defensive: model invented a tag index outside the citations array.
      parent.appendChild(document.createTextNode(match[0]));
    }
    cursor = match.index + match[0].length;
  }
  if (cursor < answer.length) {
    parent.appendChild(document.createTextNode(answer.slice(cursor)));
  }
}

function isRefusal(answer) {
  return answer.trim().startsWith(REFUSAL_PREFIX);
}

function setAnswer(answer, citations) {
  els.answer.classList.remove("empty");
  if (isRefusal(answer)) {
    els.answer.classList.add("refusal");
    clearChildren(els.answer);
    els.answer.textContent = answer;
    return;
  }
  els.answer.classList.remove("refusal");
  renderAnswerWithCitations(els.answer, answer, citations);
}

function showPlaceholder(text) {
  clearChildren(els.answer);
  const p = document.createElement("p");
  p.className = "placeholder";
  p.textContent = text;
  els.answer.appendChild(p);
  els.answer.classList.add("empty");
  els.answer.classList.remove("refusal");
}

function openCitation(n, anchorEl) {
  if (!lastResponse) return;
  const c = lastResponse.citations[n - 1];
  if (!c) return;

  document
    .querySelectorAll("a.cite.active")
    .forEach((el) => el.classList.remove("active"));
  if (anchorEl) anchorEl.classList.add("active");

  const collection = els.collection.value;
  const page = c.page_range && c.page_range[0] ? c.page_range[0] : 1;
  const src = buildPdfViewerUrl(collection, c.docid, page);

  els.panel.classList.remove("empty");
  els.panelTitle.textContent = `${c.source_pdf} · p.${page}`;
  els.closePanel.hidden = false;

  clearChildren(els.panel);
  const iframe = document.createElement("iframe");
  iframe.setAttribute("src", src);
  iframe.setAttribute("title", "source PDF");
  // pdf.js needs same-origin storage for state; allow it on the iframe.
  iframe.setAttribute("allow", "fullscreen");
  els.panel.appendChild(iframe);
}

function closePanel() {
  els.panel.classList.add("empty");
  clearChildren(els.panel);
  const p = document.createElement("p");
  p.className = "placeholder";
  p.appendChild(document.createTextNode("Click any "));
  const code = document.createElement("code");
  code.textContent = "[doc-N]";
  p.appendChild(code);
  p.appendChild(document.createTextNode(" citation in the answer to load the source PDF here."));
  els.panel.appendChild(p);
  els.panelTitle.textContent = "Source preview";
  els.closePanel.hidden = true;
  document
    .querySelectorAll("a.cite.active")
    .forEach((el) => el.classList.remove("active"));
}

async function submit(event) {
  event.preventDefault();
  const query = els.query.value.trim();
  const collection = els.collection.value;
  const k = parseInt(els.k.value, 10) || 5;
  if (!query || !collection) return;

  setBusy(true);
  els.meta.classList.add("hidden");
  showPlaceholder("Running retrieval + synthesis…");

  try {
    const t0 = performance.now();
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, collection, k }),
    });
    const clientLatencyMs = Math.round(performance.now() - t0);

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`${res.status}: ${text}`);
    }

    const body = await res.json();
    lastResponse = body;
    maybeShowProviderBanner(body.model);
    setAnswer(body.answer, body.citations);

    els.latency.textContent = `server ${body.latency_ms} ms · roundtrip ${clientLatencyMs} ms`;
    els.model.textContent = body.model;
    els.tokens.textContent = `${body.prompt_tokens}→${body.completion_tokens} tokens`;
    els.request.textContent = body.request_id;
    els.meta.classList.remove("hidden");
  } catch (err) {
    els.answer.classList.remove("empty");
    els.answer.classList.add("refusal");
    clearChildren(els.answer);
    els.answer.textContent = `Error: ${err.message}`;
    lastResponse = null;
  } finally {
    setBusy(false);
  }
}

els.form.addEventListener("submit", submit);
els.closePanel.addEventListener("click", closePanel);
if (els.themeToggle) els.themeToggle.addEventListener("click", toggleTheme);
if (els.providerBannerDismiss) {
  els.providerBannerDismiss.addEventListener("click", dismissProviderBanner);
}
loadCollections();
