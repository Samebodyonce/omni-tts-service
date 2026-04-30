const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const errorEl = document.getElementById("error");
const player = document.getElementById("player");
const downloadEl = document.getElementById("download");
const metaEl = document.getElementById("meta");

let lastObjectUrl = null;

// Tab switching
tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const name = tab.dataset.tab;
    tabs.forEach((t) => t.classList.toggle("active", t === tab));
    panels.forEach((p) => p.classList.toggle("active", p.dataset.panel === name));
    hide(errorEl);
  });
});

// Health polling
async function refreshHealth() {
  try {
    const r = await fetch("/health");
    const j = await r.json();
    const labels = {
      ok: j.mock ? "готово · mock" : `готово · ${j.mode}`,
      loading: "загрузка модели…",
      error: "ошибка загрузки",
    };
    statusEl.textContent = labels[j.status] || j.status;
    statusEl.dataset.state = j.status;
  } catch {
    statusEl.textContent = "сервер недоступен";
    statusEl.dataset.state = "error";
  }
}
refreshHealth();
setInterval(refreshHealth, 5000);

// Form submission
document.querySelectorAll("form[data-mode]").forEach((form) => {
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hide(errorEl);
    hide(resultEl);

    const mode = form.dataset.mode;
    const fd = new FormData(form);
    fd.append("mode", mode);

    const btn = form.querySelector("button[type=submit]");
    const originalLabel = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Синтез…";

    const t0 = performance.now();
    try {
      const r = await fetch("/tts/generate?fmt=wav", { method: "POST", body: fd });
      if (!r.ok) {
        const msg = await r.json().catch(() => ({ error: r.statusText }));
        throw new Error(msg.error || `HTTP ${r.status}`);
      }
      const blob = await r.blob();
      if (lastObjectUrl) URL.revokeObjectURL(lastObjectUrl);
      lastObjectUrl = URL.createObjectURL(blob);

      player.src = lastObjectUrl;
      downloadEl.href = lastObjectUrl;
      downloadEl.download = `tts_${mode}_${Date.now()}.wav`;
      const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
      const kb = (blob.size / 1024).toFixed(1);
      metaEl.textContent = `режим: ${mode} · ${kb} КБ · ${elapsed} с`;
      show(resultEl);
      player.play().catch(() => {});
    } catch (err) {
      errorEl.textContent = `Ошибка: ${err.message}`;
      show(errorEl);
    } finally {
      btn.disabled = false;
      btn.textContent = originalLabel;
    }
  });
});

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }
