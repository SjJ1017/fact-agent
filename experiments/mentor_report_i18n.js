/* 按 L 在中英之间切换。

   两种语言都在页面里，用 CSS 显示其一，所以切换是瞬时的、可打印的，
   而且不依赖运行时查表——没有"某句没匹配上"这种失败模式。

   选择偏好写进 localStorage：读者切过一次之后，重新打开还是那个语言。 */
(function () {
  const KEY = "factflow-report-lang";
  const root = document.documentElement;

  const css = document.createElement("style");
  css.textContent = `
:root[data-lang="en"] .zh, :root:not([data-lang="en"]) .en { display: none; }
#lang-hint { position: fixed; right: 14px; bottom: 14px; z-index: 60;
  font: 500 11px/1 "IBM Plex Mono", monospace; letter-spacing: .06em;
  color: var(--faint, #9aa0a6); background: var(--panel, #fff);
  border: 1px solid var(--edge, #e0dfd9); border-radius: 6px;
  padding: 7px 10px; opacity: .75; transition: opacity .2s; pointer-events: none; }
#lang-hint b { color: var(--ink, #1b1f24); font-weight: 600; }
@media print { #lang-hint { display: none; } }`;
  document.head.appendChild(css);

  let lang;
  try { lang = localStorage.getItem(KEY); } catch (e) { lang = null; }
  const set = (v) => {
    root.dataset.lang = v;
    root.setAttribute("lang", v === "en" ? "en" : "zh-Hans");
    try { localStorage.setItem(KEY, v); } catch (e) { /* private window */ }
    const h = document.getElementById("lang-hint");
    if (h) h.innerHTML = v === "en"
      ? 'press <b>L</b> for 中文' : '按 <b>L</b> 切换 English';
  };

  const hint = document.createElement("div");
  hint.id = "lang-hint";
  document.addEventListener("DOMContentLoaded", () => document.body.appendChild(hint));
  if (document.body) document.body.appendChild(hint);

  set(lang === "en" ? "en" : "zh");

  addEventListener("keydown", (e) => {
    // A bare letter must not fire while someone is typing or holding a
    // modifier for a browser shortcut.
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const t = e.target;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    if (e.key === "l" || e.key === "L") {
      e.preventDefault();
      set(root.dataset.lang === "en" ? "zh" : "en");
    }
  });
})();
