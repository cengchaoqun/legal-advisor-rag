// 统一请求封装：解析错误封套，保留 request_id 便于排查
export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.status = status;
    this.detail = detail || {};
  }
}

export async function api(path, options = {}) {
  const { method = "GET", body, form } = options;
  const init = { method, headers: {} };
  if (form) {
    init.body = form;
  } else if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const res = await fetch(path, init);
  const isJson = (res.headers.get("content-type") || "").includes("application/json");
  const data = isJson ? await res.json().catch(() => null) : await res.text();
  if (!res.ok) {
    const detail = (data && data.error) || {};
    throw new ApiError(detail.message || `请求失败（HTTP ${res.status}）`, res.status, detail);
  }
  return data;
}

export function uuid() {
  if (crypto && crypto.randomUUID) return crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

export const state = {
  sessionId: null,
  sessions: [],
  session: null,
  query: null,
  stream: null,
  evidence: new Map(),
  lastSeq: 0,
  running: false,
  docCursor: null,
  docFilters: { q: "", doc_type: "", availability: "" },
  entityCursor: null,
};

export function toast(message, ms = 2600) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.classList.remove("hidden");
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.classList.add("hidden"), ms);
}

export function fmtTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString("zh-CN", { hour12: false });
}

export function escapeHtml(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
