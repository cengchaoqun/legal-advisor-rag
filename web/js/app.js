// 单应用路由与全局状态：三个业务页面，无训练/实验/模型切换入口
import { api, state, toast } from "./api.js";
import { initChat, loadSessions, newSession, openSession } from "./chat.js";
import { initDocuments, loadDocuments } from "./documents.js";
import { initGraph, loadEntities } from "./graph.js";

const TEMPLATES = {
  chat: "tpl-chat",
  documents: "tpl-documents",
  graph: "tpl-graph",
};

let currentRoute = null;

async function router() {
  const raw = location.hash.replace(/^#\/?/, "") || "chat";
  const route = TEMPLATES[raw] ? raw : "chat";
  if (route === currentRoute) return;
  currentRoute = route;
  if (state.stream) {
    state.stream.close();
    state.stream = null;
  }
  const template = document.getElementById(TEMPLATES[route]);
  const view = document.getElementById("view");
  view.innerHTML = "";
  view.appendChild(template.content.cloneNode(true));
  document.querySelectorAll("[data-nav]").forEach((link) => {
    link.classList.toggle("active", link.dataset.nav === route);
  });

  if (route === "chat") {
    initChat();
    await loadSessions();
    if (state.sessions.length && !state.sessionId) {
      state.sessionId = state.sessions[0].session_id;
      await openSession(state.sessionId);
    } else if (!state.sessions.length) {
      await newSession();
    }
  } else if (route === "documents") {
    initDocuments();
    await loadDocuments(true);
  } else {
    initGraph();
    await loadEntities(true);
  }
}

async function refreshHealth() {
  const badge = document.getElementById("mode-badge");
  const banner = document.getElementById("banner");
  try {
    const res = await fetch("/health/ready");
    const data = await res.json();
    const status = data.status;
    const labels = { ready: "依赖就绪", degraded: "降级运行", unready: "依赖未就绪" };
    badge.textContent = labels[status] || status;
    badge.className = `badge ${status === "ready" ? "badge-ok" : status === "degraded" ? "badge-warn" : "badge-danger"}`;
    const checks = data.checks || {};
    const missing = Object.entries(checks)
      .filter(([, value]) => value !== "ok")
      .map(([name, value]) => `${name}=${value}`);
    if (status !== "ready") {
      banner.textContent = `依赖状态：${missing.join("，") || "未知"}。咨询回答需要已部署的微调模型服务，缺失时不会用辅助模型代答。`;
      banner.classList.remove("hidden");
    } else {
      banner.classList.add("hidden");
    }
  } catch (err) {
    badge.textContent = "状态未知";
    badge.className = "badge badge-danger";
  }
}

document.getElementById("new-session").addEventListener("click", async () => {
  if (location.hash !== "#/chat") location.hash = "#/chat";
  await new Promise((resolve) => setTimeout(resolve, 30));
  try {
    await newSession();
  } catch (err) {
    toast(err.message || "新建会话失败");
  }
});

window.addEventListener("hashchange", router);
window.addEventListener("DOMContentLoaded", async () => {
  await refreshHealth();
  await router();
  setInterval(refreshHealth, 30000);
});
