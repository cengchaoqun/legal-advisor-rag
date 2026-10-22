// 资料页面：DocumentListItem 列表、筛选、上传抽屉、任务轮询与重试
import { api, escapeHtml, fmtTime, state, toast, uuid } from "./api.js";

let els = {};
let pollTimer = null;

const AVAILABILITY = {
  available: { label: "可用", cls: "tag-ok" },
  limited: { label: "受限可用", cls: "tag-warn" },
  unavailable: { label: "不可用", cls: "tag-danger" },
};
const DOC_TYPE = {
  law: "法律法规",
  interpretation: "司法解释",
  case: "案例",
  guidance: "官方指引",
  unknown: "未识别",
};
const INDEX_STATE = {
  ready: { label: "就绪", dot: "dot-ok" },
  empty: { label: "无可确认关系", dot: "dot-warn" },
  failed: { label: "失败", dot: "dot-danger" },
  running: { label: "进行中", dot: "dot-warn" },
  pending: { label: "等待", dot: "" },
};

export function initDocuments() {
  els = {
    list: document.getElementById("doc-list"),
    q: document.getElementById("doc-q"),
    type: document.getElementById("doc-type"),
    availability: document.getElementById("doc-availability"),
    summary: document.getElementById("doc-summary"),
    more: document.getElementById("doc-more"),
    drawer: document.getElementById("drawer"),
    form: document.getElementById("upload-form"),
    taskPanel: document.getElementById("task-panel"),
    taskDetail: document.getElementById("task-detail"),
    taskActions: document.getElementById("task-actions"),
  };
  document.getElementById("open-upload").addEventListener("click", () => openDrawer());
  document.getElementById("close-drawer").addEventListener("click", () => closeDrawer());
  document.getElementById("refresh-docs").addEventListener("click", () => loadDocuments(true));
  els.form.addEventListener("submit", onUpload);
  els.more.addEventListener("click", () => loadDocuments(false));
  els.q.addEventListener("input", debounce(() => loadDocuments(true), 350));
  els.type.addEventListener("change", () => loadDocuments(true));
  els.availability.addEventListener("change", () => loadDocuments(true));
  els.list.addEventListener("click", onListClick);
}

function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

export async function loadDocuments(reset = true) {
  if (reset) {
    state.docCursor = null;
    state.docFilters = { q: els.q.value.trim(), doc_type: els.type.value, availability: els.availability.value };
  }
  const params = new URLSearchParams({ limit: "20" });
  if (state.docCursor) params.set("cursor", state.docCursor);
  if (state.docFilters.q) params.set("q", state.docFilters.q);
  if (state.docFilters.doc_type) params.set("doc_type", state.docFilters.doc_type);
  if (state.docFilters.availability) params.set("availability", state.docFilters.availability);
  try {
    const data = await api(`/api/documents?${params.toString()}`);
    if (reset) els.list.innerHTML = "";
    if (!data.items.length && reset) {
      els.list.innerHTML = '<div class="empty"><h3>还没有资料</h3><p class="muted">点击右上角「导入资料」上传文本型 PDF 或 Markdown。</p></div>';
    }
    for (const item of data.items) els.list.appendChild(renderRow(item));
    state.docCursor = data.next_cursor;
    els.more.classList.toggle("hidden", !data.next_cursor);
    const available = els.list.querySelectorAll('[data-availability="available"]').length;
    const limited = els.list.querySelectorAll('[data-availability="limited"]').length;
    els.summary.textContent = `当前加载 ${els.list.querySelectorAll(".doc-row").length} 条 · 可用 ${available} · 受限 ${limited}`;
  } catch (err) {
    if (err.status === 400) {
      state.docCursor = null;
      toast("游标与筛选条件不匹配，已重置列表");
      return loadDocuments(true);
    }
    toast(err.message || "加载资料失败");
  }
}

function renderRow(item) {
  const doc = item.document;
  const version = item.version;
  const task = item.latest_task;
  const availability = AVAILABILITY[item.availability] || AVAILABILITY.unavailable;
  const vectorState = version?.vector_state || task?.vector_state || "pending";
  const graphState = version?.graph_state || task?.graph_state || "pending";
  const quality = (version?.quality_flags || []).map((flag) => `<span class="tag">${escapeHtml(flagLabel(flag))}</span>`).join("");
  const row = document.createElement("div");
  row.className = "doc-row";
  row.dataset.documentId = doc.document_id;
  row.dataset.availability = item.availability;
  row.innerHTML = `
    <div>
      <h4>${escapeHtml(doc.title)}</h4>
      <div class="doc-meta">
        <span>${DOC_TYPE[doc.doc_type] || doc.doc_type}</span>
        <span>${version ? "处理指纹 " + escapeHtml(version.pipeline_revision.slice(0, 8)) : "尚无发布版本"}</span>
        <span>${version?.metadata?.document_number ? "文号 " + escapeHtml(version.metadata.document_number) : ""}</span>
        <span>${version?.metadata?.case_number ? "案号 " + escapeHtml(version.metadata.case_number) : ""}</span>
        <span>${version?.published_at ? "首次发布 " + fmtTime(version.published_at) : ""}</span>
      </div>
      <div class="doc-meta" style="margin-top:6px">${quality}</div>
    </div>
    <div class="index-state">
      <span class="tag ${availability.cls}">${availability.label}</span>
      <span><i class="dot ${INDEX_STATE[vectorState]?.dot || ""}"></i>向量索引：${INDEX_STATE[vectorState]?.label || vectorState}</span>
      <span><i class="dot ${INDEX_STATE[graphState]?.dot || ""}"></i>图谱索引：${INDEX_STATE[graphState]?.label || graphState}</span>
      <span>任务：${task ? escapeHtml(task.status) : "无"}</span>
    </div>
    <div class="doc-actions">
      ${version ? `<a class="btn btn-sm" href="/api/versions/${version.version_id}/file" target="_blank">原件</a>` : ""}
      ${task ? `<button class="btn btn-sm" data-action="task" data-task="${task.task_id}">任务详情</button>` : ""}
      ${task?.retryable ? `<button class="btn btn-sm" data-action="retry" data-task="${task.task_id}">重试</button>` : ""}
      <button class="btn btn-sm" data-action="withdraw">撤销</button>
    </div>`;
  return row;
}

function flagLabel(flag) {
  return {
    metadata_missing: "元数据缺失",
    effectivity_unknown: "效力未知",
    citation_unresolved: "引用未解析",
    graph_unavailable: "图谱不可用",
    source_access_limited: "来源受限",
    parser_partial: "解析不完整",
    citation_validation_failed: "引用校验失败",
  }[flag] || flag;
}

async function onListClick(event) {
  const button = event.target.closest("button");
  if (!button) return;
  const row = button.closest(".doc-row");
  const documentId = row?.dataset.documentId;
  if (button.dataset.action === "task") {
    openDrawer();
    await showTask(button.dataset.task);
  } else if (button.dataset.action === "retry") {
    try {
      await api(`/api/import-tasks/${button.dataset.task}/retry`, {
        method: "POST",
        body: { request_id: uuid() },
      });
      toast("已重新排队，仅恢复未完成阶段");
      await showTask(button.dataset.task);
      loadDocuments(true);
    } catch (err) {
      toast(err.message || "重试失败");
    }
  } else if (button.dataset.action === "withdraw") {
    if (!confirm("撤销后新查询不再使用该资料，已有证据快照仍可审计。继续？")) return;
    try {
      await api(`/api/documents/${documentId}`, { method: "DELETE" });
      toast("已撤销");
      loadDocuments(true);
    } catch (err) {
      toast(err.message || "撤销失败");
    }
  }
}

function openDrawer() {
  els.drawer.classList.remove("hidden");
}
function closeDrawer() {
  els.drawer.classList.add("hidden");
  clearInterval(pollTimer);
}

async function onUpload(event) {
  event.preventDefault();
  const file = document.getElementById("upload-file").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  form.append("request_id", uuid());
  const title = document.getElementById("upload-title").value.trim();
  if (title) form.append("title", title);
  form.append("doc_type", document.getElementById("upload-doc-type").value);
  const source = document.getElementById("upload-source").value.trim();
  if (source) form.append("source_url", source);
  try {
    const result = await api("/api/documents", { method: "POST", form });
    toast(result.duplicate ? "已存在相同内容，复用已有版本" : "已提交，后台自动入库");
    await showTask(result.task_id);
    loadDocuments(true);
  } catch (err) {
    toast(err.message || "上传失败");
  }
}

async function showTask(taskId) {
  clearInterval(pollTimer);
  els.taskPanel.classList.remove("hidden");
  document.getElementById("drawer-title").textContent = "导入任务";
  const render = async () => {
    try {
      const task = await api(`/api/import-tasks/${taskId}`);
      els.taskDetail.innerHTML = `
        <dt>任务</dt><dd>${escapeHtml(task.task_id)}</dd>
        <dt>状态</dt><dd>${escapeHtml(task.status)}</dd>
        <dt>阶段</dt><dd>${escapeHtml(task.stage)}</dd>
        <dt>可用性</dt><dd>${(AVAILABILITY[task.availability] || {}).label || task.availability}</dd>
        <dt>向量</dt><dd>${escapeHtml(task.vector_state)}</dd>
        <dt>图谱</dt><dd>${escapeHtml(task.graph_state)}</dd>
        <dt>尝试</dt><dd>${task.attempt}</dd>
        <dt>警告</dt><dd>${(task.warnings || []).map(flagLabel).join("、") || "无"}</dd>
        <dt>错误</dt><dd>${task.error ? escapeHtml(task.error.code + "：" + task.error.message) : "无"}</dd>
        <dt>更新时间</dt><dd>${fmtTime(task.updated_at)}</dd>`;
      els.taskActions.innerHTML = task.retryable
        ? `<button class="btn btn-sm" id="retry-task">重试未完成阶段</button>`
        : `<span class="muted small">${task.status.startsWith("succeeded") ? "已完成" : "不可重试"}</span>`;
      const retry = document.getElementById("retry-task");
      if (retry) {
        retry.addEventListener("click", async () => {
          try {
            await api(`/api/import-tasks/${taskId}/retry`, { method: "POST", body: { request_id: uuid() } });
            toast("已重新排队");
            loadDocuments(true);
          } catch (err) {
            toast(err.message || "重试失败");
          }
        });
      }
      if (["succeeded", "succeeded_with_warnings", "failed", "cancelled"].includes(task.status)) {
        clearInterval(pollTimer);
      }
    } catch (err) {
      els.taskDetail.innerHTML = `<dt>错误</dt><dd>${escapeHtml(err.message)}</dd>`;
      clearInterval(pollTimer);
    }
  };
  await render();
  pollTimer = setInterval(render, 1500);
}
