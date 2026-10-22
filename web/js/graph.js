// 知识关联：中文实体搜索 → 关系路径 → 支持原文（不借用咨询 query 的 E 编号）
import { api, escapeHtml, state, toast } from "./api.js";

let els = {};
let activeId = null;

const TYPE_LABEL = {
  LegalDocument: "法规",
  DocumentVersion: "版本",
  Provision: "条款",
  Case: "案例",
  Issue: "争议主题",
  Organization: "机构",
};

const REL_LABEL = {
  HAS_VERSION: "包含版本",
  HAS_PROVISION: "包含条款",
  CITES: "引用",
  CONCERNS: "涉及",
  ISSUED_BY: "发布机关",
  SUPERSEDES: "替代",
  SUPPORTED_BY: "原文支持",
};

export function initGraph() {
  els = {
    q: document.getElementById("entity-q"),
    list: document.getElementById("entity-list"),
    more: document.getElementById("entity-more"),
    empty: document.getElementById("graph-empty"),
    result: document.getElementById("graph-result"),
    paths: document.getElementById("graph-paths"),
    chunks: document.getElementById("graph-chunks"),
  };
  els.q.addEventListener("input", debounce(() => loadEntities(true), 300));
  els.more.addEventListener("click", () => loadEntities(false));
  els.list.addEventListener("click", (event) => {
    const item = event.target.closest("li[data-id]");
    if (item) selectEntity(item.dataset.id, item.dataset.label);
  });
}

function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

export async function loadEntities(reset = true) {
  if (reset) {
    state.entityCursor = null;
    els.list.innerHTML = "";
  }
  const params = new URLSearchParams({ limit: "20" });
  const q = els.q.value.trim();
  if (q) params.set("q", q);
  if (state.entityCursor) params.set("cursor", state.entityCursor);
  try {
    const data = await api(`/api/graph/entities?${params.toString()}`);
    if (reset && !data.items.length) {
      els.list.innerHTML = '<li class="muted small">没有匹配实体。请先导入资料，或换一个关键词。</li>';
    }
    for (const entity of data.items) {
      const li = document.createElement("li");
      li.dataset.id = entity.entity_id;
      li.dataset.label = entity.label;
      li.innerHTML = `<span>${escapeHtml(entity.label)}</span><span class="type">${TYPE_LABEL[entity.entity_type] || entity.entity_type}</span>`;
      els.list.appendChild(li);
    }
    state.entityCursor = data.next_cursor;
    els.more.classList.toggle("hidden", !data.next_cursor);
    if (data.next_cursor) els.more.dataset.cursor = data.next_cursor;
  } catch (err) {
    toast(err.message || "实体搜索失败");
  }
}

async function selectEntity(entityId, label) {
  activeId = entityId;
  els.list.querySelectorAll("li").forEach((li) => li.classList.toggle("active", li.dataset.id === entityId));
  els.empty.classList.add("hidden");
  els.result.classList.remove("hidden");
  els.paths.innerHTML = `<p class="muted small">正在读取「${escapeHtml(label)}」的关联路径…</p>`;
  els.chunks.innerHTML = "";
  try {
    const data = await api(`/api/graph/relations?entity_id=${encodeURIComponent(entityId)}&max_hops=2`);
    if (!data.paths.length) {
      els.paths.innerHTML = `<div class="empty"><h3>无匹配路径</h3><p class="muted">该实体当前没有具备原文支持的确定关系；未解析引用只作为线索保留。</p></div>`;
      els.chunks.innerHTML = "";
      return;
    }
    const nodeLabels = new Map(data.nodes.map((node) => [node.entity_id, node]));
    els.paths.innerHTML = data.paths
      .map((path) => {
        const chain = path.node_ids
          .map((id, index) => {
            const node = nodeLabels.get(id);
            const pill = `<span class="node-pill">${escapeHtml(node?.label || id)}</span>`;
            const rel = path.relation_types[index];
            return rel ? `${pill}<span class="rel">—${REL_LABEL[rel] || rel}→</span>` : pill;
          })
          .join("");
        return `<div class="path-card">
          <div class="path-chain">${chain}</div>
          <p class="muted small" style="margin-top:8px">支持原文 ${path.support_chunk_ids.length} 条 · 状态 ${path.status === "source_supported" ? "有原文支持" : "未解析"}</p>
        </div>`;
      })
      .join("");
    const chunkMap = new Map(data.support_chunks.map((chunk) => [chunk.chunk_id, chunk]));
    const used = new Set(data.paths.flatMap((path) => path.support_chunk_ids));
    els.chunks.innerHTML = [...used]
      .map((chunkId) => {
        const chunk = chunkMap.get(chunkId);
        if (!chunk) return "";
        const loc = [chunk.location?.article, chunk.location?.heading, chunk.location?.page ? `第${chunk.location.page}页` : ""]
          .filter(Boolean)
          .join(" · ");
        return `<div class="chunk-card">
          <h5>${escapeHtml(loc || "原文片段")}</h5>
          <p>${escapeHtml(chunk.content)}</p>
        </div>`;
      })
      .join("");
    if (data.warnings?.includes("GRAPH_RESULT_TRUNCATED")) {
      els.chunks.insertAdjacentHTML("afterbegin", '<p class="tag tag-warn">结果已截断，仅展示可完整定位的路径</p>');
    }
  } catch (err) {
    if (err.status === 404) {
      els.paths.innerHTML = '<div class="empty"><h3>实体不可见</h3><p class="muted">该实体可能属于已撤销或会话私有资料。</p></div>';
    } else {
      els.paths.innerHTML = `<div class="empty"><h3>图服务不可用</h3><p class="muted">${escapeHtml(err.message)}</p></div>`;
    }
    els.chunks.innerHTML = "";
  }
}
