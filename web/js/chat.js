// 咨询页面：会话、提问、SSE 渲染、证据查看
import { api, escapeHtml, fmtTime, state, toast, uuid } from "./api.js";
import { subscribeEvents } from "./events.js";

let els = {};
let currentQueryId = null;

export function initChat() {
  els = {
    list: document.getElementById("session-list"),
    thread: document.getElementById("chat-thread"),
    hero: document.getElementById("chat-hero"),
    form: document.getElementById("composer"),
    question: document.getElementById("question"),
    web: document.getElementById("web-enabled"),
    date: document.getElementById("relevant-date"),
    attachment: document.getElementById("attachment"),
    attachmentHint: document.getElementById("attachment-hint"),
    evidenceList: document.getElementById("evidence-list"),
    evidenceDetail: document.getElementById("evidence-detail"),
  };
  els.form.addEventListener("submit", onSubmit);
  els.question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      els.form.requestSubmit();
    }
  });
  document.querySelectorAll(".scene-card").forEach((card) => {
    card.addEventListener("click", () => {
      const prompts = {
        wage_arrears: "公司拖欠我 2025 年 3 月和 4 月工资，月薪 9000 元，已离职，想了解可以主张什么、需要哪些证据？",
        termination: "公司以“组织调整”为由单方通知我解除劳动合同，我工作 3 年 2 个月，没有书面协商，这种情况是否属于违法解除？",
        economic_compensation: "劳动合同到期公司不续签，我工作 4 年，月工资 12000 元，经济补偿应该怎么计算？",
      };
      els.question.value = prompts[card.dataset.scene] || "";
      els.question.focus();
    });
  });
  els.attachment.addEventListener("change", () => {
    const file = els.attachment.files[0];
    els.attachmentHint.textContent = file ? `待上传：${file.name}` : "";
  });
  els.thread.addEventListener("click", (event) => {
    const cite = event.target.closest(".cite");
    if (cite) showEvidence(cite.dataset.eid);
  });
  els.evidenceList.addEventListener("click", (event) => {
    const item = event.target.closest(".evidence-item");
    if (item) showEvidence(item.dataset.eid);
  });
}

export async function loadSessions() {
  const data = await api("/api/sessions?limit=30");
  state.sessions = data.items;
  els.list.innerHTML = state.sessions
    .map(
      (session) =>
        `<li data-id="${session.session_id}" class="${session.session_id === state.sessionId ? "active" : ""}">${escapeHtml(
          session.title,
        )}</li>`,
    )
    .join("");
  els.list.querySelectorAll("li").forEach((li) => {
    li.addEventListener("click", () => openSession(li.dataset.id));
  });
}

export async function newSession() {
  const session = await api("/api/sessions", { method: "POST", body: {} });
  state.sessionId = session.session_id;
  currentQueryId = null;
  els.thread.innerHTML = "";
  els.hero.classList.remove("hidden");
  els.evidenceList.innerHTML = '<p class="muted small">提交问题后，本轮编号证据会显示在这里。</p>';
  els.evidenceDetail.classList.add("hidden");
  await loadSessions();
  els.question.focus();
}

export async function openSession(sessionId) {
  if (state.stream) state.stream.close();
  state.sessionId = sessionId;
  state.evidence.clear();
  currentQueryId = null;
  els.thread.innerHTML = "";
  els.hero.classList.add("hidden");
  els.evidenceList.innerHTML = '<p class="muted small">提交问题后，本轮编号证据会显示在这里。</p>';
  els.evidenceDetail.classList.add("hidden");
  await loadSessions();
  const data = await api(`/api/sessions/${sessionId}/messages?limit=50`);
  const groups = new Map();
  for (const message of data.items) {
    if (!groups.has(message.query_id)) groups.set(message.query_id, {});
    groups.get(message.query_id)[message.role] = message;
  }
  for (const [queryId, pair] of groups) {
    if (pair.user) appendUser(pair.user.content);
    if (pair.assistant) {
      const card = appendAnswerCard();
      renderAnswerText(card, pair.assistant.content);
      card.dataset.queryId = queryId;
      currentQueryId = queryId;
    }
  }
  // 最后一轮若仍在执行，恢复快照并从最后序号续传
  const last = state.sessions.find((item) => item.session_id === sessionId);
  if (last && last.last_query_id) await resumeQuery(last.last_query_id);
}

async function resumeQuery(queryId) {
  const doc = await api(`/api/queries/${queryId}`);
  if (!doc || ["completed", "completed_with_warnings", "failed", "cancelled", "interrupted"].includes(doc.status)) {
    return;
  }
  let card = els.thread.querySelector(`[data-query-id="${queryId}"]`);
  if (!card) {
    card = appendAnswerCard();
    card.dataset.queryId = queryId;
  }
  clearStatus(card);
  setStatus(card, `恢复中 · ${doc.status}`, "warn");
  renderEvidence(doc.evidence || []);
  state.lastSeq = doc.last_event_seq;
  attachStream(queryId, card, doc.last_event_seq);
}

function appendUser(text) {
  const row = document.createElement("div");
  row.className = "turn";
  row.innerHTML = `<div class="bubble-user">${escapeHtml(text)}</div>`;
  els.thread.appendChild(row);
  els.thread.scrollTop = els.thread.scrollHeight;
  return row;
}

function appendAnswerCard() {
  const row = document.createElement("div");
  row.className = "turn";
  row.innerHTML = `
    <div class="answer-card">
      <div class="answer-head">
        <span class="tag" data-role="model">回答模型：待调用</span>
        <span class="tag" data-role="behavior">—</span>
        <span class="tag" data-role="citation">引用校验：—</span>
      </div>
      <div class="answer-body" data-role="body"></div>
      <div class="status-row" data-role="status"></div>
    </div>`;
  els.thread.appendChild(row);
  els.thread.scrollTop = els.thread.scrollHeight;
  return row.querySelector(".answer-card");
}

function setStatus(card, text, kind = "muted") {
  const row = card.querySelector('[data-role="status"]');
  const tag = document.createElement("span");
  tag.className = `tag tag-${kind}`;
  tag.textContent = text;
  row.appendChild(tag);
}

function clearStatus(card) {
  card.querySelector('[data-role="status"]').innerHTML = "";
}

function renderAnswerText(card, text) {
  const body = card.querySelector('[data-role="body"]');
  body.innerHTML = escapeHtml(text).replace(/\[E(\d+)\]/g, (_, n) => `<span class="cite" data-eid="E${n}">[E${n}]</span>`);
  els.thread.scrollTop = els.thread.scrollHeight;
}

async function onSubmit(event) {
  event.preventDefault();
  if (state.running) {
    toast("当前问题仍在生成，请稍候或取消");
    return;
  }
  const question = els.question.value.trim();
  if (!question) {
    toast("请输入问题");
    return;
  }
  if (!state.sessionId) await newSession();

  let attachmentIds = [];
  const file = els.attachment.files[0];
  try {
    if (file) {
      const attachment = await uploadAttachment(file);
      if (!attachment) return;
      attachmentIds = [attachment.attachment_id];
    }
    state.running = true;
    document.getElementById("send").disabled = true;
    appendUser(question);
    els.hero.classList.add("hidden");
    const card = appendAnswerCard();
    setStatus(card, "已提交，等待 worker 领取", "warn");
    els.question.value = "";
    els.attachment.value = "";
    els.attachmentHint.textContent = "";

    const payload = {
      request_id: uuid(),
      question,
      document_ids: [],
      attachment_ids: attachmentIds,
      relevant_date: els.date.value || null,
      location: null,
      web_enabled: els.web.checked,
    };
    const created = await api(`/api/sessions/${state.sessionId}/queries`, { method: "POST", body: payload });
    currentQueryId = created.query_id;
    card.dataset.queryId = created.query_id;
    state.lastSeq = 0;
    attachStream(created.query_id, card, 0);
  } catch (err) {
    state.running = false;
    document.getElementById("send").disabled = false;
    toast(err.message || "提交失败");
  }
}

async function uploadAttachment(file) {
  const form = new FormData();
  form.append("file", file);
  form.append("request_id", uuid());
  const hint = els.attachmentHint;
  hint.textContent = `上传中：${file.name}`;
  try {
    const result = await api(`/api/sessions/${state.sessionId}/attachments`, { method: "POST", form });
    hint.textContent = `解析中：${file.name}`;
    const ready = await waitAttachment(result.task_id);
    if (!ready) {
      toast("附件解析超时，请稍后在资料页查看任务状态");
      return null;
    }
    hint.textContent = `已就绪：${file.name}`;
    return result;
  } catch (err) {
    hint.textContent = "";
    toast(err.message || "附件上传失败");
    return null;
  }
}

async function waitAttachment(taskId, timeoutMs = 120000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const task = await api(`/api/import-tasks/${taskId}`);
    if (task.status === "succeeded" || task.status === "succeeded_with_warnings") return true;
    if (task.status === "failed" || task.status === "cancelled") return false;
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  return false;
}

function attachStream(queryId, card, fromSeq) {
  if (state.stream) state.stream.close();
  state.stream = subscribeEvents(queryId, {
    fromSeq,
    onEvent: (event) => handleEvent(queryId, card, event),
    onClose: (event) => {
      state.running = false;
      document.getElementById("send").disabled = false;
      if (event?.event === "stream_closed") {
        setStatus(card, "事件流已结束，可刷新读取快照", "warn");
      }
    },
    onError: (err) => {
      state.running = false;
      document.getElementById("send").disabled = false;
      if (err.status === 410) {
        setStatus(card, "事件日志已过期，请刷新读取快照", "warn");
      } else {
        setStatus(card, err.message || "事件流中断", "danger");
      }
    },
  });
}

function handleEvent(queryId, card, event) {
  const body = card.querySelector('[data-role="body"]');
  switch (event.event) {
    case "started":
      setStatus(card, "worker 已开始处理", "warn");
      break;
    case "stage": {
      const name = event.payload.name || "";
      const label = { route: "识别场景", retrieve: "检索证据", fuse: "融合排序", rerank: "精排", evidence_policy: "依据分级", generate: "生成回答" }[name] || name;
      if (name !== "generate") setStatus(card, `阶段：${label}`, "muted");
      break;
    }
    case "evidence":
      renderEvidence(event.payload.items || []);
      break;
    case "delta":
      body.dataset.raw = (body.dataset.raw || "") + (event.payload.text || "");
      renderAnswerText(card, body.dataset.raw);
      break;
    case "warning":
      setStatus(card, event.payload.message || event.payload.code, "warn");
      break;
    case "done":
      finalizeCard(card, event.payload, queryId);
      break;
    case "failed":
      setStatus(card, event.payload.message || "生成失败", "danger");
      card.querySelector('[data-role="behavior"]').textContent = event.payload.status === "interrupted" ? "已中断" : "失败";
      state.running = false;
      document.getElementById("send").disabled = false;
      break;
    case "cancelled":
      setStatus(card, "已取消", "warn");
      state.running = false;
      document.getElementById("send").disabled = false;
      break;
    default:
      break;
  }
}

async function finalizeCard(card, payload, queryId) {
  state.running = false;
  document.getElementById("send").disabled = false;
  try {
    const doc = await api(`/api/queries/${queryId}`);
    renderAnswerText(card, doc.answer || "");
    card.querySelector('[data-role="behavior"]').textContent = behaviorLabel(doc.behavior);
    card.querySelector('[data-role="citation"]').textContent = `引用校验：${citationLabel(doc.citation_validation)}`;
    const model = card.querySelector('[data-role="model"]');
    model.textContent = doc.generation_model_id ? `回答模型：${doc.generation_model_id}` : "未调用回答模型";
    model.className = `tag ${doc.generation_model_id ? "tag-ok" : "tag-warn"}`;
    renderEvidence(doc.evidence || []);
    if (doc.warnings?.length) {
      setStatus(card, `提示：${doc.warnings.join("、")}`, "warn");
    }
    if (doc.reasoning_code) setStatus(card, `决策码：${doc.reasoning_code}`, "muted");
  } catch (err) {
    setStatus(card, "读取最终结果失败，请刷新", "danger");
  }
}

function behaviorLabel(value) {
  return { answer: "回答", clarify: "需要澄清", insufficient: "依据不足" }[value] || "—";
}

function citationLabel(value) {
  return { passed: "通过", failed: "未通过", pending: "待校验", not_applicable: "不适用" }[value] || "—";
}

function renderEvidence(items) {
  state.evidence.clear();
  if (!items.length) {
    els.evidenceList.innerHTML = '<p class="muted small">本轮没有可展示的证据。</p>';
    return;
  }
  els.evidenceList.innerHTML = items
    .map((item) => {
      state.evidence.set(item.evidence_id, item);
      const loc = [item.location?.article, item.location?.heading, item.location?.page ? `第${item.location.page}页` : ""]
        .filter(Boolean)
        .join(" · ");
      const type = { local: "本地资料", web: "网页原文", web_snippet: "搜索线索", attachment: "会话附件" }[item.source_type] || item.source_type;
      return `<div class="evidence-item" data-eid="${item.evidence_id}">
        <h5>[${item.evidence_id}] ${escapeHtml(item.title)}</h5>
        <div class="loc">${type}${loc ? " · " + escapeHtml(loc) : ""}</div>
      </div>`;
    })
    .join("");
}

async function showEvidence(evidenceId) {
  if (!currentQueryId) {
    toast("请先完成一轮咨询");
    return;
  }
  try {
    const evidence = await api(`/api/queries/${currentQueryId}/evidence/${evidenceId}`);
    const meta = evidence.metadata || {};
    els.evidenceDetail.classList.remove("hidden");
    els.evidenceDetail.innerHTML = `
      <h4>[${evidence.evidence_id}] ${escapeHtml(evidence.title)}</h4>
      <p class="muted small">${escapeHtml(evidence.source_type)}${evidence.version_id ? " · 版本 " + escapeHtml(evidence.version_id.slice(0, 12)) : ""}</p>
      <p class="muted small">效力：${escapeHtml(meta.effectivity || "未知")}${meta.effective_from ? " · 生效 " + escapeHtml(meta.effective_from) : ""}</p>
      <pre>${escapeHtml(evidence.text)}</pre>
      ${evidence.quality_flags?.length ? `<p class="small">质量标记：${escapeHtml(evidence.quality_flags.join("、"))}</p>` : ""}
      ${evidence.version_id ? `<p class="small"><a href="/api/versions/${evidence.version_id}/file${evidence.source_type === "attachment" ? "?session_id=" + state.sessionId : ""}" target="_blank">查看原件</a></p>` : ""}
    `;
  } catch (err) {
    toast(err.message || "证据不可用");
  }
}
