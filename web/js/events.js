// SSE 订阅：基于 fetch 读取已持久化事件，支持 Last-Event-ID 断线续传与 seq 去重
// 浏览器 EventSource 不能设置请求头，因此这里手写解析器，保证重连不触发第二次模型调用。
export function subscribeEvents(queryId, { fromSeq = 0, onEvent, onClose, onError }) {
  const controller = new AbortController();
  let lastSeq = fromSeq;
  let closed = false;

  const close = () => {
    if (closed) return;
    closed = true;
    controller.abort();
  };

  (async () => {
    try {
      const headers = { Accept: "text/event-stream" };
      if (fromSeq > 0) headers["Last-Event-ID"] = `${queryId}:${fromSeq}`;
      const res = await fetch(`/api/queries/${queryId}/events`, {
        headers,
        signal: controller.signal,
        cache: "no-store",
      });
      if (!res.ok) {
        const data = await res.json().catch(() => null);
        throw Object.assign(new Error(data?.error?.message || `事件流不可用（HTTP ${res.status}）`), {
          status: res.status,
          code: data?.error?.code,
        });
      }
      if (!res.body) throw new Error("当前浏览器不支持流式响应");
      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (!closed) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let index;
        while ((index = buffer.indexOf("\n\n")) >= 0) {
          const chunk = buffer.slice(0, index);
          buffer = buffer.slice(index + 2);
          const event = parseSseChunk(chunk);
          if (!event) continue;
          if (event.seq !== undefined) {
            if (event.seq <= lastSeq) continue; // 去重：刷新/重连不重复渲染
            lastSeq = event.seq;
          }
          onEvent?.(event);
          if (["done", "failed", "cancelled"].includes(event.event)) {
            close();
            onClose?.(event);
            return;
          }
        }
      }
      onClose?.({ event: "stream_closed", payload: { lastSeq } });
    } catch (err) {
      if (closed) return;
      closed = true;
      onError?.(err);
    }
  })();

  return { close, get lastSeq() { return lastSeq; } };
}

function parseSseChunk(chunk) {
  const lines = chunk.split("\n");
  let data = "";
  let eventName = "";
  let id = "";
  for (const line of lines) {
    if (line.startsWith(":")) continue; // 心跳注释
    if (line.startsWith("event:")) eventName = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
    else if (line.startsWith("id:")) id = line.slice(3).trim();
  }
  if (!data) return null;
  let parsed;
  try {
    parsed = JSON.parse(data);
  } catch {
    return null;
  }
  return {
    id,
    seq: parsed.seq,
    event: parsed.event || eventName,
    payload: parsed.payload || {},
  };
}
