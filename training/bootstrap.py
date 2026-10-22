"""以真实原文生成可复现的工程银标：阅读/引用/限制表达，不虚构法律结论。"""
from __future__ import annotations

import copy
import hashlib
import random
import re


def group_assignments(sources, seed=42):
    """同发布页、同标题或相同内容归为一个连通来源组，先分组再生成。"""
    parent = list(range(len(sources)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    seen = {}
    for index, source in enumerate(sources):
        url = source["url"].split("#", 1)[0]
        title = re.sub(r"[\s：:—-]", "", source["title"])
        signatures = ["url:" + url, "title:" + title]
        signatures += ["body:" + hashlib.sha256(chunk["content"].encode()).hexdigest() for chunk in source["chunks"]]
        for key in signatures:
            if key in seen:
                parent[find(index)] = find(seen[key])
            else:
                seen[key] = index
    components = {}
    for index in range(len(sources)):
        components.setdefault(find(index), []).append(index)
    grouped = []
    for indices in components.values():
        key = hashlib.sha256("\n".join(sorted(sources[i]["version_id"] for i in indices)).encode()).hexdigest()[:24]
        grouped.append((key, indices))
    grouped.sort()
    if len(grouped) < 5:
        raise ValueError("独立来源组不足5个，不能可靠划分训练/验证/测试")
    # 按文档数与场景做预先分层；大合集不能仅因抽到验证集而占据大半数据。
    # 此处仅使用来源元数据，不读取模型预测或测试答案来挑有利划分。
    rng = random.Random(seed)
    choices = ("train", "validation", "test")
    ratios = (0.65, 0.15, 0.20)
    total = len(sources)
    best, best_score = None, float("inf")
    for _ in range(6000):
        candidate = rng.choices(choices, weights=ratios, k=len(grouped))
        sizes = dict.fromkeys(choices, 0)
        scenarios = {split: set() for split in choices}
        for split, (_, indices) in zip(candidate, grouped):
            sizes[split] += len(indices)
            scenarios[split].update(sources[index]["scenario"] for index in indices)
        missing = sum(len({source["scenario"] for source in sources} - scenarios[split]) for split in choices)
        score = missing * 1000 + sum(abs(sizes[split] / total - ratio) for split, ratio in zip(choices, ratios))
        if score < best_score:
            best, best_score = candidate, score
    assignment = {}
    for split, (key, indices) in zip(best, grouped):
        for index in indices:
            sources[index]["group_id"] = key
            assignment[sources[index]["version_id"]] = split
    return assignment


def excerpt(chunk, max_chars=300):
    text = chunk["content"]
    # 从正文句段边界选择完整前缀；过长单句不硬截，以免切掉条件。
    parts = re.split(r"(?<=[。！？；])|\n", text)
    selected, start = "", 0
    for part in parts:
        part = part.strip()
        if not part or part.startswith(("#", "来源：", "发布时间：")):
            continue
        if not selected:
            if len(part) > max_chars:
                continue
            start = text.find(part)
            selected = part
        else:
            end = text.find(part, start + len(selected))
            if end < 0 or end + len(part) - start > max_chars:
                break
            selected = text[start:end + len(part)]
    if len(selected) < 30:
        return None
    location = copy.deepcopy(chunk["location"])
    if location:
        location["start_char"] += start
        location["end_char"] = location["start_char"] + len(selected)
    return selected, location


def make_source_samples(source, split):
    evidence = []
    for chunk in source["chunks"]:
        result = excerpt(chunk)
        if result is None:
            continue
        text, location = result
        evidence.append({"evidence_id": f"E{len(evidence)+1}", "source_id": chunk["chunk_id"], "source_type": "local", "chunk_id": chunk["chunk_id"], "version_id": source["version_id"], "title": source["title"], "text": text, "url": source["url"], "location": location, "metadata": source.get("metadata"), "quality_flags": source.get("quality_flags", [])})
        if len(evidence) == 2:
            break
    if not evidence:
        return []
    first = evidence[0]
    title, text = first["title"], first["text"]
    tasks = [
        ("faithfulness", "answer", f"请仅复述《{title}》给定片段中的明确记载，保留原文限定，不补充片段外规则。", f"摘要：给定片段明确记载如下。[E1]\n依据与条件：{text}[E1]\n需补充事项：具体情形仍须核实；本回答不判断当前效力。", [first]),
        ("citation", "answer", f"《{title}》这段资料的可点击来源编号是什么？请摘录对应原文，避免引用其他材料。", f"摘要：当前来源编号为[E1]。\n依据与条件：{text}[E1]\n需补充事项：核实资料版本与适用时间。", [first]),
        ("terminology_format", "answer", f"请把《{title}》片段整理为摘要、依据与条件、需补充事项三段；依据部分直接引用原文，不作额外法律推导。", f"摘要：以下为给定资料的片段整理。[E1]\n依据与条件：{text}[E1]\n需补充事项：仅有片段，尚需核实完整上下文及适用条件。", [first]),
        ("abstention", "insufficient", f"仅凭《{title}》这个片段，可以保证所有相似案件都胜诉、且一定适用现在的法律吗？", "摘要：依据不足，无法作出这种保证。\n依据与条件：单个来源片段不足以证明所有案件的事实、法律适用和当前效力；不能保证胜诉。[E1]\n需补充事项：具体事实、相关日期、完整原文及版本依据。", [first]),
        ("abstention", "clarify", f"我想参照《{title}》判断我自己的争议，但没有说明发生时间、具体经过或诉求。你需要我补充什么？", "摘要：需澄清，现有个人事实不足。\n依据与条件：来源资料不能替代本人的事实，尚不能判断具体适用。[E1]\n需补充事项：请补充争议发生时间、劳动关系情况、具体行为、金额或诉求以及已有证据。", [first]),
    ]
    if len(evidence) == 2:
        tasks.append(("integration", "answer", f"请对照《{title}》的两个片段，各摘录一段原文并标注各自编号；不能把两个片段合成为未记载的新规则。", f"摘要：两个片段分别记载如下。\n依据与条件：片段一：{evidence[0]['text']}[E1]\n片段二：{evidence[1]['text']}[E2]\n需补充事项：两段之间的法律关系及具体适用仍需结合完整资料核实。", evidence))
    rows = []
    for index, (ability, behavior, question, answer, items) in enumerate(tasks):
        sid = hashlib.sha256(f"{source['version_id']}:{index}:bootstrap-v1".encode()).hexdigest()[:24]
        rows.append({"sample_id": sid, "split": split, "origin_group_id": source["group_id"], "scenario": source["scenario"], "ability": ability, "question": question, "evidence": copy.deepcopy(items), "expected_behavior": behavior, "target_answer": answer, "label_type": "synthetic_silver", "generator_revision": "deterministic-extractive-v1", "source_manifest_revision": "imported-snapshot-v1"})
    return rows
