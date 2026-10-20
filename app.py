import html
import json
import os
import re
import sys
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from backend.btw_handler import handle_btw
from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.rag_graph import LLM_TIMEOUT_S, build_graph, log
from backend.vector_store import add_paper, list_papers

st.set_page_config(page_title="PharmaLens · Pharma research copilot", page_icon="🔬", layout="centered")

AUTHOR = "Shivam Singh"
STACK = "LangGraph · Qdrant · Groq · Gemini"

# Public demo mode (set PAPEER_PUBLIC=1 on the host): every browser gets its own
# private session list, and API usage per visitor is capped.
PUBLIC_MODE = os.getenv("PAPEER_PUBLIC", "").lower() in ("1", "true", "yes")
MAX_QUESTIONS = int(os.getenv("PAPEER_MAX_QUESTIONS", "15"))
MAX_UPLOADS = int(os.getenv("PAPEER_MAX_UPLOADS", "3"))
MAX_FILE_MB = int(os.getenv("PAPEER_MAX_FILE_MB", "10"))

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.block-container { max-width: 880px; padding-top: 2.2rem; }
header[data-testid="stHeader"] { background: transparent; }

/* brand */
.pp-brand { font-family: Fraunces, serif; font-size: 30px; font-weight: 500; color: #FAF8F3;
            letter-spacing: -0.5px; line-height: 1; }
.pp-brand em { color: #5DCAA5; }
.pp-sub { font-size: 12px; color: #888780; margin-top: 4px; }
.pp-label { font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: #888780;
            margin: 6px 0 2px; }
.pp-doc { background: #24272E; border-radius: 8px; padding: 8px 10px; margin-bottom: 6px;
          color: #FAF8F3; font-size: 13px; line-height: 1.35; }
.pp-empty { font-size: 12px; color: #888780; }
.pp-foot { border-top: 0.5px solid #34373F; padding-top: 12px; margin-top: 8px; font-size: 12px;
           color: #888780; line-height: 1.6; }
.pp-foot b { color: #FAF8F3; font-weight: 500; }

/* sidebar session list */
[class*="st-key-sess_"] button { justify-content: flex-start; padding: 6px 10px; border-radius: 0;
                                 border-left: 2px solid transparent; width: 100%; }
[class*="st-key-sess_"] button > div, [class*="st-key-sess_"] button span { justify-content: flex-start; width: 100%; }
[class*="st-key-sess_"] button p { text-align: left; font-size: 13px; color: #B4B2A9; }
[class*="st-key-sess_"] button:hover p { color: #FAF8F3; }
[class*="st-key-sess_"] { margin-bottom: -10px; }

/* hero */
.pp-hero { border-bottom: 0.5px solid #D3D1C7; padding-bottom: 14px; margin-bottom: 8px; }
.pp-hero .t { font-family: Fraunces, serif; font-size: 34px; font-weight: 500; line-height: 1.15;
              color: #1C1B1A; letter-spacing: -0.5px; }
.pp-hero .t em { color: #0F6E56; }
.pp-hero .d { color: #5F5E5A; font-size: 15px; margin: 6px 0 12px; }

/* chips */
.pp-chip { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px;
           margin: 0 4px 4px 0; line-height: 1.6; }
.pp-green { background: #E1F5EE; color: #085041; }
.pp-amber { background: #FAEEDA; color: #633806; }
.pp-gray  { background: #EDEAE2; color: #444441; }
.pp-src { border: 0.5px solid #B4B2A9; color: #444441; background: #FFFFFF; text-decoration: none; }
a.pp-src:hover { border-color: #0F6E56; color: #0F6E56; }

/* pipeline trace */
.pp-trace { margin-top: 10px; padding-top: 8px; border-top: 0.5px dashed #D3D1C7; font-size: 12px;
            color: #888780; display: flex; flex-wrap: wrap; align-items: center; gap: 4px; }
.pp-step { background: #F1EEE6; color: #5F5E5A; border-radius: 6px; padding: 1px 8px; }
.pp-step.ok { background: #E1F5EE; color: #085041; }
.pp-step.warn { background: #FAEEDA; color: #633806; }
.pp-arrow { color: #B4B2A9; }

/* chat */
[data-testid="stChatMessage"] { border-radius: 12px; padding: 12px 14px; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: #F1EEE6; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: #FFFFFF; border: 0.5px solid #D3D1C7; }
[data-testid="stChatMessage"] h1 { font-size: 1.45rem; }
[data-testid="stChatMessage"] h2 { font-size: 1.25rem; }
[data-testid="stChatMessage"] h3 { font-size: 1.08rem; }
[class*="st-key-claim_"] { border-left: 3px solid #BA7517 !important; border-radius: 0 8px 8px 0 !important; }
[class*="st-key-suggest_"] button { text-align: left; justify-content: flex-start; min-height: 74px;
                                     background: #FFFFFF; align-items: flex-start; }
[class*="st-key-suggest_"] button > div, [class*="st-key-suggest_"] button span { justify-content: flex-start; width: 100%; }
[class*="st-key-suggest_"] button p { text-align: left; font-size: 13px; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_graph():
    return build_graph()


SESSIONS_FILE = Path("sessions.json")
_rename_llm = ChatGroq(model="openai/gpt-oss-20b", timeout=LLM_TIMEOUT_S, max_retries=2)
_store_lock = threading.Lock()
_VISITOR_RE = re.compile(r"^[0-9a-f-]{8,40}$")


def get_visitor_id() -> str:
    """Locally everything belongs to one owner; in public mode each browser gets an
    ID kept in the URL (?v=...), so a refresh keeps the visitor's own sessions."""
    if not PUBLIC_MODE:
        return "local"
    vid = st.query_params.get("v", "")
    if not _VISITOR_RE.match(vid):
        vid = uuid.uuid4().hex
        st.query_params["v"] = vid
    return vid


def _read_store() -> dict:
    """sessions.json shape: {"visitors": {visitor_id: {"sessions": {...}, "usage": {...}}}}."""
    try:
        data = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"visitors": {}}
    if "visitors" not in data:  # legacy flat {session_id: meta} file
        return {"visitors": {"local": {"sessions": data, "usage": {}}}}
    return data


def _update_visitor(fn) -> None:
    with _store_lock:
        store = _read_store()
        entry = store["visitors"].setdefault(VISITOR_ID, {"sessions": {}, "usage": {}})
        fn(entry)
        SESSIONS_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")


def load_sessions() -> dict:
    return _read_store()["visitors"].get(VISITOR_ID, {}).get("sessions", {})


def save_sessions(sessions_meta: dict) -> None:
    _update_visitor(lambda entry: entry.__setitem__("sessions", sessions_meta))


def get_usage() -> dict:
    usage = _read_store()["visitors"].get(VISITOR_ID, {}).get("usage", {})
    return {"questions": usage.get("questions", 0), "uploads": usage.get("uploads", 0)}


def bump_usage(kind: str) -> None:
    def _bump(entry):
        usage = entry.setdefault("usage", {})
        usage[kind] = usage.get(kind, 0) + 1
    _update_visitor(_bump)


def limit_reached(kind: str) -> bool:
    if not PUBLIC_MODE:
        return False
    cap = MAX_QUESTIONS if kind == "questions" else MAX_UPLOADS
    return get_usage()[kind] >= cap


LIMIT_MSG = {
    "questions": "You've used all demo questions for this browser. Thanks for trying PharmaLens!",
    "uploads": "You've reached the demo limit for adding papers. Ask questions about the ones you've loaded.",
}


def _serialize_state(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if k == "messages":
            out[k] = [
                {
                    "type": type(m).__name__,
                    "content": (
                        m.content[:300]
                        if isinstance(m.content, str)
                        else repr(m.content)[:300]
                    ),
                }
                for m in (v or [])
            ]
        elif k == "retrieved_docs":
            out[k] = [
                {"content": d.page_content[:300], "metadata": d.metadata}
                for d in (v or [])
            ]
        else:
            out[k] = v
    return out


def generate_session_name(first_message: str) -> str:
    try:
        response = _rename_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-5 word title for a research chat session "
                        "based on the user's first message. Return only the title, "
                        "no punctuation at the end, no quotes."
                    ),
                },
                {"role": "user", "content": first_message[:500]},
            ]
        )
        return response.content.strip()
    except Exception:
        return "New Session"


def maybe_rename_session(session_id: str, first_message: str) -> None:
    if st.session_state.sessions_meta.get(session_id, {}).get("is_named"):
        return
    name = generate_session_name(first_message)
    st.session_state.sessions_meta[session_id]["name"] = name
    st.session_state.sessions_meta[session_id]["is_named"] = True
    save_sessions(st.session_state.sessions_meta)


def create_session() -> str:
    sid = str(uuid.uuid4())
    st.session_state.sessions_meta[sid] = {
        "id": sid,
        "name": "New Session",
        "created_at": datetime.now().isoformat(),
        "is_named": False,
    }
    save_sessions(st.session_state.sessions_meta)
    st.session_state.chats[sid] = []
    st.session_state.turns[sid] = 0
    return sid


def load_session_chats(session_id: str) -> list[dict]:
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            return []
        chats = []
        turn = 0
        for msg in state.values.get("messages", []):
            type_name = type(msg).__name__
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if type_name == "HumanMessage":
                chats.append({"role": "user", "content": content})
            elif type_name in ("AIMessage", "AIMessageChunk"):
                # Skip the agent's tool-call messages; only final answers are shown.
                if getattr(msg, "tool_calls", None) or not content:
                    continue
                turn += 1
                chats.append({"role": "assistant", "content": content, "turn": turn, "graph_state": {}})
        return chats
    except Exception:
        return []


def switch_session(session_id: str) -> None:
    st.session_state.active_session_id = session_id
    if session_id not in st.session_state.chats:
        st.session_state.chats[session_id] = load_session_chats(session_id)
    if session_id not in st.session_state.turns:
        turn_count = sum(1 for m in st.session_state.chats[session_id] if m["role"] == "assistant")
        st.session_state.turns[session_id] = turn_count


# ── Presentation helpers ───────────────────────────────────────────────────────
ROUTE_LABELS = {
    "retrieve": "Search papers / web",
    "verify_claim": "Verify claim",
    "direct_answer": "Direct answer",
}
TOOL_LABELS = {
    "retrieve_from_vectorstore": "Vector search",
    "web_search": "Web search",
}


def _esc(text) -> str:
    return html.escape(str(text))


def fmt_math(text: str) -> str:
    """LLMs emit \\( \\) and \\[ \\] LaTeX; Streamlit markdown renders $ and $$."""
    return (
        text.replace("\\[", "$$").replace("\\]", "$$")
        .replace("\\(", "$").replace("\\)", "$")
    )


def _chip(text: str, kind: str = "gray", href: str | None = None) -> str:
    if href:
        return f'<a class="pp-chip pp-src" href="{_esc(href)}" target="_blank">{_esc(text)} ↗</a>'
    return f'<span class="pp-chip pp-{kind}">{_esc(text)}</span>'


def _is_web_doc(doc: dict) -> bool:
    return "url" in doc.get("metadata", {})


def route_badge(state: dict) -> str:
    route = state.get("route")
    if route == "verify_claim":
        if state.get("superseding_papers"):
            return _chip("Claim check · superseded by newer work", "amber")
        return _chip("Claim check · still holds", "amber")
    if route == "direct_answer":
        return _chip("Direct answer · general knowledge", "gray")
    if route == "retrieve":
        docs = state.get("retrieved_docs") or []
        if not docs or state.get("is_relevant") is False:
            return _chip("No relevant passages found", "gray")
        has_web = any(_is_web_doc(d) for d in docs)
        has_paper = any(not _is_web_doc(d) for d in docs)
        if has_web and has_paper:
            return _chip("Answered from your papers + live web", "green")
        if has_web:
            return _chip("Answered from live web", "green")
        return _chip("Answered from your papers", "green")
    return ""


def source_chips(state: dict, limit: int = 6) -> str:
    if state.get("route") != "retrieve" or state.get("is_relevant") is False:
        return ""
    seen, chips = set(), []
    for doc in state.get("retrieved_docs") or []:
        meta = doc.get("metadata", {})
        title = meta.get("title") or "Untitled"
        short = title if len(title) <= 48 else title[:45] + "…"
        if _is_web_doc(doc):
            key = meta["url"]
            label, href = short, meta["url"]
        else:
            page = meta.get("page")
            key = (title, page)
            label = f"{short} · p.{page + 1}" if isinstance(page, int) else short
            href = meta.get("source") if str(meta.get("source", "")).startswith("http") else None
        if key in seen:
            continue
        seen.add(key)
        chips.append(_chip(label, href=href) if href else f'<span class="pp-chip pp-src">{_esc(label)}</span>')
        if len(chips) >= limit:
            break
    return "".join(chips)


def trace_html(steps: list[dict], elapsed: float | None = None) -> str:
    if not steps:
        return ""
    # Collapse repeats, e.g. three vector searches in a row -> "Vector search ×3".
    grouped: list[list] = []
    for s in steps:
        if grouped and grouped[-1][0]["title"] == s["title"]:
            grouped[-1][1] += 1
        else:
            grouped.append([s, 1])
    parts = []
    for i, (s, n) in enumerate(grouped):
        cls = {True: " ok", False: " warn"}.get(s.get("ok"), "")
        label = f'{s["title"]} ×{n}' if n > 1 else s["title"]
        if i:
            parts.append('<span class="pp-arrow">→</span>')
        parts.append(f'<span class="pp-step{cls}">{_esc(label)}</span>')
    timing = f'<span style="margin-left:auto">{elapsed:.1f}s</span>' if elapsed else ""
    return f'<div class="pp-trace"><span style="margin-right:4px">Pipeline</span>{"".join(parts)}{timing}</div>'


def _messages_from_update(update) -> list:
    """A node update can be a dict or (for ToolNode returning Commands) a list of dicts."""
    items = update if isinstance(update, list) else [update]
    msgs = []
    for item in items:
        if isinstance(item, dict):
            msgs.extend(item.get("messages") or [])
        elif hasattr(item, "update") and isinstance(item.update, dict):
            msgs.extend(item.update.get("messages") or [])
    return msgs


def describe_update(node: str, update) -> list[dict]:
    """Turn a LangGraph node update into human-readable pipeline steps."""
    if node == "router":
        route = update.get("route", "")
        return [{"title": "Router", "detail": f"Classified as **{ROUTE_LABELS.get(route, route)}**"}]
    if node == "agent_node":
        msgs = _messages_from_update(update)
        calls = getattr(msgs[-1], "tool_calls", None) if msgs else None
        if calls:
            steps = []
            for tc in calls:
                args = tc.get("args", {})
                q = args.get("query") or args.get("optimized_query") or ""
                n = args.get("k") or args.get("max_results")
                tool = TOOL_LABELS.get(tc.get("name"), tc.get("name"))
                steps.append({
                    "title": "Agent",
                    "detail": f"Chose **{tool}** · “{q}”" + (f" · top {n}" if n else ""),
                    "hidden": True,
                })
            return steps
        return [{"title": "Agent", "detail": "Enough context gathered", "hidden": True}]
    if node == "retrieval":
        steps = []
        for m in _messages_from_update(update):
            # The tools build their ToolMessages without a name, so infer it from the summary text.
            content = str(m.content)
            tool = TOOL_LABELS.get(getattr(m, "name", None) or "")
            if not tool:
                tool = "Web search" if "web result" in content.lower() else "Vector search"
            steps.append({"title": tool, "detail": content})
        return steps
    if node == "relevancy_check":
        ok = bool(update.get("is_relevant"))
        return [{
            "title": "Relevant" if ok else "Low relevance",
            "detail": "Retrieved passages answer the question" if ok else "Passages look off-topic",
            "ok": ok,
        }]
    if node == "query_rewrite":
        return [{"title": "Query rewrite", "detail": f"Retrying with “{update.get('query', '')}”", "ok": False}]
    if node == "verify_claim":
        n = len(update.get("superseding_papers") or [])
        return [{"title": "Web + arXiv search", "detail": f"Found {n} newer paper(s) to compare against"}]
    if node == "generate_answer":
        return [{"title": "Answer", "detail": "Response written"}]
    return []


def render_assistant_extras(msg: dict) -> None:
    state = msg.get("graph_state") or {}
    sources = source_chips(state)
    if sources:
        st.markdown(f"<div style='margin-top:6px'>{sources}</div>", unsafe_allow_html=True)
    trace = trace_html([s for s in msg.get("steps", []) if not s.get("hidden")], msg.get("elapsed"))
    if trace:
        st.markdown(trace, unsafe_allow_html=True)
    if state:
        with st.expander("Inspect run", icon=":material/data_object:"):
            for s in msg.get("steps", []):
                st.markdown(f"**{s['title']}** — {s['detail']}")
            st.json(state, expanded=False)


def render_answer_body(msg: dict, key: str) -> None:
    state = msg.get("graph_state") or {}
    badge = route_badge(state)
    if badge:
        st.markdown(badge, unsafe_allow_html=True)
    if state.get("route") == "verify_claim":
        with st.container(border=True, key=f"claim_{key}"):
            st.markdown(fmt_math(msg["content"]))
    else:
        st.markdown(fmt_math(msg["content"]))


graph = get_graph()


STALL_DUMP_S = int(os.getenv("PAPEER_STALL_DUMP_S", "90"))


def _stream_graph(input_state: dict, config: dict, status, placeholder, steps: list[dict]) -> str:
    """Run the graph, updating the live trace and streaming the answer; returns the answer text."""
    response_text = ""
    # "updates" gives one event per finished node (for the live trace);
    # "messages" gives LLM tokens (for streaming the final answer).
    for mode, payload in graph.stream(input_state, config, stream_mode=["updates", "messages"]):
        if mode == "updates":
            for node, update in payload.items():
                if not update:
                    continue
                for step in describe_update(node, update):
                    steps.append(step)
                    status.markdown(f"**{step['title']}** — {step['detail']}")
                    status.update(label=f"{step['title']} · working…", expanded=True)
        else:
            chunk, metadata = payload
            if (
                metadata.get("langgraph_node") == "generate_answer"
                and hasattr(chunk, "content")
                and chunk.content
            ):
                if not response_text:
                    status.update(label="Writing answer…", expanded=True)
                response_text += chunk.content
                placeholder.markdown(fmt_math(response_text) + "▌")
    return response_text


VISITOR_ID = get_visitor_id()

# ── Bootstrap ──────────────────────────────────────────────────────────────────
if "sessions_meta" not in st.session_state:
    st.session_state.sessions_meta = load_sessions()
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "turns" not in st.session_state:
    st.session_state.turns = {}
if "active_session_id" not in st.session_state:
    if st.session_state.sessions_meta:
        latest = max(
            st.session_state.sessions_meta.values(),
            key=lambda s: s["created_at"],
        )
        switch_session(latest["id"])
    else:
        sid = create_session()
        st.session_state.active_session_id = sid

active_sid = st.session_state.active_session_id

# Highlight the active session in the sidebar list.
st.markdown(
    f"""<style>
.st-key-sess_{active_sid} button {{ background: #24272E; border-left: 2px solid #5DCAA5; }}
.st-key-sess_{active_sid} button p {{ color: #FAF8F3; }}
</style>""",
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="pp-brand">Pharma<em>Lens</em></div>'
        '<div class="pp-sub">Research copilot for pharma literature</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    if st.button("New research chat", icon=":material/add:", type="primary", use_container_width=True):
        new_sid = create_session()
        st.session_state.active_session_id = new_sid
        active_sid = new_sid
        st.rerun()

    st.markdown('<div class="pp-label">Sessions</div>', unsafe_allow_html=True)
    sorted_sessions = sorted(
        st.session_state.sessions_meta.values(),
        key=lambda s: s["created_at"],
        reverse=True,
    )
    for session in sorted_sessions:
        sid = session["id"]
        is_active = sid == st.session_state.active_session_id
        if st.button(
            session["name"],
            key=f"sess_{sid}",
            use_container_width=True,
            type="tertiary",
        ):
            if not is_active:
                switch_session(sid)
                st.rerun()

    st.write("")
    st.markdown('<div class="pp-label">Library</div>', unsafe_allow_html=True)
    tab_upload, tab_url, tab_arxiv = st.tabs(["Upload", "URL", "arXiv"])

    # ── Section 1: File upload ─────────────────────────────────────────────────
    with tab_upload:
        uploaded_files = st.file_uploader(
            "PDF, TXT, or Markdown",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
            key=f"uploader_{active_sid}",
            label_visibility="collapsed",
        )
        if st.button("Add to library", use_container_width=True, key="btn_add_files"):
            if uploaded_files:
                processed_key = f"processed_files_{active_sid}"
                if processed_key not in st.session_state:
                    st.session_state[processed_key] = set()
                with st.spinner("Chunking and embedding…"):
                    for f in uploaded_files:
                        if limit_reached("uploads"):
                            st.warning(LIMIT_MSG["uploads"])
                            break
                        if PUBLIC_MODE and f.size > MAX_FILE_MB * 1024 * 1024:
                            st.warning(f"{f.name} is over the {MAX_FILE_MB} MB demo limit.")
                            continue
                        if f.name in st.session_state[processed_key]:
                            st.info(f"Already loaded: {f.name}")
                            continue
                        suffix = Path(f.name).suffix
                        tmp_path = None
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                tmp.write(f.read())
                                tmp_path = tmp.name
                            docs = load_document(tmp_path)
                            for doc in docs:
                                doc.metadata["title"] = Path(f.name).stem
                            add_paper(docs, active_sid)
                            bump_usage("uploads")
                            st.session_state[processed_key].add(f.name)
                            st.success(f"Added: {f.name}")
                        except Exception as e:
                            st.error(f"Failed: {f.name} — {e}")
                        finally:
                            if tmp_path:
                                Path(tmp_path).unlink(missing_ok=True)
                st.rerun()
            else:
                st.warning("No files selected.")

    # ── Section 2: Web URL loader ──────────────────────────────────────────────
    with tab_url:
        url_input = st.text_area(
            "URLs (one per line)",
            key=f"url_area_{active_sid}",
            height=80,
            label_visibility="collapsed",
            placeholder="https://pubmed.ncbi.nlm.nih.gov/...\none URL per line",
        )
        if st.button("Load pages", use_container_width=True, key="btn_load_urls"):
            urls = [u.strip() for u in url_input.splitlines() if u.strip()]
            if urls:
                with st.spinner("Loading web pages…"):
                    for url in urls:
                        if limit_reached("uploads"):
                            st.warning(LIMIT_MSG["uploads"])
                            break
                        try:
                            docs = load_webpage(url)
                            add_paper(docs, active_sid)
                            bump_usage("uploads")
                            st.success(f"Loaded: {url[:60]}")
                        except Exception as e:
                            st.error(f"Failed: {url[:60]} — {e}")
                st.rerun()
            else:
                st.warning("Enter at least one URL.")

    # ── Section 3: ArXiv loader ────────────────────────────────────────────────
    with tab_arxiv:
        arxiv_title = st.text_input(
            "Paper title or ArXiv ID",
            key=f"arxiv_input_{active_sid}",
            label_visibility="collapsed",
            placeholder="arXiv ID or paper title",
        )
        if st.button("Fetch paper", use_container_width=True, key="btn_load_arxiv"):
            if limit_reached("uploads"):
                st.warning(LIMIT_MSG["uploads"])
            elif arxiv_title.strip():
                with st.spinner("Fetching from arXiv…"):
                    try:
                        docs = load_arxiv(arxiv_title.strip())
                        add_paper(docs, active_sid)
                        bump_usage("uploads")
                        loaded_title = docs[0].metadata.get("title") if docs else arxiv_title.strip()
                        st.success(f"Loaded: {loaded_title}")
                    except Exception as e:
                        st.error(f"Failed: {e}")
                st.rerun()
            else:
                st.warning("Enter a paper title or ArXiv ID.")

    # ── Loaded Documents list ──────────────────────────────────────────────────
    try:
        doc_titles = list_papers(active_sid)
    except Exception:
        doc_titles = None
    count = f" · {len(doc_titles)}" if doc_titles else ""
    st.markdown(f'<div class="pp-label">In this session{count}</div>', unsafe_allow_html=True)
    if doc_titles is None:
        st.markdown('<div class="pp-empty">Could not load the document list. Try refreshing.</div>',
                    unsafe_allow_html=True)
    elif doc_titles:
        st.markdown("".join(f'<div class="pp-doc">{_esc(t)}</div>' for t in doc_titles),
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="pp-empty">No papers yet. Add one above to start.</div>',
                    unsafe_allow_html=True)

    if PUBLIC_MODE:
        usage = get_usage()
        q_left = max(MAX_QUESTIONS - usage["questions"], 0)
        u_left = max(MAX_UPLOADS - usage["uploads"], 0)
        st.markdown(
            f'<div class="pp-label">Demo limits</div>'
            f'<div class="pp-empty">{q_left} of {MAX_QUESTIONS} questions · '
            f'{u_left} of {MAX_UPLOADS} papers left</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="pp-foot">Built by <b>{_esc(AUTHOR)}</b><br>{_esc(STACK)}</div>',
        unsafe_allow_html=True,
    )

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="pp-hero">'
    '<div class="t">Ask the evidence. <em>Check if it still holds.</em></div>'
    '<div class="d">Grounded answers from clinical studies and drug papers, claim checks '
    "against recent literature, and live web context when you need it.</div>"
    + _chip("Agentic RAG", "green")
    + _chip("Claim verification", "amber")
    + _chip("Live web search", "gray")
    + "</div>",
    unsafe_allow_html=True,
)

# ── Chat display ───────────────────────────────────────────────────────────────
USER_AVATAR = ":material/person:"
BOT_AVATAR = ":material/auto_stories:"

chats = st.session_state.chats.get(active_sid, [])
for i, msg in enumerate(chats):
    if msg["role"] == "user":
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            render_answer_body(msg, key=f"{active_sid}_{i}")
            render_assistant_extras(msg)

# ── Empty state ────────────────────────────────────────────────────────────────
SUGGESTIONS = [
    ("Summarize the study design, patient population and primary endpoints of my papers",
     "Summarize a study"),
    ("Verify this claim: metformin reduces cancer risk in patients with type 2 diabetes",
     "Verify a claim"),
    ("/btw What are the latest FDA drug approvals this month?", "Side question with /btw"),
]
empty_slot = st.empty()
if not chats:
    with empty_slot.container():
        st.caption("Try one of these")
        cols = st.columns(3)
        for idx, (col, (text, label)) in enumerate(zip(cols, SUGGESTIONS)):
            if col.button(f"**{label}**  \n{text}", key=f"suggest_{idx}", use_container_width=True):
                st.session_state.pending_prompt = text
                st.rerun()

# ── Chat input ─────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask about your studies, or verify a claim · /btw for side questions")
if not prompt and st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")

if prompt and limit_reached("questions"):
    empty_slot.empty()
    st.info(LIMIT_MSG["questions"], icon=":material/hourglass_empty:")
    prompt = None
elif prompt and PUBLIC_MODE:
    bump_usage("questions")

if prompt:
    empty_slot.empty()
    is_btw = prompt.strip().lower().startswith("/btw")

    if is_btw:
        query = prompt.strip()[4:].strip()

        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(_chip("Side question · not saved to this session", "gray"), unsafe_allow_html=True)
            if not query:
                st.markdown("Please add a question after `/btw`, e.g. `/btw What is a phase III trial?`")
            else:
                placeholder = st.empty()
                response_text = ""
                for chunk in handle_btw(query):
                    response_text += chunk
                    placeholder.markdown(fmt_math(response_text) + "▌")
                placeholder.markdown(fmt_math(response_text))

    else:
        if active_sid not in st.session_state.chats:
            st.session_state.chats[active_sid] = []
        if active_sid not in st.session_state.turns:
            st.session_state.turns[active_sid] = 0

        is_first_message = len(st.session_state.chats[active_sid]) == 0

        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)
        st.session_state.chats[active_sid].append({"role": "user", "content": prompt})
        st.session_state.turns[active_sid] += 1
        current_turn = st.session_state.turns[active_sid]

        if is_first_message:
            maybe_rename_session(active_sid, prompt)

        input_state = {
            "messages": [HumanMessage(content=prompt)],
            "session_id": active_sid,
            "query": prompt,
            "route": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "rewrite_count": 0,
        }
        config = {"configurable": {"thread_id": active_sid}}

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            badge_slot = st.empty()
            status = st.status("Routing your question…", expanded=True)
            placeholder = st.empty()
            steps: list[dict] = []
            started = time.perf_counter()

            # If a run is still going after STALL_DUMP_S, dump every thread's stack to the
            # app logs so a hang shows exactly where it is blocked.
            run_thread = threading.current_thread()

            def _dump_stalled_run():
                log.warning("graph run session=%s still running after %ss; thread stacks follow", active_sid[:8], STALL_DUMP_S)
                frames = sys._current_frames()
                for t in threading.enumerate():
                    frame = frames.get(t.ident)
                    if frame is not None:
                        mark = " (this run)" if t is run_thread else ""
                        log.warning("--- %s%s ---\n%s", t.name, mark, "".join(traceback.format_stack(frame)))

            watchdog = threading.Timer(STALL_DUMP_S, _dump_stalled_run)
            watchdog.daemon = True
            watchdog.start()
            log.info("graph run start session=%s first=%s", active_sid[:8], is_first_message)
            try:
                response_text = _stream_graph(input_state, config, status, placeholder, steps)
                run_error = None
            except Exception as exc:  # surface the failure instead of spinning forever
                log.exception("graph run failed session=%s", active_sid[:8])
                run_error = exc
            finally:
                watchdog.cancel()
            log.info("graph run end session=%s in %.1fs", active_sid[:8], time.perf_counter() - started)

            if run_error is not None:
                status.update(label="Pipeline failed", state="error", expanded=True)
                placeholder.error(f"Something went wrong while answering ({type(run_error).__name__}). Please try again.")
                st.stop()

            elapsed = time.perf_counter() - started
            visible = [s for s in steps if not s.get("hidden")]
            status.update(
                label=f"Pipeline · {len(visible)} steps · {elapsed:.1f}s",
                state="complete",
                expanded=False,
            )

            final_values = graph.get_state(config).values
            if not response_text:
                response_text = final_values.get("answer") or "No response generated."
            state_snapshot = _serialize_state(final_values)

            assistant_msg = {
                "role": "assistant",
                "content": response_text,
                "graph_state": state_snapshot,
                "turn": current_turn,
                "steps": steps,
                "elapsed": elapsed,
            }

            badge = route_badge(state_snapshot)
            if badge:
                badge_slot.markdown(badge, unsafe_allow_html=True)
            if state_snapshot.get("route") == "verify_claim":
                placeholder.empty()
                with st.container(border=True, key=f"claim_{active_sid}_live"):
                    st.markdown(fmt_math(response_text))
            else:
                placeholder.markdown(fmt_math(response_text))
            render_assistant_extras(assistant_msg)

        st.session_state.chats[active_sid].append(assistant_msg)

        if is_first_message:
            st.rerun()
