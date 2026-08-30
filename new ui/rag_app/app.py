"""
app.py — Document Intelligence
ChatGPT-style AI chat interface for the RAG pipeline.
"""

import streamlit as st
import time
from datetime import datetime
import os
import requests as _req
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv(".env")

st.set_page_config(
    page_title="Document Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from pipeline_client import RAGPipelineClient

# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS = {
    "chat_history":   [],
    "messages":       [],
    "view":           "chat",
    "_pending_msg":   None,
    "api_key":        os.environ.get("ANTHROPIC_API_KEY", ""),
    "top_k":          4,
    "dev_mode":       False,
    "conversations":  {},
    "active_conv_id": None,
    "conv_counter":   0,
    "last_trace":     None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Pipeline singleton ────────────────────────────────────────────────────────
def get_pipeline() -> RAGPipelineClient:
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = RAGPipelineClient("http://localhost:8000")
    p = st.session_state.rag_pipeline
    if st.session_state.api_key:
        p.set_api_key(st.session_state.api_key)
    p.config.top_k = st.session_state.top_k
    return p

# ── Backend health (direct – no stale cache) ──────────────────────────────────
def check_health() -> bool:
    try:
        r = _req.get("http://localhost:8000/api/documents", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

# ── Conversation helpers ──────────────────────────────────────────────────────
def _new_conv():
    st.session_state.conv_counter += 1
    cid = f"c{st.session_state.conv_counter}"
    st.session_state.conversations[cid] = {
        "title": "New chat", "messages": [], "chat_history": [],
        "ts": datetime.now().isoformat(),
    }
    st.session_state.active_conv_id = cid
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.last_trace = None
    st.session_state.view = "chat"

def _save():
    cid = st.session_state.active_conv_id
    if cid and cid in st.session_state.conversations:
        conv = st.session_state.conversations[cid]
        conv["messages"] = list(st.session_state.messages)
        conv["chat_history"] = list(st.session_state.chat_history)
        if conv["title"] == "New chat" and st.session_state.messages:
            for m in st.session_state.messages:
                if m["role"] == "user":
                    conv["title"] = m["content"][:30] + ("…" if len(m["content"]) > 30 else "")
                    break
        conv["ts"] = datetime.now().isoformat()

def _load(cid):
    _save()
    c = st.session_state.conversations[cid]
    st.session_state.active_conv_id = cid
    st.session_state.messages = list(c["messages"])
    st.session_state.chat_history = list(c["chat_history"])
    st.session_state.view = "chat"

def _del(cid):
    st.session_state.conversations.pop(cid, None)
    if st.session_state.active_conv_id == cid:
        st.session_state.active_conv_id = None
        st.session_state.messages = []
        st.session_state.chat_history = []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS — ChatGPT Classic dark
# KEY FIX: use :has(> span#composer-anchor) to select and fix the actual
#          Streamlit container block, because st.markdown HTML divs cannot
#          wrap st.columns children in Streamlit's DOM.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg:          #212121;
    --sb:          #171717;
    --surf:        #2F2F2F;
    --surf-h:      #3A3A3A;
    --bdr:         #424242;
    --txt:         #ECECEC;
    --txt2:        #B4B4B4;
    --txt3:        #8E8E8E;
    --green:       #10A37F;
    --user-pill:   #7C5CFC;
}

/* ── Reset chrome ── */
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stToolbar"] { display:none !important; }
[data-testid="stHeader"] { background:transparent !important; }

/* ── LOCK SIDEBAR — never collapse ── */
[data-testid="stSidebarCollapseButton"] { display:none !important; }
[data-testid="stSidebar"] {
    display:block !important;
    visibility:visible !important;
    width:260px !important;
    min-width:260px !important;
    max-width:260px !important;
    background:var(--sb) !important;
    border-right:1px solid var(--bdr) !important;
    transform:none !important;
}
[data-testid="stSidebarContent"] { padding:16px 12px !important; }

/* ── App shell ── */
html, body, .stApp { background:var(--bg) !important; color:var(--txt) !important;
    font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif !important; }

/* ── Main content: centered, fixed width ── */
.main .block-container {
    max-width:780px !important;
    margin:0 auto !important;
    padding:2rem 1.5rem 160px 1.5rem !important;
}

/* ── Chat messages reset ── */
[data-testid="stChatMessage"] { background:transparent !important; border:none !important; padding:10px 0 !important; }

/* USER bubble: RIGHT-aligned purple pill */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    display:flex !important;
    flex-direction:row !important;
    justify-content:flex-end !important;
    align-items:flex-start !important;
    gap:0 !important;
}
[data-testid="stChatMessageAvatarUser"] { display:none !important; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown {
    background:var(--user-pill) !important;
    color:#fff !important;
    border-radius:20px 20px 4px 20px !important;
    padding:12px 18px !important;
    max-width:72% !important;
    font-size:15px !important;
    line-height:1.65 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown p {
    color:#fff !important; margin:0 !important;
}

/* ASSISTANT: plain, no background */
[data-testid="stChatMessageAvatarAssistant"] {
    background:var(--surf) !important;
    border:1px solid var(--bdr) !important;
    border-radius:6px !important;
    flex-shrink:0 !important;
}

/* ── Sidebar buttons ── */
[data-testid="stSidebar"] .stButton > button {
    background:transparent !important; border:none !important;
    color:var(--txt) !important; font-size:14px !important; font-weight:500 !important;
    text-align:left !important; padding:8px 12px !important; border-radius:8px !important;
    width:100% !important; transition:background .15s !important; box-shadow:none !important; height:auto !important;
}
[data-testid="stSidebar"] .stButton > button:hover { background:var(--surf-h) !important; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background:var(--surf) !important; border:1px solid var(--bdr) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover { border-color:var(--green) !important; }

/* ── Generic inputs ── */
.stTextInput input { background:var(--surf) !important; border-color:var(--bdr) !important;
    color:var(--txt) !important; border-radius:10px !important; }
[data-testid="stFileUploader"] { background:var(--surf) !important;
    border:1px dashed var(--bdr) !important; border-radius:12px !important; padding:12px !important; }

/* ── Sources expander ── */
[data-testid="stExpander"] { background:transparent !important;
    border:1px solid var(--bdr) !important; border-radius:8px !important; margin-top:6px !important; }
[data-testid="stExpander"] summary { color:var(--txt2) !important; font-size:13px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:#424242; border-radius:4px; }

/* ── Suggestion buttons ── */
.sug-wrap > div > button {
    background:var(--surf) !important; border:1px solid var(--bdr) !important;
    border-radius:14px !important; padding:16px 18px !important; min-height:90px !important;
    text-align:left !important; transition:all .2s ease !important; box-shadow:none !important; width:100% !important;
}
.sug-wrap > div > button:hover { border-color:var(--green) !important; background:var(--surf-h) !important; }
.sug-wrap > div > button p { font-size:13px !important; color:var(--txt2) !important; margin:0 !important; }
.sug-wrap > div > button strong { font-size:14px !important; color:var(--txt) !important;
    display:block !important; margin-bottom:4px !important; }

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   COMPOSER — fixed at bottom using :has() on the real Streamlit
   container that holds the anchor span. This is the ONLY reliable
   way to fix a Streamlit block to screen position.
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
section.main div[data-testid="stVerticalBlock"]:has(> div > div > span#composer-anchor) {
    position:fixed !important;
    bottom:0 !important;
    left:260px !important;
    right:0 !important;
    z-index:500 !important;
    background:var(--bg) !important;
    padding:14px 20px 22px 20px !important;
    display:flex !important;
    flex-direction:column !important;
    align-items:center !important;
}

/* The horizontal block (columns) inside the composer */
section.main div[data-testid="stVerticalBlock"]:has(> div > div > span#composer-anchor)
    > div > div[data-testid="stHorizontalBlock"] {
    width:100% !important;
    max-width:780px !important;
    background:var(--surf) !important;
    border:1px solid var(--bdr) !important;
    border-radius:26px !important;
    padding:4px 10px !important;
    align-items:center !important;
    gap:0 !important;
    transition:border-color .15s !important;
}
section.main div[data-testid="stVerticalBlock"]:has(> div > div > span#composer-anchor)
    > div > div[data-testid="stHorizontalBlock"]:focus-within {
    border-color:var(--green) !important;
}

/* Kill inner borders/backgrounds */
section.main div[data-testid="stVerticalBlock"]:has(> div > div > span#composer-anchor) input {
    background:transparent !important; border:none !important; box-shadow:none !important;
    color:var(--txt) !important; font-size:15px !important;
}
section.main div[data-testid="stVerticalBlock"]:has(> div > div > span#composer-anchor) input::placeholder {
    color:var(--txt3) !important;
}
section.main div[data-testid="stVerticalBlock"]:has(> div > div > span#composer-anchor) button {
    background:transparent !important; border:none !important; box-shadow:none !important;
    color:var(--txt2) !important; border-radius:50% !important; min-width:34px !important; height:34px !important;
}
section.main div[data-testid="stVerticalBlock"]:has(> div > div > span#composer-anchor) button:hover {
    background:rgba(255,255,255,.08) !important; color:var(--txt) !important;
}
section.main div[data-testid="stVerticalBlock"]:has(> div > div > span#composer-anchor) [data-testid="stForm"] {
    background:transparent !important; border:none !important; padding:0 !important;
}
section.main div[data-testid="stVerticalBlock"]:has(> div > div > span#composer-anchor)
    .stTextInput > div, 
section.main div[data-testid="stVerticalBlock"]:has(> div > div > span#composer-anchor)
    .stTextInput > div > div {
    background:transparent !important; border:none !important; box-shadow:none !important;
}

/* ── Typography & misc ── */
p, li { color:var(--txt); }
.stMarkdown p { font-size:15px !important; line-height:1.7 !important; }
code { background:#1a1a1a !important; color:#e5c07b !important; border-radius:4px; padding:2px 6px; }
pre  { background:#1a1a1a !important; border-radius:8px !important; padding:16px !important; }
hr   { border-color:var(--bdr) !important; margin:10px 0 !important; }

.sec-label { font-size:11px; font-weight:600; letter-spacing:.06em; text-transform:uppercase;
    color:var(--txt3); padding:20px 4px 6px; display:block; }
.src-card { border-left:2px solid var(--green); padding:4px 0 4px 12px; margin-bottom:12px; }
.src-doc  { font-weight:600; font-size:13px; color:var(--txt); margin-bottom:3px; }
.src-txt  { font-size:13px; color:var(--txt2); line-height:1.5; }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;padding:4px 4px 18px;">'
        '<div style="width:30px;height:30px;border-radius:7px;background:#10A37F;color:#fff;'
        'display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;">DI</div>'
        '<span style="font-size:15px;font-weight:600;">Document Intelligence</span></div>',
        unsafe_allow_html=True,
    )

    if st.button("New chat", icon=":material/add:", key="new_chat_btn", use_container_width=True, type="primary"):
        _save()
        _new_conv()
        st.rerun()

    if st.button("Library", icon=":material/auto_stories:", key="nav_lib", use_container_width=True):
        _save()
        st.session_state.view = "documents"
        st.rerun()

    if st.button("Settings", icon=":material/settings:", key="nav_set", use_container_width=True):
        _save()
        st.session_state.view = "settings"
        st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Health — direct call, no cache
    if check_health():
        st.markdown("🟢 **Backend Connected**")
    else:
        st.markdown("🔴 **Backend Offline** — start uvicorn")

    # Recent chats
    recents = {cid: c for cid, c in st.session_state.conversations.items() if c.get("messages")}
    if recents:
        st.markdown('<span class="sec-label">Recent</span>', unsafe_allow_html=True)
        for cid, conv in sorted(recents.items(), key=lambda x: x[1].get("ts",""), reverse=True)[:15]:
            ca, cb = st.columns([7, 1])
            with ca:
                if st.button(conv["title"], key=f"r_{cid}", use_container_width=True):
                    _load(cid)
                    st.rerun()
            with cb:
                with st.popover("⋮", use_container_width=True):
                    if st.button("Delete", key=f"d_{cid}", use_container_width=True):
                        _del(cid)
                        st.rerun()

    st.markdown("---")

    # KB / model info
    p = get_pipeline()
    try:
        stats = p.stats()
        nd, nc = stats.get("total_documents", 0), stats.get("total_chunks", 0)
        if nc > 0:
            st.markdown(f'<div style="font-size:12px;color:#8E8E8E;padding:0 4px 10px;">● {nd} doc{"s" if nd!=1 else ""} · {nc:,} chunks</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:12px;color:#8E8E8E;padding:0 4px 10px;">○ No documents indexed</div>', unsafe_allow_html=True)
    except Exception:
        pass

    if p.config.llm_provider == "ollama":
        st.markdown(
            f'<div style="padding:10px;border-radius:8px;background:#2F2F2F;border:1px solid #424242;font-size:12px;">'
            f'🧠 <b>Local LLM</b><br><span style="color:#8E8E8E;">Model: <code style="font-size:11px;">{p.config.llm_model}</code></span></div>',
            unsafe_allow_html=True
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEWS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if st.session_state.view == "chat":

    # ── HOME (empty state) ────────────────────────────────────────────────────
    if not st.session_state.messages and not st.session_state._pending_msg:
        st.markdown(
            '<div style="text-align:center;padding:90px 0 36px;">'
            '<h1 style="font-size:28px;font-weight:600;color:#ECECEC;margin:0;">'
            "What's on the agenda today?"
            '</h1></div>',
            unsafe_allow_html=True
        )
        SUGS = [
            ("📝", "Summarize document",   "Summarize the main points of the uploaded documents."),
            ("📖", "Explain key concepts", "Explain the key concepts from the documents in simple terms."),
            ("🔍", "Find information",     "What are the most important facts mentioned in the documents?"),
            ("❓", "Ask about docs",       "Answer my questions using the uploaded documents."),
        ]
        _, mid, _ = st.columns([1, 3, 1])
        with mid:
            sc1, sc2 = st.columns(2)
            for i, (icon, label, query) in enumerate(SUGS):
                col = sc1 if i % 2 == 0 else sc2
                with col:
                    st.markdown('<div class="sug-wrap">', unsafe_allow_html=True)
                    if st.button(f"**{icon}  {label}**\n\n{query}", key=f"sug_{i}", use_container_width=True):
                        if not st.session_state.active_conv_id:
                            _new_conv()
                        st.session_state._pending_msg = query
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

    # ── ACTIVE CHAT ───────────────────────────────────────────────────────────
    else:
        for idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])
                    # Sources
                    sources = msg.get("sources", [])
                    if sources:
                        with st.expander(f"📚 Sources · {len(sources)}", expanded=False):
                            for s in sources:
                                pg = f" · p.{s.get('page_number')}" if s.get("page_number") else ""
                                st.markdown(
                                    f'<div class="src-card">'
                                    f'<div class="src-doc">📄 {s.get("document_name","Unknown")}{pg}</div>'
                                    f'<div class="src-txt">{s.get("text","")[:280]}…</div>'
                                    f'</div>', unsafe_allow_html=True
                                )
                    # Actions
                    mid = msg.get("id", str(idx))
                    fb  = msg.get("feedback")
                    a1, a2, a3, _ = st.columns([0.06, 0.06, 0.06, 0.82])
                    with a1:
                        if st.button("👍" if fb=="like" else "👍🏻", key=f"lk_{mid}", type="tertiary"):
                            msg["feedback"] = None if fb=="like" else "like"; st.rerun()
                    with a2:
                        if st.button("👎" if fb=="dislike" else "👎🏻", key=f"dl_{mid}", type="tertiary"):
                            msg["feedback"] = None if fb=="dislike" else "dislike"; st.rerun()
                    with a3:
                        esc = msg["content"].replace("`","\\`").replace("$","\\$")
                        components.html(f"""
                        <style>body{{margin:0;overflow:hidden;}}
                        button{{background:transparent;border:none;width:28px;height:28px;padding:0;
                               border-radius:6px;color:#8E8E8E;cursor:pointer;font-size:16px;transition:.15s;
                               display:flex;align-items:center;justify-content:center;}}
                        button:hover{{background:rgba(255,255,255,.08);color:#ECECEC;}}</style>
                        <button onclick="navigator.clipboard.writeText(`{esc}`);
                            this.style.color='#10A37F';
                            setTimeout(()=>this.style.color='#8E8E8E',2000);">📋</button>
                        """, height=28)


    # ── COMPOSER — always rendered at bottom via :has() CSS ───────────────────
    with st.container():
        # This span is the anchor :has() targets in CSS
        st.markdown('<span id="composer-anchor"></span>', unsafe_allow_html=True)

        cc_plus, cc_form = st.columns([0.07, 0.93])

        with cc_plus:
            with st.popover("➕", use_container_width=True):
                st.markdown("**📎 Upload Documents**")
                uploads = st.file_uploader(
                    "files", accept_multiple_files=True,
                    type=["pdf","txt","md","docx"], label_visibility="collapsed"
                )
                if uploads:
                    pipe = get_pipeline()
                    with st.spinner("Uploading…"):
                        errs = []
                        for f in uploads:
                            res = pipe.ingest_file(f.getvalue(), f.name)
                            if not res.success:
                                errs.append(f.name)
                    if errs:
                        st.error("Failed: " + ", ".join(errs))
                    else:
                        st.success("✅ Uploaded!")
                st.markdown("---")
                if st.button("🗑️ Clear chat", use_container_width=True, key="clr"):
                    if st.session_state.active_conv_id:
                        st.session_state.messages = []
                        st.session_state.chat_history = []
                        _save()
                    st.rerun()

        with cc_form:
            with st.form("chat_form", border=False, clear_on_submit=True):
                fi1, fi2 = st.columns([0.91, 0.09])
                user_input = fi1.text_input(
                    "q", label_visibility="collapsed",
                    placeholder="Ask anything about your documents…"
                )
                sent = fi2.form_submit_button("➤", use_container_width=True)
                if sent and user_input:
                    if not st.session_state.active_conv_id:
                        _new_conv()
                    st.session_state._pending_msg = user_input
                    st.rerun()

    # ── Process pending ───────────────────────────────────────────────────────
    if st.session_state._pending_msg:
        prompt = st.session_state._pending_msg
        st.session_state._pending_msg = None

        if not st.session_state.active_conv_id:
            _new_conv()

        pipeline = get_pipeline()
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            docs_list = pipeline.list_documents()
            if not docs_list:
                ans = "⚠️ No documents indexed yet. Use the **+** button to upload documents."
                st.markdown(ans)
                st.session_state.messages.append({"role":"assistant","content":ans,"sources":[]})
            else:
                try:
                    with st.spinner("Thinking…"):
                        result = pipeline.ask(query=prompt, chat_history=st.session_state.chat_history)
                    st.markdown(result.answer)

                    srcs = [
                        {"source":s.source,"document_name":s.document_name,
                         "page_number":s.page_number,"text":s.text,"score":s.score}
                        for s in result.sources
                    ]
                    st.session_state.messages.append({
                        "id": f"m{int(time.time()*1000)}",
                        "role":"assistant","content":result.answer,
                        "sources":srcs,"feedback":None
                    })
                    st.session_state.last_trace = result.trace
                except Exception as e:
                    ans = f"⚠️ **Error:** {e}"
                    st.error(ans)
                    st.session_state.messages.append({"role":"assistant","content":ans,"sources":[]})

        st.session_state.chat_history.append({"role":"user","content":prompt})
        st.session_state.chat_history.append({"role":"assistant","content":st.session_state.messages[-1]["content"]})
        _save()
        st.rerun()


# ── LIBRARY ───────────────────────────────────────────────────────────────────
elif st.session_state.view == "documents":
    st.markdown('<h2 style="font-size:24px;font-weight:600;margin-bottom:20px;">Library</h2>', unsafe_allow_html=True)
    p = get_pipeline()
    docs = p.list_documents()
    if docs:
        for d in docs:
            st.markdown(f"📄 `{d}`")
    else:
        st.markdown('<p style="color:#8E8E8E;">No documents yet. Use the + button in chat to upload.</p>', unsafe_allow_html=True)


# ── SETTINGS ─────────────────────────────────────────────────────────────────
elif st.session_state.view == "settings":
    st.markdown('<h2 style="font-size:24px;font-weight:600;margin-bottom:8px;">Settings</h2>', unsafe_allow_html=True)
    p = get_pipeline()
    st.markdown("#### 🧠 Model Configuration")

    use_cloud = st.checkbox("Use Cloud API Key", value=(p.config.llm_provider in ["claude","openai","gemini"]))
    if use_cloud:
        st.info("Local LLM will be disabled. You'll be billed by your provider.")
        pvs   = {"Anthropic Claude":"claude","OpenAI GPT":"openai","Google Gemini":"gemini"}
        cur   = next((k for k,v in pvs.items() if v==p.config.llm_provider),"Anthropic Claude")
        sname = st.selectbox("Provider", list(pvs.keys()), index=list(pvs.keys()).index(cur))
        sel   = pvs[sname]
        dflts = {"claude":"claude-3-5-sonnet-20240620","openai":"gpt-4o","gemini":"gemini-1.5-pro"}
        hints = {"claude":"sk-ant-...","openai":"sk-...","gemini":"AIzaSy..."}
        new_k = st.text_input(f"API Key ({hints[sel]})", value=st.session_state.api_key, type="password")
        new_m = st.text_input("Model", value=p.config.llm_model if p.config.llm_provider==sel else dflts[sel])
        if st.button("Save", type="primary"):
            if new_k.strip():
                st.session_state.api_key = new_k.strip()
                p.set_api_key(new_k.strip())
                p.config.llm_provider = sel
                p.config.llm_model = new_m
                st.success(f"Saved! Using {sname}.")
            else:
                st.error("Enter a valid API key.")
    else:
        if p.config.llm_provider != "ollama":
            p.config.llm_provider = "ollama"
        st.success("Using Local Ollama — no API key required.")
        nm = st.text_input("Model name", value=p.config.llm_model or "qwen2:0.5b")
        if nm != p.config.llm_model:
            p.config.llm_model = nm
            st.rerun()

    st.markdown("---")
    st.session_state.dev_mode = st.toggle("Developer Mode (RAG Trace)", value=st.session_state.dev_mode)
