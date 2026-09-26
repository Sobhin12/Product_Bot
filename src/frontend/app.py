"""Streamlit UI: ask questions about a policy, and manage the policy documents.

    streamlit run src/frontend/app.py

Talks to the API at POLICY_BOT_API_URL (default http://localhost:8765).
"""
import os
from collections.abc import Iterator
from pathlib import Path

import requests
import streamlit as st

API_URL = os.environ.get("POLICY_BOT_API_URL", "http://localhost:8765").rstrip("/")
ACTIVE = {"pending", "scanning", "parsing", "chunking", "embedding"}
QUERY_TIMEOUT = (10, 120)

# Policies offered in the chat. Choosing one sends it as the prefix that selects documents:
# "ReAssure 3.0" covers "ReAssure 3.0 Policy Wordings", "ReAssure 3.0 CIS", ...
POLICIES = ["ReAssure 3.0"]

st.set_page_config(page_title="Policy Bot", page_icon="📄", layout="wide")


def api(method: str, path: str, **kwargs) -> requests.Response | None:
    try:
        return requests.request(method, f"{API_URL}{path}", timeout=kwargs.pop("timeout", 30), **kwargs)
    except requests.RequestException:
        st.error(f"Cannot reach the API at {API_URL}. Is it running?")
        return None


def documents() -> list[dict]:
    r = api("GET", "/documents", params={"limit": 200})
    return r.json() if r is not None and r.ok else []


def detail(r: requests.Response) -> str:
    try:
        return str(r.json().get("detail", r.text))
    except ValueError:
        return r.text or f"HTTP {r.status_code}"


def stream_answer(query: str, policy: str) -> Iterator[str]:
    try:
        with requests.post(
            f"{API_URL}/query", json={"query": query, "policy": policy}, stream=True, timeout=QUERY_TIMEOUT
        ) as r:
            if not r.ok:
                yield f"Error: {detail(r)}"
                return
            r.encoding = "utf-8"
            yield from r.iter_content(chunk_size=None, decode_unicode=True)
    except requests.RequestException:
        yield "Something went wrong. Please try again in a moment."


# ---- chat ------------------------------------------------------------------


def chat_tab() -> None:
    policy = st.radio("Choose a policy", POLICIES, index=None, horizontal=True, key="policy")
    if policy != st.session_state.get("chat_policy"):
        st.session_state.chat, st.session_state.chat_policy = [], policy  # a new policy starts a fresh conversation
    if policy is None:
        st.info("Select a policy above to start asking questions.")
    else:
        st.caption(f"Answering from documents whose name starts with “{policy}”. Each question stands alone.")

    for role, text in st.session_state.chat:
        st.chat_message(role).write(text)

    if question := st.chat_input("Ask about the policy", disabled=policy is None):
        st.chat_message("user").write(question)
        with st.chat_message("assistant"):
            answer = st.write_stream(stream_answer(question, policy))
        st.session_state.chat += [("user", question), ("assistant", answer)]


# ---- documents -------------------------------------------------------------


def upload_form() -> None:
    st.subheader("Upload a document")
    file = st.file_uploader("PDF", type=["pdf"])
    name = st.text_input(
        "Document name",
        value=Path(file.name).stem if file else "",
        help="Must be unique. Start it with the policy name, e.g. 'ReAssure 3.0 Policy Wordings'.",
    )
    is_update = st.checkbox("This replaces an existing document")
    replaces = None
    if is_update:
        choices = {f"{d['display_name']} ({d['status']})": d["doc_id"] for d in documents()}
        label = st.selectbox("Document to replace", list(choices)) if choices else None
        replaces = choices.get(label) if label else None
        if not choices:
            st.caption("There are no documents to replace.")

    if st.button("Upload", type="primary", disabled=not (file and name.strip()) or (is_update and not replaces)):
        r = api(
            "POST",
            "/upload",
            files={"file": (file.name, file.getvalue(), "application/pdf")},
            data={"display_name": name.strip(), "is_update": str(is_update).lower(), "replaces_doc_id": replaces or ""},
            timeout=120,
        )
        if r is None:
            return
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code == 202:
            st.success("Uploaded. Processing has started.")
        elif r.ok and body.get("status") == "duplicate":
            st.info("This exact file is already uploaded, so nothing was changed.")
        elif r.ok and body.get("status") == "no_changes":
            st.info("The file is identical to the document it would replace: no changes.")
        else:
            st.error(detail(r))


@st.fragment(run_every=3)
def documents_panel() -> None:
    docs = documents()
    st.subheader("Documents")
    if not docs:
        st.caption("No documents yet.")
        return
    st.dataframe(
        [
            {"Name": d["display_name"], "Status": d["status"], "Retries": d["retry_count"],
             "Uploaded": d["created_at"][:16].replace("T", " ")}
            for d in docs
        ],
        hide_index=True,
        use_container_width=True,
    )
    with st.expander("Manage a document"):
        by_label = {f"{d['display_name']} ({d['status']})": d for d in docs}
        doc = by_label[st.selectbox("Document", list(by_label), key="manage")]
        if doc["error_message"]:
            st.warning(doc["error_message"])
        left, right = st.columns(2)
        if left.button("Retry", disabled=doc["status"] != "failed"):
            r = api("POST", f"/documents/{doc['doc_id']}/retry")
            if r is not None and not r.ok:
                st.error(detail(r))
            else:
                st.rerun(scope="fragment")
        if right.button("Delete", disabled=doc["status"] in ACTIVE):
            r = api("DELETE", f"/documents/{doc['doc_id']}")
            if r is not None and not r.ok:
                st.error(detail(r))
            else:
                st.rerun(scope="fragment")


def documents_tab() -> None:
    upload_form()
    st.divider()
    documents_panel()


if "chat" not in st.session_state:
    st.session_state.chat = []

chat, docs = st.tabs(["Chat", "Documents"])
with chat:
    chat_tab()
with docs:
    documents_tab()
