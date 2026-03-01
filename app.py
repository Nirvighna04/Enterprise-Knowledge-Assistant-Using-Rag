"""
app.py — Streamlit multi-page Knowledge Assistant (RAG Chatbot)

Pages:
    1. Login / Register (with OTP)
    2. Chatbot  (department selector, file uploader, chat interface)
    3. History  (past questions + answers)

Run:
    streamlit run app.py
"""

import os
import shutil
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # reads .env file

import db
import auth
import rag

db.init_db()   # ensure tables exist

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
)

# ── Session state helpers ─────────────────────────────────────────────────────
def is_logged_in() -> bool:
    return st.session_state.get("user_id") is not None

def get_user_id() -> int:
    return st.session_state["user_id"]

def get_user_email() -> str:
    return st.session_state.get("user_email", "")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Login / Register
# ══════════════════════════════════════════════════════════════════════════════

def page_login():
    st.title("🧠 Knowledge Assistant")
    st.subheader("Login or Register")

    tab_login, tab_register = st.tabs(["Login", "Register"])

    # ── LOGIN ──────────────────────────────────────────────────────────────────
    with tab_login:
        email    = st.text_input("Email",    key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", use_container_width=True):
            ok, result = auth.login_user(email, password)

            if ok:
                st.session_state["user_id"]    = result.id
                st.session_state["user_email"] = result.email
                st.session_state["page"]       = "chatbot"
                st.rerun()
            elif result == "no_user":
                st.error("No account found with that email. Please register first.")
            elif result == "bad_pass":
                st.error("Incorrect password.")

    # ── REGISTER ───────────────────────────────────────────────────────────────
    with tab_register:
        reg_email = st.text_input("Email",            key="reg_email")
        reg_pass  = st.text_input("Password",         type="password", key="reg_pass")
        reg_pass2 = st.text_input("Confirm Password", type="password", key="reg_pass2")

        if st.button("Register", use_container_width=True):
            if reg_pass != reg_pass2:
                st.error("Passwords do not match.")
            elif len(reg_pass) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                ok, result = auth.register_user(reg_email, reg_pass)
                if ok:
                    st.success("✅ Registered successfully! Please login.")
                elif result == "exists":
                    st.error("Email already registered. Please login.")




# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Chatbot
# ══════════════════════════════════════════════════════════════════════════════

DEPARTMENTS = ["Engineering", "HR", "Finance", "Sales", "Legal", "Operations", "Marketing", "General"]
JOB_ROLES   = ["Manager", "Analyst", "Engineer", "Executive", "Intern", "Consultant", "Director", "Employee"]

def page_chatbot():
    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header(f"👤 {get_user_email()}")
        st.divider()

        department = st.selectbox("Department", DEPARTMENTS)
        job_role   = st.selectbox("Job Role",   JOB_ROLES)

        st.divider()
        st.subheader("📂 Upload Documents")
        uploaded = st.file_uploader(
            "PDF / DOCX / TXT",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

        if uploaded and st.button("Ingest Documents"):
            for ufile in uploaded:
                dest = os.path.join(rag.DATA_DIR, ufile.name)
                with open(dest, "wb") as f:
                    shutil.copyfileobj(ufile, f)
                with st.spinner(f"Ingesting {ufile.name}…"):
                    n = rag.ingest_document(dest)
                db.save_document(get_user_id(), ufile.name, dest)
                st.success(f"✅ {ufile.name} — {n} chunks indexed")

        st.divider()
        st.subheader("📋 Indexed Documents")
        docs = db.get_user_documents(get_user_id())
        if docs:
            for d in docs:
                st.markdown(f"- `{d['doc_name']}`")
        else:
            st.caption("No documents yet.")

        st.divider()
        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()

    # ── Main chat area ─────────────────────────────────────────────────────────
    st.title("🧠 Knowledge Assistant")
    st.caption(f"Answering strictly from your uploaded documents | Model: `{rag.OLLAMA_MODEL}`")

    # Initialise in-session chat messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display past messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    question = st.chat_input("Ask a question about your documents…")
    if question:
        # Show user message
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("Searching documents…"):
                result = rag.rag_query(question, department, job_role)

            answer     = result["answer"]
            citations  = result["citations"]
            confidence = result["confidence"]

            st.markdown(answer)

            # Citations
            if citations:
                st.divider()
                st.markdown("**📎 Citations**")
                seen = set()
                for c in citations:
                    key = (c["doc_name"], c["page"], c["line"])
                    if key not in seen:
                        seen.add(key)
                        st.markdown(
                            f"- 📄 `{c['doc_name']}` — Page **{c['page']}**, Line **{c['line']}**"
                        )

            # Confidence badge
            colour = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(confidence, "⚪")
            st.markdown(f"**Confidence:** {colour} {confidence}")

        # Save to session and DB
        full_response = f"{answer}\n\nConfidence: {confidence}"
        st.session_state["messages"].append({"role": "assistant", "content": full_response})
        db.save_chat(get_user_id(), question, answer, department, job_role)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — History
# ══════════════════════════════════════════════════════════════════════════════

def page_history():
    st.title("📜 Chat History")
    st.caption("Your last 20 questions & answers")

    if st.button("← Back to Chat"):
        st.session_state["page"] = "chatbot"
        st.rerun()

    history = db.get_chat_history(get_user_id(), limit=20)

    if not history:
        st.info("No history yet. Start chatting!")
        return

    for i, item in enumerate(history, 1):
        with st.expander(f"Q{i}: {item['question'][:80]}…", expanded=False):
            st.markdown(f"**Department:** {item['department']} | **Role:** {item['job_role']}")
            st.markdown(f"**⏰ {item['timestamp'].strftime('%Y-%m-%d %H:%M')}**")
            st.markdown("---")
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(f"**Answer:**\n\n{item['answer']}")


# ══════════════════════════════════════════════════════════════════════════════
# Navigation
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Default page
    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    # If logged in, show top nav
    if is_logged_in():
        col1, col2, col3 = st.columns([3, 1, 1])
        with col2:
            if st.button("💬 Chat"):
                st.session_state["page"] = "chatbot"
                st.rerun()
        with col3:
            if st.button("📜 History"):
                st.session_state["page"] = "history"
                st.rerun()

    # Route
    page = st.session_state["page"]

    if not is_logged_in() and page != "login":
        page = "login"

    if page == "login":
        page_login()
    elif page == "chatbot":
        page_chatbot()
    elif page == "history":
        page_history()


if __name__ == "__main__":
    main()
