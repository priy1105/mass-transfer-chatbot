"""One-service study workspace. Run: streamlit run streamlit_app.py."""
import hashlib
import logging
import os
from dataclasses import asdict
from uuid import uuid4

import groq
import streamlit as st
from dotenv import load_dotenv
from history_store import HistoryStore, HistoryError
from attachments import VISION_MODEL, extract_attachment, prepare_image

from chatbot_core import (
    DEFAULT_MODEL, MODES, TOPICS, PDF_ONLY, PDF_AND_ONLINE, KnowledgeIndex, build_messages, dimensionless,
    export_markdown, fick_flux, load_library, read_pdf, retrieval_query, stream_answer, normalize_math_markdown, Passage,
)

load_dotenv()
LOGGER = logging.getLogger(__name__)


def setting(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return os.getenv(name, default)


@st.cache_resource(show_spinner=False)
def library_index() -> KnowledgeIndex:
    return KnowledgeIndex(load_library())


def initialize() -> None:
    defaults = {"chats": {}, "active_chat": "", "upload_index": None, "upload_digest": "", "upload_names": [], "pending": None, "source_mode": PDF_ONLY, "cloud_auth": None, "unsaved_chats": set(), "cloud_error": "", "attachment_data": {}, "attachment_previews": {}}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if not st.session_state.active_chat:
        new_chat()


def new_chat() -> None:
    chat_id = uuid4().hex
    st.session_state.chats[chat_id] = {"title": "New conversation", "messages": []}
    st.session_state.active_chat = chat_id
    st.session_state.pending = None
    st.session_state.pop("retry_question", None)
    st.session_state.workspace_nav = "Chat"
    st.session_state.source_mode = PDF_ONLY


def cloud_store() -> HistoryStore:
    return HistoryStore(setting("SUPABASE_URL"), setting("SUPABASE_PUBLISHABLE_KEY"))


def sync_history() -> None:
    if not st.session_state.cloud_auth:
        return
    try:
        store = cloud_store()
        for chat_id in list(st.session_state.unsaved_chats):
            store.save(st.session_state.cloud_auth, chat_id, st.session_state.chats[chat_id])
            st.session_state.unsaved_chats.discard(chat_id)
        st.session_state.cloud_error = ""
    except HistoryError as exc:
        st.session_state.cloud_error = str(exc)


def account_submit(action: str) -> None:
    email = st.session_state.get("account_email", "").strip()
    password = st.session_state.get("account_password", "")
    st.session_state.account_password = ""
    if not email or not password:
        st.session_state.account_notice = "Enter your email and password."
        return
    try:
        store = cloud_store()
        if action == "Create account":
            if len(password) < 8:
                st.session_state.account_notice = "Choose a password with at least 8 characters."
                return
            store.sign_up(email, password)
            st.session_state.account_notice = "Check your email to confirm your account, then sign in. If email confirmation is disabled in your project, you can sign in immediately."
            return
        session = store.sign_in(email, password)
        saved = store.load(session)
        guest = {key: chat for key, chat in st.session_state.chats.items() if chat["messages"]}
        saved.update(guest)
        st.session_state.cloud_auth = session
        st.session_state.chats = saved
        st.session_state.active_chat = next(reversed(saved), "")
        st.session_state.unsaved_chats = set(guest)
        st.session_state.account_notice = ""
        st.session_state.pending = None
        st.session_state.pop("retry_question", None)
        sync_history()
        if not saved:
            new_chat()
    except HistoryError as exc:
        st.session_state.account_notice = str(exc)


def sign_out() -> None:
    try:
        cloud_store().sign_out(st.session_state.cloud_auth)
    except HistoryError:
        pass
    st.session_state.cloud_auth = None
    st.session_state.chats = {}
    st.session_state.unsaved_chats = set()
    st.session_state.cloud_error = ""
    st.session_state.upload_index = None
    st.session_state.upload_names = []
    st.session_state.upload_digest = ""
    st.session_state.attachment_data = {}
    st.session_state.attachment_previews = {}
    st.session_state.pop("notes_upload", None)
    new_chat()


def account_controls() -> None:
    configured = bool(setting("SUPABASE_URL") and setting("SUPABASE_PUBLISHABLE_KEY"))
    if not configured:
        st.caption("Guest mode · export chats to keep a copy.")
        return
    if st.session_state.cloud_auth:
        st.caption("Signed in as " + st.session_state.cloud_auth["user"].get("email", "your account"))
        if st.session_state.cloud_error:
            st.warning(st.session_state.cloud_error)
            st.button("Retry saving history", on_click=sync_history, width="stretch")
        elif st.session_state.unsaved_chats:
            st.caption("Some conversations are waiting to be saved.")
        else:
            st.caption("Conversations saved to your account.")
        st.button("Sign out", on_click=sign_out, width="stretch")
    else:
        with st.expander("Sign in to save your history", icon=":material/account_circle:"):
            st.caption("Your conversations and cited excerpts will be stored in your account. Sign in again after a restart to restore them on any device. Existing guest chats will be saved when you sign in; PDF uploads stay temporary.")
            action = st.radio("Account", ["Sign in", "Create account"], key="account_action", label_visibility="collapsed")
            with st.form("account_form", clear_on_submit=False):
                st.text_input("Email", key="account_email")
                st.text_input("Password", type="password", key="account_password")
                st.form_submit_button(action, on_click=account_submit, args=(action,), width="stretch")
            if st.session_state.get("account_notice"):
                st.info(st.session_state.account_notice)


def select_chat(chat_id: str) -> None:
    st.session_state.active_chat = chat_id
    st.session_state.workspace_nav = "Chat"
    st.session_state.pending = None
    st.session_state.pop("retry_question", None)


def queue(question: str, mode: str | None = None) -> None:
    st.session_state.pending = question
    if mode:
        st.session_state.mode = mode


def render_sources(sources: list[dict]) -> None:
    if sources:
        with st.expander(f"Reference passages · {len(sources)}"):
            st.caption("PDF pages count from the first page of the file. These passages were retrieved for this answer; check that each supports its claims.")
            for number, source in enumerate(sources, 1):
                kind = source.get("kind", "pdf")
                location = "Image · AI transcription" if kind == "image" else f"PDF page {source['page']}" + (" · AI transcription" if kind == "scan" else "")
                st.markdown(f"**[{number}] {source['source']} · {location}**")
                st.text(source["text"])


def render_attachments(items: list[dict]) -> None:
    for item in items:
        preview = st.session_state.attachment_previews.get(item["id"])
        if preview:
            st.image(preview, width=320, alt="Attached photo: " + item["name"])
        st.caption(":material/attach_file: " + item["name"])


def ingest_attachments(files: list, api_key: str) -> tuple[list[Passage], list[dict]]:
    if len(files) > 3:
        raise ValueError("Attach up to 3 files at a time.")
    current, labels, new_entries, previews = [], [], {}, {}
    chat_id = st.session_state.active_chat
    existing = st.session_state.attachment_data.get(chat_id, {})
    # Commit session changes only after every file has been processed successfully.
    for file in files:
        data = file.getvalue()
        if not data or len(data) > 10 * 1024 * 1024:
            raise ValueError("Each attachment must be non-empty and no larger than 10 MB.")
        digest = hashlib.sha256(file.name.encode() + data).hexdigest()
        passages = existing.get(digest) or new_entries.get(digest)
        if passages is None:
            passages = extract_attachment(data, file.name, api_key, setting("GROQ_VISION_MODEL", VISION_MODEL))
            new_entries[digest] = passages
        current.extend(passages)
        labels.append({"id": digest, "name": file.name})
        if passages and passages[0].kind == "image":
            previews[digest] = prepare_image(data)
    st.session_state.attachment_data[chat_id] = {**existing, **new_entries}
    st.session_state.attachment_previews.update(previews)
    while len(st.session_state.attachment_previews) > 20:
        del st.session_state.attachment_previews[next(iter(st.session_state.attachment_previews))]
    return current, labels


def render_answer_stream(chunks) -> str:
    # Normalize the accumulated response, so delimiters split across tokens work too.
    slot = st.empty()
    raw = ""
    for chunk in chunks:
        raw += chunk
        slot.markdown(normalize_math_markdown(raw))
    return normalize_math_markdown(raw)


def start_study(question: str, mode: str) -> None:
    queue(question, mode)
    st.session_state.workspace_nav = "Chat"


def ensure_source_mode() -> None:
    if st.session_state.source_mode not in (PDF_ONLY, PDF_AND_ONLINE):
        st.session_state.source_mode = PDF_ONLY


def sidebar(api_key: str) -> tuple[str, str, str]:
    with st.sidebar:
        st.markdown("### :material/blur_on: Mass Transfer")
        st.caption("YOUR ENGINEERING STUDY SPACE")
        st.space("small")
        account_controls()
        st.button("New conversation", icon=":material/add:", on_click=new_chat, width="stretch", type="primary")
        page = st.radio("Workspace", ["Chat", "Study tools", "Library"], label_visibility="collapsed", key="workspace_nav")
        st.caption("RECENT CONVERSATIONS")
        for chat_id, chat in reversed(list(st.session_state.chats.items())):
            st.button(chat["title"], key=f"chat_{chat_id}", icon=":material/chat_bubble_outline:", width="stretch", on_click=select_chat, args=(chat_id,))
        st.divider()
        st.segmented_control("Answer sources", [PDF_ONLY, PDF_AND_ONLINE], key="source_mode", selection_mode="single", on_change=ensure_source_mode)
        if st.session_state.source_mode == PDF_AND_ONLINE:
            st.caption("Searches the web as well as your PDFs. Online information is labeled and linked. Search queries are sent through Groq's browser-search service.")
        else:
            st.caption("Default: theory grounded in your PDFs, with attached photos read as question inputs. No web search; unsupported answers are declined.")
        st.selectbox("Learning mode", list(MODES), key="mode")
        level = st.selectbox("Explanation level", ["Undergraduate", "Beginner", "Advanced"])
        topic = st.selectbox("Topic focus", TOPICS)
        with st.expander("Your notes & settings", icon=":material/tune:"):
            st.caption("Upload up to 3 text PDFs, 10 MB and 200 pages each. Uploads stay in this session. Relevant excerpts and your questions are sent to Groq for answers.")
            files = st.file_uploader("Add course PDFs", type="pdf", accept_multiple_files=True, key="notes_upload")
            if st.button("Use these notes", width="stretch", disabled=not files):
                if len(files) > 3 or any(f.size > 10 * 1024 * 1024 for f in files):
                    st.error("Choose up to 3 PDFs, each under 10 MB.")
                else:
                    try:
                        digest = hashlib.sha256(b"".join(f.name.encode() + f.getvalue() for f in files)).hexdigest()
                        if digest != st.session_state.upload_digest:
                            with st.spinner("Reading your notes…"):
                                passages = []
                                for file in files:
                                    passages.extend(read_pdf(file.getvalue(), file.name, max_pages=200))
                                st.session_state.upload_index = KnowledgeIndex(passages)
                                st.session_state.upload_digest = digest
                                st.session_state.upload_names = [f.name for f in files]
                        st.success("Your notes are ready.")
                    except Exception as exc:
                        st.error(str(exc) if isinstance(exc, ValueError) else "Could not read this PDF. Try an unlocked, text-based PDF.")
            if st.session_state.upload_index and st.button("Remove uploaded notes", width="stretch"):
                st.session_state.upload_index = None
                st.session_state.upload_names = []
                st.session_state.upload_digest = ""
                st.rerun()
            st.caption("Uploaded notes are searched first; textbook passages fill remaining reference slots.")
            st.button("Delete all conversations", width="stretch", on_click=delete_chats)
        st.badge("Groq key configured" if api_key else "Setup needed", icon=":material/check_circle:" if api_key else ":material/settings:", color="green" if api_key else "orange")
        st.caption("Signed-in chats are saved across visits. Guest chats and PDF uploads last for this session.")
    return page, level, topic


def delete_chats() -> None:
    if st.session_state.cloud_auth:
        try:
            cloud_store().clear(st.session_state.cloud_auth)
        except HistoryError as exc:
            st.session_state.cloud_error = str(exc)
            return
    st.session_state.unsaved_chats = set()
    st.session_state.attachment_data = {}
    st.session_state.attachment_previews = {}
    st.session_state.chats = {}
    new_chat()


def welcome() -> None:
    st.space("large")
    st.caption("A LITTLE CURIOSITY. A DEEPER UNDERSTANDING.")
    st.title("Make the complex\n:green[feel intuitive.]")
    st.write("Your space to explore mass transfer. Connect the theory, work through the equations, and build confidence — one question at a time.")
    st.space("medium")
    cards = [
        (":material/lightbulb:", "Build your intuition", "Understand the ideas behind the equations.", "Explain the two-film theory of mass transfer with an intuitive example.", "Explain"),
        (":material/functions:", "Work through a problem", "From the given data to a checked answer.", "Help me solve a steady-state diffusion problem. Ask me for the data you need.", "Solve a problem"),
        (":material/quiz:", "Test what you know", "A question at a time, at your own pace.", "Quiz me on the fundamentals of diffusion and mass transfer.", "Quiz me"),
    ]
    for col, (icon, title, description, prompt, mode) in zip(st.columns(3), cards):
        with col, st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(description)
            st.button("Let's explore", icon=icon, key=f"starter_{mode}", on_click=queue, args=(prompt, mode), width="stretch")
    st.space("small")
    st.caption("TEXTBOOK REFERENCES  /  WORKED EXAMPLES  /  YOUR OWN NOTES")


def answer(question: str, level: str, topic: str, api_key: str, current_passages: list[Passage] | None = None, attachments: list[dict] | None = None) -> None:
    chat = st.session_state.chats[st.session_state.active_chat]
    history = chat["messages"]
    if not api_key:
        st.warning("Add GROQ_API_KEY in Streamlit Secrets (or your local .env) to start chatting. Study tools and the library work without a key.")
        return
    if len(question) > 4000:
        st.warning("Please keep each question under 4,000 characters.")
        return
    previous = list(history)
    source_mode = PDF_AND_ONLINE if st.session_state.get("source_mode") == PDF_AND_ONLINE else PDF_ONLY
    allow_online = source_mode == PDF_AND_ONLINE
    history.append({"role": "user", "content": question, "attachments": attachments or []})
    if chat["title"] == "New conversation":
        chat["title"] = question[:42] + ("…" if len(question) > 42 else "")
    st.session_state.pop("retry_question", None)
    with st.chat_message("user"):
        st.markdown(question)
        render_attachments(attachments or [])
    with st.chat_message("assistant", avatar=":material/blur_on:"):
        output = st.empty()
        try:
            with st.spinner("Finding the right references…"):
                query = retrieval_query(question, previous, topic)
                if current_passages:
                    # Include the photo/problem terms when looking up textbook theory.
                    query += " " + " ".join(p.text[:400] for p in current_passages[:3])
                attached = current_passages or [p for group in st.session_state.attachment_data.get(st.session_state.active_chat, {}).values() for p in group]
                if not attached:
                    # Returning signed-in users retain the excerpts saved with earlier answers.
                    attached = [Passage(**p) for m in previous for p in m.get("sources", []) if p.get("kind") in ("image", "scan")]
                file_hits = KnowledgeIndex(attached).search(query, 3) if attached else []
                if current_passages and not file_hits:
                    file_hits = current_passages[:3]
                uploaded = st.session_state.upload_index
                sources = file_hits + (uploaded.search(query, 2) if uploaded else [])
                sources += library_index().search(query, max(2, 6 - len(sources)))
                messages = build_messages(question, previous, sources, st.session_state.mode, level, allow_online=allow_online)
            with output.container():
                content = render_answer_stream(stream_answer(api_key, messages, setting("GROQ_MODEL", DEFAULT_MODEL), allow_online=allow_online))
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty response")
            history.append({"role": "assistant", "content": content, "sources": [asdict(p) for p in sources], "source_mode": source_mode})
            if st.session_state.cloud_auth:
                st.session_state.unsaved_chats.add(st.session_state.active_chat)
        except Exception as exc:
            history.pop()
            st.session_state.retry_question = question
            output.empty()
            if isinstance(exc, groq.AuthenticationError):
                message = "The Groq key was rejected. Check GROQ_API_KEY in your app secrets."
            elif isinstance(exc, groq.RateLimitError):
                message = "The assistant has reached its provider usage limit. Wait a moment, then retry."
            elif isinstance(exc, (groq.APIConnectionError, groq.APITimeoutError)):
                message = "The model service could not be reached. Please try again shortly."
            elif allow_online and isinstance(exc, groq.BadRequestError):
                message = "Online search could not run with the configured Groq model. Use a browser-search-capable model such as openai/gpt-oss-20b, or switch to PDFs only."
            else:
                LOGGER.warning("Answer failed (%s)", type(exc).__name__)
                message = "The answer could not be completed. Retry, or check the app's model and library configuration."
            st.session_state.last_error = message
    sync_history()
    st.rerun()


def chat_page(level: str, topic: str, api_key: str) -> None:
    chat = st.session_state.chats[st.session_state.active_chat]
    left, right = st.columns([3, 1])
    left.caption("WORKSPACE / " + chat["title"])
    st.caption("Answer sources: " + (st.session_state.get("source_mode") or PDF_ONLY))
    if chat["messages"]:
        right.download_button("Export chat", export_markdown(chat["messages"]), "mass-transfer-chat.md", "text/markdown", icon=":material/download:", width="stretch")
    if not chat["messages"]:
        welcome()
    for index, message in enumerate(chat["messages"]):
        with st.chat_message(message["role"], avatar=":material/blur_on:" if message["role"] == "assistant" else None):
            st.markdown(normalize_math_markdown(message["content"]) if message["role"] == "assistant" else message["content"])
            if message["role"] == "user":
                render_attachments(message.get("attachments", []))
            if message["role"] == "assistant":
                st.caption("Sources: " + message.get("source_mode", "Earlier response · before source controls"))
            render_sources(message.get("sources", []))
            if message["role"] == "assistant":
                st.feedback("thumbs", key=f"feedback_{st.session_state.active_chat}_{index}")
    if st.session_state.get("retry_question"):
        st.error(st.session_state.get("last_error", "The previous answer could not be completed."))
        st.button("Retry last question", on_click=queue, args=(st.session_state.retry_question,))
    question = st.chat_input("Ask a question, explore an idea, or paste a problem…", max_chars=4000, submit_mode="disable", accept_file="multiple", file_type=["pdf", "png", "jpg", "jpeg", "webp"], max_upload_size=10)
    pending = st.session_state.pop("pending", None)
    if question or pending:
        files = question.files if question and not isinstance(question, str) else []
        text = question.text if question and not isinstance(question, str) else question
        text = text or pending or "Read the attached material and help me understand the question or main ideas."
        try:
            current, labels = [], []
            if files:
                with st.spinner("Reading your attachments…"):
                    current, labels = ingest_attachments(files, api_key)
            answer(text, level, topic, api_key, current, labels)
        except (ValueError, groq.APIError) as exc:
            if isinstance(exc, ValueError):
                st.error(str(exc))
            elif isinstance(exc, groq.PermissionDeniedError):
                st.error("Your Groq organization currently blocks the vision model. Enable " + setting("GROQ_VISION_MODEL", VISION_MODEL) + " in your Groq model permissions, then reattach the photo or scanned PDF.")
                st.link_button("Open Groq model permissions", "https://console.groq.com/settings/limits")
            elif isinstance(exc, groq.BadRequestError):
                st.error("The photo could not be processed. Check access to the configured Groq vision model, or try a smaller, clearer image.")
            else:
                st.error("The image-reading service is unavailable or has reached its usage limit. Please reattach the file and try again.")
    st.caption("Attach up to 3 PDFs or photos, 10 MB each. Chat PDFs: up to 50 pages, including at most 5 scanned pages. Photos/scans are sent to Groq for reading; check transcribed formulas and numbers.")
    st.caption("AI can make mistakes. Check the reference passages and verify important calculations.")


def tools_page() -> None:
    st.caption("WORKSPACE / STUDY TOOLS")
    st.title("A little help with the numbers.")
    st.write("Explore the fundamentals with transparent calculations. No AI key needed.")
    flux_tab, numbers_tab, revision_tab = st.tabs(["Diffusive flux", "Dimensionless numbers", "Revision studio"])
    with flux_tab:
        st.subheader("Fick’s first law")
        st.latex(r"J_A = -D_{AB}\frac{dC_A}{dz} = D_{AB}\frac{C_{A,1}-C_{A,2}}{L}")
        st.caption("Steady, one-dimensional diffusion; constant diffusivity, linear concentration gradient, no bulk flow. Positive flux points from face 1 to face 2.")
        a, b = st.columns(2)
        d = a.number_input("Diffusivity D_AB (m²/s)", min_value=1e-12, value=1e-5, format="%.3e")
        length = b.number_input("Film thickness L (m)", min_value=1e-9, value=0.001, format="%.4g")
        c1 = a.number_input("Concentration at face 1 (mol/m³)", min_value=0.0, value=10.0)
        c2 = b.number_input("Concentration at face 2 (mol/m³)", min_value=0.0, value=2.0)
        st.metric("Molar diffusive flux", f"{fick_flux(d, c1, c2, length):.5g} mol/(m²·s)")
        st.line_chart({"Position z (m)": [0.0, length], "Concentration (mol/m³)": [c1, c2]}, x="Position z (m)", y="Concentration (mol/m³)", color="#18796b", alt="Linear concentration profile across the diffusion film")
    with numbers_tab:
        st.subheader("Reynolds & Schmidt numbers")
        st.latex(r"Re = \frac{\rho u L}{\mu} \qquad Sc = \frac{\mu}{\rho D_{AB}}")
        a, b = st.columns(2)
        rho = a.number_input("Density ρ (kg/m³)", min_value=0.001, value=1000.0)
        speed = b.number_input("Velocity u (m/s)", min_value=0.0, value=0.1)
        length = a.number_input("Characteristic length (m)", min_value=1e-6, value=0.01, format="%.4g")
        mu = b.number_input("Dynamic viscosity μ (Pa·s)", min_value=1e-9, value=0.001, format="%.3e")
        d = a.number_input("Molecular diffusivity (m²/s)", min_value=1e-12, value=1e-9, format="%.3e")
        values = dimensionless(rho, speed, length, mu, d)
        a.metric("Reynolds number", f"{values['Re']:,.3g}")
        b.metric("Schmidt number", f"{values['Sc']:,.3g}")
        st.info("Re compares inertia to viscous effects. Sc compares momentum to mass diffusivity. Flow-regime thresholds and Sherwood correlations depend on geometry and boundary conditions.")
    with revision_tab:
        st.subheader("Turn a topic into a study session")
        subject = st.selectbox("Choose your topic", TOPICS[1:], key="revision_topic")
        for col, (label, mode, prompt) in zip(st.columns(3), [
            ("Make flashcards", "Flashcards", f"Make flashcards about {subject}."),
            ("Build revision notes", "Revision notes", f"Create revision notes about {subject}."),
            ("Start a quiz", "Quiz me", f"Quiz me on {subject}, one question at a time."),
        ]):
            col.button(label, on_click=start_study, args=(prompt, mode), width="stretch")
        st.caption("AI study sessions need a configured Groq key.")


def library_page() -> None:
    st.caption("WORKSPACE / LIBRARY")
    st.title("Start with the source.")
    st.write("Search your textbook and session notes, and inspect the passages behind an answer.")
    with st.spinner("Opening the reference library…"):
        index = library_index()
    names = sorted({p.source for p in index.passages})
    a, b, c = st.columns(3)
    a.metric("Reference books", len(names))
    b.metric("Searchable passages", f"{len(index.passages):,}")
    c.metric("Your session PDFs", len(st.session_state.upload_names))
    for name in names:
        st.markdown(f":material/menu_book: **{name}**")
    for name in st.session_state.upload_names:
        st.write(f"Your notes: {name}")
    query = st.text_input("Search the library", placeholder="Try ‘Fick diffusion’ or ‘packed tower absorption’", max_chars=4000)
    if query:
        uploaded = st.session_state.upload_index
        results = (uploaded.search(query, 3) if uploaded else []) + index.search(query, 5)
        if results:
            for passage in results:
                with st.container(border=True):
                    st.markdown(f"**{passage.source}** · PDF page {passage.page}")
                    st.text(passage.text)
        else:
            st.info("No matching passages. Try the full technical term or upload notes on this topic.")
    st.caption("Search uses local TF-IDF keyword ranking. Nothing is sent to an AI provider when you search here. Scanned pages without a text layer are not searchable.")


def main() -> None:
    st.set_page_config(page_title="Mass Transfer · Study Assistant", page_icon="◎", layout="wide")
    initialize()
    api_key = setting("GROQ_API_KEY")
    page, level, topic = sidebar(api_key)
    if not api_key:
        with st.expander("Connect your assistant · one-time setup", expanded=True):
            st.write("In Streamlit Community Cloud, open App settings → Secrets and add your Groq API key. Locally, put the same variable in .env.")
            st.code('GROQ_API_KEY = "your-key-here"', language="toml")
    if page == "Chat":
        chat_page(level, topic, api_key)
    elif page == "Study tools":
        tools_page()
    else:
        library_page()


if __name__ == "__main__":
    main()


