# Mass Transfer · Study Assistant

A single Streamlit app for studying chemical engineering: textbook-grounded chat, your own PDF notes, exam preparation, and transparent engineering calculators.

## What's included

- A clean sage-and-teal interface with native responsive Streamlit controls, readable typography, and a mobile sidebar.
- Streamed answers with LaTeX equations, reference excerpts, and PDF page numbers.
- Follow-up questions using the last five conversational turns.
- Explain, solve a problem, quiz, flashcard, and revision-note modes; three explanation levels and topic focus.
- Multiple independent conversations during a browser session, Markdown downloads, and response feedback.
- Up to three personal PDFs per visitor (10 MB and 200 pages each), searched alongside the textbook.
- Chat-box attachments: up to 3 photos or PDFs, 10 MB each; photos and up to 5 scanned pages per PDF use Groq vision. Chat PDFs are limited to 50 pages.
- A searchable reference library that works without an API key.
- Fick's-law diffusive-flux calculator with concentration profile, plus Reynolds and Schmidt calculators.
- Clear setup, connection, quota, and retry messages.
- **Answer sources** selector: **PDFs only** is the default for every new conversation. **PDFs + online** optionally enables real browser search through Groq using the same API key.

## Choosing answer sources

In the sidebar, use **Answer sources**:

- **PDFs only**: web tools are disabled. The tutor is instructed to use only retrieved textbook/upload passages and calculations from those passages and your inputs. If no passages match, the app returns a fixed insufficient-information response without calling the model. With partial matches, the model must decline unsupported questions instead of filling gaps from general knowledge.
- **PDFs + online**: the app requires Groq's built-in browser-search tool and asks for primary sources with clickable links, distinguishing online material from PDF citations. No additional search API key is needed. This requires a supported Groq model such as the default `openai/gpt-oss-20b`; search availability and provider limits still apply.

Each answer records its source mode, including in Markdown exports. Switching back to PDF-only mode excludes earlier online and untagged legacy assistant answers from the model context. A new conversation resets to PDFs only. Source controls apply to subsequent answers; old replies remain labeled with their original mode. Model grounding can still make mistakes, so inspect the cited passages.

Online-search integration follows [Groq's browser-search API](https://console.groq.com/docs/tool-use/built-in-tools/browser-search).

## Run locally

Use Python 3.12 or 3.13. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set GROQ_API_KEY.
.\.venv\Scripts\python -m streamlit run streamlit_app.py
```

Or, with uv installed:

```powershell
uv sync --locked
uv run streamlit run streamlit_app.py
```

The compatibility entry point `MTbot.py` also opens the same app. `GROQ_MODEL` optionally overrides the default `openai/gpt-oss-20b`.

## Deploy the entire app to Streamlit Community Cloud

No Node.js build, React server, separate API deployment, embedding download, or database setup is needed.

1. Push this project to a GitHub repository. Include `streamlit_app.py`, `chatbot_core.py`, `data/knowledge.json.gz`, `.streamlit/config.toml`, `pyproject.toml`, `uv.lock`, and `requirements.txt`. The original textbook PDF may remain in `data/`; the compressed index is sufficient at runtime.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and create an app from that repository and branch.
3. Set the main file to **`streamlit_app.py`**. In Advanced settings, choose **Python 3.12**.
4. Add this in the app's **Secrets** settings, replacing the placeholder:

   ```toml
   GROQ_API_KEY = "your-groq-api-key"
   GROQ_MODEL = "openai/gpt-oss-20b"
   ```

5. Click Deploy. Share the resulting `https://<your-app>.streamlit.app` link. The hosted app runs independently of your laptop and works from browsers on other devices. Set the app's sharing/access settings for your intended audience.

The deployment is prepared locally; these instructions do not create a hosted app or publish a repository. Community Cloud may put idle apps to sleep, and provider quotas still apply.

**Never commit `.env` or `.streamlit/secrets.toml`.** Both are ignored. `.streamlit/secrets.toml.example` shows the expected configuration. One configured server key serves all visitors; Groq usage is billed/rate-limited against that account.

Official guidance: [deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies), [secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

## Chat photos and PDFs

Use the attachment button in the chat input to add PNG, JPG, WebP, or PDF files, then type a question and send. Sending just a file also works. Text PDFs are extracted locally. Photos and scanned PDF pages are transcribed by Groq's `qwen/qwen3.8-27b` vision model using the existing API key; the text tutor then uses the extracted question and reference passages to answer. `GROQ_VISION_MODEL` optionally overrides that model. See [Groq vision documentation](https://console.groq.com/docs/vision).

If Groq returns a model-permission error, enable the vision model under your organization's [model limits/permissions](https://console.groq.com/settings/limits). A model appearing in the API's list does not guarantee the organization permits inference. Text-only PDFs still work without vision permission. This project's live photo test encountered that permission block; successful live image reading still needs the organization setting enabled.

Limits: 3 files per message; 10 MB per file; 50 pages per chat PDF; at most 5 low-text/scanned pages per PDF. For larger text PDFs, use the existing sidebar upload (200 pages). Unsupported, encrypted, invalid, and oversized files produce a clear message. Scans/photos can misread handwriting, subscripts, equations, and diagram values: check the labeled transcription in reference passages before using the answer.

Attachment text is searched within the current conversation. Image thumbnails remain temporary; up to 20 are retained per browser session. Original images and PDFs are not stored in Supabase. Signed-in history retains attachment names and cited text/transcriptions, so a restored conversation may need the original file reattached for complete follow-up searches. Uploading a photo does not enable online search: PDF-only mode keeps additional theory grounded in the file references.

## Reference index

`data/knowledge.json.gz` contains 1,986 text passages extracted from the existing 799-page Treybal textbook, with filenames and 1-based PDF page numbers. It is a portable JSON text index, not an executable pickle. The app builds a cached, read-only TF-IDF keyword index on startup. Personal uploads use a separate in-memory index per session and are never placed in that shared cache.

This replaces FAISS/Hugging Face retrieval for the deployed app to remove large model/GPU dependencies and first-run downloads. It is lexical retrieval: paraphrases and poorly extracted equations may match less well than a semantic embedding system. The original FAISS files and React frontend are preserved.

After adding or replacing text PDFs in `data/`, rebuild and commit the compressed index:

```powershell
.\.venv\Scripts\python create_memory_for_llm.py
```

PDF page numbers include front matter and can differ from printed page numbers. Scanned pages need OCR. Inspect the retrieved passages before relying on a citation. If the compressed index is absent, the app can extract the PDFs directly, but a large textbook takes longer to open.

## Sessions and privacy

Guest conversations and uploads are temporary. Optional Supabase sign-in saves conversations across restarts and devices; see [HOSTED_HISTORY.md](HOSTED_HISTORY.md) for the one-script, two-setting setup. Sign in again after returning to restore saved chats. Uploaded PDFs remain temporary in both modes. Removing notes removes the active search index; excerpts already included in a conversation remain with that conversation.

During chat, questions, recent conversation text, and selected reference excerpts are sent to Groq. In online mode, search queries also go through Groq's browser-search service. When signed in, conversations and cited excerpts are saved to the configured Supabase project with per-user access rules. Library searches and calculators run locally on the Streamlit server. Thumbs feedback stays in the current browser session.

## Validation

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```

The suite checks retrieval ranking and deduplication, follow-up context, reference exports, calculator values and invalid inputs, PDF rejection, navigation, conversations, streamed responses, study-mode routing, and failed-request retry behavior. Provider calls are mocked in automated tests.

## Optional standalone React development frontend

The existing `frontend/` and prebuilt `streamlit_component/` remain available as prior frontend work. The primary Streamlit app no longer depends on either. For the standalone React frontend:

```powershell
uv sync --extra api
uv run uvicorn api:app --reload
# In a second terminal:
cd frontend
npm install
npm run dev
```

This optional API supports the original frontend's streaming protocol. Its requests are stateless; the primary Streamlit app provides the full feature set. Set `CHATBOT_CORS_ORIGINS` for any additional allowed frontend origins. Earlier dependency manifests are backed up in `legacy/` for reference and are not used by the main deployment.
