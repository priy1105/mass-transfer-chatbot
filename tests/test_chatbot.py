"""Behavior checks for retrieval, calculations, failure recovery and app navigation."""
import io
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter
from streamlit.testing.v1 import AppTest

from chatbot_core import (
    KnowledgeIndex, Passage, build_messages, dimensionless, export_markdown,
    fick_flux, read_pdf, retrieval_query, stream_answer, PDF_ONLY, PDF_AND_ONLINE, NO_PDF_ANSWER, normalize_math_markdown,
)

PASSAGES = [
    Passage("diffusion.pdf", 2, "Fick law diffusion flux concentration gradient diffusivity constant steady diffusion."),
    Passage("diffusion.pdf", 2, "Diffusion concentration gradient molecular flux Fick diffusivity."),
    Passage("distillation.pdf", 8, "Distillation separates liquid mixtures using relative volatility and vapor liquid equilibrium."),
]


class CoreTests(unittest.TestCase):
    def test_math_delimiters_in_prose_display_and_tables(self):
        text = r"Coefficient \(h_{q}\)." + "\n" + r"\[ q = h_q\bigl(T_{\text{surface}}-T_{\text{fluid}}\bigr) \]" + "\n" + r"| \(T_s\) | \(\mathrm{W\,m^{-2}}\) | [1] |"
        normalized = normalize_math_markdown(text)
        self.assertIn("$h_{q}$", normalized)
        self.assertIn("$$\nq =", normalized)
        self.assertIn(r"$\mathrm{W\,m^{-2}}$", normalized)
        self.assertIn("[1]", normalized)
        self.assertNotIn(r"\(", normalized)
        self.assertEqual(normalize_math_markdown(normalized), normalized)

    def test_math_normalization_preserves_code_links_and_existing_math(self):
        text = r'`\(literal\)` and $J_A$ [source](https://example.edu)' + '\n```latex\n' + r'\[literal\]' + '\n```'
        self.assertEqual(normalize_math_markdown(text), text)

    def test_pdf_policy_excludes_online_and_legacy_assistant_context(self):
        history = [
            {"role": "user", "content": "Explain diffusion"},
            {"role": "assistant", "content": "Unverified web fact", "source_mode": PDF_AND_ONLINE},
            {"role": "assistant", "content": "Old general knowledge"},
            {"role": "assistant", "content": "PDF-supported fact", "source_mode": PDF_ONLY},
        ]
        messages = build_messages("Why?", history, PASSAGES[:1], "Explain", "Beginner")
        self.assertIn("PDFs ONLY", messages[0]["content"])
        self.assertNotIn("Unverified web fact", str(messages))
        self.assertNotIn("Old general knowledge", str(messages))
        self.assertIn("PDF-supported fact", str(messages))

    @patch("chatbot_core.Groq")
    def test_no_pdf_matches_refuses_without_provider_call(self, client):
        messages = build_messages("Unrelated question", [], [], "Explain", "Beginner")
        self.assertEqual("".join(stream_answer("test", messages)), NO_PDF_ANSWER)
        client.assert_not_called()

    @patch("chatbot_core.Groq")
    def test_search_tool_only_enabled_with_explicit_online_choice(self, client):
        provider = client.return_value.__enter__.return_value
        for online in (False, True):
            messages = build_messages("Explain diffusion", [], PASSAGES[:1], "Explain", "Beginner", allow_online=online)
            "".join(stream_answer("test", messages, allow_online=online))
            kwargs = provider.chat.completions.create.call_args.kwargs
            self.assertEqual(kwargs["tool_choice"], "required" if online else "none")
            self.assertEqual(kwargs["tools"], [{"type": "browser_search"}] if online else [])

    def test_retrieval_ranking_and_page_deduplication(self):
        index = KnowledgeIndex(PASSAGES)
        hits = index.search("Fick diffusion flux", 4)
        self.assertEqual(hits[0].source, "diffusion.pdf")
        self.assertEqual(len([p for p in hits if p.page == 2]), 1)
        self.assertEqual(index.search("zzzzzzzz"), [])
        self.assertEqual(index.search("diffusion", 0), [])

    def test_named_laws_preserve_ordinals(self):
        index = KnowledgeIndex([
            Passage("book.pdf", 1, "Fick first law describes flux as diffusivity times a concentration gradient."),
            Passage("book.pdf", 2, "Fick second law describes unsteady diffusion through time."),
        ])
        self.assertEqual(index.search("Fick first law")[0].page, 1)
        self.assertEqual(index.search("Fick second law")[0].page, 2)

    def test_empty_library(self):
        self.assertEqual(KnowledgeIndex([]).search("diffusion"), [])

    def test_followup_uses_conversation_and_topic(self):
        history = [{"role": "user", "content": "Explain absorption"}]
        query = retrieval_query("What are its assumptions?", history, "Diffusion")
        self.assertIn("absorption", query)
        self.assertIn("Diffusion", query)
        messages = build_messages("Why?", history, PASSAGES[:1], "Quiz me", "Beginner")
        self.assertIn("ONE question", messages[0]["content"])
        self.assertIn("PDF page 2", messages[0]["content"])
        self.assertEqual(messages[-2], history[0])
        self.assertEqual(messages[-1]["content"], "Why?")

    def test_export_includes_references(self):
        result = export_markdown([{"role": "assistant", "content": "Flux [1]", "sources": [asdict(PASSAGES[0])]}])
        self.assertIn("PDF page 2", result)
        self.assertIn("Flux [1]", result)

    def test_flux_sign_units_and_validation(self):
        self.assertAlmostEqual(fick_flux(1e-5, 10, 2, 0.001), 0.08)
        self.assertAlmostEqual(fick_flux(1e-5, 2, 10, 0.001), -0.08)
        self.assertEqual(fick_flux(1e-5, 2, 2, 0.001), 0)
        for values in [(1e-5, 10, 2, 0), (-1, 10, 2, 1), (1, -1, 2, 1), (float("nan"), 2, 1, 1)]:
            with self.assertRaises(ValueError):
                fick_flux(*values)

    def test_reynolds_schmidt(self):
        values = dimensionless(1000, 0.1, 0.01, 0.001, 1e-9)
        self.assertAlmostEqual(values["Re"], 1000)
        self.assertAlmostEqual(values["Sc"], 1000)
        with self.assertRaises(ValueError):
            dimensionless(1000, -1, 0.01, 0.001, 1e-9)

    def test_unreadable_encrypted_and_oversize_pdfs(self):
        writer = PdfWriter()
        writer.add_blank_page(100, 100)
        data = io.BytesIO()
        writer.write(data)
        with self.assertRaisesRegex(ValueError, "No readable text"):
            read_pdf(data.getvalue(), "scan.pdf")
        with self.assertRaisesRegex(ValueError, "at most"):
            read_pdf(data.getvalue(), "long.pdf", max_pages=0)
        writer.encrypt("password")
        data = io.BytesIO()
        writer.write(data)
        with self.assertRaisesRegex(ValueError, "password protected"):
            read_pdf(data.getvalue(), "locked.pdf")


class AppTests(unittest.TestCase):
    @patch("chatbot_core.load_library", return_value=PASSAGES)
    @patch("chatbot_core.stream_answer")
    def test_math_renders_when_delimiters_arrive_in_separate_chunks(self, stream, _):
        stream.return_value = iter(["Coefficient \\", "(h_q\\)", ".\n\\", "[q=h_q(T_s-T_f)\\", "]"])
        at = self.app()
        at.chat_input[0].set_value("Explain diffusion flux").run()
        self.assertFalse(at.exception)
        message = at.session_state.chats[at.session_state.active_chat]["messages"][-1]
        self.assertIn("$h_q$", message["content"])
        self.assertIn("$$\nq=", message["content"])
        # Existing session messages are normalized on display without regeneration.
        message["content"] = r"Old equation \[J=-D\frac{dC}{dz}\] and \(D_{AB}\)."
        at.run()
        self.assertFalse(at.exception)
        rendered = "\n".join(m.value for m in at.markdown)
        self.assertIn("$D_{AB}$", rendered)
        self.assertIn("$$\nJ=", rendered)

    @patch("chatbot_core.load_library", return_value=PASSAGES)
    @patch("chatbot_core.stream_answer")
    def test_source_selector_defaults_switches_and_resets(self, stream, _):
        at = self.app()
        self.assertEqual(at.session_state.source_mode, PDF_ONLY)
        at.segmented_control(key="source_mode").set_value(PDF_AND_ONLINE).run()
        stream.return_value = iter(["Online fact from [University](https://example.edu)."])
        at.chat_input[0].set_value("Explain diffusion with online references").run()
        self.assertFalse(at.exception)
        self.assertTrue(stream.call_args.kwargs["allow_online"])
        self.assertEqual(at.session_state.chats[at.session_state.active_chat]["messages"][-1]["source_mode"], PDF_AND_ONLINE)
        at.segmented_control(key="source_mode").set_value(PDF_ONLY).run()
        stream.return_value = iter(["PDF-supported answer [1]."])
        at.chat_input[0].set_value("Explain diffusion from the textbook").run()
        self.assertFalse(at.exception)
        self.assertFalse(stream.call_args.kwargs["allow_online"])
        self.assertNotIn("Online fact", str(stream.call_args.args[1]))
        next(b for b in at.button if b.label == "New conversation").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state.source_mode, PDF_ONLY)

    def app(self):
        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=30)
        at.secrets["GROQ_API_KEY"] = "test-only-key"
        return at.run()

    def test_navigation_calculators_and_new_chat(self):
        at = self.app()
        self.assertFalse(at.exception)
        at.radio(key="workspace_nav").set_value("Study tools").run()
        self.assertFalse(at.exception)
        self.assertIn("0.08", at.metric[0].value)
        next(b for b in at.button if b.label == "New conversation").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.radio(key="workspace_nav").value, "Chat")
        self.assertEqual(len(at.session_state.chats), 2)

    def test_missing_key_and_private_notes_are_session_local(self):
        first = self.app()
        first.session_state.upload_index = KnowledgeIndex(PASSAGES)
        second = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=30)
        second.secrets["GROQ_API_KEY"] = ""
        second.run()
        self.assertIsNone(second.session_state.upload_index)
        second.chat_input[0].set_value("Explain diffusion").run()
        self.assertFalse(second.exception)
        self.assertTrue(second.warning)
        second.radio(key="workspace_nav").set_value("Study tools").run()
        self.assertFalse(second.exception)
        self.assertTrue(second.metric)

    @patch("chatbot_core.load_library", return_value=PASSAGES)
    def test_library_search(self, _):
        at = self.app()
        at.radio(key="workspace_nav").set_value("Library").run()
        self.assertFalse(at.exception)
        at.text_input[0].set_value("Fick diffusion").run()
        self.assertFalse(at.exception)
        self.assertTrue(any("PDF page 2" in m.value for m in at.markdown))

    @patch("chatbot_core.load_library", return_value=PASSAGES)
    @patch("chatbot_core.stream_answer", return_value=iter(["Diffusive flux ", "uses Fick's law [1]."]))
    def test_streamed_chat_followup_and_isolation(self, stream, _):
        at = self.app()
        at.chat_input[0].set_value("Explain Fick diffusion").run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.chat_message), 2)
        chat_id = at.session_state.active_chat
        self.assertIn("Fick's law", at.session_state.chats[chat_id]["messages"][1]["content"])
        self.assertTrue(at.session_state.chats[chat_id]["messages"][1]["sources"])
        stream.return_value = iter(["It assumes steady diffusion."])
        at.chat_input[0].set_value("What are its assumptions?").run()
        self.assertEqual(len(at.chat_message), 4)
        self.assertEqual(stream.call_args.args[1][-3]["content"], "Explain Fick diffusion")
        next(b for b in at.button if b.label == "New conversation").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.chat_message), 0)
        self.assertEqual(len(at.session_state.chats[chat_id]["messages"]), 4)

    @patch("chatbot_core.load_library", return_value=PASSAGES)
    @patch("chatbot_core.stream_answer", side_effect=RuntimeError("provider failed"))
    def test_failed_answer_can_retry_without_duplicate_turn(self, stream, _):
        at = self.app()
        at.chat_input[0].set_value("Explain diffusion").run()
        self.assertFalse(at.exception)
        self.assertTrue(at.error)
        self.assertEqual(at.session_state.retry_question, "Explain diffusion")
        self.assertEqual(len(at.session_state.chats[at.session_state.active_chat]["messages"]), 0)
        stream.side_effect = None
        stream.return_value = iter(["Diffusion is molecular transport [1]."])
        next(b for b in at.button if b.label == "Retry last question").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.chat_message), 2)

    @patch("chatbot_core.load_library", return_value=PASSAGES)
    @patch("chatbot_core.stream_answer", return_value=iter(["Question 1: What is diffusion?"]))
    def test_revision_studio_routes_to_chat(self, stream, _):
        at = self.app()
        at.radio(key="workspace_nav").set_value("Study tools").run()
        next(b for b in at.button if b.label == "Start a quiz").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.radio(key="workspace_nav").value, "Chat")
        self.assertEqual(at.selectbox(key="mode").value, "Quiz me")
        self.assertEqual(len(at.chat_message), 2)


if __name__ == "__main__":
    unittest.main()
