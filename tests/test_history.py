import time
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from history_store import HistoryError, HistoryStore


def session(user="user-one"):
    return {"user": {"id": user, "email": "student@example.edu"}, "access_token": "user-token", "refresh_token": "refresh", "expires_at": time.time() + 3600}


class StoreTests(unittest.TestCase):
    def test_rejects_privileged_keys_and_insecure_urls(self):
        for url, key in [("http://example.supabase.co", "sb_publishable_test"), ("https://example.supabase.co", "sb_secret_test")]:
            with self.assertRaises(HistoryError):
                HistoryStore(url, key)

    @patch.object(HistoryStore, "_request")
    def test_save_scoped_to_user_and_preserves_answer_metadata(self, request):
        store = HistoryStore("https://example.supabase.co", "sb_publishable_test")
        chat = {"title": "Diffusion", "messages": [{"role": "assistant", "content": "$J=-D$", "source_mode": "PDFs only", "sources": []}]}
        store.save(session(), "abc", chat)
        args, kwargs = request.call_args
        self.assertEqual(args[2]["user_id"], "user-one")
        self.assertEqual(args[2]["messages"], chat["messages"])
        self.assertEqual(kwargs["token"], "user-token")
        self.assertIn("user_id,id", args[1])

    @patch.object(HistoryStore, "_request")
    def test_refresh_updates_only_the_given_session(self, request):
        store = HistoryStore("https://example.supabase.co", "sb_publishable_test")
        first, second = session(), session("user-two")
        first["expires_at"] = 0
        renewed = session()
        renewed["access_token"] = "renewed-token"
        request.return_value = renewed
        self.assertEqual(store._token(first), "renewed-token")
        self.assertEqual(second["access_token"], "user-token")
        self.assertIn("refresh_token", request.call_args.args[1])

    @patch.object(HistoryStore, "_request")
    def test_load_is_user_filtered_and_clear_is_soft_delete(self, request):
        request.return_value = [{"id": "ab-cd", "title": "Saved", "messages": []}]
        store = HistoryStore("https://example.supabase.co", "sb_publishable_test")
        self.assertIn("abcd", store.load(session()))
        self.assertIn("user_id=eq.user-one", request.call_args.args[1])
        store.clear(session())
        self.assertEqual(request.call_args.args[0], "PATCH")
        self.assertIn("archived_at", request.call_args.args[2])


class CloudAppTests(unittest.TestCase):
    def app(self):
        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=30)
        at.secrets.update({"GROQ_API_KEY": "test-only", "SUPABASE_URL": "https://example.supabase.co", "SUPABASE_PUBLISHABLE_KEY": "sb_publishable_test"})
        return at.run()

    @patch("streamlit_app.HistoryStore")
    def test_sign_in_restores_chats_and_sign_out_clears_private_state(self, unused):
        # The app imports the adapter into its execution module, so patch its source class.
        with patch("history_store.HistoryStore") as cls:
            cls.return_value.sign_in.return_value = session()
            cls.return_value.load.return_value = {"saved-id": {"title": "Previously saved chat", "messages": [{"role": "user", "content": "Explain diffusion"}]}}
            at = self.app()
            at.text_input(key="account_email").set_value("student@example.edu")
            at.text_input(key="account_password").set_value("test-password")
            next(b for b in at.button if b.label == "Sign in").click().run()
            self.assertFalse(at.exception)
            self.assertIn("saved-id", at.session_state.chats)
            self.assertEqual(at.session_state.account_password, "")
            self.assertTrue(at.session_state.cloud_auth)
            next(b for b in at.button if b.label == "Sign out").click().run()
            self.assertFalse(at.exception)
            self.assertIsNone(at.session_state.cloud_auth)
            self.assertNotIn("saved-id", at.session_state.chats)

    @patch("chatbot_core.load_library", return_value=[])
    @patch("chatbot_core.stream_answer", return_value=iter(["A saved answer."]))
    def test_failed_save_preserves_answer_and_can_retry(self, stream, library):
        with patch("history_store.HistoryStore") as cls:
            store = cls.return_value
            store.save.side_effect = HistoryError("Cloud is unavailable")
            at = self.app()
            at.session_state.cloud_auth = session()
            at.chat_input[0].set_value("Explain diffusion").run()
            self.assertFalse(at.exception)
            self.assertEqual(len(at.chat_message), 2)
            self.assertTrue(at.session_state.unsaved_chats)
            self.assertTrue(at.warning)
            store.save.side_effect = None
            next(b for b in at.button if b.label == "Retry saving history").click().run()
            self.assertFalse(at.exception)
            self.assertFalse(at.session_state.unsaved_chats)


if __name__ == "__main__":
    unittest.main()
