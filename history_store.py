"""Stateless Supabase REST adapter. Tokens belong to one Streamlit session only."""
import base64
import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


class HistoryError(RuntimeError):
    pass


class HistoryStore:
    def __init__(self, url: str, public_key: str):
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.path not in ("", "/") or parsed.username or parsed.query or parsed.fragment:
            raise HistoryError("Use the HTTPS project URL from Supabase settings.")
        if public_key.startswith("sb_secret_"):
            raise HistoryError("Use a Supabase publishable or anon key, never a secret/service-role key.")
        if public_key.count(".") == 2:
            try:
                payload = public_key.split(".")[1]
                role = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))).get("role")
            except (ValueError, TypeError):
                raise HistoryError("Invalid Supabase public key.") from None
            if role != "anon":
                raise HistoryError("Use the anon key rather than a service-role key.")
        self.url = url.rstrip("/")
        self.public_key = public_key

    def _request(self, method, path, data=None, token=None, prefer=None):
        headers = {"apikey": self.public_key, "Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        if prefer:
            headers["Prefer"] = prefer
        request = Request(self.url + path, data=json.dumps(data).encode() if data is not None else None, headers=headers, method=method)
        try:
            with urlopen(request, timeout=20) as response:
                body = response.read()
                return json.loads(body) if body else None
        except HTTPError as exc:
            if "/auth/" in path:
                raise HistoryError("Sign-in could not be completed. Check your email/password and confirm your email if required. For sign-up, check the project's email settings or try again later.") from None
            raise HistoryError("Cloud history could not be reached. Check the Supabase setup and try again; your current chats are still here.") from None
        except (URLError, TimeoutError, ValueError, OSError):
            raise HistoryError("Cloud history is unavailable. Your current chats are still available in this session.") from None

    def sign_in(self, email, password):
        data = self._request("POST", "/auth/v1/token?grant_type=password", {"email": email.strip(), "password": password})
        if not data.get("access_token") or not data.get("user", {}).get("id"):
            raise HistoryError("No authenticated session was returned.")
        data.setdefault("expires_at", time.time() + data.get("expires_in", 3600))
        return data

    def sign_up(self, email, password):
        return self._request("POST", "/auth/v1/signup", {"email": email.strip(), "password": password})

    def _token(self, session):
        if session.get("expires_at", 0) < time.time() + 60:
            refreshed = self._request("POST", "/auth/v1/token?grant_type=refresh_token", {"refresh_token": session["refresh_token"]})
            if refreshed.get("user", {}).get("id") != session["user"]["id"]:
                raise HistoryError("Please sign out and sign in again.")
            refreshed.setdefault("expires_at", time.time() + refreshed.get("expires_in", 3600))
            session.update(refreshed)
        return session["access_token"]

    def load(self, session):
        token = self._token(session)
        rows = []
        while True:
            query = urlencode({"select": "id,title,messages", "user_id": "eq." + session["user"]["id"], "archived_at": "is.null", "order": "updated_at.asc,id.asc", "limit": 100, "offset": len(rows)})
            batch = self._request("GET", "/rest/v1/conversations?" + query, token=token)
            rows.extend(batch)
            if len(batch) < 100:
                break
        return {row["id"].replace("-", ""): {"title": row["title"], "messages": row["messages"]} for row in rows}

    def save(self, session, chat_id, chat):
        token = self._token(session)
        self._request("POST", "/rest/v1/conversations?on_conflict=user_id,id", {
            "id": chat_id, "user_id": session["user"]["id"], "title": chat["title"],
            "messages": chat["messages"], "archived_at": None,
        }, token=token, prefer="resolution=merge-duplicates,return=minimal")

    def clear(self, session):
        # Soft-delete so an accidental clear can be recovered by the project owner.
        token = self._token(session)
        query = urlencode({"user_id": "eq." + session["user"]["id"], "archived_at": "is.null"})
        self._request("PATCH", "/rest/v1/conversations?" + query, {"archived_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, token=token, prefer="return=minimal")

    def sign_out(self, session):
        self._request("POST", "/auth/v1/logout?scope=local", token=self._token(session))
