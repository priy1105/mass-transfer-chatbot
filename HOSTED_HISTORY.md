# Persistent history: minimal hosted setup

The integration is optional. Without these settings, the app keeps working in guest mode.

1. Create a project at [Supabase](https://supabase.com/dashboard). In **SQL Editor**, run the contents of `supabase/setup.sql` once. The script creates the conversation table and per-user access policies.
2. Copy the **Project URL** and **publishable key** from the project's connection/API settings. A legacy **anon** key also works. Never use a secret/service-role key; the app rejects those.
3. In your Streamlit app's **Settings → Secrets**, add these alongside the existing Groq key:

   ```toml
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_PUBLISHABLE_KEY = "sb_publishable_your-key"
   ```

   Locally, put the same two variables in `.env` without TOML quotes if preferred. Restart the app after changing settings.

Keep email/password authentication enabled in Supabase Auth. Set **Auth → URL Configuration → Site URL** to your deployed Streamlit URL so confirmation emails return to the right app. For local testing, use `http://127.0.0.1:8501`. Users can create an account in the app, confirm their email if required, and sign in. Alternatively, the project owner can create confirmed users in the Supabase dashboard for a small private class. For a public launch, configure custom SMTP: Supabase's default mail service restricts recipients and sending rates, so it is intended for testing rather than public signup.

## What users get

- Each signed-in user's conversations automatically save after completed answers, with source modes, citations, and formula formatting preserved.
- On another visit/device, sign in with the same account to restore conversations. There is no persistent browser login cookie; signing in again restores history from the database.
- Guest conversations in the current tab are saved to the account when the user signs in. Other guests cannot see them.
- The app uses each user's access token and database row-level security. No shared authenticated client or service-role key is used. Expiring tokens refresh within that user's session.
- A failed save leaves the answer available locally, displays a warning, and provides **Retry saving history**. Export important unsaved conversations before leaving.
- Uploaded PDF files and their search index remain temporary. Cited excerpts included in a saved answer are stored with the conversation; re-upload the PDF for future searches after returning.
- Chat attachment names and cited transcriptions are saved; original photos, image thumbnails, and PDFs are not uploaded to Supabase. Reattach originals when needed after returning.
- Signing out clears chats, uploads, and tokens from that tab. Unsaved changes should be retried/exported first.
- **Delete all conversations** hides the account's conversations using a soft-delete timestamp. The project owner can recover them by clearing `archived_at` in Supabase; this is not permanent erasure. Full deletion/account removal is managed by the project owner in Supabase.

Existing conversations lost before this integration cannot be recovered. Cloud history becomes active only after the project is configured; creating local code does not provision a database or publish it.

## Verification before public use

The automated tests mock the remote API. After setup, sign in with two separate test accounts and confirm that each sees only its own chats. Save a chat, restart Streamlit, sign back in, and verify that it returns. This checks the actual project's email configuration and row-level policies.

References: [password authentication](https://supabase.com/docs/guides/auth/passwords), [row-level security](https://supabase.com/docs/guides/database/postgres/row-level-security), [email delivery limits and SMTP](https://supabase.com/docs/guides/auth/auth-smtp).
