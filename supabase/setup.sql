-- Run once in your Supabase SQL Editor. Uses public keys plus each user's JWT.
create table if not exists public.conversations (
  user_id uuid not null default auth.uid() references auth.users(id) on delete cascade,
  id uuid not null,
  title text not null check (length(title) <= 500),
  messages jsonb not null default '[]'::jsonb check (jsonb_typeof(messages) = 'array'),
  updated_at timestamptz not null default now(),
  archived_at timestamptz,
  primary key (user_id, id)
);
alter table public.conversations enable row level security;
revoke all on public.conversations from anon;
grant select, insert, update on public.conversations to authenticated;

drop policy if exists "Read own conversations" on public.conversations;
create policy "Read own conversations" on public.conversations for select to authenticated
  using ((select auth.uid()) = user_id);
drop policy if exists "Create own conversations" on public.conversations;
create policy "Create own conversations" on public.conversations for insert to authenticated
  with check ((select auth.uid()) = user_id);
drop policy if exists "Update own conversations" on public.conversations;
create policy "Update own conversations" on public.conversations for update to authenticated
  using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);

create or replace function public.touch_conversation_updated_at()
returns trigger language plpgsql set search_path = '' as $$
begin
  new.updated_at = now();
  return new;
end;
$$;
drop trigger if exists conversations_updated_at on public.conversations;
create trigger conversations_updated_at before update on public.conversations
for each row execute function public.touch_conversation_updated_at();
