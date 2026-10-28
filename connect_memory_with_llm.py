"""Optional command-line study assistant, using the same portable backend."""
import os
from dotenv import load_dotenv
from chatbot_core import DEFAULT_MODEL, KnowledgeIndex, build_messages, load_library, stream_answer

def main():
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise SystemExit("Set GROQ_API_KEY in .env first.")
    index = KnowledgeIndex(load_library())
    question = input("Ask about mass transfer: ").strip()
    sources = index.search(question)
    for token in stream_answer(key, build_messages(question, [], sources, "Explain", "Undergraduate"), os.getenv("GROQ_MODEL", DEFAULT_MODEL)):
        print(token, end="", flush=True)
    print("\n\nReferences:")
    for i, source in enumerate(sources, 1):
        print(f"[{i}] {source.source}, PDF page {source.page}")

if __name__ == "__main__":
    main()
