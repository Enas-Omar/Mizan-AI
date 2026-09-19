"""
chat_cli.py — Interactive CLI for Mizan AI (Libyan Legal AI Assistant).

Usage:
    python chat_cli.py
    python chat_cli.py --query "شن شروط إجازة الحج؟"
    python chat_cli.py --groq-key "gsk_..."
"""

import os
import sys
import argparse

from mizan.assistant import MizanAssistant


def main():
    parser = argparse.ArgumentParser(description="الميزان — المستشار القانوني الليبي الذكي (CLI)")
    parser.add_argument("--query", "-q", type=str, help="Single question to answer")
    parser.add_argument("--provider", "-p", type=str, default="auto",
                        choices=["auto", "groq", "ollama", "fallback"], help="LLM Provider")
    parser.add_argument("--groq-key", type=str, help="Groq API Key (or set GROQ_API_KEY env var)")
    parser.add_argument("--top-k", "-k", type=int, default=3, help="Number of legal chunks to retrieve")
    args = parser.parse_args()

    api_key = args.groq_key or os.environ.get("GROQ_API_KEY")

    print("\n" + "=" * 70)
    print("⚖️  الميزان — المستشار القانوني الليبي الذكي")
    print("=" * 70)

    assistant = MizanAssistant(
        provider=args.provider,
        api_key=api_key,
    )
    print(f"  • مزود الذكاء الاصطناعي (LLM): {assistant.generator.provider.upper()}")
    print(f"  • نمط الاسترجاع: {assistant.default_mode.upper()} (ChromaDB Vector + BM25 + RRF)")
    print("=" * 70 + "\n")

    if args.query:
        run_query(assistant, args.query, args.top_k)
        return

    # Interactive Loop — chat_history accumulates across turns for memory
    chat_history = []
    print("اكتب سؤالك بالعربية أو باللهجة الليبية (أو اكتب 'خروج' أو 'exit' للإنهاء):")
    while True:
        try:
            user_input = input("\n👤 سؤالك: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["خروج", "exit", "quit", "q"]:
                print("\nمع السلامة! نأمل أن تكون الإجابات مفيدة.")
                break
            run_query(assistant, user_input, args.top_k, chat_history)
        except (KeyboardInterrupt, EOFError):
            print("\nتم الإنهاء.")
            break


def run_query(assistant: MizanAssistant, query: str, top_k: int, chat_history: list = None):
    if chat_history is None:
        chat_history = []

    print("\n⏳ جاري البحث في نصوص القانون وتوليد الإجابة...")

    res = assistant.answer_question(query, top_k=top_k, chat_history=chat_history)

    # Update history with this turn for subsequent questions
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": res["answer"]})

    print("\n" + "#" * 70)
    print("🤖 إجابة المستشار القانوني:")
    print("#" * 70)
    print(res["answer"])

    if res.get("confidence") is not None:
        print(f"\n  (الثقة: {res['confidence']:.2f} | المزود: {res['provider'].upper()})")

    citations = res.get("citations") or []
    if citations:
        print("\n" + "-" * 70)
        print("📚 السند والمصادر القانونية المسترجعة (Citations):")
        print("-" * 70)
        for i, c in enumerate(citations, 1):
            dense_r = f"Dense#{c['dense_rank']}" if c.get('dense_rank') else ""
            bm25_r = f"BM25#{c['bm25_rank']}" if c.get('bm25_rank') else ""
            rank_info = f"[{dense_r} | {bm25_r}]" if (dense_r or bm25_r) else ""
            doc = c.get('document') or ''
            print(f"  [{i}] المادة ({c['article']}) — {doc} | ص {c['page']} {rank_info}")
    print("#" * 70)


if __name__ == "__main__":
    main()
