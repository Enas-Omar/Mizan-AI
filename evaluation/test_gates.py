# -*- coding: utf-8 -*-
"""
test_gates.py — End-to-end gating evaluation for Mizan AI.

Verifies the three-gate behavior:
  1. Legit legal questions  -> cited answer.
  2. Legal questions the knowledge base does not cover -> "لا أعرف" refusal.
  3. Non-legal questions (schools, sports, weather...) -> out-of-scope refusal.

Run:  python evaluation/test_gates.py
Add your own random scenarios to the lists below.
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from mizan.assistant import MizanAssistant

HIST_LEAVE = [
    {"role": "user", "content": "ما هي مدة الإجازة السنوية؟"},
    {"role": "assistant", "content": "الإجازة السنوية ثلاثون يوما، وخمسة وأربعون يوما لمن بلغ سن الخمسين أو تجاوزت مدة خدمته عشرين عاما."},
]

# (question, chat_history, expected)  expected: "answer" | "no_answer" | "out_of_scope"
SCENARIOS = [
    # Legit legal questions -> ANSWER
    ("ما هي مدة الإجازة السنوية؟", None, "answer"),
    ("وماذا عن العامل الذي خدم أكثر من 5 سنوات؟", HIST_LEAVE, "answer"),
    ("المادة 34", None, "answer"),
    ("شن شروط التقديم على وظيفة والتعيين؟", None, "answer"),
    ("فصلوني تعسفياً من الخدمة شن حقي في التعويض؟", None, "answer"),
    ("ما هي شروط تأسيس شركة تجارية؟", None, "answer"),
    ("إجازة خاصة بمرتب كامل للحج", None, "answer"),
    ("ما هي نسبة الضريبة على الدخل؟", None, "answer"),
    # Legal but NOT covered by the knowledge base -> "لا أعرف"
    ("وهل يختلف الأمر بالنسبة للقطاع الخاص؟", HIST_LEAVE, "no_answer"),
    ("كم معدل الرواتب في القطاع الخاص؟", HIST_LEAVE, "no_answer"),
    # Non-legal -> out-of-scope
    ("ما هي افضل مدرسة في ليبيا ؟", None, "out_of_scope"),
    ("من فاز بمباراة أمس؟", None, "out_of_scope"),
    ("ما هو الطقس اليوم في طرابلس؟", None, "out_of_scope"),
    ("أخبرني نكتة", None, "out_of_scope"),
]


def classify(assistant, q, hist):
    r = assistant.ask_stream(query=q, chat_history=hist)
    if r["type"] == "greeting":
        return "greeting"
    if r["type"] == "no_match":
        return "out_of_scope" if "خارج نطاق" in r["answer"] else "no_answer"
    tokens = list(r["stream"])
    answer = assistant.finish_stream("".join(tokens))
    print("        ->", " ".join(answer.split())[:110], "...")
    return "answer"


def main():
    assistant = MizanAssistant(provider="auto")
    print(f"Provider = {assistant.generator.provider}\n")
    passed = failed = 0
    for i, (q, hist, expected) in enumerate(SCENARIOS, 1):
        got = classify(assistant, q, hist)
        ok = got == expected
        passed += ok
        failed += not ok
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {i}. {q}\n        expected={expected} got={got}")
    print(f"\nResults: {passed} passed, {failed} failed out of {len(SCENARIOS)}")
    return failed


if __name__ == "__main__":
    sys.exit(main())
