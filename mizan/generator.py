"""
generator.py — LLM Generation Layer for Mizan AI (Multi-document).

Supports free, high-performance open-source models:
  1. Groq Free Tier (auto-detects available models, with 7-day cache) via REST API.
  2. Ollama Local (qwen2.5:7b, llama3.2, ...) via local server.
  3. Conversational Synthesizer (runs if neither external provider is configured).

Features:
  - Streaming (Groq + Ollama) with graceful fallback
  - Conversational memory (history-aware generation)
  - Follow-up query reformulation for retrieval
  - Dynamic system prompt based on retrieved document types
  - Anti-repetition post-processing engine (sentence + paragraph Jaccard dedup)
"""

import os
import re
import json
import time
from pathlib import Path

import requests

# ─── System Prompts ────────────────────────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT = """أنت "الميزان — المستشار القانوني الليبي الذكي"، مساعد قانوني متخصص في التشريعات والقوانين الليبية (قانون النشاط التجاري، القانون المدني، قانون علاقات العمل ولوائحته التنفيذية، قانون الضرائب ولائحته، قانون المصارف، مكافحة غسل الأموال، حقوق الطفل، وغيرها من التشريعات النافذة).

أسلوبك وشخصيتك في المحادثة:
1. أجب بأسلوب ودود وواضح، واشرح الفكرة القانونية بلغة مفهومة ومترابطة استناداً إلى المواد المرفقة في السياق فقط.
2. ادمج المواد القانونية ذات الصلة في سياق متصل، واذكر أرقام المواد والقانون المختص بها بدقة (مثال: "وفقاً للمادة (317) من قانون النشاط التجاري...").
3. إذا تضمن السؤال خطوات أو شروطاً أو حقوقاً، رتبها في نقاط واضحة وموجزة وغير مكررة.
4. التزم فقط بما ورد في السياق القانوني المرفق، ولا تخترع أي مواد أو إجراءات غير واردة فيه.

قواعد صارمة لمنع التكرار والحشو (أهم قاعدة):
- اكتب إجابتك مرة واحدة فقط بتركيز وإيجاز تام.
- يُمنع منعاً باتاً تكرار نفس الفكرة أو إعادة كتابة الجمل بصيغ أو مرادفات مختلفة (مثل تكرار خطوات التأسيس أو الشروط).
- يُمنع تكرار عبارات العرض أو المساعدة أو إخلاء المسؤولية. اذكر التنويه أو الخاتمة لمرة واحدة فقط وبسطر واحد في النهاية.
- بمجرد اكتمال النقاط القانونية والتنويه الأخير، أنهِ الرد فوراً وتوقف عن التوليد.

صيغة الخاتمة المعتمدة (تُكتب لمرة واحدة فقط في النهاية):
"إذا كان لديك أي استفسار آخر أنا هنا لمساعدتك.
⚖️ تنويه: هذه المعلومات استرشادية مبنية على التشريعات والقوانين الليبية المتاحة."
"""

FORMAL_SYSTEM_PROMPT = """أنت "الميزان — المستشار القانوني"، خبير في التشريعات والقوانين الليبية.
أجب بصياغة قانونية رصينة ومحكمة وموجزة.
ادمج المواد المسترجعة في مذكرة قانونية واحدة مترابطة مع ذكر أرقام المواد والقوانين ذات الصلة.
قواعد صارمة:
- التزم بالمواد المرفقة فقط.
- اكتب الإجابة لمرة واحدة دون أي تكرار أو إعادة صياغة لنفس الأفكار.
- اختتم بتنويه قانوني لمرة واحدة في النهاية وتوقف فوراً.
"""

CONCISE_SYSTEM_PROMPT = """أنت "الميزان" — مستشار قانوني سريع ومباشر في القوانين والتشريعات الليبية.
لخّص الإجابة عن سؤال المواطن في نقاط سريعة وموجزة ومباشرة بالاستناد للمواد المرفقة.
اذكر رقم المادة واسم القانون بجانب كل نقطة مباشرة.
يُمنع أي تكرار أو حشو أو إعادة صياغة لنفس المعنى، واختتم بسطر تنويه واحد فقط.
"""

PROMPT_PRESETS = {
    "ودود وتفاعلي (Friendly Chatbot) ⭐": DEFAULT_SYSTEM_PROMPT,
    "مستشار قانوني رسمي ورصين (Formal Legal)": FORMAL_SYSTEM_PROMPT,
    "موجز ومباشر بنقاط سريعة (Concise Bullets)": CONCISE_SYSTEM_PROMPT,
}

DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT  # public alias

# ─── Groq model cache ─────────────────────────────────────────────────────────
GROQ_CACHE_PATH = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".groq_model_cache.json"))
GROQ_CACHE_TTL = 7 * 86400


def build_system_prompt(doc_types: list[str] = None, base_prompt: str = None) -> str:
    """Build a dynamic system prompt based on retrieved document types."""
    prompt = base_prompt or DEFAULT_SYSTEM_PROMPT
    if doc_types:
        types_ar_map = {
            "law": "قوانين",
            "regulation": "لوائح",
            "decision": "قرارات",
            "circular": "تعاميم",
            "guide": "أدلة إرشادية",
            "guideline": "أدلة إرشادية",
        }
        types_ar = [types_ar_map.get(t, t) for t in sorted(set(doc_types)) if t]
        if types_ar:
            prompt += f"\n\nالمصادر المتاحة حالياً: {', '.join(types_ar)}."
    return prompt


class LegalAnswerGenerator:
    # Preferred models in order — first available one wins
    GROQ_PREFERRED_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama3-70b-8192",
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
        "gemma2-9b-it",
        "mixtral-8x7b-32768",
    ]

    GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(
        self,
        provider: str = "auto",
        api_key: str = None,
        model_name: str = None,
        system_prompt: str = None,
        ollama_url: str = "http://localhost:11434",
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.ollama_url = ollama_url
        self.provider = provider
        self.model_name = model_name
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        if self.provider == "auto":
            if self.api_key:
                self.provider = "groq"
                if not self.model_name:
                    self.model_name = self._pick_best_groq_model()
            elif self._check_ollama():
                self.provider = "ollama"
                self.model_name = self.model_name or "qwen2.5:7b"
            else:
                self.provider = "fallback"

    # ─── Groq model discovery ──────────────────────────────────────────────────
    def _fetch_groq_models(self) -> list[str]:
        try:
            r = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5.0,
            )
            if r.status_code == 200:
                return [m["id"] for m in r.json().get("data", [])]
        except Exception:
            pass
        return []

    def _pick_best_groq_model(self) -> str:
        # 7-day cache to avoid a models-list call on every cold start
        if GROQ_CACHE_PATH.exists():
            try:
                data = json.loads(GROQ_CACHE_PATH.read_text(encoding="utf-8"))
                if time.time() - data.get("ts", 0) < GROQ_CACHE_TTL:
                    return data["model"]
            except Exception:
                pass

        available = set(self._fetch_groq_models())
        chosen = self.GROQ_DEFAULT_MODEL

        if available:
            for preferred in self.GROQ_PREFERRED_MODELS:
                if preferred in available:
                    chosen = preferred
                    break
            else:
                for m in sorted(available):
                    if "whisper" not in m and "tts" not in m and "guard" not in m and "distil" not in m:
                        chosen = m
                        break

        try:
            GROQ_CACHE_PATH.write_text(
                json.dumps({"model": chosen, "ts": time.time()}),
                encoding="utf-8",
            )
        except Exception:
            pass

        return chosen

    def _check_ollama(self) -> bool:
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=1.0)
            return r.status_code == 200
        except Exception:
            return False

    # ─── Prompt building ───────────────────────────────────────────────────────
    def build_user_prompt(self, query: str, context_chunks: list[dict]) -> str:
        context_parts = []
        for i, c in enumerate(context_chunks, 1):
            art_label = f"المادة ({c.get('article')})" if c.get("article") else "مادة قانونية"
            doc_title = c.get("document") or ""
            year = f" سنة {c.get('year')}" if c.get("year") else ""
            section = f" | الباب: {c.get('section')}" if c.get("section") else ""
            chapter = f" | الفصل: {c.get('chapter')}" if c.get("chapter") else ""
            header = (
                f"=== سند قانوني {i}: {art_label} — {doc_title}{year}"
                f"{section}{chapter} (صفحة {c.get('page')}) ==="
            )
            context_parts.append(f"{header}\n{c.get('text', '')}")

        full_context = "\n\n".join(context_parts)
        return f"""(السياق القانوني المتاح من التشريعات الليبية):
{full_context}

----------------------------------------
سؤال المواطن:
{query}

الإجابة المدمجة والمبسطة للمواطن استناداً إلى المواد أعلاه:"""

    def _history_summary(self, chat_history: list[dict] = None, max_turns: int = 4) -> str:
        """Build a compact history summary for reformulation prompts."""
        valid_history = [
            m for m in (chat_history or [])
            if m.get("role") in ("user", "assistant")
            and len(m.get("content", "").strip()) > 0
            and not m.get("content", "").startswith("أهلاً")
        ]
        recent = valid_history[-max_turns:]
        lines = []
        for m in recent:
            role = "المواطن" if m["role"] == "user" else "المستشار"
            content = m["content"][:250].replace("\n", " ")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    # ─── Follow-up query reformulation ─────────────────────────────────────────
    def contextualize_query(self, query: str, chat_history: list[dict] = None) -> str:
        """
        If there is conversation history and the user's new question is a follow-up
        or contains pronouns/references, reformulate it into a standalone legal
        question for optimal hybrid retrieval.
        """
        if not chat_history:
            return query.strip()

        history_summary = self._history_summary(chat_history)
        if not history_summary:
            return query.strip()

        rephrase_prompt = f"""أنت خبير قانوني ليبي متخصص في فهم الأسئلة وتحديد موضوعها.
بناءً على سياق المحادثة السابق، إذا كان السؤال الحالي للمواطن مرتبطاً بالسياق السابق أو يشير إليه بضمير أو اختصار، أعد صياغته كسؤال بحثي مستقل ومكتمل للبحث في نصوص القوانين والتشريعات الليبية.
إذا كان السؤال مستقلاً بذاته بالفعل أو يطرح موضوعاً جديداً، أعده كما هو بالضبط.
مهم: أخرج السؤال المستقل فقط بدون أي مقدمات أو شروحات.

سياق المحادثة السابق:
{history_summary}

سؤال المواطن الحالي:
{query}

السؤال المستقل للبحث:"""

        if self.provider == "groq" and self.api_key:
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model_name or self.GROQ_DEFAULT_MODEL,
                        "messages": [{"role": "user", "content": rephrase_prompt}],
                        "temperature": 0.0,
                        "max_tokens": 80,
                    },
                    timeout=8.0,
                )
                if resp.status_code == 200:
                    rephrased = resp.json()["choices"][0]["message"]["content"].strip().strip('"').strip("'")
                    if rephrased and 5 < len(rephrased) < 300:
                        return rephrased
            except Exception:
                pass

        elif self.provider == "ollama":
            try:
                resp = requests.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.model_name or "qwen2.5:7b",
                        "messages": [{"role": "user", "content": rephrase_prompt}],
                        "stream": False,
                        "options": {"temperature": 0.0},
                    },
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    rephrased = resp.json()["message"]["content"].strip().strip('"').strip("'")
                    if rephrased and 5 < len(rephrased) < 300:
                        return rephrased
            except Exception:
                pass

        # Heuristic fallback if LLM offline: detect follow-up words
        followup_triggers = ["و", "ماذا عن", "كيف", "وهل", "وكيف", "شروطه", "شروطها",
                             "مدته", "مدتها", "عقوبته", "عقوبتها", "تصفيتها", "انقضائها"]
        is_followup = any(query.strip().startswith(t) for t in followup_triggers) or (
            len(query.strip()) < 30
            and not any(kw in query for kw in ["قانون", "مادة", "عمل", "شركة", "ضريبة", "طفل", "مصرف"])
        )
        if is_followup:
            last_user_q = ""
            for m in reversed(chat_history):
                if m.get("role") == "user" and m.get("content", "").strip():
                    last_user_q = m["content"]
                    break
            if last_user_q:
                return f"{query} (في سياق: {last_user_q})"

        return query.strip()

    # ─── Message assembly ──────────────────────────────────────────────────────
    def _build_messages(self, query: str, context_chunks: list[dict], chat_history: list[dict] = None) -> list[dict]:
        doc_types = [c.get("doc_type") for c in context_chunks if c.get("doc_type")]
        system_prompt = build_system_prompt(doc_types, self.system_prompt)

        messages = [{"role": "system", "content": system_prompt}]

        if chat_history:
            # Include up to last 6 messages (3 user + 3 assistant)
            recent = [
                m for m in chat_history
                if m.get("role") in ("user", "assistant")
                and len(m.get("content", "").strip()) > 0
                and not m.get("content", "").startswith("أهلاً")
            ][-6:]
            for m in recent:
                messages.append({
                    "role": m["role"],
                    "content": m["content"][:500],  # truncate to conserve context window
                })

        messages.append({"role": "user", "content": self.build_user_prompt(query, context_chunks)})
        return messages

    # ─── Generation (non-streaming) ────────────────────────────────────────────
    def generate(self, query: str, context_chunks: list[dict], chat_history: list[dict] = None) -> str:
        messages = self._build_messages(query, context_chunks, chat_history)

        if self.provider == "groq" and self.api_key:
            result = self._call_groq(messages)
            if result is not None:
                return result
            return self._conversational_fallback(query, context_chunks, chat_history)

        elif self.provider == "ollama":
            result = self._call_ollama(messages)
            if result is not None:
                return result

            fallback = self._conversational_fallback(query, context_chunks, chat_history)
            notice = (
                "\n\n---\n"
                "⚠️ **ملاحظة:** Ollama غير مشغّل على جهازك حالياً (المنفذ 11434).\n"
                "لتفعيله: افتح Ollama وشغّل `ollama run qwen2.5:7b` في الطرفية.\n"
                "في الوقت الحالي تظهر الإجابة بالنمط المدمج التلقائي.\n"
                "---"
            )
            return fallback + notice

        else:
            return self._conversational_fallback(query, context_chunks, chat_history)

    # ─── Generation (streaming) ────────────────────────────────────────────────
    def generate_stream(self, query: str, context_chunks: list[dict], chat_history: list[dict] = None):
        """Yield tokens as they arrive. Falls back to the synthesizer on failure."""
        messages = self._build_messages(query, context_chunks, chat_history)

        if self.provider == "groq" and self.api_key:
            yielded = False
            for token in self._call_groq_stream(messages):
                yielded = True
                yield token
            if not yielded:
                yield self._conversational_fallback(query, context_chunks, chat_history)
            return

        if self.provider == "ollama":
            yielded = False
            for token in self._call_ollama_stream(messages):
                yielded = True
                yield token
            if not yielded:
                fallback = self._conversational_fallback(query, context_chunks, chat_history)
                yield fallback + (
                    "\n\n---\n⚠️ **ملاحظة:** Ollama غير مشغّل حالياً (المنفذ 11434). "
                    "لتفعيله: `ollama run qwen2.5:7b`\n---"
                )
            return

        yield self._conversational_fallback(query, context_chunks, chat_history)

    # ─── Anti-repetition post-processing ───────────────────────────────────────
    @staticmethod
    def _normalize_arabic_text(text: str) -> str:
        t = text.strip().lower()
        t = re.sub(r'[إأآا]', 'ا', t)
        t = re.sub(r'[ة]', 'ه', t)
        t = re.sub(r'[ىي]', 'ي', t)
        t = re.sub(r'[\W_]+', ' ', t)
        return t.strip()

    def clean_answer(self, text: str) -> str:
        """
        Comprehensive post-processing to eliminate LLM output loops:
        1. Deduplicates repeating sentences inside paragraphs.
        2. Deduplicates near-identical paragraphs (token Jaccard overlap).
        3. Limits disclaimer / closing paragraphs to at most one single ending note.
        """
        if not text:
            return ""

        paragraphs = [b.strip() for b in re.split(r'\n{2,}', text) if b.strip()]
        cleaned_paragraphs = []
        seen_sentences = set()

        for p in paragraphs:
            # Split paragraph into sentences
            raw_sentences = re.split(r'([.!?؛\n]+)', p)
            reconstructed = []
            for i in range(0, len(raw_sentences), 2):
                s = raw_sentences[i].strip()
                punct = raw_sentences[i + 1] if i + 1 < len(raw_sentences) else ''
                if not s:
                    continue
                norm = self._normalize_arabic_text(s)
                if len(norm) > 20:
                    tokens = set(norm.split())
                    is_dup = False
                    for seen in seen_sentences:
                        seen_tokens = set(seen.split())
                        if not seen_tokens:
                            continue
                        overlap = len(tokens & seen_tokens) / max(len(tokens), len(seen_tokens))
                        if overlap > 0.65:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    seen_sentences.add(norm)
                reconstructed.append(s + punct)

            cleaned_p = ' '.join(reconstructed).strip()
            if not cleaned_p:
                continue

            # Check paragraph-level Jaccard similarity against existing paragraphs
            p_tokens = set(self._normalize_arabic_text(cleaned_p).split())
            if len(p_tokens) > 5:
                is_p_dup = False
                for prev in cleaned_paragraphs:
                    prev_tokens = set(self._normalize_arabic_text(prev).split())
                    overlap = len(p_tokens & prev_tokens) / max(len(p_tokens), len(prev_tokens))
                    if overlap > 0.65:
                        is_p_dup = True
                        break
                if is_p_dup:
                    continue

            cleaned_paragraphs.append(cleaned_p)

        # Limit closing / disclaimer paragraphs to at most ONE
        disclaimer_keywords = [
            'يرجى مراعاة', 'استشارة محام', 'استشارة محامي', 'معلومات استرشادية',
            'إذا كنت بحاجة إلى', 'اذا كنت بحاجة', 'سأكون سعيدا', 'سأكون سعيداً',
            'لا تتردد في طرح', 'تنويه:', 'إذا كان لديك أي استفسار',
        ]

        final_paragraphs = []
        seen_closing = False
        for p in cleaned_paragraphs:
            norm_p = self._normalize_arabic_text(p)
            is_closing = any(self._normalize_arabic_text(kw) in norm_p for kw in disclaimer_keywords)
            if is_closing:
                if seen_closing:
                    continue
                seen_closing = True
            final_paragraphs.append(p)

        return '\n\n'.join(final_paragraphs)

    # ─── Groq calls ────────────────────────────────────────────────────────────
    def _call_groq(self, messages: list[dict]) -> str | None:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name or self.GROQ_DEFAULT_MODEL,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1500,
            "frequency_penalty": 0.2,   # light penalty against exact repeat tokens
            "presence_penalty": 0.0,
        }
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                return self.clean_answer(raw)
            print(f"[Groq] HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        except Exception as e:
            print(f"[Groq] Exception: {e}")
            return None

    def _call_groq_stream(self, messages: list[dict]):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name or self.GROQ_DEFAULT_MODEL,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1500,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.0,
            "stream": True,
        }
        try:
            with requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=45.0,
                stream=True,
            ) as resp:
                if resp.status_code != 200:
                    print(f"[Groq Stream] HTTP {resp.status_code}")
                    return
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8")
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue
        except Exception as e:
            print(f"[Groq Stream] Exception: {e}")
            return

    # ─── Ollama calls ──────────────────────────────────────────────────────────
    def _call_ollama(self, messages: list[dict]) -> str | None:
        payload = {
            "model": self.model_name or "qwen2.5:7b",
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "repeat_penalty": 1.15,
            },
        }
        try:
            resp = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=60.0)
            if resp.status_code == 200:
                raw = resp.json()["message"]["content"].strip()
                return self.clean_answer(raw)
            return None
        except Exception:
            return None

    def _call_ollama_stream(self, messages: list[dict]):
        payload = {
            "model": self.model_name or "qwen2.5:7b",
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.2,
                "repeat_penalty": 1.15,
            },
        }
        try:
            with requests.post(f"{self.ollama_url}/api/chat", json=payload,
                               timeout=90.0, stream=True) as resp:
                if resp.status_code != 200:
                    return
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        delta = chunk.get("message", {}).get("content", "")
                        if delta:
                            yield delta
                        if chunk.get("done"):
                            break
                    except Exception:
                        continue
        except Exception as e:
            print(f"[Ollama Stream] Exception: {e}")
            return

    # ─── Offline conversational synthesizer ────────────────────────────────────
    def _conversational_fallback(self, query: str, context_chunks: list[dict], chat_history: list[dict] = None) -> str:
        """
        Conversational synthesizer when no LLM is configured.
        Weaves the articles together into one friendly narrative with conversation awareness.
        """
        articles = []
        seen_pairs = set()
        for c in context_chunks:
            a = str(c.get('article') or '').strip()
            d = (c.get('document') or '').strip()
            key = (a, d)
            if not a or key in seen_pairs:
                continue
            seen_pairs.add(key)
            articles.append(f"المادة ({a}) من {d}" if d else f"المادة ({a})")
        articles_str = " و ".join(articles) if articles else "المواد القانونية ذات الصلة"

        greeting = "أهلاً بك مجدداً! استكمالاً لمحادثتنا القانونية،" if chat_history else "أهلاً بك! يسعدني إفادتك بخصوص استفسارك."

        lines = [
            greeting,
            f"بالرجوع إلى التشريعات الليبية المتاحة، وتحديداً ما نظمته **{articles_str}**، إليك توضيح المسألة بشكل مترابط:\n",
        ]

        for i, c in enumerate(context_chunks, 1):
            art = c.get('article', 'غير محدد')
            doc = c.get('document', '')
            # Clean body lines
            raw_lines = [
                l.strip() for l in c.get('text', '').split('\n')
                if l.strip() and not l.startswith('قانون') and not l.startswith('الباب')
                and not l.startswith('الفصل') and not l.startswith('مادة')
            ]
            clean_summary = " ".join(raw_lines)
            if len(clean_summary) > 350:
                clean_summary = clean_summary[:350] + "..."

            doc_tag = f" من {doc}" if doc else ""
            if i == 1:
                lines.append(f"🔹 **وفقاً لأحكام المادة ({art}){doc_tag}:** {clean_summary}\n")
            elif i == 2:
                lines.append(f"🔹 **كما تستكمل المادة ({art}) ذلك بتوضيح:** {clean_summary}\n")
            else:
                lines.append(f"🔹 **وبالإضافة إلى ذلك، تشير المادة ({art}):** {clean_summary}\n")

        lines.append(
            "💡 **ملاحظة:** يمكنك الاطلاع على النصوص الكاملة الحرفية لكل مادة في قائمة السند القانوني بالأسفل.\n\n"
            "⚖️ تنويه: هذه المعلومات استرشادية مبنية على التشريعات الليبية المتاحة في قاعدة المعرفة."
        )

        return "\n".join(lines)
