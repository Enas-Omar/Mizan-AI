"""
app.py — Mizan AI: Streamlit Web Application for the Libyan Legal AI Assistant.

Run with:
    streamlit run app.py
"""

import os
import html
import time

import streamlit as st

from mizan.assistant import MizanAssistant
from mizan.generator import PROMPT_PRESETS
from config import DOCUMENTS

st.set_page_config(
    page_title="الميزان — المستشار القانوني الليبي الذكي",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom RTL & Arabic Styling ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
        direction: rtl;
        text-align: right;
    }
    .stChatMessage {
        direction: rtl;
        text-align: right;
    }
    .citation-card {
        background-color: #f8f9fa;
        border-right: 4px solid #008080;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 4px;
        color: #1a1a1a;
    }
    .stButton>button {
        border-radius: 8px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ إعدادات الميزان")

    provider_choice = st.selectbox(
        "مزود نموذج التوليد (LLM Provider):",
        ["Groq Cloud (مجاني وسريع جداً) ⭐", "Ollama (محلي أوفلاين)", "استرجاع مدمج (بدون مفتاح)"],
        index=0,
    )

    groq_key = ""
    model_name = "llama-3.3-70b-versatile"

    if "Groq" in provider_choice:
        default_key = os.environ.get("GROQ_API_KEY", "")
        groq_key = st.text_input(
            "Groq API Key:",
            value=default_key,
            type="password",
            help="احصل على مفتاح مجاني من console.groq.com بدون بطاقة مصرفية",
        )
        provider = "groq" if groq_key else "fallback"

        if groq_key:
            @st.cache_data(ttl=300)
            def fetch_groq_models_cached(key):
                import requests
                try:
                    r = requests.get(
                        "https://api.groq.com/openai/v1/models",
                        headers={"Authorization": f"Bearer {key}"},
                        timeout=5.0,
                    )
                    if r.status_code == 200:
                        all_ids = [m["id"] for m in r.json().get("data", [])]
                        return sorted([
                            m for m in all_ids
                            if "whisper" not in m and "tts" not in m
                            and "guard" not in m and "distil" not in m
                        ])
                except Exception:
                    pass
                return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]

            available_models = fetch_groq_models_cached(groq_key)
            preferred = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama3-70b-8192"]
            default_idx = 0
            for p in preferred:
                if p in available_models:
                    default_idx = available_models.index(p)
                    break

            model_name = st.selectbox(
                "النموذج (Model):",
                available_models,
                index=default_idx,
                help="تم جلب النماذج المتاحة الآن مباشرة من Groq",
            )
            st.success(f"✅ تم الاتصال بـ Groq — {len(available_models)} نموذج متاح")
        else:
            st.warning("⚠️ أدخل مفتاح Groq المجاني لتفعيل التوليد الذكي.")

    elif "Ollama" in provider_choice:
        import requests
        ollama_live = False
        ollama_models = []
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=1.2)
            if r.status_code == 200:
                ollama_live = True
                ollama_models = [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass

        if ollama_live:
            st.success("✅ خادم Ollama متصل وشغّال على جهازك")
            if ollama_models:
                model_name = st.selectbox("اختر نموذج Ollama المثبت:", ollama_models, index=0)
            else:
                model_name = st.text_input("اسم النموذج في Ollama:", value="qwen2.5:7b")
            provider = "ollama"
        else:
            st.warning(
                "⚠️ **خادم Ollama غير مشغّل حالياً** (المنفذ 11434).\n\n"
                "لتشغيله بدون إنترنت:\n"
                "1. افتح تطبيق Ollama أو شغّل في الطرفية:\n"
                "   `ollama run qwen2.5:7b`\n"
                "2. ثم حدّث هذه الصفحة.\n\n"
                "💡 **حالياً:** سيعمل التطبيق بالنمط المدمج التلقائي (مجاناً ومحلياً 100%) دون توقف."
            )
            model_name = "qwen2.5:7b"
            provider = "fallback"
    else:
        provider = "fallback"

    # ─── Prompt & Persona Customization ───
    st.markdown("---")
    st.subheader("🎨 أسلوب الإجابة")
    preset_choice = st.selectbox("نمط الحوار:", list(PROMPT_PRESETS.keys()), index=0)

    with st.expander("✏️ تعديل الـ System Prompt مباشرة"):
        custom_prompt = st.text_area(
            "نص تعليمات المستشار (System Prompt):",
            value=PROMPT_PRESETS[preset_choice],
            height=250,
        )

    # ─── Source Filter ───
    st.markdown("---")
    st.subheader("📚 تصفية المصادر")
    doc_options = {"كل التشريعات (12 قانوناً)": None}
    for d in DOCUMENTS:
        doc_options[d["title"]] = d["doc_id"]
    selected_source = st.selectbox("اختر المصدر:", list(doc_options.keys()), index=0)
    selected_doc_id = doc_options[selected_source]

    # ─── Retrieval Settings ───
    st.markdown("---")
    st.subheader("🔍 إعدادات محرك الاسترجاع")
    mode_map = {
        "هجين (Hybrid: Dense + BM25) ⭐": "hybrid",
        "دلالي فقط (Dense Vector)": "dense",
        "كلمات فقط (BM25)": "bm25",
    }
    mode_choice = st.selectbox("نمط البحث:", list(mode_map.keys()), index=0)
    selected_mode = mode_map[mode_choice]

    top_k = st.slider("عدد المواد المسترجعة (Top-K):", min_value=1, max_value=5, value=3)

    enable_streaming = st.checkbox("تفعيل البث التدريجي (Streaming)", value=True)

    st.markdown("---")
    st.info("""
    **🇱🇾 قاعدة المعرفة القانونية المتكاملة:**
    * **12 تشريعاً وقانوناً وقراراً ليبياً** (العمل ولائحته، المدني، التجاري، الضرائب ولائحته، المصارف، غسل الأموال، حقوق الطفل، النظام المالي، سوق الأوراق المالية).
    * **2,929 مادة ومقطع قانوني** مفهرس وموثق بأرقام الصفحات.
    * دعم كامل للهجة الليبية والمصطلحات القانونية الدقيقة.
    """)


# ─── Initialize Assistant ─────────────────────────────────────────────────────
@st.cache_resource
def get_assistant(provider_type, api_key_val, model_val, prompt_val, ret_mode):
    return MizanAssistant(
        provider=provider_type,
        api_key=api_key_val,
        model_name=model_val,
        system_prompt=prompt_val,
        retriever_mode=ret_mode,
    )

assistant = get_assistant(provider, groq_key, model_name, custom_prompt, selected_mode)


# ─── Helper: render citations ─────────────────────────────────────────────────
def render_citations(citations: list[dict]):
    if not citations:
        return
    with st.expander("📚 السند القانوني والمصادر المسترجعة (Citations)"):
        for i, c in enumerate(citations, 1):
            dense_r = f"Dense#{c.get('dense_rank')}" if c.get("dense_rank") else ""
            bm25_r = f"BM25#{c.get('bm25_rank')}" if c.get("bm25_rank") else ""
            rank_tag = f" — [{dense_r} | {bm25_r}]" if (dense_r or bm25_r) else ""
            doc = html.escape(c.get("document") or "")
            section = html.escape(c.get("section") or "")
            chapter = html.escape(c.get("chapter") or "")
            year = f" ({c.get('year')})" if c.get("year") else ""
            st.markdown(
                f"""
                <div class="citation-card">
                    <strong>📌 سند رقم ({i}): المادة ({html.escape(str(c.get('article')))}) {rank_tag}</strong><br>
                    <small><b>{doc}</b>{year} | الصفحة: {c.get('page')} | {section} | {chapter}</small>
                    <hr style="margin: 6px 0;">
                    <p style="white-space: pre-wrap; font-size: 0.9em; margin-bottom: 0;">{html.escape(c.get('text', ''))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_metrics(m: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("زمن البحث", f"{m['retrieval_time']:.2f}s")
    c2.metric("زمن التوليد", f"{m['generation_time']:.2f}s")
    c3.metric("المزود", m["provider"].upper())
    c4.metric("الثقة", f"{m['confidence']:.2f}")


# ─── Header ───────────────────────────────────────────────────────────────────
st.title("⚖️ الميزان — المستشار القانوني الليبي الذكي")
st.markdown(
    "مساعد ذكي تفاعلي للمواطنين وأصحاب الأعمال يشرح حقوقك والتزاماتك في التشريعات والقوانين الليبية "
    "بأسلوب سلس وموثق بنصوص المواد، مع دعم اللهجة الليبية."
)

# ─── Suggested Questions ──────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📊 شن شروط ضريبة الدخل والخصومات؟"):
        st.session_state["preset_q"] = "ما هي الفئات الخاضعة لضريبة الدخل في القانون الليبي وما هي الإعفاءات؟"
with col2:
    if st.button("🏢 كيف يتم تأسيس الشركات في القانون التجاري؟"):
        st.session_state["preset_q"] = "ما هي شروط تأسيس الشركات التجارية في قانون النشاط التجاري رقم 23 لسنة 2010؟"
with col3:
    if st.button("💼 شن حقي لو انفصلوني تعسفياً؟"):
        st.session_state["preset_q"] = "فصلوني تعسفياً من الخدمة، ما هي حقوقي في التعويض وفقاً لقانون علاقات العمل؟"

# ─── Chat History ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "أهلاً وسهلاً بك! 👋 أنا **الميزان**، مستشارك القانوني الليبي الذكي. "
                "قاعدة معرفتي تغطي 12 تشريعاً ليبياً: قوانين العمل والمدني والتجاري والضرائب والمصارف "
                "ومكافحة غسل الأموال وحقوق الطفل وغيرها. تفضل بطرح أي سؤال أو استفسار قانوني "
                "(بالفصحى أو باللهجة الليبية)، وسأجيبك بكل بساطة مع ذكر السند والمواد القانونية المحددة."
            ),
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            render_citations(msg["citations"])
        if msg.get("metrics"):
            render_metrics(msg["metrics"])

# ─── Input Handling ───────────────────────────────────────────────────────────
user_query = st.chat_input("اكتب سؤالك هنا (بالفصحى أو اللهجة الليبية)...")

if "preset_q" in st.session_state and st.session_state["preset_q"]:
    user_query = st.session_state.pop("preset_q")

if user_query:
    # Capture history BEFORE appending current question.
    # The generator appends the current question itself (with legal context),
    # so passing it here too would create duplicate consecutive user turns → repetition loop.
    history_before_current = list(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        filters = {"doc_id": selected_doc_id} if selected_doc_id else None

        t0 = time.time()
        result = assistant.ask_stream(
            query=user_query,
            top_k=top_k,
            mode=selected_mode,
            filters=filters,
            chat_history=history_before_current,
        )
        t_retrieval = time.time() - t0

        # ── Greeting: instant reply, no RAG ──
        if result["type"] == "greeting":
            st.markdown(result["answer"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "citations": [],
            })
            st.stop()

        # ── No relevant sources: polite no-answer ──
        if result["type"] == "no_match":
            st.markdown(result["answer"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "citations": [],
                "metrics": {
                    "retrieval_time": t_retrieval,
                    "generation_time": 0.0,
                    "provider": assistant.generator.provider,
                    "confidence": 0.0,
                },
            })
            st.stop()

        # ── Answer: stream tokens, then clean and finalize ──
        t1 = time.time()
        placeholder = st.empty()
        full_answer = ""
        if enable_streaming:
            for token in result["stream"]:
                full_answer += token
                placeholder.markdown(full_answer + "▌")
        else:
            # consume the stream without displaying intermediate updates
            for token in result["stream"]:
                full_answer += token

        answer = assistant.finish_stream(full_answer)
        placeholder.markdown(answer)
        t_gen = time.time() - t1

        metrics = {
            "retrieval_time": t_retrieval,
            "generation_time": t_gen,
            "provider": assistant.generator.provider,
            "confidence": result["confidence"],
        }
        render_metrics(metrics)
        render_citations(result["citations"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": result["citations"],
            "metrics": metrics,
        })
