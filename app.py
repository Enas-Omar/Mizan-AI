"""
app.py — Mizan AI: Streamlit Web Application for the Libyan Legal AI Assistant.

Run with:
    streamlit run app.py

Requires: streamlit >= 1.47 (for st.chat_input(accept_audio=True))
"""

import os
import html
import time

import streamlit as st
import streamlit.components.v1 as components

from mizan.assistant import MizanAssistant
from mizan.generator import PROMPT_PRESETS
from config import DOCUMENTS

from mizan.stt import transcribe_audio
from mizan.tts import text_to_speech
from mizan.tts import clean_text_for_tts

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
                "    `ollama run qwen2.5:7b`\n"
                "2. ثمّ حدّث هذه الصفحة.\n\n"
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


def render_audio_icon(audio_base64: str, autoplay: bool = False, key: str = ""):
    """
    Renders a compact circular audio icon button with autoplay support and smooth user interaction.
    """
    html_code = f"""
    <html>
    <head>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
        }}
        .audio-icon-btn {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            border: none;
            background: transparent;
            cursor: pointer;
            color: #9a9a9a;
            padding: 0;
            transition: background 0.15s ease, color 0.15s ease;
        }}
        .audio-icon-btn:hover {{
            background: rgba(120, 120, 120, 0.18);
            color: #f0f0f0;
        }}
        .audio-icon-btn svg {{
            width: 28px;
            height: 28px;
        }}
    </style>
    </head>
    <body>
        <button id="playBtn_{key}" class="audio-icon-btn" title="الاستماع للإجابة">
            <svg id="iconSvg_{key}" viewBox="0 0 24 24" fill="currentColor">
                <path d="M3 10v4h4l5 5V5l-5 5H3z"/>
                <path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/>
            </svg>
        </button>
        <audio id="ttsAudio_{key}" src="data:audio/mp3;base64,{audio_base64}"></audio>
        <script>
            const audio_{key} = document.getElementById('ttsAudio_{key}');
            const btn_{key} = document.getElementById('playBtn_{key}');
            const iconSvg_{key} = document.getElementById('iconSvg_{key}');

            const playIcon_{key} = '<path d="M3 10v4h4l5 5V5l-5 5H3z"/><path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/>';
            const pauseIcon_{key} = '<rect x="6" y="5" width="4" height="14"/><rect x="14" y="5" width="4" height="14"/>';

            btn_{key}.addEventListener('click', function() {{
                if (audio_{key}.paused) {{
                    audio_{key}.play().catch(e => console.log('Play error:', e));
                }} else {{
                    audio_{key}.pause();
                }}
            }});

            audio_{key}.addEventListener('play', function() {{
                iconSvg_{key}.innerHTML = pauseIcon_{key};
            }});
            audio_{key}.addEventListener('pause', function() {{
                iconSvg_{key}.innerHTML = playIcon_{key};
            }});
            audio_{key}.addEventListener('ended', function() {{
                iconSvg_{key}.innerHTML = playIcon_{key};
            }});

            // Handle autoplay and user interaction to bypass browser restrictions
            if ("{str(autoplay).lower()}" === "true") {{
                let p = audio_{key}.play();
                if (p !== undefined) {{
                    p.catch(error => {{
                        const unlockAudio = () => {{
                            audio_{key}.play().catch(err => console.log(err));
                            document.removeEventListener('click', unlockAudio);
                            document.removeEventListener('keydown', unlockAudio);
                        }};
                        document.addEventListener('click', unlockAudio);
                        document.addEventListener('keydown', unlockAudio);
                    }});
                }}
            }}
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=36)
def extract_query_from_input(prompt) -> tuple[str | None, bool]:
    """
    Converts output from st.chat_input(accept_audio=True) to text and determines if input was audio.
    Returns: (Extracted text, Is_audio boolean flag)
    """
    if not prompt:
        return None, False

    # If prompt is directly entered as a string via keyboard
    if isinstance(prompt, str):
        return prompt, False

    text = getattr(prompt, "text", None)
    audio = getattr(prompt, "audio", None)

    # If an audio recording is attached from the microphone
    if audio is not None:
        temp_audio_path = "temp_recorded_audio.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(audio.getvalue())
        try:
            with st.spinner("جاري معالجة التسجيل الصوتي باللهجة الليبية..."):
                transcribed = transcribe_audio(temp_audio_path)
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        if transcribed:
            return transcribed, True  # Successfully identified and transcribed audio input
        
        st.warning("ما قدرت أفهم التسجيل الصوتي، جرّب مرة ثانية أو اكتب سؤالك.")
        return text or None, False

    return text or None, False

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("⚖️ الميزان — المستشار القانوني الليبي الذكي")
st.markdown(
    "مساعد ذكي تفاعلي للمواطنين وأصحاب الأعمال يشرح حقوقك والتزاماتك في التشريعات والقوانين الليبية "
    "بأسلوب سلس وموثق بنصوص المواد، مع دعم اللهجة الليبية."
)

# ─── Suggested Questions ────────────────────────────────______________________
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 شن شروط ضريبة الدخل والخصومات؟", key="btn_q1"):
        st.session_state["preset_q"] = "ما هي الفئات الخاضعة لضريبة الدخل في القانون الليبي وما هي الإعفاءات؟"
        st.rerun()

with col2:
    if st.button("🏢 كيف يتم تأسيس الشركات في القانون التجاري؟", key="btn_q2"):
        st.session_state["preset_q"] = "ما هي شروط تأسيس الشركات التجارية في قانون النشاط التجاري رقم 23 لسنة 2010؟"
        st.rerun()

with col3:
    if st.button("💼 شن حقي لو انفصلوني تعسفياً؟", key="btn_q3"):
        st.session_state["preset_q"] = "فصلوني تعسفياً من الخدمة، ما هي حقوقي في التعويض وفقاً لقانون علاقات العمل؟"
        st.rerun()

# ─── Chat History ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "أهلاً وسهلاً بك! 👋 أنا **الميزان**، مستشارك القانوني الليبي الذكي. "
                "قاعدة معرفتي تغطي 12 تشريعاً ليبياً: قوانين العمل والمدني والتجاري والضرائب والمصارف "
                "ومكافحة غسل الأموال حقوق الطفل وغيرها. تفضل بطرح أي سؤال أو استفسار قانوني "
                "(بالفصحى أو باللهجة الليبية)، وسأجيبك بكل بساطة مع ذكر السند والمواد القانونية المحددة."
            ),
        }
    ]

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Render saved audio icon in chat history if available
        if msg.get("audio_path") and os.path.exists(msg["audio_path"]):
            import base64
            audio_path = msg["audio_path"]
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode()
            render_audio_icon(audio_base64, autoplay=msg.get("is_auto_play", False), key=f"hist_{idx}")

        if msg.get("citations"):
            render_citations(msg["citations"])
        if msg.get("metrics"):
            render_metrics(msg["metrics"])

# ─── Chat input (Text + Microphone inside input box) ───────────────────────────────────────
prompt = st.chat_input(
    "اكتب سؤالك القانوني هنا...",
    accept_audio=True,
)

# Determine if the current input is recorded audio for later auto-play
is_audio_input = False
if prompt and not isinstance(prompt, str) and getattr(prompt, "audio", None) is not None:
    is_audio_input = True

# Extract text and verify whether input came from microphone or text keyboard
typed_query, is_audio_input = extract_query_from_input(prompt)
user_query = st.session_state.pop("preset_q", None) or typed_query

# ─── Query Processing ─────────────────────────────────────────────────────────────
if user_query:
    history_before_current = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
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

            if result["type"] == "greeting":
                greeting_text = result["answer"]
                audio_file = text_to_speech(greeting_text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": greeting_text,
                    "citations": [],
                    "audio_path": audio_file,
                    "is_auto_play": is_audio_input,
                })
                st.rerun()

            if result["type"] == "no_match":
                no_match_text = result["answer"]
                audio_file = text_to_speech(no_match_text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": no_match_text,
                    "citations": [],
                    "audio_path": audio_file,
                    "is_auto_play": is_audio_input,
                    "metrics": {
                        "retrieval_time": t_retrieval,
                        "generation_time": 0.0,
                        "provider": assistant.generator.provider,
                        "confidence": 0.0,
                    },
                })
                st.rerun()

            t1 = time.time()
            if enable_streaming:
                for token in result["stream"]:
                    full_response += token
                    message_placeholder.markdown(full_response + "▌")
            else:
                for token in result["stream"]:
                    full_response += token

            answer = assistant.finish_stream(full_response)
            message_placeholder.markdown(answer)
            t_gen = time.time() - t1
            # Clean formatting and markdown symbols before TTS generation
            clean_answer = clean_text_for_tts(answer)
            # Generate speech using cleaned text
            audio_file = text_to_speech(clean_answer)
            if audio_file and os.path.exists(audio_file):
                import base64
                with open(audio_file, "rb") as f:
                    audio_bytes = f.read()
                audio_base64 = base64.b64encode(audio_bytes).decode()
                render_audio_icon(audio_base64, autoplay=is_audio_input, key=f"new_{len(st.session_state.messages)}")

            citations = result.get("citations", [])
            metrics = {
                "retrieval_time": t_retrieval,
                "generation_time": t_gen,
                "provider": assistant.generator.provider,
                "confidence": result.get("confidence", 0.0),
            }

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "metrics": metrics,
                "audio_path": audio_file,
                "is_auto_play": is_audio_input,
            })
            st.rerun()

        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة الطلب: {e}")
