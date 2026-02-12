import streamlit as st
from PyPDF2 import PdfReader
import requests
import os
import json
import plotly.graph_objects as go

# ─── Page Config ──────────────────────────────────────────────────────
st.set_page_config(page_title="Resume Analyzer Pro", page_icon="📄", layout="wide")

# ─── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .skill-tag {
        display: inline-block;
        padding: 4px 12px;
        margin: 3px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .skill-match {
        background-color: #1b4332;
        color: #95d5b2;
        border: 1px solid #2d6a4f;
    }
    .skill-miss {
        background-color: #4a1525;
        color: #f4a0b5;
        border: 1px solid #7a2040;
    }
    .score-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .score-number {
        font-size: 3rem;
        font-weight: 700;
    }
    .score-label {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("🔑 API Configuration")
api_key_input = st.sidebar.text_input("Enter your Hugging Face API Key", type="password")
if api_key_input:
    st.session_state["api_key"] = api_key_input
api_key = st.session_state.get("api_key")
if api_key:
    st.sidebar.success("✅ API key is set!")

page = st.sidebar.radio(
    "Select Page",
    ["About", "Resume Details", "Resume Matching", "Chat with Resume and Job Description"],
    index=0,
)

# ─── Shared Resume Upload (sidebar) ──────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📤 Resume Upload")
sidebar_upload = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"], key="shared_resume")
if sidebar_upload:
    st.session_state["resume_bytes"] = sidebar_upload.getvalue()
    st.session_state["resume_name"] = sidebar_upload.name
    st.sidebar.success(f"✅ {sidebar_upload.name}")
elif st.session_state.get("resume_bytes"):
    st.sidebar.success(f"✅ {st.session_state['resume_name']} (persisted)")


class UploadedFileWrapper:
    """Makes stored bytes behave like an UploadedFile for call_backend."""
    def __init__(self, data, name):
        self._data = data
        self.name = name
    def getvalue(self):
        return self._data


def get_shared_resume():
    """Get the persisted resume from session state, or None."""
    data = st.session_state.get("resume_bytes")
    name = st.session_state.get("resume_name", "resume.pdf")
    return UploadedFileWrapper(data, name) if data else None


# ─── Helper ───────────────────────────────────────────────────────────

def call_backend(endpoint, file, data):
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    response = requests.post(f"http://localhost:8000/{endpoint}", files=files, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.text}")
        return None


def call_backend_no_data(endpoint, file):
    """Call backend with only a file upload, no extra form data."""
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    response = requests.post(f"http://localhost:8000/{endpoint}", files=files)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.text}")
        return None


def get_score_color(score):
    """Return a color based on score value."""
    if score >= 80:
        return "#22c55e"
    elif score >= 60:
        return "#eab308"
    elif score >= 40:
        return "#f97316"
    else:
        return "#ef4444"


def create_gauge_chart(score, title="Overall Match Score"):
    """Create a gauge/donut chart for the overall score."""
    color = get_score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 48, "color": color}},
        title={"text": title, "font": {"size": 18, "color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "#1c1917"},
                {"range": [40, 60], "color": "#1c1917"},
                {"range": [60, 80], "color": "#1c1917"},
                {"range": [80, 100], "color": "#1c1917"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig


def create_radar_chart(scores_dict):
    """Create a radar chart for category scores."""
    categories = list(scores_dict.keys())
    values = list(scores_dict.values())
    # Close the polygon
    categories += [categories[0]]
    values += [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(99, 102, 241, 0.25)",
        line=dict(color="#818cf8", width=2),
        marker=dict(size=6, color="#818cf8"),
        name="Score",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor="#334155", linecolor="#334155",
                tickfont=dict(color="#94a3b8", size=10),
            ),
            angularaxis=dict(
                gridcolor="#334155", linecolor="#334155",
                tickfont=dict(color="#e2e8f0", size=12),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=350,
        margin=dict(l=60, r=60, t=40, b=40),
    )
    return fig


def create_bar_chart(scores_dict):
    """Create a horizontal bar chart for category breakdown."""
    categories = list(scores_dict.keys())
    values = list(scores_dict.values())
    colors = [get_score_color(v) for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=categories,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v}%" for v in values],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=13),
    ))
    fig.update_layout(
        xaxis=dict(
            range=[0, 110], gridcolor="#1e293b",
            tickfont=dict(color="#94a3b8"), title="",
        ),
        yaxis=dict(tickfont=dict(color="#e2e8f0", size=13), title=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin=dict(l=10, r=30, t=10, b=10),
    )
    return fig


# ─── App Title ────────────────────────────────────────────────────────
st.title("📄 Resume Analyzer Pro")
st.markdown("Upload your resume to analyze, match, and chat — powered by AI & RAG.")
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 1: Resume Details
# ═══════════════════════════════════════════════════════════════════════
if page == "Resume Details":
    st.header("📋 Resume Details Analysis")
    uploaded_file = get_shared_resume()
    if not uploaded_file:
        st.info("📤 Please upload a resume PDF in the sidebar to get started.")
        st.stop()

    if not api_key:
        st.warning("⚠️ Please enter your Hugging Face API key in the sidebar.")
        st.stop()

    if st.button("🔍 Analyze Resume"):
        with st.spinner("Analyzing resume..."):
            data = {"api_key": api_key}
            result = call_backend("resume_details", uploaded_file, data)
        if result:
            st.subheader("Extracted Resume Sections")
            feedback = result.get("llm_feedback", "")
            feedback = re.sub(r"\*\*(.+?)\*\*\s*:\s*", r"**\1** ", feedback)
            # Render with markdown so **bold** headings display prominently
            st.markdown("""
            <style>
                .resume-section strong {
                    font-size: 1.25rem;
                    color: #60a5fa;
                    display: block;
                    margin-top: 1rem;
                    margin-bottom: 0.25rem;
                    border-left: 3px solid #818cf8;
                    padding-left: 0.5rem;
                }
            </style>
            """, unsafe_allow_html=True)
            st.markdown(f'<div class="resume-section">\n\n{feedback}\n\n</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 2: Resume Matching (with charts)
# ═══════════════════════════════════════════════════════════════════════
elif page == "Resume Matching":
    st.header("🎯 Resume-Job Matching")
    st.markdown("Paste a job description and see how well your resume matches.")

    uploaded_file = get_shared_resume()
    if not uploaded_file:
        st.info("📤 Please upload a resume PDF in the sidebar to continue.")
        st.stop()

    job_description = st.text_area(
        "📝 Job Description",
        value=st.session_state.get("job_description", ""),
        height=200,
        placeholder="Paste the job description here...",
    )
    st.session_state.job_description = job_description

    if not job_description.strip():
        st.info("✏️ Enter a job description to continue.")
        st.stop()
    if not api_key:
        st.warning("⚠️ Please enter your Hugging Face API key in the sidebar.")
        st.stop()

    if st.button("🚀 Match Resume to Job", type="primary"):
        with st.spinner("🔄 Analyzing match with AI..."):
            data = {"api_key": api_key, "job_description": job_description}
            result = call_backend("resume_matching", uploaded_file, data)

        if result and result.get("llm_feedback"):
            feedback = result["llm_feedback"]

            # ── Extract scores from text using regex ──
            import re
            score_patterns = {
                "Total Match": r"TOTAL_MATCH_SCORE[:\s]*(\d{1,3})",
                "Skills": r"SKILLS_SCORE[:\s]*(\d{1,3})",
                "Experience": r"EXPERIENCE_SCORE[:\s]*(\d{1,3})",
                "Education": r"EDUCATION_SCORE[:\s]*(\d{1,3})",
                "Projects": r"PROJECTS_SCORE[:\s]*(\d{1,3})",
            }
            scores = {}
            for label, pattern in score_patterns.items():
                match = re.search(pattern, feedback, re.IGNORECASE)
                scores[label] = min(int(match.group(1)), 100) if match else 0

            overall = scores.get("Total Match", 0)

            # ── Remove score lines from feedback text for clean display ──
            clean_feedback = re.sub(
                r"(?:TOTAL_MATCH_SCORE|SKILLS_SCORE|EXPERIENCE_SCORE|EDUCATION_SCORE|PROJECTS_SCORE)[:\s]*\d{1,3}[/\d]*\s*",
                "", feedback
            ).strip()

            # ── Display gauge chart if we found an overall score ──
            if overall > 0:
                st.plotly_chart(create_gauge_chart(overall), use_container_width=True)

                # ── Category Score Cards ──
                category_scores = {k: v for k, v in scores.items() if k != "Total Match"}
                if any(v > 0 for v in category_scores.values()):
                    st.markdown("#### 🏆 Category Scores")
                    cols = st.columns(len(category_scores))
                    for i, (cat, score) in enumerate(category_scores.items()):
                        with cols[i]:
                            color = get_score_color(score)
                            st.markdown(
                                f"""<div class="score-card">
                                    <div class="score-number" style="color:{color}">{score}%</div>
                                    <div class="score-label">{cat}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                    # ── Charts ──
                    col_radar, col_bar = st.columns(2)
                    with col_radar:
                        st.subheader("📊 Radar Chart")
                        st.plotly_chart(create_radar_chart(category_scores), use_container_width=True)
                    with col_bar:
                        st.subheader("📈 Category Breakdown")
                        st.plotly_chart(create_bar_chart(category_scores), use_container_width=True)

                st.markdown("---")

            # ── Render feedback as markdown (## headings will display properly) ──
            st.subheader("📝 Detailed Feedback")
            st.markdown(clean_feedback, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 3: Chat with Resume and Job Description
# ═══════════════════════════════════════════════════════════════════════
elif page == "Chat with Resume and Job Description":
    st.header("💬 Chat with Resume & Job Description")
    st.markdown("Ask questions about your resume, career advice, or interview prep — powered by RAG.")

    # ── Sidebar controls for chat ──
    with st.sidebar:
        st.markdown("---")
        st.subheader("💬 Chat Settings")
        chat_source = st.selectbox(
            "Chat Context Source",
            ["All", "Resume Only", "Job Description Only", "Vector Store Only"],
            index=0,
            help="Choose what context the AI uses to answer your questions.",
        )
        source_map = {
            "All": "all",
            "Resume Only": "resume",
            "Job Description Only": "job",
            "Vector Store Only": "vectorstore",
        }
        selected_source = source_map[chat_source]

        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    uploaded_file = get_shared_resume()

    # ── Save to Vector DB ──
    if uploaded_file:
        if st.button("💾 Save Resume to Vector DB"):
            with st.spinner("Saving to vector database..."):
                result = call_backend_no_data("save_resume_to_vectorstore", uploaded_file)
            if result and result.get("success"):
                st.success(f"✅ {result['message']}")
            elif result:
                st.error(f"❌ {result.get('message', 'Failed to save.')}")

    job_description = st.text_area(
        "📝 Job Description (optional)",
        value=st.session_state.get("job_description", ""),
        height=150,
        placeholder="Paste job description for context...",
    )
    st.session_state.job_description = job_description

    if not uploaded_file:
        st.info("📤 Please upload a resume PDF in the sidebar to start chatting.")
        st.stop()

    if not api_key:
        st.warning("⚠️ Please enter your Hugging Face API key in the sidebar.")
        st.stop()

    st.markdown("---")

    # ── Chat History ──
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display existing messages
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["response"])

    # ── Chat Input ──
    user_question = st.chat_input("Ask anything about your resume, skills, or career...")

    if user_question:
        # Show user message immediately
        with st.chat_message("user"):
            st.write(user_question)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                data = {
                    "api_key": api_key,
                    "query": user_question,
                    "job_description": st.session_state.get("job_description", ""),
                    "chat_source": selected_source,
                }
                result = call_backend("chat_with_resume", uploaded_file, data)
                response = result.get("llm_feedback", "No response received.") if result else "No response received."
            st.write(response)

        # Store in history
        st.session_state.chat_history.append({
            "question": user_question,
            "response": response,
        })


# ═══════════════════════════════════════════════════════════════════════
# PAGE 4: About
# ═══════════════════════════════════════════════════════════════════════
elif page == "About":
    st.header("ℹ️ About Resume Analyzer Pro")
    st.markdown("""
    ### Welcome to Resume Analyzer Pro!

    This AI-powered application helps you analyze your resume against job descriptions and provides personalized recommendations to improve your career prospects.

    ### Features:
    - **Intelligent Resume Processing**: Upload PDF resumes with section-wise parsing and content visualization
    - **LLM-Powered Extraction**: AI-based extraction of skills, education, projects, work experience, and certifications via the Hugging Face Inference API
    - **Smart Job Matching**: Semantic matching using sentence-transformer embeddings and FAISS, with visual score charts
    - **Enhanced Chat Interface**: Context-aware conversations powered by a Retrieval-Augmented Generation (RAG) flow
    - **Vector Store Integration**: Save your resume into the FAISS vector database for enhanced retrieval
    - **Flexible Chat Sources**: Choose to chat with your resume, job descriptions, or the full vector store

    ### Technology Stack:
    - **RAG Pipeline**: Retrieval-Augmented Generation using LangChain text splitting and context assembly
    - **Sentence Transformers**: `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings` for semantic embeddings
    - **FAISS**: Local vector stores for efficient similarity search (`vector_store/`)
    - **LLM Inference**: Hugging Face Inference API with `mistralai/Mistral-7B-Instruct-v0.2`
    - **FastAPI**: Backend service exposing analysis, matching, chat, and vector store endpoints
    - **Streamlit**: Interactive web interface with Plotly charts
    - **PyPDF2**: PDF text extraction

    ### How It Works:
    1. **Upload & Process**: Upload your PDF resume; text is extracted locally with PyPDF2 and cleaned
    2. **Embedding & Retrieval**: Resume/job text is split into chunks, embedded with a sentence transformer, and searched in local FAISS stores
    3. **Smart Matching**: The LLM evaluates resume vs. job description with retrieved context to generate structured scores and feedback
    4. **Chat**: Ask questions; the system augments your query with relevant retrieved context and responds via the LLM
    5. **Save to Vector Store**: Store your resume in the FAISS database for future retrieval-augmented conversations

    ### Privacy Notice:
    Your resume data is processed locally and stored only for your session. We do not share your personal information with third parties.

    ### Contact:
    For support or feedback, please contact: @garvity
    """)

    st.markdown("---")
    st.markdown("Resume Analyzer Pro © 2026 | Powered by AI and Machine Learning")
