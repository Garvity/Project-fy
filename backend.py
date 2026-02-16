import os
import re
import numpy as np
from PyPDF2 import PdfReader
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from datetime import datetime, timezone
import uvicorn
from dotenv import load_dotenv

# os.environ['HF_HUB_DISABLE_XET'] = '1'
# # Optionally disable transfer for extra safety (helps with resume/chunk issues)
# os.environ['HF_HUB_DISABLE_TRANSFER'] = '1'

load_dotenv()
api_key = os.getenv("HF_TOKEN")

app = FastAPI()

# CORS middleware to allow requests from Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your Streamlit port if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

# Load embeddings model and both vector stores once
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
job_vectorstore = FAISS.load_local(
    "vector_store/job_faiss", embeddings, allow_dangerous_deserialization=True
)
resume_vectorstore = FAISS.load_local(
    "vector_store/resume_faiss", embeddings, allow_dangerous_deserialization=True
)

#  Extract text from uploaded PDF resume
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Clean the text data
def clean_text(text):
    """Cleans text by converting to lowercase, removing special characters,
    and handling whitespace."""
    try:
        text = str(text)
    except Exception as e:
        print(f"Error converting text to string: {e}")
        return text
    text = text.lower()
    text = re.sub(r"[^\x00-\x7f]", r"", text)
    text = re.sub(r"\t", r"", text).strip()
    text = re.sub(r"(\n|\r)+", r"\n", text).strip()
    text = re.sub(r" +", r" ", text).strip()
    return text


def get_llm_response(api_key, prompt, model="mistralai/Mistral-7B-Instruct-v0.2"):
    """
    Sends a prompt to Hugging Face Inference API and returns the generated text.
    """
    client = InferenceClient(provider= "featherless-ai", token=api_key)
    
    try:
        completion = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        response = completion.choices[0].message.content.strip()
        return response

    except Exception as e:
        return f"❌ Error generating response: {str(e)}"

# Page 1: Resume Details
@app.post("/resume_details")
async def resume_details(file: UploadFile = File(...), api_key: str = Form(...)):
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    prompt = f"""
    You are a resume parser. Extract and label ONLY these sections from the resume text below:
    - Name
    - Email
    - LinkedIn Profile
    - GitHub Profile
    - Portfolio
    - Phone Number
    - Education
    - Skills
    - Experience
    - Projects
    - Achievements
    - Certifications
    - Extra-curricular Activities

    Rules:
    - Output in bullet points: **Section Name**: Extracted content (keep concise).
    - Include ONLY sections present in the text—skip missing ones.
    - If no relevant text, output nothing for that section.
    - Use bold for section names.
    - Do NOT add any extra commentary or information.
    - Ensure the output is clean and easy to read.
    - If the resume is empty or unreadable, respond with "No content found in the resume."
    - The section headings should be in bold and big compared to the rest of the text.
    - Display the name in capitalized format.
    - Don't include any labels like "•" or "-".
    - Don't display the project github links.

    Resume Text:
    {resume_text[:4000]}  # Slightly longer limit

    Start output directly with bullets—no intro text.
    """

    feedback = get_llm_response(api_key, prompt)
    print(feedback)
    if "Error" in feedback:
        return {"error": feedback}
    return {"llm_feedback": feedback}

# Page 2: Resume Matching
@app.post("/resume_matching")
async def resume_matching(
    file: UploadFile = File(...), 
    job_description: str = Form(...), 
    api_key: str = Form(...)
):
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    prompt = f"""
    You are an expert resume evaluator. Analyze the resume against the job description.

    Job Description:
    {job_description}

    Resume:
    {resume_text[:3000]}

    You MUST start your response with exactly these score lines (each on its own line, with a number 0-100):
    TOTAL_MATCH_SCORE: <number>
    SKILLS_SCORE: <number>
    EXPERIENCE_SCORE: <number>
    EDUCATION_SCORE: <number>
    PROJECTS_SCORE: <number>

    Then provide detailed feedback using these exact section headings (use markdown ## for headings):

    ## Areas of Strength
    List the candidate's strongest matching qualifications as bullet points.

    ## Areas of Weakness
    List gaps or weaker areas as bullet points.

    ## Missing Skills & Qualifications
    List specific skills or qualifications from the job description that are missing from the resume.

    ## Matching Skills & Qualifications
    List skills and qualifications that match between the resume and job description.

    ## Suggestions for Improvement
    Provide specific, actionable suggestions to improve the resume match score.

    Rules:
    - Be concise and professional.
    - Use bullet points (- ) for items within each section.
    - Every section heading must use ## markdown format.
    - Scores should be realistic and well-justified.
    """
    feedback = get_llm_response(api_key, prompt)
    if "Error" in feedback:
        return {"error": feedback}
    return {"llm_feedback": feedback}

# Page 3: Chat with Resume and Job Description
@app.post("/chat_with_resume")
async def chat_with_resume(
    query: str = Form(...),
    api_key: str = Form(...),
    job_description: str = Form(""),
    chat_source: str = Form("all"),
    file: UploadFile = File(...)
):  
    # ── Extract and clean the user's uploaded resume ──
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    # ── Build context based on the selected chat_source ──
    context_sections = []

    # ALWAYS include the user's uploaded resume so the LLM knows who the candidate is
    context_sections.append(
        f"--- Candidate's Resume (uploaded by user) ---\n{resume_text[:4000]}"
    )

    if chat_source in ("all", "job") and job_description.strip():
        jd_cleaned = clean_text(job_description)
        context_sections.append(
            f"--- Job Description (provided by user) ---\n{jd_cleaned[:3000]}"
        )

    if chat_source in ("all", "vectorstore"):
        # ── Embed the RESUME CONTENT (not the query!) to find relevant matches ──
        # This ensures an AI/ML resume finds AI/ML jobs, not random PHP/WordPress ones.
        resume_chunks = splitter.split_text(resume_text)
        if resume_chunks:
            chunk_embeddings = [embeddings.embed_query(chunk) for chunk in resume_chunks]
            resume_embedding = np.mean(chunk_embeddings, axis=0).tolist()
        else:
            resume_embedding = embeddings.embed_query(resume_text[:2000])

        # Search for job descriptions that semantically match the resume's skills/content
        job_results = job_vectorstore.similarity_search_by_vector(resume_embedding, k=5)
        # Search for similar resumes in the database for comparison
        resume_results = resume_vectorstore.similarity_search_by_vector(resume_embedding, k=3)

        if job_results:
            job_context = "\n\n".join([doc.page_content for doc in job_results])
            context_sections.append(
                f"--- Matching Job Descriptions (from database, matched to candidate's skills) ---\n{job_context}"
            )
        if resume_results:
            resume_context = "\n\n".join([doc.page_content for doc in resume_results])
            context_sections.append(
                f"--- Similar Resumes (from database, for comparison) ---\n{resume_context}"
            )

    combined_context = "\n\n".join(context_sections)

    # ── Create the LLM prompt ──
    prompt = f"""You are an intelligent career assistant. Answer the user's question based ONLY on the context provided below.

User's Question: {query}

{combined_context}

Instructions:
- Answer the question based strictly on the context above.
- When the user asks about the candidate, resume, skills, projects, education, or experience, refer ONLY to the "Candidate's Resume" section.
- When the user asks about matching job descriptions, refer to the "Matching Job Descriptions" section — these were found by matching the candidate's resume skills against a database of real job postings.
- Do NOT invent or assume information that is not present in the context.
- If the context does not contain enough information to answer, say so clearly.
- Be concise and professional.
- When referring to the person whose resume was uploaded, call them "the candidate".
"""

    feedback = get_llm_response(api_key, prompt)
    if "Error" in feedback:
        return {"error": feedback}
    return {"llm_feedback": feedback}


# Page 3 helper: Save resume to vector store
@app.post("/save_resume_to_vectorstore")
async def save_resume_to_vectorstore(file: UploadFile = File(...)):
    """Extract text from uploaded resume PDF, chunk it, and add to the resume FAISS store."""
    global resume_vectorstore
    try:
        resume_text = extract_text_from_pdf(file.file)
        resume_text = clean_text(resume_text)

        if not resume_text.strip():
            return {"success": False, "message": "Could not extract text from the PDF."}

        chunks = splitter.split_text(resume_text)
        if not chunks:
            return {"success": False, "message": "No text chunks generated from the resume."}

        from langchain.schema import Document
        docs = [
            Document(page_content=chunk, metadata={"source": file.filename})
            for chunk in chunks
        ]
        resume_vectorstore.add_documents(docs)
        resume_vectorstore.save_local("vector_store/resume_faiss")

        return {
            "success": True,
            "message": f"Resume saved! {len(docs)} chunks added to vector store.",
        }
    except Exception as e:
        return {"success": False, "message": f"Error saving resume: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════
# Compare Resumes
# ═══════════════════════════════════════════════════════════════════════
@app.post("/compare_resumes")
async def compare_resumes(
    files: List[UploadFile] = File(...),
    job_description: str = Form(...),
    api_key: str = Form(...)
):
    """Score each uploaded resume against the job description and return ranked results."""
    results = []

    for file in files:
        resume_text = extract_text_from_pdf(file.file)
        resume_text = clean_text(resume_text)

        if not resume_text.strip():
            results.append({
                "filename": file.filename,
                "total_score": 0,
                "skills": 0,
                "experience": 0,
                "education": 0,
                "projects": 0,
                "summary": "Could not extract text from this PDF.",
            })
            continue

        prompt = f"""You are an expert resume evaluator. Score this resume against the job description.

Job Description:
{job_description[:3000]}

Resume ({file.filename}):
{resume_text[:3000]}

You MUST start your response with exactly these score lines (each on its own line, number 0-100):
TOTAL_MATCH_SCORE: <number>
SKILLS_SCORE: <number>
EXPERIENCE_SCORE: <number>
EDUCATION_SCORE: <number>
PROJECTS_SCORE: <number>

Then write a SHORT 2-3 sentence summary of the candidate's fit for this role.
Keep the summary concise — no bullet points, no section headings.
"""

        feedback = get_llm_response(api_key, prompt)

        # Parse scores from response
        scores = {}
        for label, pattern in {
            "total_score": r"TOTAL_MATCH_SCORE[:\s]*(\d{1,3})",
            "skills": r"SKILLS_SCORE[:\s]*(\d{1,3})",
            "experience": r"EXPERIENCE_SCORE[:\s]*(\d{1,3})",
            "education": r"EDUCATION_SCORE[:\s]*(\d{1,3})",
            "projects": r"PROJECTS_SCORE[:\s]*(\d{1,3})",
        }.items():
            match = re.search(pattern, feedback, re.IGNORECASE)
            scores[label] = min(int(match.group(1)), 100) if match else 0

        # Extract summary (everything after the score lines)
        summary = re.sub(
            r"(?:TOTAL_MATCH_SCORE|SKILLS_SCORE|EXPERIENCE_SCORE|EDUCATION_SCORE|PROJECTS_SCORE)[:\s]*\d{1,3}[/\d]*\s*",
            "", feedback
        ).strip()

        results.append({
            "filename": file.filename,
            **scores,
            "summary": summary,
        })

        # Reset file pointer for potential reuse
        file.file.seek(0)

    # Sort by total_score descending
    results.sort(key=lambda x: x["total_score"], reverse=True)
    return {"results": results}


@app.post("/chat_with_comparison")
async def chat_with_comparison(
    files: List[UploadFile] = File(...),
    query: str = Form(...),
    job_description: str = Form(...),
    api_key: str = Form(...)
):
    """Chat with comparative context from all uploaded resumes + job description."""
    context_sections = [
        f"--- Job Description ---\n{clean_text(job_description)[:3000]}"
    ]

    for i, file in enumerate(files, 1):
        resume_text = extract_text_from_pdf(file.file)
        resume_text = clean_text(resume_text)
        context_sections.append(
            f"--- Resume {i}: {file.filename} ---\n{resume_text[:2500]}"
        )
        file.file.seek(0)

    combined_context = "\n\n".join(context_sections)

    prompt = f"""You are an expert career advisor comparing multiple resumes against a job description.

User's Question: {query}

{combined_context}

Instructions:
- Answer based strictly on the resumes and job description above.
- When comparing candidates, refer to them by their resume filename.
- Be specific about why one candidate is stronger or weaker than another.
- If asked for rankings, provide scores and clear justifications.
- Be concise and professional.
"""

    feedback = get_llm_response(api_key, prompt)
    if "Error" in feedback:
        return {"error": feedback}
    return {"llm_feedback": feedback}


# ═══════════════════════════════════════════════════════════════════════
# Skill Gap Analysis
# ═══════════════════════════════════════════════════════════════════════
@app.post("/skill_gap_analysis")
async def skill_gap_analysis(file: UploadFile = File(...), api_key: str = Form(...)):
    """Extract candidate skills, find market-demanded skills from job vector store, compare."""
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    # 1) Extract candidate skills via LLM
    skill_prompt = f"""Extract ALL technical and professional skills from this resume.
Return ONLY a JSON array of skill strings, nothing else. Example: ["Python", "Machine Learning", "SQL"]

Resume:
{resume_text[:4000]}

Return ONLY the JSON array:"""

    skill_response = get_llm_response(api_key, skill_prompt)
    try:
        # Find the JSON array in the response
        match = re.search(r'\[.*?\]', skill_response, re.DOTALL)
        candidate_skills = json.loads(match.group(0)) if match else []
    except Exception:
        candidate_skills = []

    # 2) Search job vector store for relevant jobs using resume embedding
    resume_chunks = splitter.split_text(resume_text)
    if resume_chunks:
        chunk_embeddings = [embeddings.embed_query(c) for c in resume_chunks]
        resume_embedding = np.mean(chunk_embeddings, axis=0).tolist()
    else:
        resume_embedding = embeddings.embed_query(resume_text[:2000])

    job_results = job_vectorstore.similarity_search_by_vector(resume_embedding, k=10)
    job_texts = "\n\n".join([doc.page_content for doc in job_results])

    # 3) Extract market-demanded skills from matched jobs via LLM
    market_prompt = f"""Extract ALL technical and professional skills mentioned across these job descriptions.
Return ONLY a JSON array of skill strings, nothing else. Example: ["Python", "AWS", "Docker"]

Job Descriptions:
{job_texts[:5000]}

Return ONLY the JSON array:"""

    market_response = get_llm_response(api_key, market_prompt)
    try:
        match = re.search(r'\[.*?\]', market_response, re.DOTALL)
        market_skills = json.loads(match.group(0)) if match else []
    except Exception:
        market_skills = []

    # 4) Compare: normalize to lowercase for matching
    candidate_lower = {s.lower().strip() for s in candidate_skills}
    market_lower = {s.lower().strip() for s in market_skills}

    matched = sorted(candidate_lower & market_lower)
    missing = sorted(market_lower - candidate_lower)
    extra = sorted(candidate_lower - market_lower)

    return {
        "candidate_skills": sorted(candidate_lower),
        "market_skills": sorted(market_lower),
        "matched_skills": matched,
        "missing_skills": missing,
        "extra_skills": extra,
        "match_percentage": round(len(matched) / max(len(market_lower), 1) * 100),
    }


# ═══════════════════════════════════════════════════════════════════════
# ATS Score Estimator (LLM-Powered Semantic Analysis)
# ═══════════════════════════════════════════════════════════════════════
@app.post("/ats_score")
async def ats_score(file: UploadFile = File(...), api_key: str = Form(...)):
    """Use the LLM for semantic ATS analysis instead of regex pattern matching."""
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    prompt = f"""You are the MOST CRITICAL and unforgiving ATS evaluator. You represent a Fortune 500 company that rejects 95% of applicants.

BE BRUTALLY HONEST. Your job is to find EVERY flaw. The average resume scores 30-50% total. A "good" resume scores 50-65%. Only the top 5% of resumes score above 70%.

MANDATORY Scoring Constraints:
- NEVER give more than 70% of the max score in ANY category unless it is truly flawless.
- If you would rate something as "good", give it 40-50% of max.
- If you would rate something as "okay", give it 25-35% of max.
- If it's "present but not impressive", give 15-25% of max.
- Your TOTAL score should almost never exceed 65/100.

Categories and max scores with HARSH criteria:
1. Contact Info (max 20) — Requires ALL FOUR: email, phone, LinkedIn, GitHub/portfolio. Missing even ONE = max 10. Having only email+phone = max 7. No LinkedIn = automatic cap at 12.
2. Section Structure (max 25) — Requires CLEAR, LABELED, STANDARD headings for Education, Experience, Skills, Projects, and Summary. Missing any major section = max 12. Sections present but poorly organized = max 15. Creative/non-standard headings that confuse ATS parsers = max 10.
3. Resume Length (max 15) — Ideal is 400-600 words with high information density. Too short (<300) = max 4. Too long (>800) = max 6. Right length but with filler/fluff = max 8.
4. Impact & Metrics (max 15) — Demands SPECIFIC, QUANTIFIED results in EVERY bullet (e.g., "increased revenue by 23%", "reduced load time by 40%"). Vague verbs like "worked on", "helped with", "responsible for" = max 3. Some metrics but not all bullets = max 6. Good metrics but not in every bullet = max 9.
5. Keyword Relevance (max 15) — Keywords must appear IN CONTEXT with demonstrated application. A plain skills list = max 4. Keywords present but only in a skills section without project/experience context = max 6. Some contextual usage = max 9.
6. Formatting & Readability (max 10) — Must be perfectly clean, single-column, consistent bullet style, uniform fonts. ANY inconsistency = max 5. Minor issues = max 6. Good but not perfect = max 7.

Resume:
{resume_text[:4000]}

You MUST respond with ONLY valid JSON in this exact format, no other text:
{{
  "Contact Info": {{"score": <number>, "max": 20, "details": "<one-line critical explanation>"}},
  "Section Structure": {{"score": <number>, "max": 25, "details": "<one-line critical explanation>"}},
  "Resume Length": {{"score": <number>, "max": 15, "details": "<one-line critical explanation>"}},
  "Impact & Metrics": {{"score": <number>, "max": 15, "details": "<one-line critical explanation>"}},
  "Keyword Relevance": {{"score": <number>, "max": 15, "details": "<one-line critical explanation>"}},
  "Formatting & Readability": {{"score": <number>, "max": 10, "details": "<one-line critical explanation>"}}
}}
"""

    response = get_llm_response(api_key, prompt)

    # Parse the JSON from the LLM response
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        breakdown = json.loads(json_match.group(0)) if json_match else {}
    except Exception:
        breakdown = {
            "Contact Info": {"score": 0, "max": 20, "details": "Could not parse LLM response"},
            "Section Structure": {"score": 0, "max": 25, "details": "Could not parse LLM response"},
            "Resume Length": {"score": 0, "max": 15, "details": "Could not parse LLM response"},
            "Impact & Metrics": {"score": 0, "max": 15, "details": "Could not parse LLM response"},
            "Keyword Relevance": {"score": 0, "max": 15, "details": "Could not parse LLM response"},
            "Formatting & Readability": {"score": 0, "max": 10, "details": "Could not parse LLM response"},
        }

    # Ensure scores are clamped to their max values
    for cat in breakdown:
        if isinstance(breakdown[cat], dict):
            max_val = breakdown[cat].get("max", 100)
            breakdown[cat]["score"] = min(int(breakdown[cat].get("score", 0)), max_val)

    total = sum(v.get("score", 0) for v in breakdown.values())
    total_max = sum(v.get("max", 0) for v in breakdown.values())

    return {
        "total_score": total,
        "total_max": total_max,
        "percentage": round(total / max(total_max, 1) * 100),
        "breakdown": breakdown,
    }


# ═══════════════════════════════════════════════════════════════════════
# Resume Enhancement: AI Resume Rewriter
# ═══════════════════════════════════════════════════════════════════════
@app.post("/rewrite_resume")
async def rewrite_resume(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    job_description: str = Form(...)
):
    """Rewrite resume bullet points to better match the target job description."""
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    prompt = f"""You are an expert resume writer and career coach who values HONESTY above all.

Given the candidate's resume and target job description, your job is to HONESTLY assess the alignment and then enhance ONLY what is genuinely relevant.

Job Description:
{job_description[:3000]}

Original Resume:
{resume_text[:4000]}

CRITICAL Instructions — follow this exact structure:

**⚠️ Alignment Assessment**
First, honestly assess how well the resume's ACTUAL experience and projects match this JD. If the candidate's background is in a DIFFERENT domain (e.g., resume is AI/ML but JD is frontend, or resume is backend but JD is data science), clearly state:
- The resume's primary domain/expertise
- The JD's domain requirements
- The honest match level (Strong / Partial / Weak / Mismatch)
- Which parts of the resume genuinely relate to this JD and which don't

**✅ Relevant Experience — Enhanced**
ONLY rewrite bullet points from sections that GENUINELY align with the JD. Use strong action verbs and add metrics where truthful. Do NOT fabricate frontend experience from ML projects or vice versa. For each rewritten bullet, note WHY in [brackets].

**⚡ Transferable Skills**
If there's a domain mismatch, highlight bullet points that show transferable skills (e.g., problem-solving, system design, collaboration) without pretending they're direct matches.

**❌ Sections That Cannot Be Enhanced for This JD**
List any experience/projects that are NOT relevant to this JD and explain why rewriting them would be dishonest. Suggest what the candidate should do instead (e.g., build relevant projects, take courses).

**💡 Recommendations**
If the resume is a weak match for the JD, provide honest advice: should the candidate apply? What should they add to their resume first? What projects/skills would bridge the gap?

Remember: NEVER fabricate relevance. If an AI/ML project has nothing to do with a frontend JD, say so — don't rewrite it to sound like frontend work."""

    feedback = get_llm_response(api_key, prompt)
    if "Error" in feedback:
        return {"error": feedback}
    return {"llm_feedback": feedback}


# ═══════════════════════════════════════════════════════════════════════
# Resume Enhancement: Keyword Optimizer
# ═══════════════════════════════════════════════════════════════════════
@app.post("/keyword_optimizer")
async def keyword_optimizer(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    job_description: str = Form(...)
):
    """Identify missing JD keywords and suggest where to naturally insert them."""
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    prompt = f"""You are an expert ATS keyword analyst. You must be EXTREMELY PRECISE and HONEST.

CRITICAL RULE: Before classifying ANY keyword as "present" or "missing", you MUST verify it actually appears in the resume text below. Do NOT guess or assume — look for the EXACT keyword or a very close variant. If "Angular" is not written anywhere in the resume, it is MISSING — even if React is present.

Job Description:
{job_description[:3000]}

Resume (search this text carefully for each keyword):
{resume_text[:4000]}

Follow this EXACT structure:

**⚠️ Domain Alignment Check**
First, assess whether the resume's domain matches the JD. If the resume is AI/ML-focused but the JD asks for Frontend/Backend (or vice versa), clearly state this mismatch. This affects how realistic the keyword suggestions are.

**✅ Keywords VERIFIED Present in Resume**
List ONLY keywords from the JD that you can CONFIRM appear in the resume text above (exact match or very close variant). For each, quote the exact line from the resume where it appears.

**❌ Keywords Confirmed MISSING from Resume**
List keywords from the JD that are genuinely NOT in the resume. For each:
- **Keyword**: [the keyword]
- **Importance**: [High/Medium/Low]
- **Can Be Added Honestly?**: [Yes — the candidate has this skill/experience based on their resume] OR [No — adding this would be fabricating experience the candidate doesn't have]
- **Suggested Placement** (only if "Yes" above): Where and how to naturally insert it
- **Example Rewrite** (only if "Yes" above): Show the specific rephrased sentence

**🚫 Keywords That Should NOT Be Added**
List any JD keywords that the candidate clearly lacks experience in. Be honest — if the resume shows no Angular experience, don't suggest "just add Angular to your skills section." Instead, suggest learning resources or projects to build that skill.

**📊 Keyword Match Score**
Give an honest keyword match percentage based on verified matches only.

REMEMBER: Accuracy over helpfulness. A wrong "present" classification is worse than a harsh "missing" one."""

    feedback = get_llm_response(api_key, prompt)
    if "Error" in feedback:
        return {"error": feedback}
    return {"llm_feedback": feedback}


# ═══════════════════════════════════════════════════════════════════════
# Resume Enhancement: Cover Letter Generator
# ═══════════════════════════════════════════════════════════════════════
@app.post("/cover_letter")
async def cover_letter(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    job_description: str = Form(...),
    company_name: str = Form("the company"),
    tone: str = Form("professional")
):
    """Generate a tailored cover letter based on resume + job description."""
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    prompt = f"""You are an expert cover letter writer.

Write a compelling, tailored cover letter for the candidate applying to {company_name}.

Job Description:
{job_description[:3000]}

Candidate's Resume:
{resume_text[:4000]}

Tone: {tone}

Instructions:
- Write 3-4 paragraphs (300-400 words total).
- Opening: Express enthusiasm for the specific role and company. Hook the reader.
- Body (1-2 paragraphs): Connect 2-3 specific experiences/skills from the resume to the JD requirements. Use concrete examples with results.
- Closing: Reiterate interest, mention availability, call to action.
- Do NOT use generic phrases like "I am writing to apply for..." or "I believe I would be a great fit...".
- Make it sound human, confident, and specific to THIS role — not a template.
- Use the candidate's actual achievements from the resume, do not fabricate.
- Format as a proper letter with [Your Name] etc. placeholders for personal details."""

    feedback = get_llm_response(api_key, prompt)
    if "Error" in feedback:
        return {"error": feedback}
    return {"llm_feedback": feedback}


# ═══════════════════════════════════════════════════════════════════════
# Resume Enhancement: Resume Summary Generator
# ═══════════════════════════════════════════════════════════════════════
@app.post("/resume_summary")
async def resume_summary(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    job_description: str = Form(""),
    summary_type: str = Form("professional_summary")
):
    """Generate a professional summary/objective statement tailored to a specific role."""
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    jd_context = f"\nTarget Job Description:\n{job_description[:2000]}" if job_description.strip() else ""

    type_instructions = {
        "professional_summary": "Write a 3-4 sentence PROFESSIONAL SUMMARY that highlights the candidate's most relevant experience, key technical skills, and career achievements. It should read like a mini elevator pitch.",
        "objective": "Write a 2-3 sentence CAREER OBJECTIVE that states the candidate's career goals and what they aim to bring to the target role. Focus on value proposition.",
        "headline": "Write a single powerful HEADLINE (one line, under 15 words) that captures the candidate's professional identity. Example: 'Full-Stack Developer | 5+ Years in Scalable SaaS Products | AWS Certified'"
    }

    instruction = type_instructions.get(summary_type, type_instructions["professional_summary"])

    prompt = f"""You are an expert resume writer.

{instruction}

Candidate's Resume:
{resume_text[:4000]}
{jd_context}

Rules:
- Use ONLY information from the resume. Do not fabricate skills, years of experience, or achievements.
- If a JD is provided, tailor the summary to emphasize skills relevant to that role.
- Use strong, confident language. Avoid cliches like "hard-working" or "team player".
- Include specific technologies, tools, or domains from the resume.
- Generate exactly 3 different versions labeled Version 1, Version 2, Version 3 so the candidate can choose their favorite."""

    feedback = get_llm_response(api_key, prompt)
    if "Error" in feedback:
        return {"error": feedback}
    return {"llm_feedback": feedback}


# ═══════════════════════════════════════════════════════════════════════
# Job Search: Job Recommendation Feed
# ═══════════════════════════════════════════════════════════════════════
@app.post("/job_recommendations")
async def job_recommendations(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    top_n: int = Form(10)
):
    """Find best-matching jobs from the vector store based on resume embeddings."""
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    # Build a composite embedding from resume chunks
    resume_chunks = splitter.split_text(resume_text)
    if resume_chunks:
        chunk_embeddings = [embeddings.embed_query(c) for c in resume_chunks]
        resume_embedding = np.mean(chunk_embeddings, axis=0).tolist()
    else:
        resume_embedding = embeddings.embed_query(resume_text[:2000])

    # Search job vector store
    results_with_scores = job_vectorstore.similarity_search_with_score_by_vector(
        resume_embedding, k=min(top_n, 20)
    )

    jobs = []
    for doc, score in results_with_scores:
        # FAISS returns L2 distance — convert to similarity percentage
        similarity = float(round(max(0, (1 / (1 + score))) * 100, 1))
        jobs.append({
            "content": doc.page_content,
            "metadata": doc.metadata if doc.metadata else {},
            "similarity": similarity,
        })

    # Use LLM to generate a brief relevance summary for the top results
    if jobs:
        top_jobs_text = "\n\n".join([
            f"Job {i+1} (Similarity: {j['similarity']}%):\n{j['content'][:500]}"
            for i, j in enumerate(jobs[:5])
        ])

        prompt = f"""You are a career advisor. Given the candidate's resume and top matching jobs from the database, write a brief 2-3 sentence analysis for EACH of the top 5 jobs explaining WHY it's a good match.

Resume (key skills):
{resume_text[:2000]}

Top Matching Jobs:
{top_jobs_text}

For each job, write:
**Job [number]**: [2-3 sentence explanation of why this is a good fit]
"""
        summary = get_llm_response(api_key, prompt)
    else:
        summary = "No matching jobs found in the database."

    return {
        "jobs": jobs,
        "summary": summary,
        "total_found": len(jobs),
    }


# ═══════════════════════════════════════════════════════════════════════
# Job Search: Batch JD Matching
# ═══════════════════════════════════════════════════════════════════════
@app.post("/batch_jd_match")
async def batch_jd_match(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    job_descriptions: str = Form(...)
):
    """Score a single resume against multiple JDs and rank them by fit."""
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    # Parse JDs — split by the delimiter "---JD---"
    jd_list = [jd.strip() for jd in job_descriptions.split("---JD---") if jd.strip()]

    results = []
    for i, jd in enumerate(jd_list):
        prompt = f"""You are an expert resume-to-job matcher. Score how well this resume matches the job description.

Resume:
{resume_text[:3000]}

Job Description:
{jd[:2500]}

You MUST start your response with exactly these score lines (each on its own line, number 0-100):
MATCH_SCORE: <number>
SKILLS_FIT: <number>
EXPERIENCE_FIT: <number>

Then write a SHORT 2-3 sentence summary of why this job is or isn't a good fit for this candidate.
"""
        feedback = get_llm_response(api_key, prompt)

        # Parse scores
        scores = {}
        for label, pattern in {
            "match_score": r"MATCH_SCORE[:\s]*(\d{1,3})",
            "skills_fit": r"SKILLS_FIT[:\s]*(\d{1,3})",
            "experience_fit": r"EXPERIENCE_FIT[:\s]*(\d{1,3})",
        }.items():
            match = re.search(pattern, feedback, re.IGNORECASE)
            scores[label] = min(int(match.group(1)), 100) if match else 0

        # Extract summary
        summary = re.sub(
            r"(?:MATCH_SCORE|SKILLS_FIT|EXPERIENCE_FIT)[:\s]*\d{1,3}[/\d]*\s*",
            "", feedback
        ).strip()

        # Extract a short title from the JD (first line or first 80 chars)
        jd_title = jd.split('\n')[0].strip()[:80] or f"Job Description {i+1}"

        results.append({
            "jd_index": i + 1,
            "jd_title": jd_title,
            "jd_preview": jd[:200],
            **scores,
            "summary": summary,
        })

    # Sort by match_score descending
    results.sort(key=lambda x: x["match_score"], reverse=True)
    return {"results": results}


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    
# --- IGNORE ---
# # Code to create and save vector stores (run once)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vectorstore = FAISS.load_local("vector_store/job_faiss", embeddings, allow_dangerous_deserialization=True)
# resume_docs = [Document(page_content=text, metadata={"source": file.filename}) for text in chunks]
# vectorstore.add_documents(resume_docs)
# vectorstore.save_local("vector_store/job_faiss")
