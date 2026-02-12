import os

# os.environ['HF_HUB_DISABLE_XET'] = '1'
# # Optionally disable transfer for extra safety (helps with resume/chunk issues)
# os.environ['HF_HUB_DISABLE_TRANSFER'] = '1'

import re
import numpy as np
from PyPDF2 import PdfReader
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from datetime import datetime
import uvicorn
from dotenv import load_dotenv
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

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    
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
