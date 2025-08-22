# app.py
import streamlit as st
from io import BytesIO
import os
import re
import base64
from typing import List, Dict
from pathlib import Path

# Parsing libs
import pdfplumber
import docx
from docx import Document

# Text processing / optional ML libs
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    SBT_AVAILABLE = True
except Exception:
    SBT_AVAILABLE = False

# ---------- Configuration ----------
MODEL_DIR = Path("models/generator")  # put a transformers-compatible model here for LLM mode
EMBEDDING_MODEL_DIR = Path("models/embeddings")  # optional sentence-transformers model
# A small packed skills list (extendable)
COMMON_SKILLS = [
    "Python","Java","C++","SQL","JavaScript","HTML","CSS","React","Node.js","Django",
    "Flask","Machine Learning","Deep Learning","NLP","TensorFlow","PyTorch","Pandas",
    "Scikit-learn","Git","AWS","GCP","Azure","Docker","Kubernetes","CI/CD","Agile",
    "Scrum","Leadership","Communication","Excel","Tableau","Power BI"
]

st.set_page_config(page_title="AI Resume & Cover Letter Generator — Local", page_icon="📄", layout="wide")

# ---------- Helpers ----------
def read_pdf(file_bytes: bytes) -> str:
    text = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def read_docx(file_bytes: bytes) -> str:
    doc = docx.Document(BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_upload(uploaded_file) -> str:
    content = uploaded_file.read()
    if uploaded_file.name.lower().endswith(".pdf"):
        return read_pdf(content)
    elif uploaded_file.name.lower().endswith(".docx") or uploaded_file.name.lower().endswith(".doc"):
        return read_docx(content)
    else:
        # assume plain text
        try:
            return content.decode("utf-8")
        except Exception:
            return ""

def simple_skill_extraction(text: str, skills_list: List[str]) -> List[str]:
    found = []
    txt = text.lower()
    for s in skills_list:
        if s.lower() in txt:
            found.append(s)
    # dedupe and maintain order
    seen = set()
    ordered = []
    for s in found:
        if s not in seen:
            ordered.append(s)
            seen.add(s)
    return ordered

def highlight_relevant_sentences(resume_text: str, jd_text: str, top_k: int = 6) -> List[str]:
    """
    If sentence-transformers available, do semantic similarity; otherwise simple keyword overlap.
    """
    if SBT_AVAILABLE and EMBEDDING_MODEL_DIR.exists():
        try:
            embedder = SentenceTransformer(str(EMBEDDING_MODEL_DIR))
            resume_sents = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', resume_text) if len(s.strip())>10]
            jd_emb = embedder.encode(jd_text, convert_to_tensor=True)
            sent_embs = embedder.encode(resume_sents, convert_to_tensor=True)
            cos_scores = util.cos_sim(jd_emb, sent_embs)[0]
            top_idx = cos_scores.argsort(descending=True)[:top_k].cpu().tolist()
            return [resume_sents[i] for i in top_idx]
        except Exception:
            pass

    # fallback: keyword overlap ranking
    jd_words = set(re.findall(r"\w+", jd_text.lower()))
    sents = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', resume_text) if len(s.strip())>10]
    scored = []
    for s in sents:
        words = set(re.findall(r"\w+", s.lower()))
        score = len(words & jd_words)
        if score > 0:
            scored.append((score, s))
    scored.sort(reverse=True)
    return [s for _, s in scored[:top_k]]

def generate_template_cover_letter(name: str, role: str, company: str, top_skills: List[str], tone: str, bullet_points: List[str]) -> str:
    # A strong, reusable template that produces decent output offline
    intro = f"Dear {company} Hiring Team," if company else f"Dear Hiring Manager,"
    skill_line = ""
    if top_skills:
        skill_line = f" I bring experience in {', '.join(top_skills[:3])} and a track record of delivering measurable results."
    body = (
        f"My name is {name}. I'm excited to apply for the {role} role at {company}." if name or company else f"I am excited to apply for the {role} role."
    )
    body += skill_line + f" I am particularly drawn to this opportunity because of the alignment between my experience and your needs."

    bullets = "\n".join([f"- {b}" for b in bullet_points]) if bullet_points else ""
    closing = {
        "professional": "Sincerely,\n" + (name or "Candidate"),
        "friendly": "Best regards,\n" + (name or "Candidate"),
        "confident": "Looking forward to contributing,\n" + (name or "Candidate")
    }.get(tone, "Sincerely,\n" + (name or "Candidate"))

    return f"{intro}\n\n{body}\n\nRelevant achievements:\n{bullets}\n\n{closing}"

def call_local_llm(prompt: str, model_dir: Path, max_length: int = 512) -> str:
    """
    Use transformers pipeline on a local seq2seq model (e.g. Flan-T5). The user must download and place model files under model_dir.
    This function gracefully fails back to template generation if no transformer available.
    """
    if not TRANSFORMERS_AVAILABLE or not model_dir.exists():
        raise RuntimeError("Local model not available")

    # Load pipeline (simple generate with seq2seq)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
    out = gen(prompt, max_length=max_length, truncation=True)
    return out[0]["generated_text"]

def build_tailoring_prompt(resume_text: str, jd_text: str, name: str, role: str, tone: str, instructions: str) -> str:
    prompt = (
        "You are a resume-tailoring assistant. Given the user's existing resume content and a job description, produce:\n"
        "1) A tailored resume summary (2-4 sentences) highlighting most relevant skills and achievements.\n"
        "2) A tailored cover letter (2 short paragraphs + closing).\n"
        "3) 3 suggested bullet points to add to experience section tailored for the role.\n\n"
        f"User name: {name}\n"
        f"Target role: {role}\n"
        f"Tone: {tone}\n"
        f"Extra instructions: {instructions}\n\n"
        "Job Description:\n" + jd_text + "\n\n"
        "Resume Text:\n" + resume_text + "\n\n"
        "Format the output clearly with headings: RESUME_SUMMARY, COVER_LETTER, SUGGESTED_BULLETS.\n"
    )
    return prompt

def create_docx_resume(original_text: str, tailored_summary: str, suggested_bullets: List[str]) -> bytes:
    doc = Document()
    doc.add_heading("Tailored Resume", level=1)
    doc.add_paragraph(tailored_summary)
    doc.add_paragraph("Suggested bullet points to include:")
    for b in suggested_bullets:
        doc.add_paragraph(b, style='List Bullet')
    doc.add_page_break()
    doc.add_heading("Original / Parsed Resume Text", level=2)
    for para in original_text.splitlines():
        if para.strip():
            doc.add_paragraph(para)
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.read()

def make_download_link(file_bytes: bytes, filename: str, label: str):
    b64 = base64.b64encode(file_bytes).decode()
    href = f'<a download="{filename}" href="data:application/octet-stream;base64,{b64}">{label}</a>'
    return href

# ---------- UI ----------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#f7fbff,#ffffff); }
    .header { display:flex; align-items:center; gap:16px; }
    .card { padding:18px; border-radius:12px; box-shadow: 0 6px 18px rgba(16,24,40,0.06); background: white; }
    </style>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([1,2])
with col1:
    st.image("https://raw.githubusercontent.com/samkreter/sample-assets/main/resume.png", width=160) if st.button("Load demo image (optional)") else None

with col2:
    st.markdown("<div class='header'><h1>AI Resume & Cover Letter Generator — Local</h1></div>", unsafe_allow_html=True)
    st.markdown("Upload resume (PDF/DOCX/TXT) or paste LinkedIn profile. Provide the Job Description and get a tailored resume + cover letter. Runs offline — works even without an LLM model (template fallback).")

st.sidebar.title("Settings & Mode")
use_local_llm = False
if MODEL_DIR.exists() and TRANSFORMERS_AVAILABLE:
    use_local_llm = st.sidebar.checkbox("Enable local LLM (if model present)", value=True)
else:
    st.sidebar.info("No local generator model found or transformers not installed. App will use offline template mode.")
st.sidebar.markdown("**Tone**")
tone = st.sidebar.selectbox("Select tone", ["professional","friendly","confident"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Team / Fine-tune**")
fine_tune_path = st.sidebar.text_input("Path to team fine-tuned model (optional)", value=str(MODEL_DIR))
st.sidebar.markdown("If your team fine-tuned a model, point this to the model dir to use it.")

st.header("1) Provide Resume / LinkedIn")
uploaded = st.file_uploader("Upload your resume (PDF/DOCX/TXT). If you want, paste LinkedIn profile below.", type=['pdf','docx','doc','txt'])
resume_text = ""
if uploaded:
    resume_text = extract_text_from_upload(uploaded)
    st.success("Resume uploaded and parsed.")
else:
    linkedin = st.text_area("Or paste LinkedIn profile / CV text (leave blank if uploading file)", height=180)
    if linkedin.strip():
        resume_text = linkedin.strip()

if resume_text:
    st.markdown("**Parsed resume preview (first 400 chars):**")
    st.write(resume_text[:1000] + ("..." if len(resume_text)>1000 else ""))

st.header("2) Provide Job Description")
jd_input = st.text_area("Paste the job description or upload a JD", height=260)
if not jd_input:
    st.info("Tip: paste the full job description or link (paste text). For best results include responsibilities & requirements.")

st.header("3) Tailoring options")
name = st.text_input("Your name (optional)")
target_role = st.text_input("Target role / title (e.g. 'Frontend Engineer')")
company = st.text_input("Company name (optional)")
extra_instructions = st.text_area("Extra instructions for tailoring (tone, must-include keywords, metrics to emphasize)", height=80)

if st.button("Generate tailored resume & cover letter"):
    if not resume_text or not jd_input:
        st.error("Please provide both resume (or LinkedIn) and job description.")
    else:
        with st.spinner("Analyzing & tailoring..."):
            # Extract skills
            detected_skills = simple_skill_extraction(resume_text, COMMON_SKILLS)
            relevant_sents = highlight_relevant_sentences(resume_text, jd_input, top_k=6)

            # Prepare suggestions
            suggested_bullets = []
            for s in relevant_sents:
                # create short bullet if too long
                b = s.strip()
                if len(b) > 180:
                    b = b[:177] + "..."
                suggested_bullets.append(b)
            if not suggested_bullets:
                suggested_bullets = ["Delivered X result using Y", "Improved process by N% through Z", "Led a team of X to achieve Y"]

            tailored_summary = ""
            generated_cover_letter = ""

            if use_local_llm and Path(fine_tune_path).exists() and TRANSFORMERS_AVAILABLE:
                try:
                    prompt = build_tailoring_prompt(resume_text, jd_input, name, target_role, tone, extra_instructions)
                    llm_out = call_local_llm(prompt, Path(fine_tune_path), max_length=512)
                    # naive parse if the model followed headings
                    if "RESUME_SUMMARY" in llm_out:
                        parts = re.split(r'RESUME_SUMMARY|RESUME_SUMMARY|COVER_LETTER|SUGGESTED_BULLETS', llm_out)
                        # fallback simple
                        tailored_summary = llm_out[:600]
                        generated_cover_letter = llm_out[:1000]
                    else:
                        # use whole output for cover letter and summary fallback
                        tailored_summary = llm_out[:400]
                        generated_cover_letter = llm_out[400:1400]
                except Exception as e:
                    st.warning(f"Local LLM generation failed: {e}. Falling back to template mode.")
                    tailored_summary = f"{target_role} with experience in {', '.join(detected_skills[:4])}."
                    generated_cover_letter = generate_template_cover_letter(name, target_role, company, detected_skills, tone, suggested_bullets[:3])
            else:
                # Template / deterministic fallback
                tailored_summary = f"{target_role} with experience in {', '.join(detected_skills[:5])} who has a strong track record delivering results aligned to the job description."
                generated_cover_letter = generate_template_cover_letter(name or "Candidate", target_role, company, detected_skills, tone, suggested_bullets[:3])

            # Show results
            st.subheader("Tailored Resume Summary")
            st.write(tailored_summary)

            st.subheader("Suggested bullet points to add to your resume")
            for b in suggested_bullets[:6]:
                st.markdown(f"- {b}")

            st.subheader("Tailored Cover Letter")
            st.write(generated_cover_letter)

            # Create downloadable docx resume
            docx_bytes = create_docx_resume(resume_text, tailored_summary, suggested_bullets[:6])
            st.markdown(make_download_link(docx_bytes, "tailored_resume.docx", "📥 Download tailored resume (DOCX)"), unsafe_allow_html=True)

            # Cover letter download
            cl_bytes = generated_cover_letter.encode("utf-8")
            st.markdown(make_download_link(cl_bytes, "cover_letter.txt", "📥 Download cover letter (TXT)"), unsafe_allow_html=True)

            st.success("Done — review and tweak any outputs before sending to employers.")

st.markdown("---")
st.markdown("**Notes & tips**: The app runs entirely locally. To enable a richer LLM-driven tailoring, download a seq2seq model (e.g. Flan-T5) and place it under `models/generator` (or point the sidebar to your fine-tuned model path). If no model is present the app uses deterministic templates + matching which works offline and reliably.")




