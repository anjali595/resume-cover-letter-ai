import pdfkit
from jinja2 import Environment, FileSystemLoader
import io

# --- AI Integration Placeholder ---
from google.generativeai import TextGenerationClient
import os

# Set your Google Gemini API key as environment variable
API_KEY = os.getenv("GOOGLE_API_KEY")
client = TextGenerationClient(api_key=API_KEY)

def extract_data_from_resume(uploaded_file=None, linkedin_url=None, job_description=None):
    """
    Extracts or generates resume data from uploaded file or LinkedIn URL using AI
    """
    prompt = f"""
    Extract name, email, phone, skills, education, experience, projects, achievements, and extra-curricular
    from the uploaded resume or LinkedIn URL. Tailor it for the following job description:
    {job_description}
    """
    
    response = client.generate_text(
        model="gemini-1.5",
        prompt=prompt,
        max_output_tokens=800
    )
    # Expected format: JSON-like string from AI
    try:
        import json
        data = json.loads(response.text)
    except:
        # Fallback dummy data
        data = {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "+91 9876543210",
            "skills": "Python, AI, ML",
            "education": "B.Tech in Computer Science",
            "experience": "Software Engineer at XYZ",
            "projects": "AI Resume Generator",
            "achievements": "Hackathon Winner",
            "extra_curricular": "Football, Music"
        }
    return data

def generate_resume_pdf(form_data):
    """
    Generates PDF resume from form_data dictionary
    """
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('resume_template.html')
    html_out = template.render(form_data)
    pdf_out = pdfkit.from_string(html_out, False)
    return pdf_out

def generate_cover_letter(user_data, job_description):
    """
    Generates AI-powered cover letter PDF
    """
    prompt = f"""
    Write a professional cover letter for a candidate with the following data:
    {user_data}
    Tailored to this job description:
    {job_description}
    """
    response = client.generate_text(
        model="gemini-1.5",
        prompt=prompt,
        max_output_tokens=500
    )
    cover_letter_text = response.text or "Dear Hiring Manager,\n\nI am excited to apply for this position..."
    pdf_out = pdfkit.from_string(cover_letter_text, False)
    return pdf_out
