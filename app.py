# app.py
import streamlit as st
from io import BytesIO
from fpdf import FPDF
import base64
import os
import json
from pathlib import Path

# Optional AI libraries (transformers, openai, etc.)
try:
    from transformers import pipeline
    ai_pipeline_available = True
except:
    ai_pipeline_available = False

# --- APP CONFIG ---
st.set_page_config(page_title="AI Resume & Cover Letter Generator", page_icon=":briefcase:", layout="centered")

# --- CSS STYLING ---
st.markdown("""
<style>
/* Golden premium button */
.premium-btn {
    background-color: #FFD700;
    color: black;
    font-weight: bold;
    font-size: 18px;
    padding: 10px 20px;
    border-radius: 12px;
    text-align: center;
    display: inline-block;
    cursor: pointer;
    margin: 10px 0;
}

/* Dark / Light mode toggle switch */
.switch {
  position: relative;
  display: inline-block;
  width: 60px;
  height: 34px;
}
.switch input {display:none;}
.slider {
  position: absolute;
  cursor: pointer;
  top:0;
  left:0;
  right:0;
  bottom:0;
  background-color:#ccc;
  transition: .4s;
  border-radius: 34px;
}
.slider:before {
  position: absolute;
  content:"";
  height:26px;
  width:26px;
  left:4px;
  bottom:4px;
  background-color:white;
  transition:.4s;
  border-radius:50%;
}
input:checked + .slider {background-color:#2196F3;}
input:checked + .slider:before {transform: translateX(26px);}
</style>
""", unsafe_allow_html=True)

# --- THEME TOGGLE ---
theme_choice = st.radio("Select Theme:", ["Light", "Dark"])
if theme_choice == "Dark":
    st.markdown('<style>body {background-color: #0e1117; color: white;}</style>', unsafe_allow_html=True)
else:
    st.markdown('<style>body {background-color: white; color: black;}</style>', unsafe_allow_html=True)

st.title("🤖 AI-Powered Resume & Cover Letter Generator")

# --- UPLOAD / LINKEDIN ---
st.header("Upload Your Resume or LinkedIn Profile")
resume_file = st.file_uploader("Upload Resume (PDF/DOCX):", type=["pdf", "docx"])
linkedin_url = st.text_input("Or paste LinkedIn Profile URL:")

# --- JOB DESCRIPTION ---
job_description = st.text_area("Paste Job Description Here:")

# --- CUSTOM RESUME GENERATION ---
st.header("Generate Resume From Details")
with st.expander("Enter Your Details"):
    full_name = st.text_input("Full Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone Number")
    education = st.text_area("Education")
    experience = st.text_area("Experience")
    skills = st.text_area("Skills")
    achievements = st.text_area("Achievements")
    interests = st.text_area("Interests / Extra-Curriculars")

# --- PREMIUM PAYMENT BUTTON (RAZORPAY) ---
st.markdown('<div class="premium-btn" id="premium-btn">Upgrade to Premium</div>', unsafe_allow_html=True)

# Razorpay Integration
razorpay_key = "YOUR_RAZORPAY_KEY"  # Replace with your Razorpay API key
premium_script = f"""
<script src="https://checkout.razorpay.com/v1/checkout.js"></script>
<script>
var btn = document.getElementById('premium-btn');
btn.onclick = function(e){{
    var options = {{
        "key": "{razorpay_key}",
        "amount": 49900,  // 499 INR
        "currency": "INR",
        "name": "AI Resume Generator Premium",
        "description": "Premium subscription",
        "handler": function(response){{
            document.querySelector("#payment_status").innerText = "✅ Payment Successful! Premium features unlocked.";
        }},
        "theme": {{
            "color": "#F37254"
        }}
    }};
    var rzp1 = new Razorpay(options);
    rzp1.open();
    e.preventDefault();
}};
</script>
<div id="payment_status" style="color:green;font-weight:bold;margin-top:10px;"></div>
"""
st.components.v1.html(premium_script, height=150)

# --- GENERATE RESUME BUTTON ---
if st.button("Generate Resume PDF"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, full_name, ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.multi_cell(0, 8, f"Email: {email}\nPhone: {phone}\n\nEducation:\n{education}\n\nExperience:\n{experience}\n\nSkills:\n{skills}\n\nAchievements:\n{achievements}\n\nInterests:\n{interests}")
    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer)
    pdf_bytes = pdf_buffer.getvalue()

    # Download link
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="resume.pdf">📄 Download Your Resume PDF</a>'
    st.markdown(href, unsafe_allow_html=True)

# --- AI COVER LETTER / RESUME GENERATION (PREMIUM) ---
if st.button("Generate AI-Tailored Resume & Cover Letter"):
    if not job_description:
        st.warning("Please enter a job description first!")
    else:
        st.info("Generating AI resume & cover letter...")
        if ai_pipeline_available:
            # Example using a summarization / text generation model
            generator = pipeline("text2text-generation", model="google/flan-t5-small")
            prompt_resume = f"Optimize this resume for the following job description:\n{job_description}\n\nResume Info:\nName: {full_name}\nExperience: {experience}\nSkills: {skills}\nAchievements: {achievements}"
            ai_resume = generator(prompt_resume, max_length=500)[0]['generated_text']

            prompt_cover_letter = f"Generate a professional cover letter for the following job description:\n{job_description}\n\nCandidate Name: {full_name}\nExperience: {experience}\nSkills: {skills}\nAchievements: {achievements}"
            ai_cover_letter = generator(prompt_cover_letter, max_length=500)[0]['generated_text']

            st.subheader("AI-Tailored Resume")
            st.text(ai_resume)
            st.subheader("AI-Generated Cover Letter")
            st.text(ai_cover_letter)
        else:
            st.warning("AI libraries not installed. Please install 'transformers' and restart the app.")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit | Premium payments via Razorpay integration")

