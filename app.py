import streamlit as st
from utils import generate_resume_pdf, generate_cover_letter
import base64

# Page configuration
st.set_page_config(
    page_title="AI Resume & Cover Letter Generator",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Theme Toggle ---
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
    <style>
    .stApp {background-color: #1E1E1E; color: white;}
    button {background-color: goldenrod; color: white;}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {background-color: #FFFFFF; color: black;}
    button {background-color: goldenrod; color: white;}
    </style>
    """, unsafe_allow_html=True)

st.title("💼 AI Powered Resume & Cover Letter Generator")

# --- Sidebar Premium Upgrade ---
st.sidebar.markdown("### Premium Features")
if st.sidebar.button("Upgrade to Premium ✨"):
    st.markdown("""
    <script src="https://checkout.razorpay.com/v1/checkout.js"></script>
    <script>
    var options = {
        "key": "YOUR_RAZORPAY_KEY",
        "amount": "50000", 
        "currency": "INR",
        "name": "AI Resume App",
        "description": "Premium Upgrade",
        "handler": function(response){
            alert("Payment successful: " + response.razorpay_payment_id);
        },
        "theme": {"color": "goldenrod"}
    };
    var rzp1 = new Razorpay(options);
    rzp1.open();
    </script>
    """, unsafe_allow_html=True)

# --- Upload Resume or LinkedIn ---
st.header("Step 1: Upload Your Resume / LinkedIn")
uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
linkedin_url = st.text_input("Or enter LinkedIn profile URL")

# --- Job Description ---
st.header("Step 2: Enter Job Description")
job_description = st.text_area("Paste job description here...")

# --- AI Customization ---
st.header("Step 3: Generate Tailored Resume & Cover Letter")
if st.button("Generate"):
    if uploaded_file or linkedin_url:
        with st.spinner("Generating AI-tailored resume and cover letter..."):
            resume_pdf = generate_resume_pdf(uploaded_file, linkedin_url, job_description)
            cover_letter_pdf = generate_cover_letter(uploaded_file, linkedin_url, job_description)
            st.success("Generated successfully!")

            # Download buttons
            st.download_button("Download Resume", resume_pdf, file_name="Tailored_Resume.pdf", mime="application/pdf")
            st.download_button("Download Cover Letter", cover_letter_pdf, file_name="Cover_Letter.pdf", mime="application/pdf")
    else:
        st.warning("Please upload a resume or enter LinkedIn URL!")

# --- Generate Resume from Form ---
st.header("Or Create a Resume from Form")
with st.form("resume_form"):
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone")
    skills = st.text_area("Skills (comma-separated)")
    education = st.text_area("Education")
    experience = st.text_area("Experience")
    projects = st.text_area("Projects")
    achievements = st.text_area("Achievements")
    extra_curricular = st.text_area("Extra-Curricular Activities")
    submit = st.form_submit_button("Generate Resume PDF")

    if submit:
        resume_pdf = generate_resume_pdf(form_data={
            "name": name,
            "email": email,
            "phone": phone,
            "skills": skills,
            "education": education,
            "experience": experience,
            "projects": projects,
            "achievements": achievements,
            "extra_curricular": extra_curricular
        })
        st.success("Resume Generated!")
        st.download_button("Download Resume PDF", resume_pdf, file_name="Custom_Resume.pdf", mime="application/pdf")



