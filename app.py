import streamlit as st
from io import BytesIO
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import razorpay

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
/* App background */
.stApp {
    background: linear-gradient(180deg, #eef4ff, #ffffff);
    font-family: 'Segoe UI', sans-serif;
}
/* Header */
.header {
    font-size: 34px;
    font-weight: bold;
    color: #003366;
    padding: 15px 0;
    text-align: center;
}
/* Section container */
.section {
    background-color: white;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    margin-bottom: 25px;
    transition: all 0.3s ease;
}
.section:hover {
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    transform: translateY(-2px);
}
/* Buttons */
.stButton button {
    background: linear-gradient(90deg, #00509e, #0073e6);
    color: white;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    border: none;
    transition: all 0.3s ease;
    font-weight: bold;
}
.stButton button:hover {
    background: linear-gradient(90deg, #0073e6, #00509e);
    transform: scale(1.05);
}
/* Download link */
a.download-link {
    display: inline-block;
    background: linear-gradient(90deg, #28a745, #218838);
    color: white;
    padding: 10px 20px;
    border-radius: 8px;
    text-decoration: none;
    font-weight: bold;
    transition: all 0.3s ease;
}
a.download-link:hover {
    background: linear-gradient(90deg, #218838, #28a745);
    transform: scale(1.05);
}
/* Premium badge */
.premium-badge {
    display: inline-block;
    background: gold;
    color: black;
    padding: 5px 12px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 0.85rem;
    margin-left: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- PDF GENERATOR FUNCTION ----------
def generate_pdf(name, email, phone, skills, experience):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # HEADER
    p.setFillColor(colors.HexColor("#003366"))
    p.rect(0, height - 80, width, 80, fill=True, stroke=False)
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 24)
    p.drawString(50, height - 50, name)

    # CONTACT INFO
    p.setFillColor(colors.HexColor("#00509e"))
    p.rect(0, height - 100, width, 20, fill=True, stroke=False)
    p.setFillColor(colors.white)
    p.setFont("Helvetica", 10)
    p.drawString(50, height - 95, f"Email: {email}  |  Phone: {phone}")

    y = height - 140

    # SKILLS SECTION
    p.setFillColor(colors.HexColor("#e6f0ff"))
    p.rect(40, y - 20, width - 80, 20, fill=True, stroke=False)
    p.setFillColor(colors.HexColor("#003366"))
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, y - 15, "Skills")
    y -= 40
    p.setFillColor(colors.black)
    p.setFont("Helvetica", 12)
    for skill in skills.split(","):
        p.drawString(60, y, f"• {skill.strip()}")
        y -= 15

    # EXPERIENCE SECTION
    y -= 20
    p.setFillColor(colors.HexColor("#e6f0ff"))
    p.rect(40, y - 20, width - 80, 20, fill=True, stroke=False)
    p.setFillColor(colors.HexColor("#003366"))
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, y - 15, "Experience")
    y -= 40
    p.setFillColor(colors.black)
    p.setFont("Helvetica", 12)
    for exp in experience.split("\n"):
        p.drawString(60, y, f"• {exp.strip()}")
        y -= 15

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# ---------- RAZORPAY SETUP ----------
razorpay_client = razorpay.Client(auth=("YOUR_RAZORPAY_KEY_ID", "YOUR_RAZORPAY_SECRET"))

# ---------- APP HEADER ----------
st.markdown('<div class="header">AI Resume Generator <span class="premium-badge">Premium</span></div>', unsafe_allow_html=True)

# ---------- USER FORM ----------
with st.form("resume_form"):
    st.markdown('<div class="section">', unsafe_allow_html=True)
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone")
    skills = st.text_area("Skills (comma separated)")
    experience = st.text_area("Experience (each on a new line)")
    st.markdown('</div>', unsafe_allow_html=True)
    submitted = st.form_submit_button("Generate Resume")

if submitted:
    pdf_buffer = generate_pdf(name, email, phone, skills, experience)
    b64 = base64.b64encode(pdf_buffer.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="resume.pdf" class="download-link">📄 Download Resume</a>'
    st.markdown(href, unsafe_allow_html=True)

# ---------- PREMIUM UPGRADE BUTTON ----------
if st.button("Upgrade to Premium 💳"):
    order = razorpay_client.order.create({
        "amount": 50000,  # amount in paise (500 INR)
        "currency": "INR",
        "payment_capture": "1"
    })
    st.write(f"Click the link below to pay via Razorpay:")
    st.markdown(f"[Pay Now](https://rzp.io/i/{order['id']})")





