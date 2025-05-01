import streamlit as st
import joblib
import os
import re
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
import requests
import json
import time

model = joblib.load('decision_tree_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
category_map = {0: "Peoplesoft", 1: "React.js Developer", 2: "SQL Lighting Insight", 3: "WorkDay"}

cleaned_details = pd.read_excel('cleaned_details.xlsx')

API_URL = "https://api.mistral.ai/v1/chat/completions"
API_KEY = "E2wXeQFM9n1TBhcRFZM21TZgoc4IxlO6"
MODEL_ID = "mistral-small-latest"

PREDEFINED_ROLES = {
    "Data Scientist": [
        "python", "machine learning", "nlp", "data science", "statistics", 
        "data visualization", "pandas", "numpy", "deep learning", "sql",
    ],
    "Front-End Developer": [
        "html", "css", "javascript", "react.js", "angular", 
        "vue.js", "sass", "bootstrap", "jquery", "responsive design",
    ],
    "Data Analyst": [
        "excel", "sql", "power bi", "tableau", "data cleaning", 
        "data visualization", "statistical analysis", "r", "python", "pivot tables",
    ],
    "Back-End Developer": [
        "java", "c#", "python", "node.js", "ruby", 
        "sql", "api development", "database management", "django", "flask",
    ],
    "Full Stack Developer": [
        "html", "css", "javascript", "react.js", "node.js", 
        "express.js", "mongodb", "postgresql", "git", "docker",
    ],
    "Mobile App Developer": [
        "java", "kotlin", "swift", "react native", "flutter", 
        "android", "ios", "firebase", "apis", "mobile testing",
    ],
    "DevOps Engineer": [
        "docker", "kubernetes", "terraform", "jenkins", "ansible", 
        "aws", "azure", "gcp", "linux", "ci/cd pipelines",
    ],
    "Cloud Architect": [
        "aws", "azure", "gcp", "cloud security", "cloud migration", 
        "networking", "virtualization", "terraform", "docker", "kubernetes",
    ],
    "UI/UX Designer": [
        "ux research", "wireframing", "prototyping", "figma", "sketch", 
        "adobe xd", "user testing", "responsive design", "user flows", "interaction design",
    ],
    "Cybersecurity Analyst": [
        "network security", "firewall", "penetration testing", "encryption", "ethical hacking", 
        "incident response", "malware analysis", "cryptography", "risk management", "compliance",
    ],
    "Product Manager": [
        "agile", "scrum", "product lifecycle", "roadmapping", "user stories", 
        "market research", "stakeholder management", "product vision", "data analysis", "prioritization",
    ],
    "HR Specialist": [
        "recruitment", "talent acquisition", "employee relations", "hr management", "performance management", 
        "training and development", "employee benefits", "compensation analysis", "onboarding", "compliance",
    ],
    "Sales Manager": [
        "sales strategy", "account management", "customer relationship", "negotiation", "crm", 
        "lead generation", "salesforce", "closing deals", "sales forecasting", "market analysis",
    ],
    "Marketing Manager": [
        "seo", "sem", "content marketing", "social media", "email marketing", 
        "brand management", "market research", "google analytics", "pay-per-click", "campaign management",
    ],
    "Financial Analyst": [
        "financial modeling", "forecasting", "excel", "data analysis", "accounting", 
        "budgeting", "financial statements", "valuation", "risk analysis", "investment analysis",
    ],
    "Content Writer": [
        "copywriting", "blogging", "editing", "proofreading", "seo writing", 
        "content strategy", "social media content", "research", "creative writing", "technical writing",
    ],
    "Network Engineer": [
        "network protocols", "routing", "switching", "firewalls", "vpn", 
        "dns", "dhcp", "network security", "wireshark", "network monitoring",
    ],
    "AI Engineer": [
        "python", "deep learning", "tensorflow", "keras", "pytorch", 
        "machine learning", "ai algorithms", "computer vision", "nlp", "reinforcement learning",
    ],
}

st.set_page_config(page_title="Resume Management App", page_icon="📄", layout="wide")

st.sidebar.title("Navigation")
menu = st.sidebar.radio("Choose a page", ["Home", "About", "Resume Screening", "Resume Classification", "Contact Us"])

def create_prompt(text):
    return f"""
You are a resume parser. Extract **all useful information** from the resume below in clean, structured JSON format. 
Include fields like name, email, phone, location, education, experience, skills, projects, profile links, certifications, etc.

Resume:
{text}
"""

def parse_resume_with_mistral(text):
    prompt = create_prompt(text)
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            content = response.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            st.success("Candidates Information Extracted Successfully.")
            st.code(response.json()["choices"][0]["message"]["content"])
            return None
    else:
        st.error(f"Mistral API Error {response.status_code}: {response.text}")
        return None

def extract_text_from_file(file):
    text = ""
    if file.name.endswith('.pdf'):
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    elif file.name.endswith('.docx'):
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

def extract_resume_details(resume_text, predefined_roles):
    matched_role = None
    extracted_skills = []

    for role, skills in predefined_roles.items():
        matched_skills = [skill for skill in skills if skill.lower() in resume_text.lower()]
        if len(matched_skills) > len(extracted_skills):
            extracted_skills = matched_skills
            matched_role = role

    name = re.search(r"Name:\s*([A-Za-z\s]+)", resume_text)
    education = re.findall(r"(B\.\w+\s+in\s+\w+|M\.\w+\s+in\s+\w+)", resume_text)
    experience = re.search(r"(\d+)\s+years?\s+of\s+experience", resume_text)
    certifications = re.findall(r"(Certified\s+\w+|AWS Certified|Azure Certified)", resume_text)

    return {
        "Name": name.group(1).strip() if name else "Not Found",
        "Education": education if education else ["Not Found"],
        "Skills": extracted_skills if extracted_skills else ["Not Found"],
        "Experience": experience.group(1) + " years" if experience else "Not Found",
        "Certifications": certifications if certifications else ["Not Found"],
        "Matched Role": matched_role if matched_role else "Not Found",
    }

def categorize_resumes(uploaded_files, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    results = []

    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(('.pdf', '.docx')):
            text = extract_text_from_file(uploaded_file)
            
            if text:
                cleaned_resume = text.strip()

                parsed = parse_resume_with_mistral(cleaned_resume)
                if parsed:
                    parsed["filename"] = uploaded_file.name
                    results.append(parsed)
                    st.json(parsed)

    return results

if menu == "Home":
    st.title("Welcome to the Resume Management App")
    st.markdown("""
        ## Introduction
        The *Resume Management App* is designed to streamline and automate the tedious process of managing and screening resumes for HR professionals and recruiters. Our platform offers the following features to help you efficiently handle resume data:

        - *Resume Screening*: Extract relevant information such as name, education, skills, experience, and certifications from resumes in various formats (PDF, DOCX).
        - *Resume Classification*: Classify resumes based on predefined roles like People Soft, WorkDay, SQL Developer, and React JS Developer.
        - *Automated Categorization*: The app uses machine learning models to categorize resumes accurately based on the content.

        ## Features:
        - *Simple Upload Interface*: Upload resumes easily and receive categorized data in just a few steps.
        - *Automatic Text Extraction*: Extract text from both PDF and DOCX files.
        - *Skills-based Role Matching*: The system matches resumes with predefined roles based on the skills and experience found in the resumes.
        - *Downloadable Results*: View the results in a table format and download categorized resumes as CSV files.

        ## Why Choose This App?
        - *Saves Time*: Automates the manual task of resume screening.
        - *Increases Accuracy*: Ensures consistent classification and minimizes human error.
        - *User-Friendly*: Intuitive interface designed for HR professionals, recruiters, and hiring managers.

        ### Get Started Now!
        Upload your resumes and let the app do the rest. Start classifying and analyzing resumes today!

        """, unsafe_allow_html=True)

elif menu == "About":
    st.title("About the App")
    st.markdown("""
        This app is designed to assist HR professionals in:
        - Screening resumes to extract key details.
        - Classifying resumes into job roles based on predefined roles like People Soft, WorkDay, SQL Developer, and React JS Developer.
        
        *Features:*
        - Automatic extraction of details like Name, Skills, Education, and Experience.
        - Categorization based on predefined skills for various roles.
        
        This tool enhances hiring workflows by reducing manual effort and ensuring accuracy.
    """)
    st.image("D:/Data Science ExcelR/Projects/Project - 473 RESUME CLASSIFICATION/Resume Classification Project Modules/screening/resume_screening_deployment/resume-screening-software.png", caption="Automated Resume Screening", use_container_width=True)

elif menu == "Resume Screening":
    st.title("Resume Screening")
    uploaded_files = st.file_uploader("Upload Resume(s) (PDF or DOCX):", type=["pdf", "docx"], accept_multiple_files=True)
    
    results = []
    if uploaded_files:
        start_time = time.time()
        with st.spinner("Processing resumes..."):
            results = categorize_resumes(uploaded_files, "output_resumes")
        
        end_time = time.time()
        total_time = round(end_time - start_time, 2)
        st.info(f"Extracted information from resumes in {total_time} seconds.")

elif menu == "Resume Classification":
    st.title("Resume Classification")
    uploaded_files = st.file_uploader("Upload Resumes (.pdf, .docx):", type=["pdf", "docx"], accept_multiple_files=True)

    if uploaded_files:
        classification_results = []

        with st.spinner("Classifying resumes..."):
            for uploaded_file in uploaded_files:
                resume_text = extract_text_from_file(uploaded_file)
                if resume_text:
                    cleaned_text = resume_text.strip()
                    vectorized = vectorizer.transform([cleaned_text])
                    prediction = model.predict(vectorized)[0]
                    category = category_map.get(prediction, "Unknown")

                    classification_results.append({
                        "Filename": uploaded_file.name,
                        "Predicted Category": category,
                        "Raw Text": cleaned_text[:500] + "..."  
                    })

        if classification_results:
            df_classified = pd.DataFrame(classification_results)
            st.success(f"Classified {len(df_classified)} resumes successfully.")
            st.dataframe(df_classified)

            csv = df_classified.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, file_name="classified_resumes.csv", mime="text/csv")
        else:
            st.warning("No resumes were classified.")


elif menu == "Contact Us":
    st.title("Get in Touch")
    st.markdown("""
        ## Meet Our Team
        If you have any questions or need support, feel free to reach out to any of our team members:
    """)

    st.header("👤 Our Team :\n1. Mulagapati Padmavathi\n2. Shaik Mohammed Suhail\n3. Jishnu Prasad\n4. PAVAN T S\n5. Ajay patil")
    st.subheader("👤 Member 1: Mulagapati Padmavathi")
    st.write("""
        - *Email*: [mulagapatipadmavathi@gmail.com](mailto : mulagapatipadmavathi@gmail.com)
        - *LinkedIn*: [Padmavathi's LinkedIn](https://www.linkedin.com/in/mulagapati-padmavathi-15017b273)
    """)
    st.subheader("👤 Member 2: Shaik Mohammed Suhail")
    st.write("""
        - *Email*: [shaiksuhail9876@gmail.com](mailto : shaiksuhail9876@gmail.com)
    """)

    st.markdown("""
        ---
        *Note*: Feel free to reach out to any of us via email, phone, or LinkedIn for inquiries, collaborations, or assistance.
    """)