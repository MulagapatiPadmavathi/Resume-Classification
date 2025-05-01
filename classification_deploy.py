import streamlit as st
import joblib
import os
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document

model = joblib.load('decision_tree_model.pkl')  
vectorizer = joblib.load('tfidf_vectorizer.pkl') 
category_map = {0: "Peoplesoft", 1: "React.js Developer", 2: "SQL Lighting Insight", 3: "WorkDay"} 

cleaned_details = pd.read_excel('cleaned_details.xlsx')

def clean_text(text):
    cleaned_text = text.strip()
    return cleaned_text

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

def categorize_resumes(uploaded_files, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    results = []

    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(('.pdf', '.docx')):  
            text = extract_text_from_file(uploaded_file)
            
            if text:
                cleaned_resume = clean_text(text)
                
                input_features = vectorizer.transform([cleaned_resume])
                
                prediction_id = model.predict(input_features)[0]
                category_id = category_map.get(prediction_id)

                file_name_parts = uploaded_file.name.split('_')
                
                if len(file_name_parts) > 1:
                    employee_name = file_name_parts[1].replace('.docx', '').replace('.pdf', '')
                else:
                    employee_name = uploaded_file.name.replace('.docx', '').replace('.pdf', '')

                matching_rows = cleaned_details[cleaned_details['File_Name'].str.contains(uploaded_file.name.split('.')[0], case=False, na=False)]

                if not matching_rows.empty:
                    matching_rows = matching_rows.drop(columns=['category'], errors='ignore') 
                    matching_rows['Predicted Category'] = category_id
                    matching_rows['File Name'] = uploaded_file.name

                    category_folder = os.path.join(output_directory, category_id)
                    if not os.path.exists(category_folder):
                        os.makedirs(category_folder)

                    target_file = os.path.join(category_folder, uploaded_file.name)
                    with open(target_file, 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                    results.append(matching_rows)

    if results:
        final_results_df = pd.concat(results, ignore_index=True)
        return final_results_df
    else:
        return pd.DataFrame()

st.title("Resume Classification App")
st.write("Upload resumes to classify them into one of the following categories: Peoplesoft, React.js Developer, SQL Lighting Insight, and WorkDay.")

output_directory = "categorized_resumes"

uploaded_files = st.file_uploader("Upload Resume(s) (.pdf, .docx):", type=['pdf', 'docx'], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing the uploaded resumes..."):
        results_df = categorize_resumes(uploaded_files, output_directory)
        
        if not results_df.empty:
            st.success("Resumes successfully categorized!")
            st.write("Here are the categorized resumes with details:")
            st.dataframe(results_df)

            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Categorized Resumes with Details as CSV",
                data=csv,
                file_name="categorized_resumes_with_details.csv",
                mime="text/csv"
            )
        else:
            st.error("No valid resumes found for categorization.")
