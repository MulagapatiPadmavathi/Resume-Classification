import streamlit as st
import joblib
import os
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document

# Load the trained model, vectorizer, and category mapping
model = joblib.load('decision_tree_model.pkl')  # Your trained model
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Your TF-IDF vectorizer
category_map = {0: "Peoplesoft", 1: "React.js Developer", 2: "SQL Lighting Insight", 3: "WorkDay"}  # Category mapping

# Load the cleaned details Excel file
cleaned_details = pd.read_excel('cleaned_details.xlsx')

# Function to clean and preprocess text (can add more text-cleaning logic)
def clean_text(text):
    # Example: Remove extra spaces, newlines, etc.
    cleaned_text = text.strip()
    return cleaned_text

# Function to extract text from uploaded files
def extract_text_from_file(file):
    text = ""
    if file.name.endswith('.pdf'):
        # Extract text from PDF using PyPDF2
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    elif file.name.endswith('.docx'):
        # Extract text from DOCX using python-docx
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

# Function to categorize resumes and store them in category folders
def categorize_resumes(uploaded_files, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    results = []

    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(('.pdf', '.docx')):  # check for valid file types
            # Extract text based on file type
            text = extract_text_from_file(uploaded_file)
            
            if text:
                # Clean and preprocess the extracted text
                cleaned_resume = clean_text(text)
                
                # Vectorize the cleaned resume text
                input_features = vectorizer.transform([cleaned_resume])
                
                # Predict the category using the trained model
                prediction_id = model.predict(input_features)[0]
                category_id = category_map.get(prediction_id)

                # Extract the employee name from the file name (assuming it's in the format: category_name_employee_name.docx)
                file_name_parts = uploaded_file.name.split('_')
                
                # Check if there are at least two parts in the split name to avoid index errors
                if len(file_name_parts) > 1:
                    employee_name = file_name_parts[1].replace('.docx', '').replace('.pdf', '')
                else:
                    # Handle case where there isn't an underscore (you may want to set a default name or skip)
                    employee_name = uploaded_file.name.replace('.docx', '').replace('.pdf', '')

                # Check if the file name (or part of it) matches with 'File_Name' in the cleaned details dataframe
                matching_rows = cleaned_details[cleaned_details['File_Name'].str.contains(uploaded_file.name.split('.')[0], case=False, na=False)]

                if not matching_rows.empty:
                    # Exclude the existing category column and add the predicted category and file name
                    matching_rows = matching_rows.drop(columns=['category'], errors='ignore')  # Drop 'category' column
                    matching_rows['Predicted Category'] = category_id
                    matching_rows['File Name'] = uploaded_file.name

                    # Save the resume in the appropriate category folder
                    category_folder = os.path.join(output_directory, category_id)
                    if not os.path.exists(category_folder):
                        os.makedirs(category_folder)

                    # Save the file in the category folder
                    target_file = os.path.join(category_folder, uploaded_file.name)
                    with open(target_file, 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                    # Append the employee details to the results list
                    results.append(matching_rows)

    # Concatenate all results into a single DataFrame
    if results:
        final_results_df = pd.concat(results, ignore_index=True)
        return final_results_df
    else:
        return pd.DataFrame()

# Streamlit App
st.title("Resume Classification App")
st.write("Upload resumes to classify them into one of the following categories: Peoplesoft, React.js Developer, SQL Lighting Insight, and WorkDay.")

# Define the output directory
output_directory = "categorized_resumes"

# File uploader for multiple files
uploaded_files = st.file_uploader("Upload Resume(s) (.pdf, .docx):", type=['pdf', 'docx'], accept_multiple_files=True)

if uploaded_files:
    # Show a spinner while processing the files
    with st.spinner("Processing the uploaded resumes..."):
        # Categorize resumes and get the results as a DataFrame
        results_df = categorize_resumes(uploaded_files, output_directory)
        
        # Show the results in a table
        if not results_df.empty:
            st.success("Resumes successfully categorized!")
            st.write("Here are the categorized resumes with details:")
            st.dataframe(results_df)

            # Add a download button for the CSV file
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Categorized Resumes with Details as CSV",
                data=csv,
                file_name="categorized_resumes_with_details.csv",
                mime="text/csv"
            )
        else:
            st.error("No valid resumes found for categorization.")
