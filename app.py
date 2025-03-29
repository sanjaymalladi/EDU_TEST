# app.py
import streamlit as st
from edu_agent import CombinedEducationAgent
from exp_agent import CombinedExperienceAgent
import io

import io

# Helper function to extract text from a PDF file
def extract_text_from_pdf(file):
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

# Helper function to extract text from a DOCX file
def extract_text_from_docx(file):
    try:
        from docx import Document
        doc = Document(io.BytesIO(file.read()))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return None

# Function to read content from uploaded files
def read_file_content(file):
    if file is None:
        return None

    file_extension = file.name.split('.')[-1].lower()

    try:
        if file_extension == 'pdf':
            return extract_text_from_pdf(file)
        elif file_extension == 'docx':
            return extract_text_from_docx(file)
        elif file_extension == 'txt':
            return file.read().decode('utf-8').strip()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    except Exception as e:
        st.error(f"Error reading file {file.name}: {str(e)}")
        return None

# Streamlit application
def main():
    st.title("📄 JD & Resume Analyzer")
    st.write("Upload a job description and resume to analyze if the candidate meets the Education and Experience criteria.")

    col1, col2 = st.columns(2)

    with col1:
        jd_file = st.file_uploader("Upload Job Description (PDF, DOCX, or TXT)", type=['pdf', 'docx', 'txt'])
        if jd_file:
            st.session_state['jd_text'] = read_file_content(jd_file)
            st.text_area("Job Description Content", st.session_state.get('jd_text', ''), height=300)

    with col2:
        resume_file = st.file_uploader("Upload Resume (PDF, DOCX, or TXT)", type=['pdf', 'docx', 'txt'])
        if resume_file:
            st.session_state['resume_text'] = read_file_content(resume_file)
            st.text_area("Resume Content", st.session_state.get('resume_text', ''), height=300)

    if st.button("Analyze"):
        if jd_file and resume_file:
            with st.spinner("Analyzing documents..."):
                edu_agent = CombinedEducationAgent()
                edu_result = edu_agent.run(st.session_state['jd_text'], st.session_state['resume_text'])

                exp_result = None
                if CombinedExperienceAgent:
                    exp_agent = CombinedExperienceAgent()
                    exp_result = exp_agent.run(st.session_state['jd_text'], st.session_state['resume_text'])
                else:
                    st.warning("Experience analysis will be skipped as `exp_agent.py` was not found.")

                if "error" in edu_result:
                    st.error(f"Education Analysis Error: {edu_result['error']}")
                else:
                    display_education_analysis(edu_result)

                if exp_result:
                    if "error" in exp_result:
                        st.error(f"Experience Analysis Error: {exp_result['error']}")
                    else:
                        display_experience_analysis(exp_result)
        else:
            st.warning("Please upload both a job description and resume.")

def display_education_analysis(result: dict):
    st.header("Education Analysis Results")

    with st.expander("🎯 Education Criteria Questions", expanded=True):
        st.write(result['aspects'])

    with st.expander("🔍 Resume Education Details", expanded=True):
        st.write(result['clarifications'])

    st.subheader("📊 Education Match Score")
    st.write(result['evaluation'])
    if 'Rating' in result['evaluation']:
        try:
            rating_match = int(result['evaluation'].split("Rating:")[-1].split("**")[0].strip())
            st.progress(rating_match / 120.0)
        except:
            pass # Handle cases where rating might not be properly formatted

def display_experience_analysis(result: dict):
    st.header("Experience Analysis Results")

    with st.expander("💼 Experience Criteria Aspects", expanded=True):
        st.write(result['aspects'])

    with st.expander("📝 Resume Experience Details", expanded=True):
        st.write(result['clarifications'])

    st.subheader("📈 Experience Match Score")
    st.write(result['evaluation'])
    if 'Rating' in result['evaluation']:
        try:
            rating_match = int(result['evaluation'].split("Rating:")[-1].split("**")[0].strip())
            st.progress(rating_match / 120.0)
        except:
            pass # Handle cases where rating might not be properly formatted

if __name__ == "__main__":
    main()