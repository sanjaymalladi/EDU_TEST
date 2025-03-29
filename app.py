# app.py
import streamlit as st
from edu_agent import CombinedEducationAgent
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
    st.title("📄 JD & Resume Education Analyzer")
    st.write("Upload a job description and resume to analyze if the candidate meets the Education criteria.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        jd_file = st.file_uploader("Upload Job Description (PDF, DOCX, or TXT)", type=['pdf', 'docx', 'txt'])
        if jd_file:
            jd_text = read_file_content(jd_file)
            st.text_area("Job Description Content", jd_text, height=300)
    
    with col2:
        resume_file = st.file_uploader("Upload Resume (PDF, DOCX, or TXT)", type=['pdf', 'docx', 'txt'])
        if resume_file:
            resume_text = read_file_content(resume_file)
            st.text_area("Resume Content", resume_text, height=300)

    if st.button("Analyze Education"):
        if jd_file and resume_file:
            with st.spinner("Analyzing documents..."):
                agent = CombinedEducationAgent()
                result = agent.run(jd_text, resume_text)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    display_analysis_result(result)
        else:
            st.warning("Please upload both a job description and resume.")

def display_analysis_result(result: dict):
    st.title("Education Analysis Results")
    
    with st.expander("🎯 Aspects (Education Criteria)", expanded=True):
        st.write(result['aspects'])
    
    with st.expander("🔍 Clarifications (Resume)", expanded=True):
        st.write(result['clarifications'])
    
    st.header("📊 Final Evaluation")
    st.write(result['evaluation'])

if __name__ == "__main__":
    main()
