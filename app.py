# app.py
import streamlit as st
from edu_agent import CombinedEducationAgent
from exp_agent import CombinedExperienceAgent
from skills_agent import CombinedSkillsAgent
from supervisor_agent import SupervisorAgent # Import the SupervisorAgent
# from mh_agent import CombinedMHAgent # Removed import
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
    st.write("Upload a job description and resume to analyze if the candidate meets the Education, Experience, and Skills criteria.")

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
                    st.warning("Experience analysis will be skipped as exp_agent.py was not found.")

                skills_result = None
                if CombinedSkillsAgent:
                    skills_agent = CombinedSkillsAgent()
                    skills_result = skills_agent.run(st.session_state['jd_text'], st.session_state['resume_text'])
                else:
                    st.warning("Skills analysis will be skipped as skills_agent.py was not found.")

                supervisor_agent = SupervisorAgent()
                weights, weight_reasoning = supervisor_agent.get_section_weights(st.session_state['jd_text'])

                edu_rating = 0
                if edu_result and "evaluation" in edu_result and 'Rating' in edu_result['evaluation']:
                    evaluation_str = edu_result['evaluation']
                    if evaluation_str.startswith("- "):
                        evaluation_str = evaluation_str[2:]
                    try:
                        edu_rating = int(evaluation_str.split("Rating:")[-1].split("**")[0].strip())
                    except:
                        edu_rating = 0 

                exp_rating = 0
                exp_rationale = ""
                if exp_result and "evaluation" in exp_result and 'Rating' in exp_result['evaluation']:
                    evaluation_str = exp_result['evaluation']
                    if evaluation_str.startswith("- "):
                        evaluation_str = evaluation_str[2:]
                    try:
                        exp_rating = int(evaluation_str.split("Rating:")[-1].split("**")[0].strip())
                        exp_rationale = exp_result.get('evaluation', '')
                    except:
                        exp_rating = 0
                if exp_result and 'evaluation' in exp_result and 'Evidence' in exp_result['evaluation']:
                    exp_rationale = exp_result['evaluation']

                skills_rating = 0
                skills_rationale = ""
                if skills_result and "evaluation" in skills_result and 'Rating' in skills_result['evaluation']:
                    evaluation_str = skills_result['evaluation']
                    if evaluation_str.startswith("- "):
                        evaluation_str = evaluation_str[2:]
                    try:
                        skills_rating = int(evaluation_str.split("Rating:")[-1].split("**")[0].strip())
                        skills_rationale = skills_result.get('evaluation', '')
                    except:
                        skills_rating = 0
                if skills_result and 'evaluation' in skills_result and 'Evidence' in skills_result['evaluation']:
                    skills_rationale = skills_result['evaluation']


                # Overall Score Calculation
                overall_rating = "NA"
                overall_category = "NA"

                if edu_rating != 0 or exp_rating != 0 or skills_rating != 0:
                    experience_weight = weights.get('experience', 0)
                    skills_weight = weights.get('skills', 0)
                    education_weight = weights.get('education_and_certification', 0)

                    total_weight = experience_weight + skills_weight + education_weight

                    if total_weight != 100.0 and total_weight > 0:
                        experience_weight = (experience_weight / total_weight) * 100
                        skills_weight = (skills_weight / total_weight) * 100
                        education_weight = (education_weight / total_weight) * 100
                        total_weight = 100.0

                    experience_score = (exp_rating * experience_weight) / 100
                    skills_score = (skills_rating * skills_weight) / 100
                    education_score = (edu_rating * education_weight) / 100

                    overall_rating = round(experience_score + skills_score + education_score)
                    overall_category = supervisor_agent.find_category(overall_rating)

                overall_summary = supervisor_agent.generate_summary(
                    experience_rationale=exp_rationale,
                    skills_rationale=skills_rationale,
                    education_rationale=edu_result.get('evaluation', '') if edu_result else ''
                )
                st.header("Overall Candidate Analysis")
                st.subheader(f"Overall Rating: {overall_rating}")
                st.write(f"**Section Weights:**")
                for section, weight in weights.items():
                    st.write(f"- {section.replace('_', ' ').title()}: {weight}%")
                st.subheader("Summary")
                st.write(overall_summary)

                
                st.header("Education Analysis Results")
                if "error" in edu_result:
                    st.error(f"Education Analysis Error: {edu_result['error']}")
                else:
                    st.subheader("🎯 Education Criteria Questions")
                    st.write(edu_result['aspects'])
                    st.subheader("🔍 Resume Education Details")
                    st.write(edu_result['clarifications'])
                    st.subheader("📊 Education Match Score")
                    st.write(edu_result['evaluation'])
                    if 'Rating' in edu_result['evaluation']:
                        try:
                            rating_match_edu = int(edu_result['evaluation'].split("Rating:")[-1].split("**")[0].strip())
                            st.progress(rating_match_edu / 120.0)
                        except:
                            pass

                if exp_result:
                    st.header("Experience Analysis Results")
                    if "error" in exp_result:
                        st.error(f"Experience Analysis Error: {exp_result['error']}")
                    else:
                        st.subheader("💼 Experience Criteria Aspects")
                        st.write(exp_result['aspects'])
                        st.subheader("📝 Resume Experience Details")
                        st.write(exp_result['clarifications'])
                        st.subheader("📈 Experience Match Score")
                        st.write(exp_result['evaluation'])
                        if 'Rating' in exp_result['evaluation']:
                            try:
                                rating_match_exp = int(exp_result['evaluation'].split("Rating:")[-1].split("**")[0].strip())
                                st.progress(rating_match_exp / 120.0)
                            except:
                                pass

                if skills_result:
                    st.header("Skills Analysis Results")
                    if "error" in skills_result:
                        st.error(f"Skills Analysis Error: {skills_result['error']}")
                    else:
                        st.subheader("💪 Skills Criteria Aspects")
                        st.write(skills_result['aspects'])
                        st.subheader("🛠️ Resume Skills Details")
                        st.write(skills_result['clarifications'])
                        st.subheader("🧠 Skills Match Score")
                        st.write(skills_result['evaluation'])
                        if 'Rating' in skills_result['evaluation']:
                            try:
                                rating_match_skills = int(skills_result['evaluation'].split("Rating:")[-1].split("**")[0].strip())
                                st.progress(rating_match_skills / 100.0)
                            except:
                                pass

        else:
            st.warning("Please upload both a job description and resume.")

if __name__ == "__main__":
    main()