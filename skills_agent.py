import os
from langchain_google_genai import ChatGoogleGenerativeAI  # Import Gemini model
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variables for model API key and model type
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Initialize the chat model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Specify the Gemini model name
    google_api_key=api_key,
    temperature=0.0,
    max_output_tokens=4000,  # Use max_output_tokens instead of max_tokens
    top_p=1
)

# Define the combined agent class
class CombinedSkillsAgent:
    def __init__(self):
        self.model = model

        # Define the prompts for each step
        self.aspects_prompt = PromptTemplate(
    input_variables=["job_description"],
    template="""
You are an expert recruiter specializing in analyzing resumes against job descriptions (JDs). Your task is to formulate only skills verification checkpoints that will generate factual insights in the next step, helping to analyze the candidate’s technical and domain skills in relation to the job requirements. Keep in consideration that resumes may list skills without detailed examples.

**Input**: Job Description (JD)
**Job Description**:
{job_description}
**Output**: Formulate 3-5 evaluation checkpoints/criteria focused solely on the candidate's technical and domain skills as they would typically be listed in a resume's "Skills" section. These checkpoints/criteria will serve as criteria for the next step, where the candidate's resume will be checked for evidence of these skills.

###Steps:

    1) Understand the JD and identify the key technical and domain skills explicitly mentioned or strongly implied for success in the role. Determine the number of checkpoints (between 3-5) based on the number of critical skills.
    2) Focus on skills that are typically listed in a "Skills" section of a resume, such as programming languages, tools, technologies, methodologies, and specific areas of expertise. Avoid creating checkpoints related to qualifications, certifications, experience levels (unless the JD specifies a skill as "X years of experience with Y"), roles, or responsibilities.

**Guidelines**: Ensure that the output set of checkpoints/criteria should adhere to each of the following guidelines:

    1) Directly address must-have or critical technical/domain skills explicitly mentioned in the JD.
        a. Example: If the JD requires "Python," a checkpoint could be "Verify if the candidate lists Python as a skill."
    2) Account for implicit skill requirements necessary for success in the role that would typically be listed as skills.
        a. Example: If the role involves machine learning, a checkpoint could be "Check if the candidate lists any machine learning algorithms or frameworks (e.g., scikit-learn, TensorFlow)."
    3) Focus on the skills themselves, not on how or where they were applied (this will be assessed implicitly if the skills are listed).
    4) Include one checkpoint for each major category of required skills mentioned in the JD.
        a. Example: If the JD mentions "Programming Languages," "Databases," and "Data Visualization Tools," aim for at least one checkpoint for each category.
    5) Differentiate between specific skills if the JD is specific.
        a. Example: If the JD requires "SQL" and "NoSQL," create separate checkpoints for each.

*Important note:* Focus on creating concise, actionable checkpoints that directly ask about the presence of specific technical and domain skills that a candidate would typically list in their resume's "Skills" section. Adjust the number of checkpoints/criteria dynamically between 3-5 depending on the specific skill requirements outlined in the JD. Think step by step about what skills a candidate would list.

### Output Format:
    sample output 1:
        Checkpoint 1: [Verify if the candidate lists proficiency in Python.]
        Checkpoint 2: [Check if the candidate mentions experience with SQL databases.]
        Checkpoint 3: [Confirm if the candidate lists any data visualization tools like Tableau or PowerBI.]
    sample output 2:
        Checkpoint 1: [Verify if the candidate lists expertise in machine learning.]
        Checkpoint 2: [Check if the candidate mentions specific ETL tools.]
        Checkpoint 3: [Confirm if the candidate lists cloud platforms like AWS or Azure.]
    """
    )

        self.clarification_prompt = PromptTemplate(
            input_variables=["checkpoints", "resume"],
            template="""
    You are an expert recruiter specializing in reading resumes against job descriptions. Your task is to read the checkpoints provided to you and extract objective and factual information (if available) from the resume to clarify these checkpoints.
You are required to take a pragmatic and holistic approach considering the context of the resume and understand the implied aspects as well. Do not provide non-factual information or information that does not explicitly or implicitly exist in the resume. Do not include clarifications with either positive or negative bias. Simply assess if the resume explicitly or implicitly contains information relevant to the skills mentioned in the checkpoints in an accurate and unbiased manner. Do not include any information about experience, roles, years, education, or certifications unless they are directly part of the listed skills.

**Input**:
- Checkpoints: {checkpoints}
- Resume: {resume}
**Output**: The output should be a set of clarifications that should be factual for each checkpoint provided in the "Checkpoints", focusing solely on skills.

**Guidelines** : Ensure that the factual clarifications should adhere to each of the following guidelines:

    1) Provide clarifications to directly address the must-have or critical skills outlined in the checkpoints explicitly or implicitly.
    2) Wherever possible, the reasoning/clarification should help to uncover the presence and potential application of the skills rather than just theoretical understanding.
    3) Make sure to help understand the core domain expertise of the skills with respect to the role specified.

*Important note:*
        1) Your task is to provide objective reasoning with factual pointers that support or refute the presence of the required skills for each checkpoint. Do not provide subjective opinions or assumptions.
        2) If the resume does not contain enough information to clarify a skill-related checkpoint, mention this in your response.
        3) Never hallucinate or provide information not grounded in the resume regarding the candidate's skills. Do not infer skills based on experience or roles unless the skill is explicitly mentioned in those sections.
### Output Format:
    Checkpoint 1: [Factual reasoning about the skill]
    Checkpoint 2: [Factual reasoning about the skill]
    """
        )

        self.evaluation_prompt = PromptTemplate(
            input_variables=["job_description", "candidates_profile", "checkpoints", "answer_script"],
            template="""
You are an expert recruiter specializing in evaluating resumes against job descriptions.
Your task is to evaluate and assign a numeric rating for the candidate's skills based solely on the "Checkpoints" and "Answer Script" to determine how well they align with the JD’s skill requirements. Provide a factual 70-100 word justification focusing only on skills, avoiding any mention of experience, years, or roles. Think step by step.

**Input**:
- Job Description: {job_description}
- Checkpoints: {checkpoints}
- Answer Script: {answer_script}

**Output**: A score and a 70-100 word justification explaining the candidate’s skill alignment or gaps with the JD, using specific skill examples from the "Checkpoints" and "Answer Script." Do not mention experience, years, or roles.

### Steps:

1) **Understand the Job Description with Focus on Skills Required:**
    i) Identify must-have skills and competencies critical to the role.
    ii) Recognize additional skills that enhance suitability but are not mandatory.

2) **Analyze and Score the Candidate’s Skills from "Checkpoints" and "Answer Script":**
    **Factors to Consider While Scoring:**
    i) **Depth of Expertise**: Assess the depth of proficiency in must-have and additional skills relative to the JD. Prioritize core domain skills over additional ones.
    ii) **Presence and Mention**: Evaluate if the candidate explicitly mentions the required skills in their resume.
    iii) **Specificity of Skills**: Higher ratings should be given if the candidate mentions specific tools, technologies, or methodologies that align with the JD's skill requirements.
    iv) **Relevance of Skills**: Consider the relevance of the mentioned skills to the specific role outlined in the JD.

3) **Assign a Rating:**
    i) 1–40: Lacks must-have skills.
    ii) 41–60: Basic alignment with some of the required skills present.
    iii) 61–100: Strong alignment with most or all required skills explicitly mentioned.

4) **Provide Factual Justification:**
    a) Write a 70-100 word justification explaining why the candidate’s skills align or do not align with the JD. Use specific skill examples from the "Checkpoints" and "Answer Script." Do not mention experience, years, projects, or roles. Focus solely on the presence and relevance of the skills.

### Output Format:
- **Rating:** Numeric score between 1 and 100.
- **Evidence:** A 70-100 word justification focusing solely on skills, their presence, specificity, and relevance, supported by examples from the input.

### Constraints:
- Avoid referencing experience (e.g., years worked, projects completed, leadership roles), non-skill factors (e.g., education, certifications), or the candidate's professional history unless explicitly part of the JD’s skill requirements (e.g., "5+ years experience with Python" would imply a skill). Focus solely on the skills listed or implied.
"""
        )

    def run(self, jd_text: str, resume_text: str) -> dict:
        try:
            # Step 1: Generate aspects (checkpoints) from JD
            aspects = self.generate_aspects(jd_text)
            if not aspects:
                return {"error": "Failed to generate aspects."}

            # Step 2: Generate clarifications (based on resume)
            clarifications = self.generate_clarifications(aspects, resume_text)
            if not clarifications:
                return {"error": "Failed to generate clarifications."}

            # Step 3: Perform evaluation (based on aspects and clarifications)
            evaluation = self.evaluate(jd_text, resume_text, aspects, clarifications)
            if not evaluation:
                return {"error": "Failed to perform evaluation."}

            return {
                'aspects': aspects,
                'clarifications': clarifications,
                'evaluation': evaluation
            }
        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}

    def generate_aspects(self, job_description: str) -> str:
        prompt_text = self.aspects_prompt.format(job_description=job_description)
        response = self.model.invoke([HumanMessage(content=prompt_text)])
        return response.content.strip()

    def generate_clarifications(self, checkpoints: str, resume: str) -> str:
        prompt_text = self.clarification_prompt.format(checkpoints=checkpoints, resume=resume)
        response = self.model.invoke([HumanMessage(content=prompt_text)])
        return response.content.strip()

    def evaluate(self, job_description: str, profile: str, checkpoints: str, answer_script: str) -> str:
        prompt_text = self.evaluation_prompt.format(
            job_description=job_description,
            candidates_profile=profile,
            checkpoints=checkpoints,
            answer_script=answer_script
        )
        response = self.model.invoke([HumanMessage(content=prompt_text)])
        return response.content.strip()


if __name__ == "__main__":
    agent = CombinedSkillsAgent()

    # Test with a sample JD and resume
    jd_text = """
    We are looking for a Data Scientist - Team Lead. The ideal candidate should have a Master's degree in Data Science or related field with a strong emphasis on statistical modeling and machine learning.
    Preferred certifications include: AWS Certified Machine Learning – Specialty, Google Cloud Professional Data Scientist.
    A Bachelor's degree in Computer Science or Statistics is also desirable. Must have leadership experience of at least 5 years managing data engineering and analytics teams. Experience working in a European, ideally German, company with cross-cultural collaboration skills. Proficiency in data modeling, ETL processes, and data warehousing. Advanced programming skills in SQL and Python. Expertise with advanced analytics tools and techniques, including machine learning and data visualization.
    """
    resume_text = """
    John Doe holds a Master of Science in Data Science from XYZ University, graduated in May 2022. His master's thesis focused on statistical modeling.
    He is certified in AWS Certified Machine Learning – Specialty (valid until December 2025).
    He also has a Bachelor of Science degree in Statistics from ABC University.
    Led a team of data scientists for the past 3 years at PQR Corp, where he was responsible for developing and deploying machine learning models using Python and R. Successfully delivered several data-driven solutions that improved business outcomes. Prior to that, he worked as a Data Engineer at a multinational company for 7 years, where he built data pipelines and managed data warehouses. Skills: Python (ML - scikit-learn, Pandas, NumPy, SciPy), SQL, Data Modeling, ETL, Data Warehousing, Machine Learning, Data Visualization, AWS.
    """

    result = agent.run(jd_text, resume_text)
    print(result)