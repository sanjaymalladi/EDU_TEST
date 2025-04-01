import os
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Dict

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

class AspectsAgent:
    def __init__(self):
        self.model = model

        # Prompt for Education Aspects (copied from edu_agent.py)
        self.edu_aspects_prompt = PromptTemplate(
            input_variables=["job_description"],
            template="""
You are an expert recruiter specializing in analyzing resumes against job descriptions (JDs).
Your task is to formulate a set of 1–3 detailed 'Checkpoints' from the JD that focus solely on the candidate's
education and certifications. These Checkpoints will be used in the next step to evaluate the
candidate's suitability. Aim for specific and granular checkpoints.

Important:
- Think through your reasoning privately (step by step) but do not include that chain-of-thought
in your final output.
- Provide only the final set of 1–3 concise, relevant, and detailed Checkpoints.

**Input**: Job Description (JD)
**Job Description**:
{job_description}
**Output**: A set of 1 to 3 detailed checkpoints/criteria focused solely on the candidate's education
and certifications, serving as verification criteria for the next step.

### Steps:

1. Read and deeply understand the JD, specifically focusing on the Education and Certification sections.
   Identify all explicit must-have and preferred education and certification requirements.
   - If specific fields of study or specializations are mentioned, capture them.
   - If specific institutions or levels of prestige are implied or stated, note them.
   - If certain types or levels of certifications are required, make them explicit.
   - For preferred qualifications, think about how they would ideally look in a candidate's profile.
2. Frame 1–3 detailed checkpoints that are most relevant to the JD's education and certification requirements.
   - Each checkpoint should ideally focus on a specific aspect (e.g., specific degree, specific certification, specific field of study).
   - Ensure the checkpoints are actionable and can be clearly verified against a resume.
3. Keep these guidelines in mind:
   - Check for the exact names of required degrees, certifications, and any specified specializations.
   - If the JD mentions preferred qualifications, formulate checkpoints that reflect the ideal candidate's educational background and certifications.
   - Consider the level of detail provided in the JD. If it's highly specific, your checkpoints should also be highly specific.
   - If no formal requirements are stated but the role implies a certain educational level or type of certification, formulate a checkpoint based on that inference (e.g., "Bachelor's degree in a relevant field").

### Output Format Examples:

Checkpoint 1: Must possess a Master's degree in Computer Science with a specialization in Machine Learning.
Checkpoint 2: Hold a valid certification in AWS Certified Machine Learning – Specialty.
Checkpoint 3: Bachelor's degree in Electrical Engineering or a closely related field from a recognized university.

OR (if fewer checkpoints)
Checkpoint 1: Required: Bachelor's degree in Accounting. Preferred: CPA certification.
Checkpoint 2: Must have completed a Data Science Bootcamp with a focus on Python and statistical modeling.
    """
        )

        # Prompt for Experience Aspects (copied from exp_agent.py)
        self.exp_aspects_prompt = PromptTemplate(
            input_variables=["job_description"],
            template="""
You are an expert recruiter specializing in analyzing resumes against job descriptions (JDs). Your task is to formulate 3 to 10 (from JD) check points that will help generate factual insights from the CV in the next step to analyse the quality and suitability of the past professional experience strictly from the resume. Ensure that these checkpoints focus solely on the candidate's work history, including the application of skills and qualifications within their roles. Do not create checkpoints that solely ask about the presence of educational qualifications or certifications.

**Input**: Job Description (JD)
**Job Description**:
{job_description}
**Output**: Formulate 3 to 10 checkpoints/criteria focused solely on the candidate's past professional experience and how their skills and qualifications were applied.
These checkpoints/criteria will serve as criteria for the next step, where the candidate's resume will be checked for evidence and factual reasoning related to their work history.

###Steps:
    1) Understand the JD and determine the number of checkpoints (between 3-10) required depending on the complexity and experience specifications from the JD.
        a. For instance, roles requiring less experience may need fewer checkpoints (between 3-5).
        b. Cover the contents listed below in order to understand the suitability of the professional experience to the job description/job.
        c. For JDs which are not written in detail regarding experience, keep the number of checkpoints relatively low.

    2) With a holistic and pragmatic approach, formulate the checkpoints that cover the verifiable aspects of professional experience usually available from resumes. Note that behavioral aspects, cultural fit, thinking process, or future plans should not be part of this exercise. Focus strictly on past work experience and the application of skills.
**Guidelines**: Ensure that the output set of checkpoints/criteria should adhere to each of the following guidelines:

    1) Directly address must-have or critical *experience-related* requirements explicitly or implicitly outlined in the JD, focusing on how these were applied in past roles.

    2) Include at least one checkpoint for understanding the actual years of experience required for key technologies/core areas and overall years of professional experience in the field.

    3) Include at least one checkpoint to assess the relevance of recent professional experience, responsibilities, and alignment with the core domain specified in the JD.
        a. Checkpoint Example: Check if the candidate's most recent role and responsibilities align with the key responsibilities outlined in the JD. Highlight deviations from the core role and their duration.

    4) Include at least one checkpoint to analyze career stability, such as job switches, career gaps (exceeding two months), and whether the candidate is currently employed and for how long.
        a. Checkpoint Example: Examine the chronology of job switches and career gaps. Are there gaps longer than two months between jobs? Has the candidate demonstrated frequent job changes? Are they currently employed in a relevant role? **Think step by step.**

    5) Address industry-specific requirements if applicable. If the JD specifies the role to be industry-specific, include a checkpoint for assessing relevant industry experience. Skip this if the role is not industry-specific.

    6) Include one checkpoint to evaluate the candidate's career progression and responsibilities in past professional roles, ensuring a logical sequence in work history and growth in responsibilities. This checkpoint is more important in roles which have a higher number of years of experience and can be ignored for beginners.

    7) Include checkpoints that probe into detailed subtopics or concepts relevant to success in the role, eliciting comprehensive insights about the candidate's professional qualifications and their application in practical scenarios.

    8) Include a checkpoint to uncover the candidate's achievements or individual contributions to past projects or assignments for a nuanced understanding of their professional role and impact.

    9) Core Domain expertise/experience with respect to the role specified in the JD, focusing on the practical application of this expertise in previous roles.
        a. Example: For a Workday HCM professional, differentiate their practical experience from Workday Finance implementations.

    *Important note:* Focus on creating the most relevant checkpoints/criteria that will guide to uncover the Professional Experience requirements mentioned in the JD, emphasizing the application of skills and qualifications within their work history. Adjust the number of checkpoints/criteria dynamically between 3 to 10 depending on the specific requirements outlined in the JD while following the provided guidelines. Think step by step.

### Output Format:
    sample output 1:
        Checkpoint 1: [Description of checkpoint related to professional experience and application of skills]
        Checkpoint 2: [Description of checkpoint related to professional experience and application of skills]
        ....
        Checkpoint 8: [Description of checkpoint related to professional experience and application of skills]
    sample output 2:
        Checkpoint 1: [Description of checkpoint related to professional experience and application of skills]
        Checkpoint 2: [Description of checkpoint related to professional experience and application of skills]
        ....
        Checkpoint 6: [Description of checkpoint related to professional experience and application of skills]
    sample output 3:
        Checkpoint 1: [Description of checkpoint related to professional experience and application of skills]
        Checkpoint 2: [Description of checkpoint related to professional experience and application of skills]
        ....
        Checkpoint 9: [Description of checkpoint related to professional experience and application of skills]
    """
        )

        # Prompt for Must-Have Aspects (copied from mh_agent.py)
        self.mh_aspects_prompt = PromptTemplate(
            input_variables=["job_description"],
            template="""
    You are an expert recruiter specialized in analyzing resumes against job descriptions (JDs). Your task is to formulate checkpoints that focus on verifying criteria that are explicitly mentioned as must-have in the JD. These checkpoints will help generate insightful responses in the next step, ensuring the resume is analyzed against the critical, non-negotiable requirements, if any, outlined in the JD.

    **Input**: Job Description (JD)
**Job Description**:
{job_description}
    **Output**: Formulate 2 to 3 evaluation checkpoints/criteria focused solely on the must-have requirements. These checkpoints/criteria will serve as evaluation criteria for the next stage, where the candidate's resume will be checked for evidence and reasoning.

    ### Steps:
    1) Understand the JD and determine the number of checkpoints (between 2-3) required depending on the specifications from the JD and the context of the role. For freshers/career beginners, the number of checkpoints could be less in number.
    2) With a holistic and pragmatic approach, formulate the checkpoints that cover the verifiable aspects usually available from resumes. Note that the cultural aspects or thinking process or future plans of the candidate should not be part of this exercise.

    **Guidelines**:
    1. Identify parameters explicitly marked as must-have in the JD.
        a. Consider the context and include aspects labeled as "required," "mandatory," "essential," "prerequisite," or similar if appropriate to be considered as must-have.
        b. Focus only on very critical criteria that, if missing, should lead to disqualification of the candidate.
    2. Clearly differentiate between must-haves and good-to-haves/preferences.
        a. Exclude any parameters described as "preferred," "nice-to-have," or optional.
    3. If specific education, certification, or experience is not explicitly mentioned as a must-have, do not include it in this section.
    
    **Output Format:**
    Checkpoint 1: [Description of checkpoint]
    Checkpoint 2: [Description of checkpoint]"""
        )

        # Prompt for Skills Aspects (copied from skills_agent.py)
        self.skills_aspects_prompt = PromptTemplate(
            input_variables=["job_description"],
            template="""
You are an expert recruiter specializing in analyzing resumes against job descriptions (JDs). Your task is to formulate only skills verification checkpoints that will generate factual insights in the next step, helping to analyze the candidate's technical and domain skills in relation to the job requirements. Keep in consideration that resumes may list skills without detailed examples.

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

    def _generate_single_aspect(self, prompt_template: PromptTemplate, job_description: str) -> str:
        """Helper function to generate aspects using a specific prompt."""
        try:
            prompt_text = prompt_template.format(job_description=job_description)
            response = self.model.invoke([HumanMessage(content=prompt_text)])
            return response.content.strip()
        except Exception as e:
            print(f"Error generating aspect: {e}")
            return f"Error generating aspects: {str(e)}"

    def generate_all_aspects(self, job_description: str) -> Dict[str, str]:
        """Generates aspects for all categories based on the job description."""
        
        aspects = {}
        
        print("Generating Education aspects...")
        aspects['edu'] = self._generate_single_aspect(self.edu_aspects_prompt, job_description)
        
        print("Generating Experience aspects...")
        aspects['exp'] = self._generate_single_aspect(self.exp_aspects_prompt, job_description)
        
        print("Generating Must-Have aspects...")
        aspects['mh'] = self._generate_single_aspect(self.mh_aspects_prompt, job_description)
        
        print("Generating Skills aspects...")
        aspects['skills'] = self._generate_single_aspect(self.skills_aspects_prompt, job_description)
        
        print("Aspect generation complete.")
        return aspects

if __name__ == "__main__":
    agent = AspectsAgent()

    # Test with a sample JD 
    jd_text = """
    We are looking for a Data Scientist - Team Lead. The ideal candidate should have a Master's degree in Data Science or related field with a strong emphasis on statistical modeling and machine learning.
    Preferred certifications include: AWS Certified Machine Learning – Specialty, Google Cloud Professional Data Scientist.
    A Bachelor's degree in Computer Science or Statistics is also desirable. Must have leadership experience of at least 5 years managing data engineering and analytics teams. Experience working in a European, ideally German, company with cross-cultural collaboration skills. Proficiency in data modeling, ETL processes, and data warehousing. Advanced programming skills in SQL and Python. Expertise with advanced analytics tools and techniques, including machine learning and data visualization. MUST HAVE DEGREE IN DATA SCIENCE OR RELATED FIELD.
    """
    
    all_aspects = agent.generate_all_aspects(jd_text)
    
    print("\n--- Generated Aspects ---")
    for category, aspect_text in all_aspects.items():
        print(f"\n{category.upper()} Aspects:\n{aspect_text}")
        print("-" * 20) 