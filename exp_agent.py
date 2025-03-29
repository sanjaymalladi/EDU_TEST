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
class CombinedExperienceAgent:
    def __init__(self):
        self.model = model

        # Define the prompts for each step
        self.aspects_prompt = PromptTemplate(
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
        a. Checkpoint Example: Check if the candidate’s most recent role and responsibilities align with the key responsibilities outlined in the JD. Highlight deviations from the core role and their duration.

    4) Include at least one checkpoint to analyze career stability, such as job switches, career gaps (exceeding two months), and whether the candidate is currently employed and for how long.
        a. Checkpoint Example: Examine the chronology of job switches and career gaps. Are there gaps longer than two months between jobs? Has the candidate demonstrated frequent job changes? Are they currently employed in a relevant role? **Think step by step.**

    5) Address industry-specific requirements if applicable. If the JD specifies the role to be industry-specific, include a checkpoint for assessing relevant industry experience. Skip this if the role is not industry-specific.

    6) Include one checkpoint to evaluate the candidate's career progression and responsibilities in past professional roles, ensuring a logical sequence in work history and growth in responsibilities. This checkpoint is more important in roles which have a higher number of years of experience and can be ignored for beginners.

    7) Include checkpoints that probe into detailed subtopics or concepts relevant to success in the role, eliciting comprehensive insights about the candidate’s professional qualifications and their application in practical scenarios.

    8) Include a checkpoint to uncover the candidate’s achievements or individual contributions to past projects or assignments for a nuanced understanding of their professional role and impact.

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

        self.clarification_prompt = PromptTemplate(
            input_variables=["checkpoints", "resume"],
            template= """
You are an expert recruiter specializing in reading resumes against job descriptions. Your task is to read the checkpoints provided to you and extract objective and factual information (if available) strictly from the resume related to the candidate's professional experience to clarify these checkpoints.
You are required to take a pragmatic and holistic approach considering the context of the resume and understand the implied aspects of their work history as well as the application of their skills. Do not provide non-factual information or information that does not explicitly or implicitly exist in the resume. Simply provide information about their professional experience and the application of their skills that is present in the resume explicitly or implicitly in an accurate and unbiased manner. Do not include any information about education or certifications unless they are directly mentioned as part of a job responsibility or achievement within a work experience entry.

**Note: Instructions for Reference: **
        1) For any details related to understanding the checkpoints to analyze for, refer to the provided "Checkpoints".
        2) To find factual information, refer to the "Resume".
Use this source as necessary to make informed reasoning and strictly follow the Guidelines given below.

**Input**:
- Checkpoints: {checkpoints}
- Resume: {resume}
**Output**: The output should be a set of clarifications that should be factual for each checkpoint provided in the "Checkpoints", focusing only on professional experience and the application of skills.

Follow the guidelines to ensure the quality of the output.
**Guidelines** : Ensure that the factual clarifications should adhere to each of the following guidelines:
    1) Provide clarifications to directly address the must-have or critical *experience-related* requirements outlined in the checkpoints explicitly or implicitly, focusing on how the candidate demonstrated these in their work history.

    2) Career Stability and Gaps: - Study the chronology of job switches and calculate any gap of more than two months between roles.
    - If a gap beyond two months exists or the candidate is currently unemployed, report this as a gap.
    - *Think step by step* to assess the career stability of the candidate based on their work history.

    3) Industry Relevance: Evaluate the organizations the candidate has worked for and their industries. Do not rely solely on explicit mentions by the candidate, as industries are often implied by context within their professional experience.

    4) Progressive Exploration: - Explore the candidate’s responsibilities in past professional roles progressively, ensuring logical sequencing and growth in their career.
    - Uncover achievements or individual contributions to previous projects or assignments for a nuanced understanding of their professional role and impact.

    5) Domain Expertise: Clarify the candidate's domain expertise and experience with respect to the role specified in the JD, providing details of how this expertise was applied in their previous roles.

6) While providing clarifications related to the number of years of experience, do not go by the number of years too strictly. Consider the overall professional experience, the quality of experience, and other relevant factors to give some favorable consideration if the number of years slightly falls short of the requirement or if it exceeds the requirement by not a significant margin.

*Important note: * 1) Your task is to provide objective reasoning with factual pointers that support or refute the suitability of the candidate for each checkpoint based on their professional experience and the application of their skills. Do not provide subjective opinions or assumptions.
    2) If the resume does not explicitly or implicitly contain enough information about their professional experience to clarify a checkpoint, mention this in your response. However, consider the context of the responsibilities mentioned and understand the implied aspects as well as the potential application of their skills. Do not look for standalone education or certification details.

    3) Never hallucinate or provide information not grounded in the resume regarding their professional background and how they applied their skills.
### Output Format:
    Checkpoint 1: [Factual reasoning of checkpoint based on professional experience and application of skills]
    Checkpoint 2: [Factual reasoning of checkpoint based on professional experience and application of skills]
    """
        )

        self.evaluation_prompt = PromptTemplate(
            input_variables=["job_description", "candidates_profile", "checkpoints", "answer_script"],
            template="""
    You are an expert recruiter specializing in evaluating resumes against job descriptions.
Your task is to evaluate and assign a numeric rating for the candidate's resume based on the checkpoint and answer script provided to understand how well their professional experience strictly aligns with the JD. Provide a factual 70-100 word justification focusing only on their work history and the demonstrated application of their skills. Think step by step.

**Input**:
- Job Description: {job_description}
- Checkpoints: {checkpoints}
- Answer Script: {answer_script}

**Output**: The output should be a score and a summary of evidence and reasoning explaining the observation and the reasons for rating, focusing solely on professional experience and the application of skills.

    ### Steps:
    1) Step 1. **Understand the Job Description and Candidate's Profile (Focus on Experience and Applied Skills):**
        A. Understand "JD" and "checkpoints" to understand if any specific professional experience (e.g., specific roles, industries, technologies, and achievements) and the application of specific skills are explicitly mentioned as must-have or essential.
        B. If must-haves are not explicitly specified in the JD, understand the implicit needs of the role with the main focus on years of professional experience, essential experiences, industries, roles, technologies, relevant achievements, and any specific target/reputation of companies or any other relevant aspects related to work history and skill application. Think role specific and Think step by step.
        C. Understand additional requirements or inferred priorities like years of professional experience or any other role-specific relevant aspects that contribute to success in the role, considering how skills are typically applied.
        D. While evaluating years of professional experience, consider the context of the role, current position, carefully check for gaps between job switches, match with responsibilities handled by the candidate, etc., taking a holistic and pragmatic approach.

    2) Step 2. Compare the candidate's professional experience and demonstrated skills from the given "Answer Script" and "Candidate's Profile" with the analysis done in step 1 with the following scope of interest:

A.  Give proportionately higher weightage to must-have and critical professional experiences and demonstrated skills which have a higher impact on the probability of success in the role.
B.  While evaluating the years of professional experience checkpoint, do not go by the number of years too strictly. Consider the overall experience, the quality of experience, and other relevant factors to give some favorable consideration if the number of years slightly falls short of the requirement or if it exceeds the requirement by not a significant margin.
        C. Alignment with required industries for roles which are industry-specific, current or recent roles, technologies, or any experience with preferred kind/reputation of companies.
        D. **Give higher weightage to current and most recent assignments while evaluating professional experience and skills. Reduce ratings proportionately if the key responsibilities of recent roles are unrelated to the JD, even if older experiences align well.**
        E. Check whether the core domain experience of the candidate and their demonstrated expertise match with the core requirements of the JD.
        F. Read any gaps in the work history patterns or frequent job changes to other companies to gauge their potential impact on suitability. **Significant employment gaps (more than 2 months) should be counted against the candidate, especially if there are multiple unexplained gaps or gaps exceeding 6 months. Pay special attention to employment gaps identified in the Employment Gap Analysis section if provided.**
        G. Significance of career progression, achievements, and responsibilities in their professional history, highlighting the impact of their actions and the skills utilized.

    3) Step 3. **Assess and Score:** Consider steps #1 & #2 and assign a numeric rating based on how well the candidate meets the requirements for professional experience and demonstrated skills:
        i. 1–40: The candidate lacks critical areas of professional experience or must-have experience and has not demonstrated the application of key skills. There are frequent company changes or multiple unexplained gaps in relevant work history. Little to no relevant experience in required industries, roles, domain expertise, or technologies.
        ii. 41–60: The candidate meets some key professional experience requirements but may lack certain critical qualifications or have **significant gaps (more than 2 months)** in experience. Their experience is somewhat relevant but does not fully align with the essential aspects of the JD, or their recent roles do not align with the role explained in the JD. The application of key skills might be limited.
        iii. 61–100: The candidate meets or exceeds most job requirements for professional experience, including all must-have/critical qualifications. They have consistent and relevant experience with demonstrated achievements, leadership, and career progression that align closely with the role, including strong soft skills demonstrated through their work history and clear application of required skills.
        iv. Above 100: The candidate holds significantly higher positions compared to the job role in the JD based on their professional experience and demonstrated expertise. Their experience may surpass the requirements to the extent that they might not find the role sufficiently challenging or satisfying. This rating needs to be applied only in cases of exceptionally high disparity between requirement and actual professional experience/positions.

    4) Step 4. **Provide factual evidence:**
Provide justification pointers for the rating, explaining why the candidate's professional experience and demonstrated skills align or do not align with the job requirements. Include specific examples from the "checkpoints" or "Answer script" related to their work history to support your evaluation. Do not mention standalone education or certifications unless they were integral to specific work responsibilities or achievements.

**Note: If an Employment Gap Analysis section is provided, be sure to reference it in your evidence, noting how the identified employment gaps influenced your rating.**

    ### Output Format:
    - **Rating:** Assign a numeric rating between 1 and 120 based on your evaluation of their professional experience and demonstrated skills.
    - **Evidence:** For each evaluation aspect, provide a concise justification of 70-100 words, explaining why the candidate's professional experience and demonstrated skills align or do not align with the job requirements. Include specific examples from the resume related to their work history to support your evaluation. Do not include details about standalone education or certifications.
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
    agent = CombinedExperienceAgent()

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
    Led a team of data scientists for the past 3 years at PQR Corp, where he was responsible for developing and deploying machine learning models using Python and R. Successfully delivered several data-driven solutions that improved business outcomes. Prior to that, he worked as a Data Engineer at a multinational company for 7 years, where he built data pipelines and managed data warehouses.
    """

    result = agent.run(jd_text, resume_text)
    print(result)