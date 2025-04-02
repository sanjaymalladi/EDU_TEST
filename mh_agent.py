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
    top_p=1,
    top_k=1
)

# Define the combined agent class
class CombinedMHAgent:
    def __init__(self):
        self.model = model

        # Define the prompts for each step
        self.clarification_prompt = PromptTemplate(
            input_variables=["checkpoints", "resume"],
            template= """
    You are an expert recruiter specializing in reading resumes against job descriptions. Your task is to read the checkpoints provided and extract objective and factual information (if available) from the resume to clarify these checkpoints.

    **Guidelines:**
    1. Analyze both explicit and implicit meanings from the resume.
    2. For must-have certifications, consider only those explicitly mentioned. Do not assume.
    3. For industry relevance, assess the organizations listed and determine their industries.
    4. For education and certifications, verify if they match stated requirements.
    5. Provide objective reasoning with factual pointers from the resume.
    6. Do not hallucinate or include information not grounded in the resume.
    7. If the resume lacks enough information, mention this explicitly.

    **Checkpoints:**
    {checkpoints}

    **Resume:**
    {resume}

    **Output Format:**
    Checkpoint 1: [Factual reasoning from resume based on checkpoint]
    Checkpoint 2: [Factual reasoning from resume based on checkpoint]
    """
        )

        self.evaluation_prompt = PromptTemplate(
            input_variables=["job_description", "candidates_profile", "checkpoints", "answer_script"],
            template="""
            You are an expert recruiter specializing in evaluating resumes against job descriptions.
            Your task is to evaluate and assign a categorisation for the candidate's resume based on the "checkpoints" and "answer_script" provided to understand how well it aligns with the JD. Also provide a brief reasoning. You are required to take a pragmatic and holistic approach considering the context of the resume and understand the implied aspects as well.  For example, if the JD specifies requirement of Graduation and the resume mentions post-graduation, it is implied that the person holds graduation and should be considered as such.

            Think step by step and follow the instructions provided below:

            **Input**:
            - Job Description: {job_description}
            - Checkpoints: {checkpoints}
            - Answer Script: {answer_script}
            **Output**: The output should be category of the resume and a summary of evidence and reasoning explaining the observation and the reasons for the categorisation.

            ### Steps:

            Step1 : *Understand the Job Description along with checkpoints and answer script provided*

            Step 2 : Analyse if the answers from the checkpoints and answer script satisfy the must-haves while considering the following aspects :

            a)	If there are any checkpoints related to years of experience, do not strictly focus on just the number of years in the literary sense. The number of years should be given due importance, but more importantly, considered holistically  taking into account the context of the role, responsibilities handled by the candidate, etc. Minor deviations in number of years of experience should not lead to disqualification.

            b)	While considering the other checkpoints and answers, take a holistic and a pragmatic approach in understanding and give due importance to both the explicit and implied experiences and skills based on the role/responsibilities, context of the roles and past experiences of the individual. 

            C) If there are any checkpoints related to Education, consider the context while evaluating. For example, if the JD specifies requirement of Graduation and the resume mentions post-graduation, it is implied that the person qualifies for that checkpoint.

            Step 3 : Based on the understanding from Step 1 and Step 2, categorise the resume following the criteria specified below :

            a)	Category I : JD does not explicitly or implicitly specify any must-haves or essentials for the resume to be considered for the role.
            b)	Category II: Satisfies all must-haves explicitly or implicitly. If there is slight uncertainty, give benefit of doubt to the candidate and place him/her in Category II.
            c)	Category III: Lacks one or more must-haves mentioned in the JD.

            Step 4. *Provide factual evidence:*
            Provide reasoning for the rating along with observations and explaining why the candidate has been assigned to a particular category. Include specific examples from the "checkpoints" and "Answer script" to support your categorisation.

            ### Output Format:
            ### Output Format:
            **category**: category I/II/III based on the evidence.
            **evidence**: Provide a concise justification for categorization in only 40-50 words explaining why the candidate's relevant skills and expertise does or does not align with the skills required for the role outlined in the job description.
    """
        )

    def run(self, jd_text: str, resume_text: str, aspects: dict) -> dict:
        try:
            # Step 1: Use provided aspects (checkpoints) from JD
            aspects_text = aspects.get('mh', '')
            if not aspects_text:
                return {"error": "No must-have aspects provided."}

            # Step 2: Generate clarifications (based on resume)
            clarifications = self.generate_clarifications(aspects_text, resume_text)
            if not clarifications:
                return {"error": "Failed to generate clarifications."}

            # Step 3: Perform evaluation (based on aspects and clarifications)
            evaluation = self.evaluate(jd_text, resume_text, aspects_text, clarifications)
            if not evaluation:
                return {"error": "Failed to perform evaluation."}

            return {
                'aspects': aspects_text,
                'clarifications': clarifications,
                'evaluation': evaluation
            }
        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}

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
    agent = CombinedMHAgent()

    # Test with a sample JD and resume
    jd_text = """
    We are looking for a Data Scientist - Team Lead. The ideal candidate should have a Master's degree in Data Science or related field with a strong emphasis on statistical modeling and machine learning.
    Preferred certifications include: AWS Certified Machine Learning – Specialty, Google Cloud Professional Data Scientist.
    A Bachelor's degree in Computer Science or Statistics is also desirable. Must have leadership experience of at least 5 years managing data engineering and analytics teams. Experience working in a European, ideally German, company with cross-cultural collaboration skills. Proficiency in data modeling, ETL processes, and data warehousing. Advanced programming skills in SQL and Python. Expertise with advanced analytics tools and techniques, including machine learning and data visualization. MUST HAVE DEGREE IN DATA SCIENCE OR RELATED FIELD.
    """
    resume_text = """
    John Doe holds a Master of Science in Data Science from XYZ University, graduated in May 2022. His master's thesis focused on statistical modeling.
    He is certified in AWS Certified Machine Learning – Specialty (valid until December 2025).
    He also has a Bachelor of Science degree in Statistics from ABC University.
    Led a team of data scientists for the past 3 years at PQR Corp, where he was responsible for developing and deploying machine learning models using Python and R. Successfully delivered several data-driven solutions that improved business outcomes. Prior to that, he worked as a Data Engineer at a multinational company for 7 years, where he built data pipelines and managed data warehouses.
    """

    # Create a sample aspects dictionary (in a real scenario, this would come from the AspectsAgent)
    aspects = {
        'mh': """
        Checkpoint 1: Must have a degree in Data Science or related field.
        Checkpoint 2: Must have leadership experience of at least 5 years managing data engineering and analytics teams.
        """
    }

    result = agent.run(jd_text, resume_text, aspects)
    print(result)
