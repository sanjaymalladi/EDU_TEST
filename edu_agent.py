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
class CombinedEducationAgent:
    def __init__(self):
        self.model = model

        # Define the prompts for each step
        self.clarification_prompt = PromptTemplate(
            input_variables=["checkpoints", "resume"],
            template= """
You are an expert recruiter specializing in reading resumes against job descriptions. Your task is to review the given detailed 'Checkpoints' and extract objective, factual information
from the candidate's resume to provide a detailed clarification for each checkpoint.

Important:
- Think step by step internally, but do not share that chain-of-thought.
- Provide detailed factual clarifications relevant to each checkpoint, exactly as the resume states
or implies. Include specific details like the name of the degree, major/specialization, university, graduation date (if available), certification name, issuing organization, and validity period (if mentioned).
- Do not include personal opinions, assumptions, or biases.
- Be careful and consider industry jargon while reading resumes. For example CA or ICAI are usually
written instead of Chartered Accountant. ICWAI is now called ICMAI. Abbreviations are quite often
written for various certifications as well.
- Consider and understand the implied qualifications. For example : If a candidate is a post graduate in a particular specialisation, it implies that he is a graduate as well.

**Input**:
- Checkpoints: {checkpoints}
- Resume: {resume}

**Output**:
- A set of detailed clarifications, each tied to one of the provided Checkpoints.

### Guidelines:
1. For each checkpoint, meticulously scan the resume for matching or related information.
2. Extract specific details:
   - For degrees, note the full name of the degree, the major or specialization (if any), the name of the institution/university, and the graduation date or expected graduation date if mentioned.
   - For certifications, note the full name of the certification, the issuing organization, the date of issuance, and any expiration date or validity information.
   - If a checkpoint mentions a specific field of study, look for keywords or phrases in the education section of the resume that indicate this specialization.
3. If the resume provides partial fulfillment of a checkpoint (e.g., "pursuing a Master's degree"), note this factually with the relevant details.
4. If a checkpoint mentions a preferred qualification, and the resume shows it, provide the specific details. If the resume shows a similar but not exact qualification, note the details of what is present.
5. If no information is available for a particular checkpoint, explicitly state that the resume does not contain enough data regarding that specific requirement.
6. Never fabricate or guess details. Only provide clarifications you can directly trace back to actual resume content.

### Output Format:
Checkpoint 1: [Detailed factual clarification based on resume, e.g., Holds a Master of Science in Computer Science with a specialization in Machine Learning from Stanford University, graduated in June 2023.]
Checkpoint 2: [Detailed factual clarification based on resume, e.g., Possesses an AWS Certified Machine Learning – Specialty certification issued by Amazon Web Services, valid until December 2025.]
Checkpoint 3: [Detailed factual clarification based on resume, e.g., Earned a Bachelor of Science in Electrical Engineering from the University of California, Berkeley.]
    """
        )

        self.evaluation_prompt = PromptTemplate(
            input_variables=["job_description", "candidates_profile", "checkpoints", "answer_script"],
            template="""You are an expert recruiter specializing in evaluating resumes against job descriptions.
Your task is to pragmatically evaluate and assign a numeric rating for the candidate's education and certifications
based solely on the detailed 'Checkpoints' and 'Answer Script' to determine how well they align with the JD's
education/certification requirements.
**Important**: Think through the reasoning internally, but do not provide or invent detailed chain-
of-thought. Instead, provide a concise 70-100 word justification focusing on factual details from the
'Checkpoints' and 'Answer Script', explicitly mentioning the level of detail and alignment.

**Input**:
- Job Description: {job_description}
- Checkpoints: {checkpoints}
- Answer Script: {answer_script}

**Output**:
1) A numeric 'Rating' (1-120)
2) An 'Evidence' section with a 70-100 word concise justification, focusing only on education and
certifications mentioned in the 'Checkpoints' and 'Answer Script' and how they align with the JD. Highlight the specific details from the resume that match or don't match the requirements.Explicitly mention the degree names, specializations, universities, certification names, and issuing organizations where applicable. Avoid discussing experience unless explicitly part of the JD's education requirements. Do not include your private chain-of-thought.

### Steps:
1) Understand the Job Description with a strong focus on the detailed education/certification requirements.
2) Compare the candidate's qualifications from the Answer Script with the detailed Checkpoints, paying attention to the specific details extracted.
3) If the JD does not specify any certification requirements and the candidate has relevant certifications award a bonus score. If the candidate does not possess any certifications that are not specified in the JD, do not reduce score. 
4) If the JD gives general direction such as ‘any graduation’ consider graduation in specialisation to be a match. Do not reduce score in such instances.
5) Assign a score:
    i) 1–40: Must-have education/certifications explicitly mentioned with specific details are unmet or lacking.
    ii) 41–60: Meets some must-haves at a basic level but significant details or specific requirements are missing. Certifications may add minor points if relevant.
    iii) 61–100: Meets most or all must-haves with good alignment in terms of details. May meet some preferred qualifications. Relevant certifications, even if not explicitly mentioned, enhance the rating.
    iv) >100: Significantly exceeds requirements, possessing all must-haves with specific details and likely several preferred qualifications. Relevant certifications, even if not directly requested, are highly valued.
4) Provide a factual justification (70-100 words) referencing specific examples from the 'Checkpoints'
    and the detailed 'Answer Script'. Explicitly mention the degree names, specializations, universities, certification names, and issuing organizations where applicable. Avoid discussing experience unless explicitly part of the JD's education requirements. Do not include your private chain-of-thought.
"""
        )

    def run(self, jd_text: str, resume_text: str, aspects: dict) -> dict:
        try:
            # Step 1: Use provided aspects (checkpoints) from JD
            aspects_text = aspects.get('edu', '')
            if not aspects_text:
                return {"error": "No education aspects provided."}

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
    agent = CombinedEducationAgent()

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
        'edu': """
        Checkpoint 1: Must possess a Master's degree in Data Science or a closely related field.
        Checkpoint 2: Hold a valid certification in AWS Certified Machine Learning – Specialty.
        Checkpoint 3: Bachelor's degree in Computer Science or Statistics is desirable.
        """
    }

    result = agent.run(jd_text, resume_text, aspects)
    print(result)