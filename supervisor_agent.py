# supervisor_agent.py
import os
from dotenv import load_dotenv
load_dotenv()
import json
from typing import Dict, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, OutputFixingParser # Updated import
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains import LLMChain

# Define Pydantic models for structured output
class SectionWeightsStructure(BaseModel):
    weight: int = Field(
        ...,
        description="The weight of the section in percentage."
    )
    reasoning: str = Field(
        ...,
        description="The detailed factual reasoning of about 50-60 words for the weightage assigned of this section."
    )

class SectionWeightsResult(BaseModel):
    """
    Model to represent the evaluation result for a candidate.
    """
    experience : SectionWeightsStructure = Field(
        ...,
        description="The weight and reasoning for Experience."
    )
    skills : SectionWeightsStructure = Field(
        ...,
        description="The weight and reasoning for Skills."
    )
    education_certification : SectionWeightsStructure = Field(
        ...,
        description="The weight and reasoning for Education/Certification."
    )

class SummaryModel(BaseModel):
    """
    Model to represent the summary of a candidate's evaluation.
    """
    summary: str = Field(description="A concise 2-3 line summary of the candidate's suitability for the role")

class SupervisorAgent:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.0,
            max_output_tokens=4000,  # Use max_output_tokens instead of max_tokens
            top_p=1,
            top_k=1
        )

    def get_section_weights(self, job_description: str) -> Tuple[Dict, Dict]:
        experience_schema = ResponseSchema(name="experience", description="The weight and reasoning for Experience.", type="object")
        skills_schema = ResponseSchema(name="skills", description="The weight and reasoning for Skills.", type="object")
        education_certification_schema = ResponseSchema(name="education_certification", description="The weight and reasoning for Education/Certification.", type="object")

        response_schemas = [experience_schema, skills_schema, education_certification_schema]

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt = PromptTemplate(
            template="""You are an expert recruiter specializing in resume evaluation and scoring based on job descriptions (JDs). Your task is to determine the relative weights for three key sections—Experience, Skills, and Education/Certifications—based on the JD. These weights will be used to calibrate the overall candidate rating.

### Input:The input for this task will be a job description (JD).
**Job Description**:
{job_description}

### Output: Your output should be a set of three numerical weights (expressed as percentages summing to 100%) representing the importance of Experience, Skills, and Education/Certifications for evaluating candidates.

### Guidelines for Weight Distribution:
1) Experience Weight (X%)
Assign higher weight (e.g., 50-70%) if the JD emphasizes past work experience, industry expertise, or specific job responsibilities.
Assign lower weight (e.g., 30-50%) if the JD is open to freshers or prioritizes skills over prior experience.

2) Skills Weight (Y%)
Assign higher weight (e.g., 30-50%) if the JD requires strong technical or specialized skills (e.g., programming languages, software tools, methodologies).
Assign lower weight (e.g., 20-40%) if experience in a domain is more critical than specific skill sets.

3) Education/Certification Weight (Z%)
Assign higher weight (e.g., 20-40%) if the JD explicitly requires degrees, certifications, or licenses (e.g., CPA, PMP, AWS Certified).
Assign lower weight (e.g., 10-20%) if practical experience and skills are emphasized over formal education.

### Special Considerations:
1) Ensure the three weights sum up to 100%.
2) If the JD is senior-level or leadership-focused, prioritize Experience more heavily.
3) If the JD is for highly technical roles, increase Skills weight.
4) If the JD mandates specific degrees or certifications, adjust the Education/Certification weight accordingly.
5) Consider industry norms (e.g., IT roles may prioritize skills, while medical/legal roles may require strong educational backgrounds).

{format_instructions}
""",
            input_variables=["job_description"],
            partial_variables={"format_instructions": format_instructions}
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            response = chain.run(job_description=job_description)
            print(f"DEBUG - Raw LLM response for weights: {response}") # Added debug print
            parsed_output = output_parser.parse(response)
            print(f"DEBUG - Parsed output: {parsed_output}") # Added debug print

            weights = {
                "experience": parsed_output["experience"]["weight"],
                "skills": parsed_output["skills"]["weight"],
                "education_and_certification": parsed_output["education_certification"]["weight"],
            }
            reasoning = {
                "experience": parsed_output["experience"]["reasoning"], # Changed from 'description' to 'reasoning'
                "skills": parsed_output["skills"]["reasoning"],     # Changed from 'description' to 'reasoning'
                "education_and_certification": parsed_output["education_certification"]["reasoning"], # Changed from 'description' to 'reasoning'
            }
            return weights, reasoning
        except Exception as e:
            print(f"Error getting section weights: {e}")
            return {"experience": 33, "skills": 34, "education_and_certification": 33}, {"experience": "Default weights due to error.", "skills": "Default weights due to error.", "education_and_certification": "Default weights due to error."}

    def find_category(self, rating: int) -> str:
        if rating == "NA":
            return "NA"
        elif rating <= 40:
            return "Poor Match"
        elif 41 <= rating <= 60:
            return "Moderate Match"
        elif 61 <= rating <= 100:
            return "Good Match"
        else:
            return "Overqualified"

    def generate_summary(self, experience_rationale: str, skills_rationale: str, education_rationale: str) -> str:
        prompt = PromptTemplate(
            template="""You are an expert recruiter tasked with creating a concise executive summary of a candidate's evaluation.

Here's the evaluation data for different sections:

**Experience:** {experience_rationale}
**Skills:** {skills_rationale}
**Education and Certification:** {education_rationale}

Your goal is to synthesize this evidence into a brief, informative summary. Focus on providing a balanced assessment that highlights key strengths and areas of concern, while giving a clear indication of the candidate's overall fit for the position.

Provide a 2-3 line summary highlighting the candidate's strengths and weaknesses based on the evidence.
""",
            input_variables=["experience_rationale", "skills_rationale", "education_rationale"]
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        summary = chain.run(experience_rationale=experience_rationale, skills_rationale=skills_rationale, education_rationale=education_rationale)
        return summary

    def calculate_overall_rating(self, edu_rating: int, exp_rating: int, skills_rating: int, weights: Dict, mh_category: str = None) -> Tuple[int, str]:
        """
        Calculate the overall rating and category based on individual ratings and weights.
        
        Args:
            edu_rating (int): Education rating (0-100)
            exp_rating (int): Experience rating (0-100)
            skills_rating (int): Skills rating (0-100)
            weights (Dict): Dictionary containing weights for each section
            mh_category (str): Category from Must-Have analysis (I, II, or III)
            
        Returns:
            Tuple[int, str]: Overall rating and category
        """
        if edu_rating == 0 and exp_rating == 0 and skills_rating == 0:
            return "NA", "NA"

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

        # Apply penalty for missing must-have requirements
        if mh_category and "III" in mh_category:
            overall_rating = max(0, overall_rating - 20)  # Reduce by 20 points, but not below 0

        overall_category = self.find_category(overall_rating)

        return overall_rating, overall_category
