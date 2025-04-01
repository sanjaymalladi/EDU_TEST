from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from langchain_google_genai import ChatGoogleGenerativeAI
import io
import os
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
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

# Import our agents
from aspects_agent import AspectsAgent
from edu_agent import CombinedEducationAgent
from exp_agent import CombinedExperienceAgent
from skills_agent import CombinedSkillsAgent
from supervisor_agent import SupervisorAgent
from mh_agent import CombinedMHAgent

# Initialize FastAPI app with CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Resume Analysis API",
    description="API for analyzing resumes against job descriptions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class AspectRequest(BaseModel):
    job_description: str

class AspectResponse(BaseModel):
    section_aspects: Dict

class EvaluationRequest(BaseModel):
    job_description: str
    resume: str
    section_aspects: Dict

class RatingAndEvidence(BaseModel):
    evidence: List[str]
    rating: int

class CategoryAndEvidence(BaseModel):
    category: int
    evidence: List[str]

class EvaluationResponse(BaseModel):
    experience: RatingAndEvidence
    skills: RatingAndEvidence
    education_and_certification: RatingAndEvidence
    must_haves: CategoryAndEvidence
    overall_rating: int
    overall_category: str
    section_weights: Dict[str, float]
    overall_summary: str

class AnalysisResponse(BaseModel):
    overall_rating: int
    overall_category: str
    section_weights: Dict[str, int]
    overall_summary: str
    education_analysis: Optional[Dict]
    experience_analysis: Optional[Dict]
    skills_analysis: Optional[Dict]
    must_have_analysis: Optional[Dict]

async def extract_text_from_pdf(file: bytes) -> str:
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

async def extract_text_from_docx(file: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(file))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading DOCX: {str(e)}")

async def read_file_content(file: UploadFile) -> str:
    content = await file.read()
    file_extension = file.filename.split('.')[-1].lower()

    try:
        if file_extension == 'pdf':
            return await extract_text_from_pdf(content)
        elif file_extension == 'docx':
            return await extract_text_from_docx(content)
        elif file_extension == 'txt':
            return content.decode('utf-8').strip()
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file {file.filename}: {str(e)}")

def extract_rating(evaluation_str: str) -> int:
    """Extract numeric rating from evaluation string."""
    if not evaluation_str or not isinstance(evaluation_str, str):
        return 0
    
    evaluation_str = evaluation_str.strip()
    if 'Rating:' in evaluation_str:
        try:
            rating_part = evaluation_str.split('Rating:')[-1].strip()
            numbers = re.findall(r'\d+', rating_part)
            if numbers:
                return int(numbers[0])
        except Exception:
            pass
    return 0

def calculate_overall_rating(experience_rating: int, skills_rating: int, education_rating: int, 
                           weights: Dict[str, float], mh_category: Optional[int] = None) -> tuple:
    """Calculate overall rating and category."""
    try:
        # Normalize weights to sum to 100
        total_weight = sum(weights.values())
        if total_weight > 0:
            normalized_weights = {k: (v/total_weight) * 100 for k, v in weights.items()}
        else:
            normalized_weights = {
                'experience': 40,
                'skills': 35,
                'education_and_certification': 25
            }

        # Calculate weighted score
        weighted_score = (
            normalized_weights['experience'] * experience_rating / 100 +
            normalized_weights['skills'] * skills_rating / 100 +
            normalized_weights['education_and_certification'] * education_rating / 100
        )

        # Apply must-have penalty if category is 3
        if mh_category == 3:
            weighted_score = max(0, weighted_score - 20)

        # Round the final rating
        overall_rating = round(weighted_score)

        # Determine category
        if overall_rating < 41:
            category = "Not Suitable"
        elif overall_rating < 61:
            category = "Moderate Match"
        elif overall_rating < 81:
            category = "Good Match"
        else:
            category = "Excellent Match"

        return overall_rating, category

    except Exception as e:
        return 0, "Error"

async def run_in_threadpool(executor: ThreadPoolExecutor, func, *args):
    """Run a synchronous function in a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, func, *args)

@app.post("/aspects", response_model=AspectResponse)
async def generate_aspects(request: AspectRequest) -> AspectResponse:
    """Generate aspects for all sections from job description."""
    try:
        # Initialize agents
        agents = {
            'skills': AspectsAgent(),
            'education': AspectsAgent(),
            'experience': AspectsAgent(),
            'musthave': AspectsAgent()
        }

        all_aspects = {}

        # Create thread pool executor
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Create tasks for each section
            tasks = []
            for section_name, agent in agents.items():
                task = run_in_threadpool(
                    executor,
                    agent.generate_all_aspects,
                    request.job_description
                )
                tasks.append((section_name, task))

            # Wait for all tasks to complete
            for section_name, task in tasks:
                try:
                    result = await task
                    all_aspects[section_name] = result
                except Exception as e:
                    all_aspects[section_name] = f"Error generating aspects: {str(e)}"

        return AspectResponse(section_aspects=all_aspects)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Aspect generation failed: {str(e)}"
        )

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_resume(request: EvaluationRequest) -> EvaluationResponse:
    """Evaluate a resume against a job description using the provided aspects."""
    try:
        # Initialize agents
        edu_agent = CombinedEducationAgent()
        exp_agent = CombinedExperienceAgent()
        skills_agent = CombinedSkillsAgent()
        mh_agent = CombinedMHAgent()
        supervisor_agent = SupervisorAgent()

        # Create thread pool executor
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Run evaluations in parallel
            tasks = {
                'education': run_in_threadpool(
                    executor, 
                    edu_agent.run, 
                    request.job_description, 
                    request.resume,
                    request.section_aspects.get('education', {})
                ),
                'experience': run_in_threadpool(
                    executor,
                    exp_agent.run,
                    request.job_description,
                    request.resume,
                    request.section_aspects.get('experience', {})
                ),
                'skills': run_in_threadpool(
                    executor,
                    skills_agent.run,
                    request.job_description,
                    request.resume,
                    request.section_aspects.get('skills', {})
                ),
                'must_haves': run_in_threadpool(
                    executor,
                    mh_agent.run,
                    request.job_description,
                    request.resume,
                    request.section_aspects.get('musthave', {})
                )
            }

            # Get section weights
            weights, _ = await run_in_threadpool(
                executor,
                supervisor_agent.get_section_weights,
                request.job_description
            )

            # Wait for all evaluations
            results = {
                section: await task
                for section, task in tasks.items()
            }

            # Extract ratings and evidence
            edu_rating = extract_rating(results['education'].get('evaluation', ''))
            exp_rating = extract_rating(results['experience'].get('evaluation', ''))
            skills_rating = extract_rating(results['skills'].get('evaluation', ''))
            mh_category = results['must_haves'].get('category', 1)

            # Calculate overall rating
            overall_rating, overall_category = calculate_overall_rating(
                exp_rating, skills_rating, edu_rating, weights, mh_category
            )

            # Generate overall summary
            overall_summary = supervisor_agent.generate_summary(
                experience_rationale=results['experience'].get('evaluation', ''),
                skills_rationale=results['skills'].get('evaluation', ''),
                education_rationale=results['education'].get('evaluation', '')
            )

            return EvaluationResponse(
                experience=RatingAndEvidence(
                    evidence=results['experience'].get('evidence', []),
                    rating=exp_rating
                ),
                skills=RatingAndEvidence(
                    evidence=results['skills'].get('evidence', []),
                    rating=skills_rating
                ),
                education_and_certification=RatingAndEvidence(
                    evidence=results['education'].get('evidence', []),
                    rating=edu_rating
                ),
                must_haves=CategoryAndEvidence(
                    evidence=results['must_haves'].get('evidence', []),
                    category=mh_category
                ),
                overall_rating=overall_rating,
                overall_category=overall_category,
                section_weights=weights,
                overall_summary=overall_summary
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_resume(
    jd_file: UploadFile = File(...),
    resume_file: UploadFile = File(...)
):
    """
    Analyze a resume against a job description.
    
    Parameters:
    - jd_file: Job description file (PDF, DOCX, or TXT)
    - resume_file: Resume file (PDF, DOCX, or TXT)
    
    Returns:
    - AnalysisResponse containing the analysis results
    """
    try:
        # Read file contents
        jd_text = await read_file_content(jd_file)
        resume_text = await read_file_content(resume_file)

        # Generate aspects
        aspects_agent = AspectsAgent()
        aspects = aspects_agent.generate_all_aspects(jd_text)

        # Initialize agents
        edu_agent = CombinedEducationAgent()
        exp_agent = CombinedExperienceAgent()
        skills_agent = CombinedSkillsAgent()
        mh_agent = CombinedMHAgent()
        supervisor_agent = SupervisorAgent()

        # Run analyses
        edu_result = edu_agent.run(jd_text, resume_text, aspects)
        exp_result = exp_agent.run(jd_text, resume_text, aspects)
        skills_result = skills_agent.run(jd_text, resume_text, aspects)
        mh_result = mh_agent.run(jd_text, resume_text, aspects)

        # Get section weights
        weights, weight_reasoning = supervisor_agent.get_section_weights(jd_text)

        # Extract ratings
        edu_rating = extract_rating(edu_result.get('evaluation', '')) if edu_result else 0
        exp_rating = extract_rating(exp_result.get('evaluation', '')) if exp_result else 0
        skills_rating = extract_rating(skills_result.get('evaluation', '')) if skills_result else 0

        # Get MH category
        mh_category = None
        if mh_result and "evaluation" in mh_result:
            evaluation_str = mh_result['evaluation']
            if isinstance(evaluation_str, str):
                if "Category III" in evaluation_str:
                    mh_category = "III"
                elif "Category II" in evaluation_str:
                    mh_category = "II"
                elif "Category I" in evaluation_str:
                    mh_category = "I"

        # Calculate overall rating
        overall_rating, overall_category = supervisor_agent.calculate_overall_rating(
            edu_rating=edu_rating,
            exp_rating=exp_rating,
            skills_rating=skills_rating,
            weights=weights,
            mh_category=mh_category
        )

        # Generate summary
        overall_summary = supervisor_agent.generate_summary(
            experience_rationale=exp_result.get('evaluation', '') if exp_result else '',
            skills_rationale=skills_result.get('evaluation', '') if skills_result else '',
            education_rationale=edu_result.get('evaluation', '') if edu_result else ''
        )

        return AnalysisResponse(
            overall_rating=overall_rating,
            overall_category=overall_category,
            section_weights=weights,
            overall_summary=overall_summary,
            education_analysis=edu_result,
            experience_analysis=exp_result,
            skills_analysis=skills_result,
            must_have_analysis=mh_result
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 