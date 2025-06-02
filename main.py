# enhanced_main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import pdfplumber
from docx import Document
from typing import Dict, Optional, List
import logging
from datetime import datetime
import json
from dataclasses import asdict

from rag_engine import EnhancedRAGEngine, ChunkConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced CV Scoring System", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Enhanced storage for CVs and JD
class CVStore:
    def __init__(self):
        self.cvs: Dict[str, str] = {}
        self.jd: Optional[str] = None
        self.metadata: Dict[str, dict] = {}
        self.rag_engine = EnhancedRAGEngine(
            chunk_config=ChunkConfig(chunk_size=256, overlap=50, strategy="recursive")
        )
    
    def add_cv(self, cv_id: str, text: str, filename: str):
        self.cvs[cv_id] = text
        self.metadata[cv_id] = {
            "filename": filename,
            "upload_time": datetime.now().isoformat(),
            "text_length": len(text),
            "word_count": len(text.split())
        }
    
    def set_jd(self, text: str, filename: str):
        self.jd = text
        self.metadata["jd"] = {
            "filename": filename,
            "upload_time": datetime.now().isoformat(),
            "text_length": len(text),
            "word_count": len(text.split())
        }
    
    def get_status(self):
        return {
            "cvs_uploaded": len(self.cvs),
            "cv_ids": list(self.cvs.keys()),
            "jd_uploaded": self.jd is not None,
            "metadata": self.metadata
        }

cv_store = CVStore()

def extract_text_from_pdf(file_path: str) -> str:
    """Enhanced PDF text extraction with better error handling."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + f"\n--- Page {page_num + 1} ---\n"
                except Exception as e:
                    logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Error reading PDF file: {e}")
        raise HTTPException(status_code=400, detail=f"Error reading PDF file: {str(e)}")
    
    return text.strip()

def extract_text_from_docx(file_path: str) -> str:
    """Enhanced DOCX text extraction."""
    try:
        doc = Document(file_path)
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text.strip())
        
        # Also extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    paragraphs.append(" | ".join(row_text))
        
        return "\n".join(paragraphs)
    except Exception as e:
        logger.error(f"Error reading DOCX file: {e}")
        raise HTTPException(status_code=400, detail=f"Error reading DOCX file: {str(e)}")

def save_and_extract(file: UploadFile) -> tuple[str, str]:
    """Enhanced file saving and text extraction."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    ext = file.filename.split(".")[-1].lower()
    if ext not in ["pdf", "docx"]:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
    
    filename = f"{uuid.uuid4()}.{ext}"
    path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        with open(path, "wb") as f:
            content = file.file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Empty file uploaded")
            f.write(content)
        
        # Extract text based on file type
        if ext == "pdf":
            extracted_text = extract_text_from_pdf(path)
        else:  # docx
            extracted_text = extract_text_from_docx(path)
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the file")
        
        return extracted_text, filename
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Clean up uploaded file
        if os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass

@app.get("/")
def read_root():
    return {"message": "Enhanced CV Scoring System API", "version": "2.0.0"}

@app.get("/status")
def get_status():
    """Get current system status."""
    return cv_store.get_status()

@app.post("/upload-cv/")
async def upload_cv(cv_id: str, file: UploadFile = File(...)):
    """Upload a CV with custom ID."""
    try:
        text, filename = save_and_extract(file)
        cv_store.add_cv(cv_id, text, filename)
        
        return {
            "message": f"CV {cv_id} uploaded successfully",
            "cv_id": cv_id,
            "filename": filename,
            "text_preview": text[:300] + "..." if len(text) > 300 else text,
            "word_count": len(text.split()),
            "character_count": len(text)
        }
    except Exception as e:
        logger.error(f"Error uploading CV {cv_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-jd/")
async def upload_jd(file: UploadFile = File(...)):
    """Upload Job Description."""
    try:
        text, filename = save_and_extract(file)
        cv_store.set_jd(text, filename)
        
        return {
            "message": "Job Description uploaded successfully",
            "filename": filename,
            "text_preview": text[:300] + "..." if len(text) > 300 else text,
            "word_count": len(text.split()),
            "character_count": len(text)
        }
    except Exception as e:
        logger.error(f"Error uploading JD: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score-cvs/")
async def score_cvs(background_tasks: BackgroundTasks):
    """Enhanced scoring with semantic similarity + weighted keyword matching."""

    if not cv_store.jd:
        raise HTTPException(status_code=400, detail="Please upload a Job Description first")
    if not cv_store.cvs:
        raise HTTPException(status_code=400, detail="Please upload at least one CV")

    # Validate JD text quality
    jd_word_count = len(cv_store.jd.split())
    if jd_word_count < 10:
        raise HTTPException(
            status_code=400,
            detail="Job Description is too short for meaningful analysis. Please upload a more detailed JD."
        )

    # --- Extract weighted keywords from JD ---
    try:
        jd_keywords_with_weights = EnhancedRAGEngine.extract_jd_keywords_with_weights(cv_store.jd, top_n=20)
    except Exception as e:
        logger.error(f"Error extracting JD keywords: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract keywords from JD.")

    results = []
    detailed_results = {}
    processing_errors = []

    for cv_id, cv_text in cv_store.cvs.items():
        try:
            logger.info(f"Processing CV: {cv_id}")

            # Validate CV text
            cv_word_count = len(cv_text.split())
            if cv_word_count < 10:
                error_msg = f"CV {cv_id} is too short for analysis"
                processing_errors.append(error_msg)
                results.append({
                    "cv_id": cv_id,
                    "error": error_msg,
                    "filename": cv_store.metadata[cv_id]["filename"]
                })
                continue

            # --- Semantic similarity score ---
            score_data = cv_store.rag_engine.calculate_comprehensive_score(cv_text, cv_store.jd)
            if "error" in score_data:
                error_msg = f"Scoring failed for CV {cv_id}: {score_data['error']}"
                processing_errors.append(error_msg)
                results.append({
                    "cv_id": cv_id,
                    "error": score_data["error"],
                    "filename": cv_store.metadata[cv_id]["filename"]
                })
                continue

            semantic_score = score_data["final_score"] * 100  # Convert to 0-100 scale

            # --- Weighted keyword match score ---
            keyword_score = EnhancedRAGEngine.score_cv_by_weighted_keywords(cv_text, jd_keywords_with_weights)

            # --- Combine scores (70% semantic, 30% keyword) ---
            final_score = EnhancedRAGEngine.combine_scores(
                semantic_score, keyword_score, semantic_weight=0.8, keyword_weight=0.2
            )

            # Determine match quality
            if final_score >= 80:
                match_quality = "Excellent Match"
            elif final_score >= 65:
                match_quality = "Good Match"
            elif final_score >= 50:
                match_quality = "Fair Match"
            else:
                match_quality = "Poor Match"

            cv_result = {
                "cv_id": cv_id,
                "filename": cv_store.metadata[cv_id]["filename"],
                "final_score": round(final_score, 2),
                "match_quality": match_quality,
                "semantic_score": round(semantic_score, 2),
                "keyword_score": round(keyword_score, 2),
                "max_similarity": round(score_data["max_similarity"] * 100, 2),
                "avg_similarity": round(score_data["avg_similarity"] * 100, 2),
                "coverage_score": round(score_data["coverage_score"] * 100, 2)
            }

            results.append(cv_result)
            detailed_results[cv_id] = score_data

        except Exception as e:
            error_msg = f"Unexpected error processing CV {cv_id}: {str(e)}"
            logger.error(error_msg)
            processing_errors.append(error_msg)
            results.append({
                "cv_id": cv_id,
                "error": str(e),
                "filename": cv_store.metadata[cv_id]["filename"]
            })

    # Check if we have any successful results
    successful_results = [r for r in results if "error" not in r]
    if not successful_results:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process any CVs successfully. Errors: {'; '.join(processing_errors)}"
        )

    # Sort successful results by score
    successful_results.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    return {
        "ranked_candidates": results,
        "summary": {
            "total_cvs_processed": len(cv_store.cvs),
            "successful_processing": len(successful_results),
            "failed_processing": len(processing_errors),
            "highest_score": max([r.get("final_score", 0) for r in successful_results], default=0),
            "average_score": round(
                sum([r.get("final_score", 0) for r in successful_results]) / len(successful_results)
                if successful_results else 0, 2
            ),
            "processing_errors": processing_errors if processing_errors else None
        }
    }

@app.delete("/clear-data/")
async def clear_data():
    """Clear all uploaded CVs and JD."""
    cv_store.cvs.clear()
    cv_store.jd = None
    cv_store.metadata.clear()
    return {"message": "All data cleared successfully"}

@app.get("/cv/{cv_id}")
async def get_cv_details(cv_id: str):
    """Get details for a specific CV."""
    if cv_id not in cv_store.cvs:
        raise HTTPException(status_code=404, detail="CV not found")
    
    return {
        "cv_id": cv_id,
        "metadata": cv_store.metadata[cv_id],
        "text_preview": cv_store.cvs[cv_id][:500] + "..." if len(cv_store.cvs[cv_id]) > 500 else cv_store.cvs[cv_id]
    }


