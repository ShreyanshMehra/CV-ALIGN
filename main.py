from fastapi import FastAPI, File, UploadFile
import os
import uuid
import pdfplumber
from docx import Document

from rag_engine import chunk_text, build_faiss_index, retrieve_top_k_chunks

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

cv1_text = ""
cv2_text = ""
cv3_text = ""
jd_text_global = ""

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def save_and_extract(file: UploadFile):
    ext = file.filename.split(".")[-1].lower()
    filename = f"{uuid.uuid4()}.{ext}"
    path = os.path.join(UPLOAD_DIR, filename)

    with open(path, "wb") as f:
        f.write(file.file.read())

    if ext == "pdf":
        extracted = extract_text_from_pdf(path)
    elif ext == "docx":
        extracted = extract_text_from_docx(path)
    else:
        raise ValueError("Unsupported file type")

    return extracted, filename

# Upload CV1
@app.post("/upload-cv1/")
async def upload_cv1(file: UploadFile = File(...)):
    global cv1_text
    try:
        text, filename = save_and_extract(file)
        cv1_text = text
        return {"message": "CV1 uploaded", "filename": filename, "text_preview": text[:300]}
    except Exception as e:
        return {"error": str(e)}

# Upload CV2
@app.post("/upload-cv2/")
async def upload_cv2(file: UploadFile = File(...)):
    global cv2_text
    try:
        text, filename = save_and_extract(file)
        cv2_text = text
        return {"message": "CV2 uploaded", "filename": filename, "text_preview": text[:300]}
    except Exception as e:
        return {"error": str(e)}

# Upload CV3
@app.post("/upload-cv3/")
async def upload_cv3(file: UploadFile = File(...)):
    global cv3_text
    try:
        text, filename = save_and_extract(file)
        cv3_text = text
        return {"message": "CV3 uploaded", "filename": filename, "text_preview": text[:300]}
    except Exception as e:
        return {"error": str(e)}

# Upload JD
@app.post("/upload-jd/")
async def upload_jd(file: UploadFile = File(...)):
    global jd_text_global
    try:
        text, filename = save_and_extract(file)
        jd_text_global = text
        return {"message": "JD uploaded", "filename": filename, "text_preview": text[:300]}
    except Exception as e:
        return {"error": str(e)}

# Score all 3 CVs against the JD and rank
@app.get("/score-cvs/")
def score_cvs():
    if not jd_text_global:
        return {"error": "Upload JD first."}

    cvs = {
        "CV1": cv1_text,
        "CV2": cv2_text,
        "CV3": cv3_text
    }
    missing = [name for name, text in cvs.items() if not text]
    if missing:
        return {"error": f"Upload these CVs first: {', '.join(missing)}"}

    results = []
    for name, text in cvs.items():
        chunks = chunk_text(text)
        index, embeddings, chunk_list = build_faiss_index(chunks)
        top_chunks = retrieve_top_k_chunks(jd_text_global, index, chunk_list, embeddings)

        avg_score = sum(score for _, score in top_chunks) / len(top_chunks)
        results.append({
            "candidate": name,
            "similarity_score": round(avg_score * 100, 2),
            "rag_based_match": "Strong match!" if avg_score > 0.75 else "Needs improvement."
        })

    results.sort(key=lambda x: x["similarity_score"], reverse=True)

    return {"ranked_candidates": results}
