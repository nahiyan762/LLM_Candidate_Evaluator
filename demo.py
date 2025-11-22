import re
import os
from pathlib import Path
from typing import List
import pdfplumber
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

# Import evaluation functions from extract_clean_pdf
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

app = FastAPI(title="LLM Candidate Evaluator")

# Configuration for Docker host gateway
API_BASE_URL = os.getenv("API_BASE_URL", "http://host.docker.internal:8000")

# Response model for evaluation results
class EvaluationResult(BaseModel):
    evaluation: dict
    status: str

def clean_text(text: str) -> str:
    """Clean and normalize extracted PDF text for readability."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[•▪●♦■□–—−]', '-', text)
    text = re.sub(r'[-_]{3,}', '-', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text.strip()


def load_from_text_file(file_path: str) -> List[Document]:
    """Load content from text file and split into chunks."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    doc = Document(page_content=content, metadata={"source": file_path})
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents([doc])
    return chunks


def create_or_update_vector_store(all_docs: List[Document], persist_dir="./chroma_db"):
    """Create or update a ChromaDB vector store with all documents."""
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vectorstore


def evaluate_candidate_text(cv_text: str, persist_dir="./chroma_db") -> dict:
    """Retrieve context from Chroma and evaluate candidate using Ollama."""
    try:
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )

        llm = ChatOllama(model="deepseek-r1", temperature=0.7)
        
        # Use a short query to find relevant evaluation criteria
        query = "Android developer evaluation criteria requirements skills"
        docs = vectorstore.similarity_search(query, k=4)
        
        # Combine retrieved context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create evaluation prompt
        evaluation_prompt = f"""
You are an expert Android hiring manager.
Based on the evaluation criteria and candidate CV, evaluate them and return a JSON with:

{{
"strengths": [],
"weaknesses": [],
"projects": [
    "Project Name"
],
"experience_years": number,
"score": number (1-10),
"decision": if score above 5 "Hire" else "Reject"
}}

Evaluation Criteria:
{context}

Candidate CV:
{cv_text[:2000]}...

Provide a thorough evaluation based on the criteria above.
"""

        # Get evaluation from LLM
        result = llm.invoke(evaluation_prompt)
        
        # Try to parse JSON from the response
        import json
        try:
            # Extract JSON from the response
            response_text = result.content
            # Find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return {"error": "Could not extract JSON from response", "raw_response": response_text}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in response", "raw_response": result.content}
            
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}


def initialize_knowledge_base():
    """Initialize the knowledge base with evaluation criteria files."""
    kb_files = [
        "data/android_architecture.txt",
        "data/android_core_skills.txt", 
        "data/android_tools.txt",
        "data/senior_expectations.txt"
    ]
    
    all_docs = []
    for kb_file in kb_files:
        if Path(kb_file).exists():
            all_docs.extend(load_from_text_file(kb_file))
    
    if all_docs:
        create_or_update_vector_store(all_docs)
        return True
    return False


@app.on_event("startup")
async def startup_event():
    """Initialize knowledge base on startup."""
    if initialize_knowledge_base():
        print("✅ Knowledge base initialized successfully")
    else:
        print("⚠️ Warning: No knowledge base files found in data/ directory")


@app.get("/", response_class=PlainTextResponse)
async def root():
    """Root health endpoint returning plain text (not JSON)."""
    return "FastAPI LLM Candidate Evaluator is running"
    """Upload PDF and extract text only."""
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        text_content = ""
        # Read the uploaded PDF file
        pdf_reader = pdfplumber.open(file.file)
        for page in pdf_reader.pages:
            text_content += page.extract_text() or ""
        pdf_reader.close()

        if not text_content.strip():
            raise HTTPException(status_code=400, detail="No text found in the PDF")

        # Clean the extracted text
        cleaned_text = clean_text(text_content)

        return JSONResponse(content={
            "filename": file.filename,
            "text": cleaned_text,
            "status": "success"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")


@app.post("/evaluate-candidate", response_model=EvaluationResult)
async def evaluate_candidate(file: UploadFile = File(...)):
    """Upload PDF and get complete LLM evaluation."""
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        text_content = ""
        # Read the uploaded PDF file
        pdf_reader = pdfplumber.open(file.file)
        for page in pdf_reader.pages:
            text_content += page.extract_text() or ""
        pdf_reader.close()

        if not text_content.strip():
            raise HTTPException(status_code=400, detail="No text found in the PDF")

        # Clean the extracted text
        cleaned_text = clean_text(text_content)
        
        # Extract candidate name (basic extraction from first few lines)
        candidate_name = "Unknown"
        first_lines = cleaned_text.split('\n')[:5]
        for line in first_lines:
            if any(keyword in line.lower() for keyword in ['name', 'cv', 'resume']):
                # Simple name extraction
                words = line.split()
                if len(words) >= 2:
                    candidate_name = ' '.join(words[:3])  # Take first 3 words as name
                break
        
        # If no name found in structured way, take first line that looks like a name
        if candidate_name == "Unknown":
            for line in first_lines:
                words = line.strip().split()
                if len(words) >= 2 and len(words) <= 4:  # Likely a name
                    candidate_name = line.strip()
                    break

        # Evaluate the candidate
        evaluation_result = evaluate_candidate_text(cleaned_text)

        return EvaluationResult(
            evaluation=evaluation_result,
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating candidate: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "LLM Candidate Evaluator"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "demo:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
