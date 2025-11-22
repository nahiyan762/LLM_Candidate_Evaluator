import pdfplumber
import re
import sys
import os
from pathlib import Path
from typing import List
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a local PDF file."""
    text_content = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        sys.exit(1)

    if not text_content.strip():
        print("⚠️ No readable text found in the PDF.")
    return text_content


def clean_text(text: str) -> str:
    """Clean and normalize extracted PDF text for readability."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[•▪●♦■□–—−]', '-', text)
    text = re.sub(r'[-_]{3,}', '-', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text.strip()


def save_clean_text(output_path: str, text: str):
    """Save cleaned text to a .txt file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Cleaned text saved to: {output_path}")


def load_from_text_file(file_path: str) -> List[Document]:
    """Load content from text file and split into chunks."""
    print(f"📄 Loading text file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    doc = Document(page_content=content, metadata={"source": file_path})

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents([doc])
    print(f"📝 Created {len(chunks)} text chunks from {Path(file_path).name}")
    return chunks


def create_or_update_vector_store(all_docs: List[Document], persist_dir="./chroma_db"):
    """Create or update a ChromaDB vector store with all documents."""
    print("🔄 Creating or updating vector store...")

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print(f"✅ Vector store created/updated successfully at: {persist_dir}")
    return vectorstore


def evaluate_candidate(cv_text: str, persist_dir="./chroma_db"):
    """Retrieve context from Chroma and evaluate candidate using Ollama (DeepSeek)."""

    print("🔎 Loading vectorstore...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    # LLM — you can replace deepseek-coder with another Ollama model (e.g., llama3)
    llm = ChatOllama(model="deepseek-r1", temperature=0.7)

    print("🧠 Evaluating candidate... (this may take a few seconds)")
    
    # Instead of using RetrievalQA, we'll manually retrieve context and create our own prompt
    # This gives us more control over input length
    
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
"score": number (1-10),
"decision": "Hire" or "Reject"
}}

Evaluation Criteria:
{context}

Candidate CV:
{cv_text[:2000]}...

Provide a thorough evaluation based on the criteria above.
"""

    # Get evaluation from LLM
    result = llm.invoke(evaluation_prompt)

    print("\n📊 Evaluation Result:")
    print("=" * 40)
    print(result.content)
    print("=" * 40)


def main(pdf_path: str):
    if not pdf_path or not Path(pdf_path).exists():
        print("❌ PDF file not found or path not provided.")
        sys.exit(1)

    print(f"📄 Reading PDF: {pdf_path}")

    # 1️⃣ Extract & clean
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)

    # 2️⃣ Save cleaned text
    output_file = Path(pdf_path).with_suffix(".txt")
    save_clean_text(output_file, cleaned_text)

    # 3️⃣ Load knowledge base files
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
        else:
            print(f"⚠️ Skipped missing file: {kb_file}")

    # 4️⃣ Create or update vector store once (not 4 times)
    create_or_update_vector_store(all_docs)

    print("\n🔍 Preview of cleaned PDF text:")
    print("=" * 40)
    print(cleaned_text[:1000])
    print("\n...")
    evaluate_candidate(cleaned_text)


if __name__ == "__main__":
    main("/Users/bs01381/Downloads/Mohammad Sultan Al Nahiyan.pdf")
