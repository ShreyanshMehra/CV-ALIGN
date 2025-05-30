from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=100):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def build_faiss_index(chunks):
    if not chunks:
        raise ValueError("No chunks to build index.")
    
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)  # Normalize for cosine similarity
    embeddings = embeddings.astype('float32')
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product for cosine similarity (since vectors normalized)
    index.add(embeddings)
    return index, embeddings, chunks

def retrieve_top_k_chunks(jd_text, index, chunks, embeddings, k=3):
    query_embedding = model.encode([jd_text], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    
    k = min(k, len(chunks))
    D, I = index.search(query_embedding, k)
    
    top_chunks = [chunks[i] for i in I[0]]
    top_scores = D[0].tolist()  # Inner product similarity scores between 0 and 1
    
    return list(zip(top_chunks, top_scores))
