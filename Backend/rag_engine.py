import os
import requests
import faiss
import numpy as np
import re
from typing import List, Tuple, Dict
from dataclasses import dataclass
import logging
from functools import lru_cache
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
logger = logging.getLogger(__name__)

API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

@dataclass
class ChunkConfig:
    chunk_size: int = 256
    overlap: int = 50
    strategy: str = "simple"
    min_chunk_length: int = 10

class EnhancedRAGEngine:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EnhancedRAGEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, chunk_config: ChunkConfig = None):
        if hasattr(self, 'initialized'):
            return
        self.chunk_config = chunk_config or ChunkConfig()
        self.dimension = 384
        self.initialized = True

    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single string using Hugging Face API."""
        if not text:
            return np.zeros(self.dimension, dtype='float32')
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": [text]}
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Hugging Face API error: {response.text}")
            raise e
        embedding = response.json()
        if isinstance(embedding[0], list) and isinstance(embedding[0][0], list):
            embedding = embedding[0][0]
        else:
            embedding = embedding[0]
        return np.array(embedding, dtype='float32')


    @staticmethod
    def get_embeddings(texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of strings using Hugging Face API."""
        if not texts:
            raise ValueError("No texts provided for embedding.")
        
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": texts}
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Hugging Face API error: {response.text}")
            raise e
        
        embeddings = response.json()

        # If API returns shape [N, 1, D], flatten it
        if isinstance(embeddings[0], list) and isinstance(embeddings[0][0], list):
            embeddings = [e[0] for e in embeddings]
        
        return np.array(embeddings, dtype='float32')

    def preprocess_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\"\'\/\@\#\$\%\&\*\+\=]', ' ', text)
        return text.strip()

    def chunk_text(self, text: str) -> List[str]:
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return []
        
        sentences = re.split(r'[.!?]\s+', processed_text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) <= self.chunk_config.chunk_size:
                current_chunk = (current_chunk + " " + sentence).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(sentence) > self.chunk_config.chunk_size:
                    words = sentence.split()
                    for i in range(0, len(words), self.chunk_config.chunk_size):
                        chunk = " ".join(words[i:i + self.chunk_config.chunk_size])
                        if len(chunk) >= self.chunk_config.min_chunk_length:
                            chunks.append(chunk)
                else:
                    current_chunk = sentence
                    
        if current_chunk and len(current_chunk) >= self.chunk_config.min_chunk_length:
            chunks.append(current_chunk)
            
        return chunks

    def build_optimized_faiss_index(self, chunks: List[str]) -> Tuple[faiss.Index, np.ndarray, List[str]]:
        if not chunks:
            raise ValueError("No chunks provided to build index")
        
        batch_size = 32
        all_embeddings = []
        valid_chunks = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch = [chunk for chunk in batch if chunk and len(chunk.strip()) >= 3]
            if not batch:
                continue
            
            embeddings = self.get_embeddings(batch)
            all_embeddings.append(embeddings)
            valid_chunks.extend(batch)
        
        if not valid_chunks:
            raise ValueError("No valid chunks after filtering")
        
        embeddings = np.vstack(all_embeddings)
        
        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings)
        
        return index, embeddings, valid_chunks

    def calculate_comprehensive_score(self, cv_text: str, jd_text: str, top_k: int = 5) -> Dict:
        """Calculate similarity scores between CV and JD text. This is a synchronous function."""
        if not cv_text or not jd_text:
            return {"error": "Empty CV or JD text provided"}
            
        cv_chunks = self.chunk_text(cv_text)
        jd_chunks = self.chunk_text(jd_text)
        
        if not cv_chunks or not jd_chunks:
            return {"error": "Unable to create valid chunks from text"}
            
        try:
            cv_index, cv_embeddings, cv_chunk_list = self.build_optimized_faiss_index(cv_chunks)
            
            batch_size = 10
            all_similarities = []
            jd_coverage_scores = []
            
            for i in range(0, len(jd_chunks), batch_size):
                batch = jd_chunks[i:i + batch_size]
                batch_embeddings = self.get_embeddings(batch)
                
                similarities = np.dot(batch_embeddings, cv_embeddings.T)
                
                for sim_scores in similarities:
                    top_k_scores = np.sort(sim_scores)[-top_k:]
                    all_similarities.extend(top_k_scores)
                    jd_coverage_scores.append(np.mean(top_k_scores))
            
            if not all_similarities:
                return {"error": "No similarities could be calculated"}
                
            max_similarity = float(np.max(all_similarities))
            avg_similarity = float(np.mean(all_similarities))
            coverage_score = float(np.mean(jd_coverage_scores))
            
            final_score = (0.4 * max_similarity + 
                         0.4 * avg_similarity + 
                         0.2 * coverage_score)
            
            return {
                "final_score": round(final_score, 4),
                "max_similarity": round(max_similarity, 4),
                "avg_similarity": round(avg_similarity, 4),
                "coverage_score": round(coverage_score, 4),
                "total_chunks_processed": len(cv_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive scoring: {str(e)}")
            return {"error": f"Scoring failed: {str(e)}"}

    @staticmethod
    def extract_jd_keywords_with_weights(jd_text: str, top_n=20, ngram_range=(1, 2)) -> list[tuple[str, float]]:
        """
        Extract top_n keyword phrases from a job description using Hugging Face cloud embeddings.
        Returns a list of (keyword, score), where score is cosine similarity to the JD.
        """
        vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit([jd_text])
        candidates = vectorizer.get_feature_names_out()

        if not candidates.any():
            return []

        inputs = [jd_text] + list(candidates)
        response = requests.post(API_URL, headers=HEADERS, json={"inputs": inputs})
        response.raise_for_status()
        embeddings = response.json()

        doc_embedding = np.array(embeddings[0])
        candidate_embeddings = np.array(embeddings[1:])

        scores = cosine_similarity([doc_embedding], candidate_embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_n]

        keywords = [(candidates[i], float(scores[i])) for i in top_indices]
        return keywords

    @staticmethod
    def score_cv_by_semantic_keywords(cv_text, jd_keywords_with_weights, embedding_func, threshold=0.6):
        """
        Score CV using semantic similarity between JD keywords and CV sentences.
        """
        cv_sentences = re.split(r'[.\n]', cv_text)
        cv_sentences = [s.strip() for s in cv_sentences if s.strip()]
        if not cv_sentences:
            return 0.0
        cv_embeddings = np.array([embedding_func(s) for s in cv_sentences])
        total_weight = sum(weight for _, weight in jd_keywords_with_weights)
        if total_weight == 0:
            return 0.0
        matched_weight = 0.0
        for phrase, weight in jd_keywords_with_weights:
            phrase_embedding = embedding_func(phrase)
            similarities = np.dot(cv_embeddings, phrase_embedding.T).flatten()
            max_similarity = np.max(similarities)
            if max_similarity >= threshold:
                matched_weight += weight * max_similarity  # Weight by similarity
        score = (matched_weight / total_weight) * 100
        return round(score, 2)

    @staticmethod
    def combine_scores(semantic_score, keyword_score, semantic_weight=0.7, keyword_weight=0.3):
        final = (semantic_score * semantic_weight) + (keyword_score * keyword_weight)
        return round(final, 2)
