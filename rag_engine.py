from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from typing import List, Tuple
from dataclasses import dataclass
import logging
from keybert import KeyBERT

logger = logging.getLogger(__name__)

@dataclass
class ChunkConfig:
    chunk_size: int = 256
    overlap: int = 50
    strategy: str = "recursive"
    min_chunk_length: int = 10

class EnhancedRAGEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_config: ChunkConfig = None):
        self.model = SentenceTransformer(model_name)
        self.chunk_config = chunk_config or ChunkConfig()
        self.dimension = 384

    def preprocess_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text.strip())
        if len(text) < 3:
            return ""
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\"\'\/\@\#\$\%\&\*\+\=]', ' ', text)
        text = re.sub(r'\.{3,}', '...', text)
        text = re.sub(r'\-{3,}', '---', text)
        return text.strip()

    def chunk_text_safe(self, text: str) -> List[str]:
        if not text or len(text.strip()) < self.chunk_config.min_chunk_length:
            return []
        words = text.split()
        if len(words) == 0:
            return []
        chunks = []
        chunk_size = self.chunk_config.chunk_size
        overlap = self.chunk_config.overlap
        for i in range(0, len(words), max(1, chunk_size - overlap)):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) > 0:
                chunk_text = ' '.join(chunk_words).strip()
                if len(chunk_text) >= self.chunk_config.min_chunk_length:
                    chunks.append(chunk_text)
        if not chunks and len(text.strip()) >= self.chunk_config.min_chunk_length:
            chunks = [text.strip()]
        return chunks

    def chunk_text_recursive_safe(self, text: str) -> List[str]:
        if not text or len(text.strip()) < self.chunk_config.min_chunk_length:
            return []
        separators = ['\n\n', '\n', '. ', '! ', '? ', '; ', ', ', ' ']
        def safe_split_text(text: str, separators: List[str]) -> List[str]:
            if not separators or not text:
                return [text] if text and len(text.strip()) >= self.chunk_config.min_chunk_length else []
            separator = separators[0]
            remaining_separators = separators[1:]
            if separator not in text:
                return safe_split_text(text, remaining_separators)
            parts = text.split(separator)
            result = []
            current_chunk = ""
            for part in parts:
                if not part:
                    continue
                test_chunk = current_chunk + separator + part if current_chunk else part
                if len(test_chunk.split()) <= self.chunk_config.chunk_size:
                    current_chunk = test_chunk
                else:
                    if current_chunk and len(current_chunk.strip()) >= self.chunk_config.min_chunk_length:
                        result.append(current_chunk.strip())
                    if len(part.split()) > self.chunk_config.chunk_size:
                        sub_chunks = safe_split_text(part, remaining_separators)
                        result.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = part
            if current_chunk and len(current_chunk.strip()) >= self.chunk_config.min_chunk_length:
                result.append(current_chunk.strip())
            return [chunk for chunk in result if chunk and len(chunk.strip()) >= self.chunk_config.min_chunk_length]
        chunks = safe_split_text(text, separators)
        if not chunks and len(text.strip()) >= self.chunk_config.min_chunk_length:
            chunks = [text.strip()]
        return chunks

    def chunk_text(self, text: str) -> List[str]:
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return []
        if self.chunk_config.strategy == "recursive":
            chunks = self.chunk_text_recursive_safe(processed_text)
        else:
            chunks = self.chunk_text_safe(processed_text)
        valid_chunks = [chunk for chunk in chunks if chunk and len(chunk.strip()) >= self.chunk_config.min_chunk_length]
        if not valid_chunks:
            if len(processed_text) >= self.chunk_config.min_chunk_length:
                valid_chunks = [processed_text]
        return valid_chunks

    def build_optimized_faiss_index(self, chunks: List[str]) -> Tuple[faiss.Index, np.ndarray, List[str]]:
        if not chunks:
            raise ValueError("No chunks provided to build index")
        valid_chunks = [chunk for chunk in chunks if chunk and len(chunk.strip()) >= 3]
        if not valid_chunks:
            raise ValueError("No valid chunks after filtering")
        embeddings = self.model.encode(
            valid_chunks, 
            convert_to_numpy=True, 
            normalize_embeddings=True,
            show_progress_bar=False
        )
        embeddings = embeddings.astype('float32')
        num_chunks = len(valid_chunks)
        if num_chunks < 1000:
            index = faiss.IndexFlatIP(self.dimension)
        else:
            nlist = min(int(np.sqrt(num_chunks)), 100)
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
            index.nprobe = min(10, nlist)
        index.add(embeddings)
        return index, embeddings, valid_chunks

    def retrieve_top_k_chunks(self, query_text: str, index: faiss.Index, chunks: List[str], 
                            embeddings: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        if not query_text or not chunks:
            return []
        processed_query = self.preprocess_text(query_text)
        if not processed_query:
            return []
        query_embedding = self.model.encode(
            [processed_query], 
            convert_to_numpy=True, 
            normalize_embeddings=True
        ).astype('float32')
        k = min(k, len(chunks), index.ntotal)
        if k <= 0:
            return []
        scores, indices = index.search(query_embedding, k)
        top_chunks = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(chunks):
                top_chunks.append((chunks[idx], float(score)))
        return top_chunks

    def calculate_comprehensive_score(self, cv_text: str, jd_text: str, top_k: int = 10) -> dict:
        if not cv_text or not jd_text:
            return {"error": "Empty CV or JD text provided"}
        if len(cv_text.strip()) < 10 or len(jd_text.strip()) < 10:
            return {"error": "CV or JD text too short for meaningful analysis"}
        cv_chunks = self.chunk_text(cv_text)
        jd_chunks = self.chunk_text(jd_text)
        if not cv_chunks:
            return {"error": "Unable to create valid chunks from CV text"}
        if not jd_chunks:
            return {"error": "Unable to create valid chunks from JD text"}
        cv_index, cv_embeddings, cv_chunk_list = self.build_optimized_faiss_index(cv_chunks)
        all_similarities = []
        jd_coverage_scores = []
        for jd_chunk in jd_chunks:
            top_matches = self.retrieve_top_k_chunks(
                jd_chunk, cv_index, cv_chunk_list, cv_embeddings, 
                k=min(3, len(cv_chunks))
            )
            if top_matches:
                best_score = top_matches[0][1]
                all_similarities.append(best_score)
                avg_top_matches = np.mean([score for _, score in top_matches])
                jd_coverage_scores.append(avg_top_matches)
        if not all_similarities:
            return {"error": "No similarities could be calculated"}
        max_similarity = float(np.max(all_similarities))
        avg_similarity = float(np.mean(all_similarities))
        median_similarity = float(np.median(all_similarities))
        coverage_score = float(np.mean(jd_coverage_scores)) if jd_coverage_scores else 0.0
        final_score = (0.3 * max_similarity + 0.4 * avg_similarity + 
                      0.2 * median_similarity + 0.1 * coverage_score)
        return {
            "final_score": round(final_score, 4),
            "max_similarity": round(max_similarity, 4),
            "avg_similarity": round(avg_similarity, 4),
            "median_similarity": round(median_similarity, 4),
            "coverage_score": round(coverage_score, 4),
            "total_chunks_processed": len(cv_chunks),
            "jd_requirements_count": len(jd_chunks)
        }

    @staticmethod
    def extract_jd_keywords_with_weights(jd_text, top_n=20, ngram_range=(1,2)):
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(
            jd_text,
            keyphrase_ngram_range=ngram_range,
            stop_words='english',
            top_n=top_n
        )
        return keywords

    @staticmethod
    def score_cv_by_semantic_keywords(cv_text, jd_keywords_with_weights, model, threshold=0.6):
        """
        Score CV using semantic similarity between JD keywords and CV sentences.
        """
        cv_sentences = re.split(r'[.\n]', cv_text)
        cv_sentences = [s.strip() for s in cv_sentences if s.strip()]
        if not cv_sentences:
            return 0.0
        cv_embeddings = model.encode(cv_sentences)
        total_weight = sum(weight for _, weight in jd_keywords_with_weights)
        if total_weight == 0:
            return 0.0
        matched_weight = 0.0
        for phrase, weight in jd_keywords_with_weights:
            phrase_embedding = model.encode([phrase])
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
