# fixed_enhanced_rag_engine.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ChunkConfig:
    chunk_size: int = 256
    overlap: int = 50
    strategy: str = "recursive"
    min_chunk_length: int = 10  # Minimum chunk length to avoid empty chunks

class EnhancedRAGEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_config: ChunkConfig = None):
        try:
            self.model = SentenceTransformer(model_name)
            self.chunk_config = chunk_config or ChunkConfig()
            self.dimension = 384
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise ValueError(f"Failed to initialize sentence transformer: {e}")
        
    def preprocess_text(self, text: str) -> str:
        """Safely clean and normalize text."""
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # Remove excessive whitespace and normalize
            text = re.sub(r'\s+', ' ', text.strip())
            
            # Only proceed if we have actual content
            if len(text) < 3:
                return ""
            
            # Remove problematic characters safely
            text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\"\'\/\@\#\$\%\&\*\+\=]', ' ', text)
            text = re.sub(r'\.{3,}', '...', text)
            text = re.sub(r'\-{3,}', '---', text)
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"Error in text preprocessing: {e}")
            return text.strip() if text else ""
    
    def chunk_text_safe(self, text: str) -> List[str]:
        """Safe chunking with proper error handling."""
        if not text or len(text.strip()) < self.chunk_config.min_chunk_length:
            logger.warning("Text too short or empty for chunking")
            return []
        
        try:
            # Simple word-based chunking as fallback
            words = text.split()
            if len(words) == 0:
                return []
            
            chunks = []
            chunk_size = self.chunk_config.chunk_size
            overlap = self.chunk_config.overlap
            
            for i in range(0, len(words), max(1, chunk_size - overlap)):
                chunk_words = words[i:i + chunk_size]
                if len(chunk_words) > 0:  # Ensure chunk is not empty
                    chunk_text = ' '.join(chunk_words).strip()
                    if len(chunk_text) >= self.chunk_config.min_chunk_length:
                        chunks.append(chunk_text)
            
            # Ensure we have at least one chunk
            if not chunks and len(text.strip()) >= self.chunk_config.min_chunk_length:
                chunks = [text.strip()]
            
            logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in chunking: {e}")
            # Return the entire text as a single chunk if chunking fails
            if len(text.strip()) >= self.chunk_config.min_chunk_length:
                return [text.strip()]
            return []
    
    def chunk_text_recursive_safe(self, text: str) -> List[str]:
        """Safe recursive chunking with bounds checking."""
        if not text or len(text.strip()) < self.chunk_config.min_chunk_length:
            return []
        
        try:
            separators = ['\n\n', '\n', '. ', '! ', '? ', '; ', ', ', ' ']
            
            def safe_split_text(text: str, separators: List[str]) -> List[str]:
                if not separators or not text:
                    return [text] if text and len(text.strip()) >= self.chunk_config.min_chunk_length else []
                
                separator = separators[0]
                remaining_separators = separators[1:]
                
                if separator not in text:
                    return safe_split_text(text, remaining_separators)
                
                try:
                    parts = text.split(separator)
                    result = []
                    current_chunk = ""
                    
                    for part in parts:
                        if not part:  # Skip empty parts
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
                    
                except Exception as e:
                    logger.warning(f"Error in recursive splitting: {e}")
                    return safe_split_text(text, remaining_separators)
            
            chunks = safe_split_text(text, separators)
            
            # Ensure we have at least one valid chunk
            if not chunks and len(text.strip()) >= self.chunk_config.min_chunk_length:
                chunks = [text.strip()]
                
            return chunks
            
        except Exception as e:
            logger.error(f"Error in recursive chunking: {e}")
            return self.chunk_text_safe(text)  # Fallback to simple chunking
    
    def chunk_text(self, text: str) -> List[str]:
        """Main chunking method with error handling."""
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            logger.warning("No valid text to chunk after preprocessing")
            return []
        
        try:
            if self.chunk_config.strategy == "recursive":
                chunks = self.chunk_text_recursive_safe(processed_text)
            else:
                chunks = self.chunk_text_safe(processed_text)
            
            # Final validation
            valid_chunks = [chunk for chunk in chunks if chunk and len(chunk.strip()) >= self.chunk_config.min_chunk_length]
            
            if not valid_chunks:
                logger.warning("No valid chunks created, using entire text")
                if len(processed_text) >= self.chunk_config.min_chunk_length:
                    valid_chunks = [processed_text]
            
            return valid_chunks
            
        except Exception as e:
            logger.error(f"Error in chunk_text: {e}")
            # Last resort: return the entire text if it's long enough
            if len(processed_text) >= self.chunk_config.min_chunk_length:
                return [processed_text]
            return []
    
    def build_optimized_faiss_index(self, chunks: List[str]) -> Tuple[faiss.Index, np.ndarray, List[str]]:
        """Build FAISS index with comprehensive error handling."""
        if not chunks:
            raise ValueError("No chunks provided to build index")
        
        # Filter out empty or invalid chunks
        valid_chunks = [chunk for chunk in chunks if chunk and len(chunk.strip()) >= 3]
        
        if not valid_chunks:
            raise ValueError("No valid chunks after filtering")
        
        try:
            logger.info(f"Building FAISS index for {len(valid_chunks)} chunks")
            
            # Generate embeddings with error handling
            embeddings = self.model.encode(
                valid_chunks, 
                convert_to_numpy=True, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            if embeddings.size == 0:
                raise ValueError("Failed to generate embeddings")
            
            embeddings = embeddings.astype('float32')
            
            # Build index based on size
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
            logger.info(f"Successfully built FAISS index with {index.ntotal} vectors")
            
            return index, embeddings, valid_chunks
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            raise ValueError(f"Failed to build FAISS index: {e}")
    
    def retrieve_top_k_chunks(self, query_text: str, index: faiss.Index, chunks: List[str], 
                            embeddings: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Safe retrieval with bounds checking."""
        if not query_text or not chunks:
            return []
        
        try:
            processed_query = self.preprocess_text(query_text)
            if not processed_query:
                logger.warning("Empty query after preprocessing")
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
            
            # Safely extract results with bounds checking
            top_chunks = []
            for idx, score in zip(indices[0], scores[0]):
                if 0 <= idx < len(chunks):  # Bounds check
                    top_chunks.append((chunks[idx], float(score)))
            
            return top_chunks
            
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return []
    
    def calculate_comprehensive_score(self, cv_text: str, jd_text: str, top_k: int = 10) -> dict:
        """Calculate comprehensive score with extensive error handling."""
        try:
            # Validate inputs
            if not cv_text or not jd_text:
                return {"error": "Empty CV or JD text provided"}
            
            if len(cv_text.strip()) < 10 or len(jd_text.strip()) < 10:
                return {"error": "CV or JD text too short for meaningful analysis"}
            
            # Process texts
            cv_chunks = self.chunk_text(cv_text)
            jd_chunks = self.chunk_text(jd_text)
            
            if not cv_chunks:
                return {"error": "Unable to create valid chunks from CV text"}
            
            if not jd_chunks:
                return {"error": "Unable to create valid chunks from JD text"}
            
            logger.info(f"Processing {len(cv_chunks)} CV chunks and {len(jd_chunks)} JD chunks")
            
            # Build index
            cv_index, cv_embeddings, cv_chunk_list = self.build_optimized_faiss_index(cv_chunks)
            
            # Calculate similarities
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
            
            # Calculate metrics
            max_similarity = float(np.max(all_similarities))
            avg_similarity = float(np.mean(all_similarities))
            median_similarity = float(np.median(all_similarities))
            coverage_score = float(np.mean(jd_coverage_scores)) if jd_coverage_scores else 0.0
            
            # Weighted final score
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
            
        except Exception as e:
            logger.error(f"Error in comprehensive scoring: {e}")
            return {"error": f"Scoring failed: {str(e)}"}

# Backward compatibility functions
def chunk_text(text: str, chunk_size: int = 100) -> List[str]:
    try:
        engine = EnhancedRAGEngine(chunk_config=ChunkConfig(chunk_size=chunk_size, strategy="recursive"))
        return engine.chunk_text(text)
    except Exception as e:
        logger.error(f"Error in chunk_text wrapper: {e}")
        return []

def build_faiss_index(chunks: List[str]) -> Tuple[faiss.Index, np.ndarray, List[str]]:
    try:
        engine = EnhancedRAGEngine()
        return engine.build_optimized_faiss_index(chunks)
    except Exception as e:
        logger.error(f"Error in build_faiss_index wrapper: {e}")
        raise

def retrieve_top_k_chunks(query_text: str, index: faiss.Index, chunks: List[str], 
                         embeddings: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
    try:
        engine = EnhancedRAGEngine()
        return engine.retrieve_top_k_chunks(query_text, index, chunks, embeddings, k)
    except Exception as e:
        logger.error(f"Error in retrieve_top_k_chunks wrapper: {e}")
        return []
