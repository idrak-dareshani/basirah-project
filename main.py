import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle
from dataclasses import dataclass
from datetime import datetime

# Core libraries for RAG system
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TafsirDocument:
    """Data class for Tafsir documents"""
    surah_no: int
    ayah_no: int
    author: str
    text: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None
    
class ArabicTafsirRAG:
    """
    RAG System for Arabic Tafsir Data
    
    This system processes Arabic Tafsir texts and metadata to create
    a searchable knowledge base with semantic search capabilities.
    """
    
    def __init__(self, 
                 data_path: str = "data",
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 index_path: str = "tafsir_index",
                 use_gpu: bool = True):
        """
        Initialize the RAG system
        
        Args:
            data_path: Path to the main data folder
            model_name: Sentence transformer model for embeddings
            index_path: Path to save/load the FAISS index
            use_gpu: Whether to use GPU for processing
        """
        self.data_path = Path(data_path)
        self.index_path = Path(index_path)
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(model_name)
        if self.use_gpu:
            self.embedding_model = self.embedding_model.cuda()
        
        # Storage for documents and embeddings
        self.documents: List[TafsirDocument] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.Index] = None
        
        # Create index directory if it doesn't exist
        self.index_path.mkdir(exist_ok=True)
        
        logger.info(f"RAG System initialized with model: {model_name}")
        logger.info(f"GPU available: {self.use_gpu}")
    
    def load_author_data(self, author_path: Path) -> List[TafsirDocument]:
        """
        Load data for a specific author
        
        Args:
            author_path: Path to author's folder
            
        Returns:
            List of TafsirDocument objects
        """
        author_name = author_path.name
        documents = []
        
        logger.info(f"Loading data for author: {author_name}")
        
        # Get all JSON files (metadata files)
        json_files = list(author_path.glob("*.json"))
        
        for json_file in json_files:
            try:
                # Extract surah number from filename
                surah_no = int(json_file.stem)
                
                # Load metadata
                with open(json_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Find corresponding text files
                text_files = list(author_path.glob(f"{surah_no}_*.txt"))
                
                for text_file in text_files:
                    try:
                        # Extract ayah number from filename
                        ayah_no = int(text_file.stem.split('_')[1])
                        
                        # Load text content
                        with open(text_file, 'r', encoding='utf-8') as f:
                            text_content = f.read().strip()
                        
                        # Create document object
                        doc = TafsirDocument(
                            surah_no=surah_no,
                            ayah_no=ayah_no,
                            author=author_name,
                            text=text_content,
                            metadata=metadata
                        )
                        documents.append(doc)
                        
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error processing {text_file}: {e}")
                        continue
                        
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning(f"Error processing {json_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(documents)} documents for {author_name}")
        return documents
    
    def load_all_data(self) -> None:
        """Load data from all authors"""
        logger.info("Loading all Tafsir data...")
        
        author_folders = [p for p in self.data_path.iterdir() if p.is_dir()]
        
        for author_path in author_folders:
            author_documents = self.load_author_data(author_path)
            self.documents.extend(author_documents)
        
        logger.info(f"Total documents loaded: {len(self.documents)}")
    
    def create_embeddings(self, batch_size: int = 32) -> None:
        """
        Create embeddings for all documents
        
        Args:
            batch_size: Batch size for processing
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_all_data() first.")
        
        logger.info("Creating embeddings for all documents...")
        
        # Extract texts for embedding
        texts = [doc.text for doc in self.documents]
        
        # Create embeddings in batches
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        self.embeddings = np.vstack(embeddings)
        
        # Store embeddings in document objects
        for i, doc in enumerate(self.documents):
            doc.embedding = self.embeddings[i]
        
        logger.info(f"Created embeddings with shape: {self.embeddings.shape}")
    
    def build_faiss_index(self) -> None:
        """Build FAISS index for fast similarity search"""
        if self.embeddings is None:
            raise ValueError("Embeddings not created. Call create_embeddings() first.")
        
        logger.info("Building FAISS index...")
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Add embeddings to index
        self.faiss_index.add(self.embeddings)
        
        logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def save_index(self) -> None:
        """Save the index and documents to disk"""
        logger.info("Saving index to disk...")
        
        # Save FAISS index
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(self.index_path / "faiss_index.bin"))
        
        # Save documents and embeddings
        with open(self.index_path / "documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'num_documents': len(self.documents),
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else None,
            'created_at': datetime.now().isoformat()
        }
        
        with open(self.index_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Index saved successfully")
    
    def load_index(self) -> bool:
        """
        Load the index from disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Loading index from disk...")
            
            # Load FAISS index
            faiss_path = self.index_path / "faiss_index.bin"
            if faiss_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_path))
            
            # Load documents
            documents_path = self.index_path / "documents.pkl"
            if documents_path.exists():
                with open(documents_path, 'rb') as f:
                    self.documents = pickle.load(f)
            
            # Extract embeddings from documents
            if self.documents and self.documents[0].embedding is not None:
                self.embeddings = np.array([doc.embedding for doc in self.documents])
            
            logger.info("Index loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def search(self, query: str, k: int = 5, author_filter: Optional[str] = None) -> List[Dict]:
        """
        Search for relevant Tafsir passages
        
        Args:
            query: Search query
            k: Number of results to return
            author_filter: Optional author name to filter results
            
        Returns:
            List of search results with metadata
        """
        if self.faiss_index is None:
            raise ValueError("Index not built. Call build_faiss_index() first.")
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        search_k = k * 10 if author_filter else k  # Get more results if filtering
        scores, indices = self.faiss_index.search(query_embedding, search_k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # No more results
                break
            
            doc = self.documents[idx]
            
            # Apply author filter if specified
            if author_filter and doc.author != author_filter:
                continue
            
            result = {
                'score': float(score),
                'surah_no': doc.surah_no,
                'ayah_no': doc.ayah_no,
                'author': doc.author,
                'text': doc.text,
                'metadata': doc.metadata
            }
            results.append(result)
            
            if len(results) >= k:
                break
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        if not self.documents:
            return {"message": "No documents loaded"}
        
        # Count documents by author
        author_counts = {}
        surah_counts = {}
        
        for doc in self.documents:
            author_counts[doc.author] = author_counts.get(doc.author, 0) + 1
            surah_counts[doc.surah_no] = surah_counts.get(doc.surah_no, 0) + 1
        
        stats = {
            'total_documents': len(self.documents),
            'total_authors': len(author_counts),
            'total_surahs': len(surah_counts),
            'authors': author_counts,
            'model_name': self.model_name,
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else None,
            'index_built': self.faiss_index is not None
        }
        
        return stats
    
    def get_random_sample(self, n: int = 3) -> List[Dict]:
        """Get random sample of documents for inspection"""
        if not self.documents:
            return []
        
        import random
        sample_docs = random.sample(self.documents, min(n, len(self.documents)))
        
        result = []
        for doc in sample_docs:
            # Handle different metadata formats
            if isinstance(doc.metadata, dict):
                metadata_info = list(doc.metadata.keys())
            elif isinstance(doc.metadata, list):
                metadata_info = f"List with {len(doc.metadata)} items"
            else:
                metadata_info = f"Type: {type(doc.metadata).__name__}"
            
            result.append({
                'surah_no': doc.surah_no,
                'ayah_no': doc.ayah_no,
                'author': doc.author,
                'text': doc.text[:200] + "..." if len(doc.text) > 200 else doc.text,
                'metadata_info': metadata_info
            })
        
        return result

# Example usage and testing
if __name__ == "__main__":
    # Initialize RAG system
    rag = ArabicTafsirRAG(
        data_path="data",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # # Build index (first time only)
    # rag.load_all_data()
    # rag.create_embeddings()
    # rag.build_faiss_index()
    # rag.save_index()

    try:
        # Try to load existing index
        if not rag.load_index():
            print("No existing index found. Building new index...")
            
            # Load all data
            rag.load_all_data()
            
            if not rag.documents:
                print("No documents loaded. Please check your data structure.")
                exit(1)
            
            # Create embeddings
            rag.create_embeddings()
            
            # Build FAISS index
            rag.build_faiss_index()
            
            # Save index
            rag.save_index()
            
        # Display statistics
        stats = rag.get_statistics()
        print("\n=== System Statistics ===")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        
        # # Show sample documents
        # print("\n=== Sample Documents ===")
        # samples = rag.get_random_sample(3)
        # for i, sample in enumerate(samples, 1):
        #     print(f"\nSample {i}:")
        #     print(f"Author: {sample['author']}")
        #     print(f"Surah: {sample['surah_no']}, Ayah: {sample['ayah_no']}")
        #     print(f"Text: {sample['text']}")
        #     print(f"Metadata info: {sample['metadata_info']}")
            
            # # Show actual metadata structure for debugging
            # if i == 1:  # Show metadata structure for first sample only
            #     doc = rag.documents[0]
            #     print(f"Metadata type: {type(doc.metadata)}")
            #     if isinstance(doc.metadata, list) and len(doc.metadata) > 0:
            #         print(f"First metadata item: {doc.metadata[0]}")
            #         print(f"First metadata item type: {type(doc.metadata[0])}")
            #     elif isinstance(doc.metadata, dict):
            #         print(f"Metadata keys: {list(doc.metadata.keys())}")
            #     else:
            #         print(f"Metadata content: {doc.metadata}")
        
        # Example search
        print("\n=== Example Search ===")
        query = "الجنة"  # Prayer in Arabic
        results = rag.search(query, k=3)
        
        print(f"Search query: {query}")
        print(f"Found {len(results)} results:")
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result['score']:.4f}):")
            print(f"Author: {result['author']}")
            print(f"Surah: {result['surah_no']}, Ayah: {result['ayah_no']}")
            print(f"Text: {result['text'][:150]}...")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise