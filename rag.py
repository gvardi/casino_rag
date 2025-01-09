__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import tiktoken
import json
import chromadb
import os
from chromadb.utils import embedding_functions
from openai import OpenAI
from typing import List, Dict, Tuple, Optional
from itertools import zip_longest
import logging
from dataclasses import dataclass
import re
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QAContext:
    """Data class to store Q&A context with metadata"""
    question: str
    answer: str
    qa_id: str
    source_file: str
    topic: Optional[str] = None
    references: List[str] = None
    section: Optional[str] = None

class DocumentProcessor:
    """Handles document preprocessing and structure analysis"""
    
    def __init__(self):
        self.topic_patterns = {
            "financial": r"financ|budget|cost|revenue|investment",
            "regulatory": r"regulat|compliance|law|requirement",
            "operational": r"operat|gaming|facility|casino",
            "licensing": r"licens|permit|approval",
        }
        self.reference_pattern = r"Q\.?\s*\d+|Question\s*\d+"
        
    def extract_topic(self, text: str) -> str:
        """Extract main topic based on keyword patterns"""
        for topic, pattern in self.topic_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return topic
        return "general"
        
    def extract_references(self, text: str) -> List[str]:
        """Extract referenced question numbers"""
        refs = re.findall(self.reference_pattern, text, re.IGNORECASE)
        return [ref.strip() for ref in refs]

    def process_qa_pair(self, qa_id: str, question: str, answer: str, source_file: str) -> QAContext:
        """Process a single Q&A pair to extract metadata"""
        # Extract topic and references
        combined_text = f"{question} {answer}"
        topic = self.extract_topic(combined_text)
        references = self.extract_references(combined_text)
        
        return QAContext(
            question=question,
            answer=answer,
            qa_id=qa_id,
            source_file=source_file,
            topic=topic,
            references=references
        )

class EnhancedRAGQuerySystem:
    def __init__(self, api_key: str, min_relevance_score: float = 0.4):
        """
        Initialize the enhanced RAG system
        
        Args:
            api_key (str): OpenAI API key
            min_relevance_score (float): Minimum relevance score threshold for context
        """
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-ada-002"
        )
        
        # Create multiple collections for different types of embeddings
        self.collections = {
            "full": self.chroma_client.get_or_create_collection(
                name="qa_collection_full",
                embedding_function=self.embedding_function
            ),
            "questions": self.chroma_client.get_or_create_collection(
                name="qa_collection_questions",
                embedding_function=self.embedding_function
            ),
            "answers": self.chroma_client.get_or_create_collection(
                name="qa_collection_answers",
                embedding_function=self.embedding_function
            )
        }
        
        self.client = OpenAI(api_key=api_key)
        self.min_relevance_score = min_relevance_score
        self.enc = tiktoken.encoding_for_model("gpt-4")
        self.doc_processor = DocumentProcessor()
        self.qa_contexts = {}  # Store processed QA contexts
        
    def chunk_text(self, text: str, max_tokens: int = 6000) -> List[str]:
        """
        Chunk text into smaller pieces that fit within token limit.
        Uses a smaller max_tokens to provide safety margin.
        """
        if not text.strip():
            return []
            
        try:
            tokens = self.enc.encode(text)
            chunks = []
            current_chunk = []
            current_length = 0
            
            for token in tokens:
                if current_length + 1 > max_tokens:
                    # Convert current chunk back to text
                    chunk_text = self.enc.decode(current_chunk)
                    if chunk_text.strip():  # Only add non-empty chunks
                        chunks.append(chunk_text)
                    current_chunk = [token]
                    current_length = 1
                else:
                    current_chunk.append(token)
                    current_length += 1
            
            # Add the last chunk if it exists and is non-empty
            if current_chunk:
                chunk_text = self.enc.decode(current_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text)
                    
            return chunks
        except Exception as e:
            logger.error(f"Error in chunk_text: {str(e)}")
            # Return original text as single chunk if chunking fails
            return [text] if text.strip() else []

    def batch_add_to_collection(self, collection, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """
        Add documents to collection in batches with proper error handling
        """
        batch_size = 50  # Adjust based on your needs
        
        # Pre-validate document lengths
        filtered_batches = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            # Filter out any empty documents
            valid_indices = [j for j, doc in enumerate(batch_docs) if doc.strip()]
            if valid_indices:
                filtered_batches.append((
                    [batch_docs[j] for j in valid_indices],
                    [batch_meta[j] for j in valid_indices],
                    [batch_ids[j] for j in valid_indices]
                ))
        
        # Process filtered batches
        for batch_docs, batch_meta, batch_ids in filtered_batches:
            try:
                # Double check none of the documents are too long
                for doc in batch_docs:
                    tokens = len(self.enc.encode(doc))
                    if tokens > 7000:
                        logger.warning(f"Document with {tokens} tokens found after chunking. Skipping.")
                        continue
                        
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_meta,
                    ids=batch_ids
                )
                logger.info(f"Successfully added batch of {len(batch_docs)} documents")
                
            except Exception as e:
                error_msg = str(e)
                if "maximum context length" in error_msg.lower():
                    logger.error("Token limit exceeded even after chunking. Skipping batch.")
                else:
                    logger.error(f"Error adding batch to collection: {error_msg}")
                # Log the problematic batch for debugging
                logger.debug(f"Problematic batch size: {len(batch_docs)}")
                # Continue with next batch
                continue

    def load_qa_data(self, json_paths: List[str]):
        """
        Load and process QA JSON files with enhanced preprocessing and chunking
        """
        full_documents = []
        question_documents = []
        answer_documents = []
        metadatas = []
        ids = []
        doc_id = 0
        
        for json_path in json_paths:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
                
                # Process each Q&A pair
                for qa_id, qa_pair in qa_data.items():
                    try:
                        # Process QA pair with metadata extraction
                        qa_context = self.doc_processor.process_qa_pair(
                            qa_id=qa_id,
                            question=qa_pair["question"],
                            answer=qa_pair["answer"],
                            source_file=json_path
                        )
                        
                        # Store processed context
                        self.qa_contexts[qa_id] = qa_context
                        
                        # Prepare documents for different embeddings
                        full_text = f"Question: {qa_context.question}\nAnswer: {qa_context.answer}"
                        
                        metadata = {
                            "qa_id": qa_id,
                            "topic": qa_context.topic,
                            "references": json.dumps(qa_context.references),
                            "source_file": json_path
                        }
                        
                        # Chunk the full text if needed
                        full_text_chunks = self.chunk_text(full_text)
                        question_chunks = self.chunk_text(qa_context.question)
                        answer_chunks = self.chunk_text(qa_context.answer)
                        
                        # Add each chunk to the document lists
                        for chunk_idx, (full_chunk, q_chunk, a_chunk) in enumerate(
                            zip_longest(full_text_chunks, question_chunks, answer_chunks, fillvalue="")):
                            
                            # Update metadata for chunk
                            chunk_metadata = metadata.copy()
                            chunk_metadata["chunk_index"] = chunk_idx
                            
                            full_documents.append(full_chunk)
                            question_documents.append(q_chunk)
                            answer_documents.append(a_chunk)
                            metadatas.append(chunk_metadata)
                            ids.append(f"doc_{doc_id}_chunk_{chunk_idx}")
                        
                        doc_id += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing QA pair {qa_id}: {str(e)}")
                        continue
                
            except Exception as e:
                logger.error(f"Error processing file {json_path}: {str(e)}")
                continue
        
        # Add documents to collections in batches
        if full_documents:
            for collection_name, documents in [
                ("full", full_documents),
                ("questions", question_documents),
                ("answers", answer_documents)
            ]:
                logger.info(f"Adding documents to {collection_name} collection...")
                self.batch_add_to_collection(
                    self.collections[collection_name],
                    documents,
                    metadatas,
                    ids
                )
            logger.info(f"Added {len(full_documents)} chunks to the vector database")
    
    def get_related_context(self, qa_id: str) -> List[str]:
        """Get context from related Q&As based on references"""
        if qa_id not in self.qa_contexts:
            return []
            
        context = self.qa_contexts[qa_id]
        related_contexts = []
        
        # Add referenced Q&As
        if context.references:
            for ref in context.references:
                ref_id = ref.replace('Q', '').replace('.', '').strip()
                if ref_id in self.qa_contexts:
                    ref_context = self.qa_contexts[ref_id]
                    related_contexts.append(
                        f"Referenced Q{ref_id}:\nQuestion: {ref_context.question}\nAnswer: {ref_context.answer}"
                    )
                    
        return related_contexts

    def query(self, question: str, n_results: int = 5) -> str:
        """
        Enhanced query processing with multi-stage retrieval
        """
        try:
            # Stage 1: Initial broad search across different embeddings
            results = {}
            for collection_name, collection in self.collections.items():
                results[collection_name] = collection.query(
                    query_texts=[question],
                    n_results=n_results
                )
            
            # Stage 2: Combine and deduplicate results
            seen_qa_ids = set()
            combined_contexts = []
            
            for collection_name, result in results.items():
                documents = result.get('documents', [])[0] if result.get('documents') else []
                metadatas = result.get('metadatas', [])[0] if result.get('metadatas') else []
                
                for doc, metadata in zip(documents, metadatas):
                    qa_id = metadata.get('qa_id')
                    if qa_id not in seen_qa_ids:
                        seen_qa_ids.add(qa_id)
                        combined_contexts.append(doc)
                        
                        # Add related context
                        related_contexts = self.get_related_context(qa_id)
                        combined_contexts.extend(related_contexts)
            
            if not combined_contexts:
                return "I couldn't find relevant information in the context. Please try rephrasing your question."
            
            # Stage 3: Generate response with enhanced prompting
            context = "\n\n---\n\n".join(combined_contexts)
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a knowledgeable assistant specialized in interpreting regulatory "
                        "documents and Q&As. When answering:\n"
                        "1. Synthesize information from multiple related answers when available\n"
                        "2. Maintain regulatory accuracy and context\n"
                        "3. Acknowledge any cross-references\n"
                        "4. Clearly state if information comes from multiple sources\n"
                        "Base your answer only on the provided context."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question: {question}\n\n"
                        f"Please provide a comprehensive answer based on the context provided, "
                        f"synthesizing information from multiple sources if applicable."
                    )
                }
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in query processing: {str(e)}")
            return "Sorry, I encountered an error while processing your query."

def main():
    # Initialize with your OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
        
    # Initialize enhanced RAG system
    rag_system = EnhancedRAGQuerySystem(api_key)
    
    # Load QA data
    input_jsons = ["qa_output2023.json", "qa_output2024.json"]
    logger.info("Loading QA data into vector database...")
    rag_system.load_qa_data(input_jsons)
    
    # Example query
    question = "What specific factors will the Board consider?"
    logger.info(f"Testing with question: {question}")
    answer = rag_system.query(question)
    print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()
