import sqlite3

import tiktoken
import json
import chromadb
import os
from chromadb.utils import embedding_functions
from openai import OpenAI
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError, Future

class RAGQuerySystem:
    def __init__(self, api_key: str, min_relevance_score: float = 0.7):
        """
        Initialize the improved RAG system
        
        Args:
            api_key (str): OpenAI API key
            min_relevance_score (float): Minimum relevance score threshold for context
        """
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")  
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-ada-002"
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="qa_collection",
            embedding_function=self.embedding_function
        )
        self.client = OpenAI(api_key=api_key)
        self.min_relevance_score = min_relevance_score
        self.enc = tiktoken.encoding_for_model("gpt-4")

    def chunk_text_by_tokens(self, 
                           text: str,
                           chunk_size: int = 6000,
                           overlap: int = 200) -> List[str]:
        """
        Chunk text by token count for more accurate splitting
        """
        tokens = self.enc.encode(text)
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunks.append(self.enc.decode(chunk_tokens))
            start += chunk_size - overlap
            
        return chunks

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _embed_and_store(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Retry wrapper for embedding and storage"""
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def load_qa_data(self, json_paths: List[str]):
        """
        Load and process QA JSON files with improved chunking and error handling
        """
        documents = []
        metadatas = []
        ids = []
        doc_id = 0

        for json_path in json_paths:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)

                for qa_id, qa_pair in qa_data.items():
                    question = qa_pair["question"]
                    answer = qa_pair["answer"]
                    context = f"Question: {question}\nAnswer: {answer}"

                    chunked_contexts = self.chunk_text_by_tokens(context)

                    for chunk_context in chunked_contexts:
                        metadata = {
                            "question": question,
                            "answer": answer,
                            "source_file": json_path,
                            "chunk_id": f"{qa_id}_chunk_{doc_id}"
                        }

                        documents.append(chunk_context)
                        metadatas.append(metadata)
                        ids.append(f"doc_{doc_id}")
                        doc_id += 1

                print(f"Processed {json_path}")

            except Exception as e:
                print(f"Error processing {json_path}: {str(e)}")
                continue

        # Batch process documents
        if documents:
            try:
                self._embed_and_store(documents, metadatas, ids)
                print(f"Added {len(documents)} documents to the vector database")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error adding documents to collection: {e}")

    def query(self, question: str, n_results: int = 5) -> str:
        """Query with basic error handling instead of retries"""
        """
        Query the system with improved context management and retry logic
        """
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )

        # Get documents
        documents = results['documents'][0]

        if not documents:
            return "I cannot find relevant information in the context to answer this question accurately."

        # Prepare context
        context = "\n\n".join(documents)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Use the provided context to answer the question. "
                    "Base your answer only on the provided context. If the context doesn't contain "
                    "relevant information, say so."
                )
            },
            {
                "role": "user", 
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\n"
                           f"Please answer the question based on the context provided, "
                           f"citing relevance scores where helpful."
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"Error in GPT query: {str(e)}")
            return "Sorry, I encountered an error while generating the answer."

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection"""
        return {
            "total_documents": len(self.collection.get()["ids"]),
            "embedding_function": self.embedding_function.model_name
        }
    
def main():
    # Provide your OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")

    # File paths
    input_jsons = [
        "qa_output2023.json",
        "qa_output2024.json"
    ]

    try:
        # Initialize RAG system
        rag_system = RAGQuerySystem(api_key)

        # Load QA data
        print("Loading QA data into vector database...")
        rag_system.load_qa_data(input_jsons)

        # Example query
        question = "According to the documents, what are the requirements for a gaming facility to commence operations?"
        print(f"\nTesting with question: {question}")
        answer = rag_system.query(question)
        print(f"\nAnswer: {answer}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
