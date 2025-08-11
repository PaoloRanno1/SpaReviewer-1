import logging
from typing import List, Optional, Dict

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

class ChunkingError(RuntimeError):
    """Custom exception for errors during document chunking."""
    pass

class EmbeddingError(RuntimeError):
    """Custom exception for errors during embedding."""
    pass

###############################################################
#####################################################################

## This class is used to chunk an SPA file using semantic chunking
class DocumentChunker:
    def __init__(
        self,
        pdf_path: str,
        SPA_name: str,
        min_chunk_size: int = 1100,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 0.8,
        recursive_chunk_size: int = 2000,
        recursive_chunk_overlap: int = 200,
        max_semantic_chunk_size: Optional[int] = None,
        api_key: Optional[str] = None
    ):
        """
        Initializes the DocumentChunker.

        :param pdf_path: Path to the PDF file to load.
        :param SPA_name: Name of the SPA document for metadata.
        :param min_chunk_size: Minimum size for merged chunks (default: 1100).
        :param breakpoint_threshold_type: Type of breakpoint threshold for semantic chunking (default: "percentile").
        :param breakpoint_threshold_amount: Amount for the breakpoint threshold (default: 0.8).
        :param recursive_chunk_size: Max size for recursive splitting (default: 2000).
        :param recursive_chunk_overlap: Overlap for recursive splitting (default: 200).
        :param max_semantic_chunk_size: Optional max size for semantic chunks; if set, large chunks are split recursively.
        :param api_key: Optional Google API key for embeddings.
        """
        try:
            self.loader = PyMuPDFLoader(pdf_path)
            self.documents = self.loader.load()
        except Exception as e:
            raise ValueError(f"Failed to load PDF: {e}")
        if not self.documents:
            logger.warning("No documents loaded from PDF.")
        self.SPA_name = SPA_name
        self.embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        self.chunker = SemanticChunker(
            self.embedder,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount
        )
        self.min_chunk_size = min_chunk_size
        self.recursive_chunk_size = recursive_chunk_size
        self.recursive_chunk_overlap = recursive_chunk_overlap
        self.max_semantic_chunk_size = max_semantic_chunk_size

    def chunk_documents(self) -> List[Document]:
        """
        Chunks the loaded documents semantically, merges small chunks, applies overlap, and adds metadata.

        :return: List of chunked Document objects.
        """
        if not self.documents:
            logger.info("No documents to chunk.")
            return []
        logger.info("Chunking documents...")
        try:
            chunks = self.chunker.split_documents(self.documents)
            logger.info(f"Semantic chunking produced {len(chunks)} chunks.")
        except Exception as e:
            raise ChunkingError(f"Semantic chunking failed: {e}")
        # Optionally split large semantic chunks
        if self.max_semantic_chunk_size:
            large_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.max_semantic_chunk_size,
                chunk_overlap=0,
                separators=["\n\n", "\n", " ", ""]
            )
            adjusted_chunks = []
            for chunk in chunks:
                if len(chunk.page_content) > self.max_semantic_chunk_size:
                    adjusted_chunks.extend(large_splitter.split_documents([chunk]))
                else:
                    adjusted_chunks.append(chunk)
            chunks = adjusted_chunks
            logger.info(f"After splitting large chunks: {len(chunks)} chunks.")
        logger.info("Merging small chunks...")
        merged_chunks = []
        current_chunk = ""
        current_metadata = {}
        for chunk in chunks:
            if not current_chunk:
                current_metadata = chunk.metadata.copy()
            if current_chunk and len(current_chunk) + len(chunk.page_content) < self.min_chunk_size:
                current_chunk += " " + chunk.page_content
            else:
                if current_chunk:
                    new_chunk = chunk.model_copy()
                    new_chunk.page_content = current_chunk
                    new_chunk.metadata = current_metadata
                    merged_chunks.append(new_chunk)
                current_chunk = chunk.page_content
                current_metadata = chunk.metadata.copy()
        if current_chunk:
            last_chunk = chunks[-1].model_copy() if chunks else None
            if last_chunk:
                last_chunk.page_content = current_chunk
                last_chunk.metadata = current_metadata
                merged_chunks.append(last_chunk)
        logger.info(f"Merged into {len(merged_chunks)} chunks.")
        overlap_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.recursive_chunk_size,
            chunk_overlap=self.recursive_chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        final_chunks = overlap_splitter.split_documents(merged_chunks)
        logger.info(f"Final chunks after overlap splitting: {len(final_chunks)}.")
        # Add SPA_name metadata
        for chunk in final_chunks:
            chunk.metadata["document_name"] = self.SPA_name
        return final_chunks

###############################################################
#####################################################################

class VectorStoreEmbedder:
    def __init__(self, persist_directory: str = "DATABASE_SPA", api_key: Optional[str] = None):
        """
        Initializes the VectorStoreEmbedder.

        :param persist_directory: Directory for persisting the Chroma store (default: "ChromaSPADATABASE").
        :param api_key: Optional Google API key for embeddings.
        """
        self.persist_directory = persist_directory
        self.doc_embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="RETRIEVAL_DOCUMENT",
            google_api_key=api_key
        )
        self.query_embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="RETRIEVAL_QUERY",
            google_api_key=api_key
        )
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.query_embeddings_model
        )

    def embed_chunks(self, chunks: List[Document], overwrite: bool = False) -> Chroma:
        """
        Embeds chunks into the Chroma store, skipping if the document already exists unless overwrite is True.

        :param chunks: List of Document objects to embed.
        :param overwrite: If True, delete existing entries for this document before embedding (default: False).
        :return: The updated Chroma vector store.
        """
        if not chunks:
            logger.warning("No chunks provided to embed.")
            return self.vector_store
        document_name = chunks[0].metadata.get("document_name")
        if not document_name:
            logger.warning("No 'document_name' metadata found; proceeding with embedding.")
        else:
            existing = self.vector_store.get(where={"document_name": document_name})
            if existing and len(existing['ids']) > 0:
                if overwrite:
                    logger.info(f"Overwriting existing embeddings for '{document_name}'.")
                    self.delete_by_metadata({"document_name": document_name})
                else:
                    logger.info(f"SPA '{document_name}' already embedded; skipping.")
                    return self.vector_store
        logger.info(f"Embedding {len(chunks)} chunks for '{document_name or 'unknown'}'...")
        try:
            texts = [c.page_content for c in chunks]
            metadatas = [c.metadata for c in chunks]
            vectors = self.doc_embeddings_model.embed_documents(texts)
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                embeddings=vectors
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to embed chunks: {e}")
        logger.info("Chunks embedded successfully.")
        return self.vector_store

    def delete_by_metadata(self, filter: Dict[str, str]) -> None:
        """
        Deletes documents from the vector store based on metadata filter.

        :param filter: Metadata filter for deletion (e.g., {"document_name": "SPA_x"}).
        """
        try:
            self.vector_store.delete(where=filter)
            logger.info(f"Deleted documents matching filter: {filter}.")
        except Exception as e:
            raise EmbeddingError(f"Failed to delete by metadata: {e}")

###############################################################
#####################################################################

# The following class is used to retrieve documents from the vector store

class VectorStoreRetriever:
    def __init__(self, persist_directory: str = "DATABASE_SPA", api_key: Optional[str] = None):
        """
        Initializes the VectorStoreRetriever.

        :param persist_directory: Directory for the Chroma store (default: "ChromaSPADATABASE").
        :param api_key: Optional Google API key for embeddings.
        """
        self.persist_directory = persist_directory
        self.query_embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="RETRIEVAL_QUERY",
            google_api_key=api_key
        )
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.query_embeddings_model
        )

    def query_similarity(self, query: str, k: int = 5, filter: Optional[Dict[str, str]] = None) -> List[Document]:
        """
        Performs similarity search on the vector store.

        :param query: Query string.
        :param k: Number of results to return (default: 5).
        :param filter: Optional metadata filter.
        :return: List of retrieved Document objects.
        """
        logger.info(f"Querying with similarity search: '{query}'")
        try:
            results = self.vector_store.similarity_search(query, k=k, filter=filter)
            logger.info(f"Retrieved {len(results)} results.")
            return results
        except Exception as e:
            raise RuntimeError(f"Similarity query failed: {e}")

    def query_with_scores(self, query: str, k: int = 5, filter: Optional[Dict[str, str]] = None) -> List[tuple[Document, float]]:
        """
        Performs similarity search and returns results with scores.

        :param query: Query string.
        :param k: Number of results to return (default: 5).
        :param filter: Optional metadata filter.
        :return: List of (Document, score) tuples.
        """
        logger.info(f"Querying with similarity search and scores: '{query}'")
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k, filter=filter)
            logger.info(f"Retrieved {len(results)} results with scores.")
            return results
        except Exception as e:
            raise RuntimeError(f"Query with scores failed: {e}")

    def query_mmr(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        """
        Performs Max Marginal Relevance search for diversified results.

        :param query: Query string.
        :param k: Number of results to return (default: 5).
        :param fetch_k: Initial documents to fetch (default: 20).
        :param lambda_mult: Diversity factor (0=max diversity, 1=max relevance; default: 0.5).
        :param filter: Optional metadata filter.
        :return: List of retrieved Document objects.
        """
        logger.info(f"Querying with MMR search: '{query}'")
        try:
            results = self.vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
            )
            logger.info(f"Retrieved {len(results)} diversified results.")
            return results
        except Exception as e:
            raise RuntimeError(f"MMR query failed: {e}")

    def query_by_vector(self, query_vector: List[float], k: int = 5, filter: Optional[Dict[str, str]] = None) -> List[Document]:
        """
        Performs similarity search using a pre-embedded query vector.

        :param query_vector: Embedded query vector.
        :param k: Number of results to return (default: 5).
        :param filter: Optional metadata filter.
        :return: List of retrieved Document objects.
        """
        logger.info("Querying by vector.")
        try:
            results = self.vector_store.similarity_search_by_vector(query_vector, k=k, filter=filter)
            logger.info(f"Retrieved {len(results)} results by vector.")
            return results
        except Exception as e:
            raise RuntimeError(f"Vector query failed: {e}")

# Example usage:
# chunker = DocumentChunker(pdf_path="example.pdf", SPA_name="ExampleSPA")
# chunks = chunker.chunk_documents()
# embedder = VectorStoreEmbedder()
# embedder.embed_chunks(chunks, overwrite=True)
# retriever = VectorStoreRetriever()
# results = retriever.query_similarity("What is the purchase price?", k=3, filter={"document_name": "ExampleSPA"})

####################################
####################################
# The Following class defines a CHATBOT for Q&A Retrieval
# Make sure these are imported
# import logging
# from typing import Optional, Dict, List
# from langchain_core.documents import Document
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.prompts import PromptTemplate

# Make sure these are imported
import logging
from typing import Optional, Dict, List
# from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate

# logger = logging.getLogger(__name__)

class QAChatbot:
    def __init__(self, retriever: 'VectorStoreRetriever', model="gemini-2.5-flash", api_key: Optional[str] = None):
        self.llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.1)

        # 1. Main prompt - This remains the same, instructing the LLM on its overall task.
        self.main_prompt_template = """
        You are a specialized AI assistant for analyzing Share Purchase Agreements (SPAs). Your primary function is to accurately answer questions based EXCLUSIVELY on the provided context.

        ### Instructions:
        1.  **Strictly Adhere to Context:** Base your answer solely on the text provided in the 'Context' blocks.
        2.  **Cite Your Sources:** After providing a direct answer and a supporting quote, you MUST cite the SPA document and page number (e.g., "SPA Name: InterIT, Page: 15").
        3.  **Handle Missing Information:** If the answer cannot be found, state: "The provided context does not contain information on this topic."
        4.  **No Legal Advice:** You are an information retrieval tool, not a legal advisor.

        ---
        ### Context:
        {context}

        ### Question:
        {question}

        ### Answer:
        """
        prompt = PromptTemplate.from_template(self.main_prompt_template)

        # 2. NEW: Document Prompt - This defines how EACH document is formatted BEFORE being stuffed into the main prompt.
        #    This is where we include the metadata.
        #    NOTE: The variable names {source} and {page} MUST match the keys in your Document's metadata dictionary.
        #    From your error log, I see you have 'source' and 'page' keys.
        document_prompt = PromptTemplate.from_template(
            "--- SPA Name: {document_name}, Page: {page} ---\nContent: {page_content}\n---------------------------------"
        )

        # 3. CORRECTED CHAIN: We now pass the document_prompt to the chain constructor.
        self.qa_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt,
            document_prompt=document_prompt, # This is the key addition
        )

        self.retriever = retriever


    def answer_question(self, query: str, k: int = 10, filter: Optional[Dict[str, str]] = None) -> Dict:
        logger.info(f"Answering question: '{query}'")

        # 1. Retrieve relevant documents (this remains the same)
        docs = self.retriever.query_similarity(query, k=k, filter=filter)

        if not docs:
            logger.warning("No documents were retrieved for the query.")
            return {
                "question": query,
                "output_text": "I could not find any relevant information to answer your question.",
                "input_documents": []
            }

        # 2. REVERTED: We no longer need to manually format the context.
        #    We pass the raw list of Document objects directly.
        try:
            answer = self.qa_chain.invoke({"context": docs, "question": query})

            return {
                "question": query,
                "output_text": answer,
                "input_documents": docs
            }
        except Exception as e:
            logger.error(f"Failed to get answer from QA chain: {e}")
            raise


################
################
# Multi SPA Chatbot:
class MultiSPAChatbot:
    def __init__(self, retriever: 'VectorStoreRetriever', model="gemini-2.5-flash", api_key: Optional[str] = None):
        """
        Initialize the Multi-SPA Chatbot for comparative analysis.

        :param retriever: VectorStoreRetriever instance
        :param model: Google Generative AI model name
        :param api_key: Google API key
        """
        self.llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.1)
        self.retriever = retriever

        # Comparative analysis prompt template
        self.comparative_prompt_template = """
        # Persona
        You are a specialized AI assistant for comparative analysis of Share Purchase Agreements (SPAs). You work at Strada Partners, a mid cap Private equity firm with 250 M AUM located in Belgium, Antwerp.
         # Task
         Your task is to analyze and compare multiple SPAs based on the provided context.

        ### Instructions:
        1. **Strictly Adhere to Context:** Base your analysis solely on the text provided in the 'Context' sections below.
        2. **Structured Comparison:** Organize your response to clearly compare the requested aspects across all SPAs.
        3. **Cite Sources:** For each point, cite the specific SPA document and page number (e.g., "SPA Name: InterIT, Page: 15").
        4. **Handle Missing Information:** If information is missing for any SPA, explicitly state which SPAs lack the requested information.
        5. **Comparative Format:** Structure your answer to highlight similarities and differences between the SPAs.
        6. **No Legal Advice:** You are an information analysis tool, not a legal advisor.

        ### SPAs Being Analyzed:
        {spa_list}

        ---
        ### Context from All SPAs:
        {context}

        ### Comparative Question:
        {question}

        ### Comparative Analysis:
        Please provide a structured comparison addressing the question. Format your response as follows:
        - **Summary:** Detailed overview of findings across all SPAs
        - **Individual SPA Analysis:** Detailed findings for each SPA
        - **Key Differences:** Main distinctions between the SPAs
        - **Key Similarities:** Common elements across the SPAs
        """

        prompt = PromptTemplate.from_template(self.comparative_prompt_template)

        # Document prompt for better context formatting
        document_prompt = PromptTemplate.from_template(
            "\n=== SPA: {document_name} | Page: {page} ===\n{page_content}\n" + "="*50
        )

        # Create the chain for comparative analysis
        self.comparative_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt,
            document_prompt=document_prompt,
        )

    def answer_comparative_question(
        self,
        query: str,
        spa_names: List[str],
        k_per_spa: int = 5,
        retrieval_method: str = "similarity"
    ) -> Dict:
        """
        Answer a comparative question across multiple SPAs.

        :param query: The comparative question to ask
        :param spa_names: List of SPA names to compare
        :param k_per_spa: Number of documents to retrieve per SPA
        :param retrieval_method: Method to use for retrieval ("similarity", "mmr")
        :return: Dictionary containing the question, answer, and supporting documents
        """
        logger.info(f"Answering comparative question across {len(spa_names)} SPAs: '{query}'")

        all_documents = []
        spa_document_counts = {}

        # Retrieve documents from each SPA
        for spa_name in spa_names:
            logger.info(f"Retrieving documents from SPA: {spa_name}")

            # Create filter for this specific SPA
            spa_filter = {"document_name": spa_name}

            # Retrieve documents using the specified method
            if retrieval_method == "similarity":
                spa_docs = self.retriever.query_similarity(query, k=k_per_spa, filter=spa_filter)
            elif retrieval_method == "mmr":
                spa_docs = self.retriever.query_mmr(query, k=k_per_spa, filter=spa_filter)
            else:
                raise ValueError(f"Unsupported retrieval method: {retrieval_method}")

            spa_document_counts[spa_name] = len(spa_docs)
            all_documents.extend(spa_docs)

            logger.info(f"Retrieved {len(spa_docs)} documents from {spa_name}")

        # Check if we have documents from all requested SPAs
        missing_spas = [spa for spa, count in spa_document_counts.items() if count == 0]
        if missing_spas:
            logger.warning(f"No documents found for SPAs: {missing_spas}")

        if not all_documents:
            logger.warning("No documents were retrieved for any of the specified SPAs.")
            return {
                "question": query,
                "spa_names": spa_names,
                "output_text": f"I could not find any relevant information in the specified SPAs ({', '.join(spa_names)}) to answer your question.",
                "input_documents": [],
                "document_counts_per_spa": spa_document_counts
            }

        # Create SPA list string for the prompt
        spa_list_str = ", ".join(spa_names)

        try:
            # Generate comparative analysis
            answer = self.comparative_chain.invoke({
                "context": all_documents,
                "question": query,
                "spa_list": spa_list_str
            })

            return {
                "question": query,
                "spa_names": spa_names,
                "output_text": answer,
                "input_documents": all_documents,
                "document_counts_per_spa": spa_document_counts,
                "total_documents": len(all_documents)
            }

        except Exception as e:
            logger.error(f"Failed to get comparative answer: {e}")
            raise

    def get_available_spas(self) -> List[str]:
        """
        Get a list of all available SPA names in the database.

        :return: List of available SPA document names
        """
        try:
            # Query all documents to get unique document names
            all_docs = self.retriever.vector_store.get()

            if 'metadatas' in all_docs and all_docs['metadatas']:
                document_names = set()
                for metadata in all_docs['metadatas']:
                    if 'document_name' in metadata:
                        document_names.add(metadata['document_name'])
                return sorted(list(document_names))
            else:
                logger.warning("No documents found in the database.")
                return []

        except Exception as e:
            logger.error(f"Failed to retrieve available SPAs: {e}")
            return []

    def validate_spa_names(self, spa_names: List[str]) -> Dict[str, bool]:
        """
        Validate that the provided SPA names exist in the database.

        :param spa_names: List of SPA names to validate
        :return: Dictionary mapping SPA names to their existence status
        """
        available_spas = self.get_available_spas()
        return {spa_name: spa_name in available_spas for spa_name in spa_names}


############################################################
############################################################

'''
Tool to visualize the SPA Database
'''


def Database_counter_deluxe(path="DATABASE_SPA", show_chart=True):
    import chromadb
    from collections import Counter
    from IPython.display import display, Markdown, HTML
    import pandas as pd

    # Optional: Import plotting libraries if available
    if show_chart:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.style.use('default')
        except ImportError:
            show_chart = False
            print("⚠️ Matplotlib not available. Skipping charts.")

    client = chromadb.PersistentClient(path=path)
    collection = client.get_collection(name="langchain")
    results = collection.get(include=["metadatas"])

    document_names = []
    for metadata in results['metadatas']:
        if metadata and 'document_name' in metadata:
            document_names.append(metadata['document_name'])

    # Get unique document names
    unique_spa_names = list(set(document_names))
    unique_spa_names.sort()

    # Count occurrences of each document
    document_counts = Counter(document_names)

    # Create styled markdown content
    markdown_content = f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
<h1 style="margin: 0; text-align: center;">📊 SPA Database Analysis</h1>
</div>

## 🎯 Key Metrics

<div style="display: flex; justify-content: space-around; margin: 20px 0;">
<div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; margin: 5px;">
<h3 style="margin: 0; color: #2c3e50;">📁 Unique SPAs</h3>
<h2 style="margin: 10px 0; color: #3498db;">{len(unique_spa_names)}</h2>
</div>
<div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; margin: 5px;">
<h3 style="margin: 0; color: #2c3e50;">📄 Total Chunks</h3>
<h2 style="margin: 10px 0; color: #e74c3c;">{len(document_names)}</h2>
</div>
<div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; margin: 5px;">
<h3 style="margin: 0; color: #2c3e50;">📈 Avg per SPA</h3>
<h2 style="margin: 10px 0; color: #27ae60;">{len(document_names) / len(unique_spa_names):.1f}</h2>
</div>
</div>

---

## 📋 All SPAs (Alphabetical)

<div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;">
"""

    # Add SPAs in columns for better space usage
    col_size = (len(unique_spa_names) + 2) // 3  # 3 columns

    markdown_content += '<div style="display: flex; flex-wrap: wrap;">\n'

    for i, spa_name in enumerate(unique_spa_names):
        chunk_count = document_counts[spa_name]
        if i % col_size == 0 and i > 0:
            markdown_content += '</div><div style="flex: 1; margin-right: 20px;">\n'
        elif i == 0:
            markdown_content += '<div style="flex: 1; margin-right: 20px;">\n'

        markdown_content += f"• <span style='color: #000; font-weight: bold;'>{spa_name}</span> <span style='color: #000;'>({chunk_count})</span><br>\n"

    markdown_content += '</div></div></div>\n\n---\n\n'

    # Top 10 table
    top_10 = document_counts.most_common(10)
    markdown_content += "## 🏆 Top 10 SPAs by Chunk Count\n\n"
    markdown_content += "| 🥇 Rank | 📄 SPA Name | 📊 Chunks | 📈 % of Total |\n"
    markdown_content += "|:-------:|-------------|:---------:|:------------:|\n"

    total_chunks = len(document_names)
    for rank, (spa_name, count) in enumerate(top_10, 1):
        percentage = (count / total_chunks) * 100
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}"
        markdown_content += f"| {emoji} | {spa_name} | **{count}** | {percentage:.1f}% |\n"

    # Display the markdown
    display(Markdown(markdown_content))

    # Show chart if requested
    if show_chart and len(document_counts) > 0:
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Top 10 bar chart
        top_10_names = [name[:30] + '...' if len(name) > 30 else name for name, _ in top_10]
        top_10_counts = [count for _, count in top_10]

        bars = ax1.bar(range(len(top_10_counts)), top_10_counts, color='skyblue', alpha=0.8)
        ax1.set_title('Top 10 SPAs by Chunk Count', fontsize=14, fontweight='bold')
        ax1.set_xlabel('SPA Name')
        ax1.set_ylabel('Number of Chunks')
        ax1.set_xticks(range(len(top_10_names)))
        ax1.set_xticklabels(top_10_names, rotation=45, ha='right')

        # Add value labels on bars
        for bar, count in zip(bars, top_10_counts):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(count), ha='center', va='bottom', fontweight='bold')

        # Distribution histogram
        chunk_counts_list = list(document_counts.values())
        ax2.hist(chunk_counts_list, bins=min(20, len(set(chunk_counts_list))),
                 color='lightcoral', alpha=0.7, edgecolor='black')
        ax2.set_title('Distribution of Chunks per SPA', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Chunks')
        ax2.set_ylabel('Number of SPAs')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Return DataFrame for further analysis
    df = pd.DataFrame([
        {
            'SPA_Name': spa_name,
            'Chunk_Count': count,
            'Percentage': f"{(count / total_chunks) * 100:.1f}%",
            'Percentage_Numeric': (count / total_chunks) * 100
        }
        for spa_name, count in document_counts.most_common()
    ])

    return df

######################################
#####################################
'''
Enhanced chatbot with memory capabilities, abulity to retireve more documents and has memory

'''
import logging
from typing import Optional, Dict, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    timestamp: datetime
    query: str
    spa_names: List[str]
    answer: str
    retrieved_doc_ids: Set[str]
    retrieval_method: str
    k_per_spa: int

@dataclass
class ConversationHistory:
    """Manages conversation history and retrieved documents"""
    turns: List[ConversationTurn] = field(default_factory=list)
    all_retrieved_doc_ids: Set[str] = field(default_factory=set)

    def add_turn(self, turn: ConversationTurn):
        """Add a new conversation turn"""
        self.turns.append(turn)
        self.all_retrieved_doc_ids.update(turn.retrieved_doc_ids)

    def get_recent_context(self, n_turns: int = 3) -> str:
        """Get recent conversation context as a formatted string"""
        if not self.turns:
            return ""

        recent_turns = self.turns[-n_turns:]
        context_parts = []

        for i, turn in enumerate(recent_turns, 1):
            context_parts.append(f"Previous Q{i}: {turn.query}")
            context_parts.append(f"Previous A{i}: {turn.answer[:200]}...")  # Truncate for brevity

        return "\n".join(context_parts)

    def get_last_spa_names(self) -> List[str]:
        """Get SPA names from the last query"""
        if self.turns:
            return self.turns[-1].spa_names
        return []



#######################
#######################
######################


class EnhancedMultiSPAChatbot:
    def __init__(self, retriever: 'VectorStoreRetriever', model="gemini-2.5-flash", api_key: Optional[str] = None):
        """
        Initialize the Enhanced Multi-SPA Chatbot with memory and smart retrieval.

        :param retriever: VectorStoreRetriever instance
        :param model: Google Generative AI model name
        :param api_key: Google API key
        """
        self.llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.1)
        self.retriever = retriever
        self.conversation_history = ConversationHistory()

        # Enhanced comparative analysis prompt template with conversation context
        self.comparative_prompt_template = """
        # Persona
        You are a specialized AI assistant for comparative analysis of Share Purchase Agreements (SPAs). You work at Strada Partners, a mid cap Private equity firm with 250 M AUM located in Belgium, Antwerp.

        # Task
        Your task is to analyze and compare multiple SPAs based on the provided context, taking into account the conversation history.

        ### Instructions:
        1. **Strictly Adhere to Context:** Base your analysis solely on the text provided in the 'Context' sections below.
        2. **Consider Conversation History:** Use the previous conversation context to provide more relevant and connected responses.
        3. **Structured Comparison:** Organize your response to clearly compare the requested aspects across all SPAs.
        4. **Cite Sources:** For each point, cite the specific SPA document and page number (e.g., "SPA Name: InterIT, Page: 15").
        5. **Handle Missing Information:** If information is missing for any SPA, explicitly state which SPAs lack the requested information.
        6. **Comparative Format:** Structure your answer to highlight similarities and differences between the SPAs.
        7. **No Legal Advice:** You are an information analysis tool, not a legal advisor.

        ### Previous Conversation Context:
        {conversation_context}

        ### SPAs Being Analyzed:
        {spa_list}

        ---
        ### Context from All SPAs:
        {context}

        ### Current Question:
        {question}

        ### Comparative Analysis:
        Please provide a structured comparison addressing the question. Consider the conversation history to provide continuity. Format your response as follows:
        - **Summary:** Detailed overview of findings across all SPAs
        - **Individual SPA Analysis:** Detailed findings for each SPA
        - **Key Differences:** Main distinctions between the SPAs
        - **Key Similarities:** Common elements across the SPAs
        - **Connection to Previous Discussion:** How this relates to our previous conversation (if applicable)
        """

        prompt = PromptTemplate.from_template(self.comparative_prompt_template)

        # Document prompt for better context formatting
        document_prompt = PromptTemplate.from_template(
            "\n=== SPA: {document_name} | Page: {page} ===\n{page_content}\n" + "="*50
        )

        # Create the chain for comparative analysis
        self.comparative_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt,
            document_prompt=document_prompt,
        )

    def _extract_document_ids(self, documents: List[Document]) -> Set[str]:
        """Extract unique identifiers from documents"""
        doc_ids = set()
        for doc in documents:
            # Create a unique ID based on document name, page, and content hash
            content_preview = doc.page_content[:100]  # First 100 chars
            doc_id = f"{doc.metadata.get('document_name', 'unknown')}_{doc.metadata.get('page', 'unknown')}_{hash(content_preview)}"
            doc_ids.add(doc_id)
        return doc_ids

    def _filter_new_documents(self, documents: List[Document], exclude_ids: Set[str]) -> List[Document]:
        """Filter out documents that have already been retrieved"""
        new_documents = []
        for doc in documents:
            content_preview = doc.page_content[:100]
            doc_id = f"{doc.metadata.get('document_name', 'unknown')}_{doc.metadata.get('page', 'unknown')}_{hash(content_preview)}"
            if doc_id not in exclude_ids:
                new_documents.append(doc)
        return new_documents

    def answer_comparative_question(
        self,
        query: str,
        spa_names: Optional[List[str]] = None,
        k_per_spa: int = 5,
        retrieval_method: str = "similarity",
        use_conversation_context: bool = True,
        avoid_previous_docs: bool = False
    ) -> Dict:
        """
        Answer a comparative question across multiple SPAs with conversation memory.

        :param query: The comparative question to ask
        :param spa_names: List of SPA names to compare (if None, uses last SPAs from conversation)
        :param k_per_spa: Number of documents to retrieve per SPA
        :param retrieval_method: Method to use for retrieval ("similarity", "mmr")
        :param use_conversation_context: Whether to include conversation history in the prompt
        :param avoid_previous_docs: Whether to avoid previously retrieved documents
        :return: Dictionary containing the question, answer, and supporting documents
        """
        # Use previous SPAs if none specified
        if spa_names is None:
            spa_names = self.conversation_history.get_last_spa_names()
            if not spa_names:
                raise ValueError("No SPA names provided and no previous conversation to reference")

        logger.info(f"Answering comparative question across {len(spa_names)} SPAs: '{query}'")

        all_documents = []
        spa_document_counts = {}
        exclude_ids = self.conversation_history.all_retrieved_doc_ids if avoid_previous_docs else set()

        # Retrieve documents from each SPA
        for spa_name in spa_names:
            logger.info(f"Retrieving documents from SPA: {spa_name}")

            # Create filter for this specific SPA
            spa_filter = {"document_name": spa_name}

            # Try multiple retrieval strategies to get diverse results
            spa_docs = []

            # Primary retrieval method
            if retrieval_method == "similarity":
                primary_docs = self.retriever.query_similarity(query, k=k_per_spa * 2, filter=spa_filter)
            elif retrieval_method == "mmr":
                primary_docs = self.retriever.query_mmr(query, k=k_per_spa * 2, filter=spa_filter)
            else:
                raise ValueError(f"Unsupported retrieval method: {retrieval_method}")

            # Filter out previously retrieved documents if requested
            if avoid_previous_docs:
                primary_docs = self._filter_new_documents(primary_docs, exclude_ids)

            spa_docs.extend(primary_docs[:k_per_spa])

            # If we don't have enough documents and we're avoiding previous docs, try alternative methods
            if len(spa_docs) < k_per_spa and avoid_previous_docs:
                logger.info(f"Only found {len(spa_docs)} new documents for {spa_name}, trying alternative retrieval...")

                # Try MMR if we used similarity, or vice versa
                alt_method = "mmr" if retrieval_method == "similarity" else "similarity"
                if alt_method == "similarity":
                    alt_docs = self.retriever.query_similarity(query, k=k_per_spa * 2, filter=spa_filter)
                else:
                    alt_docs = self.retriever.query_mmr(query, k=k_per_spa * 2, filter=spa_filter)

                alt_docs = self._filter_new_documents(alt_docs, exclude_ids)

                # Add alternative documents until we reach k_per_spa
                for doc in alt_docs:
                    if len(spa_docs) >= k_per_spa:
                        break
                    if doc not in spa_docs:  # Avoid duplicates
                        spa_docs.append(doc)

            spa_document_counts[spa_name] = len(spa_docs)
            all_documents.extend(spa_docs)

            logger.info(f"Retrieved {len(spa_docs)} documents from {spa_name}")

        # Check if we have documents from all requested SPAs
        missing_spas = [spa for spa, count in spa_document_counts.items() if count == 0]
        if missing_spas:
            logger.warning(f"No documents found for SPAs: {missing_spas}")

        if not all_documents:
            logger.warning("No documents were retrieved for any of the specified SPAs.")
            return {
                "question": query,
                "spa_names": spa_names,
                "output_text": f"I could not find any relevant information in the specified SPAs ({', '.join(spa_names)}) to answer your question.",
                "input_documents": [],
                "document_counts_per_spa": spa_document_counts
            }

        # Prepare conversation context
        conversation_context = ""
        if use_conversation_context:
            conversation_context = self.conversation_history.get_recent_context()

        # Create SPA list string for the prompt
        spa_list_str = ", ".join(spa_names)

        try:
            # Generate comparative analysis
            answer = self.comparative_chain.invoke({
                "context": all_documents,
                "question": query,
                "spa_list": spa_list_str,
                "conversation_context": conversation_context
            })

            # Store this conversation turn
            retrieved_doc_ids = self._extract_document_ids(all_documents)
            turn = ConversationTurn(
                timestamp=datetime.now(),
                query=query,
                spa_names=spa_names,
                answer=answer,
                retrieved_doc_ids=retrieved_doc_ids,
                retrieval_method=retrieval_method,
                k_per_spa=k_per_spa
            )
            self.conversation_history.add_turn(turn)

            return {
                "question": query,
                "spa_names": spa_names,
                "output_text": answer,
                "input_documents": all_documents,
                "document_counts_per_spa": spa_document_counts,
                "total_documents": len(all_documents),
                "new_documents_retrieved": len(retrieved_doc_ids - exclude_ids) if avoid_previous_docs else len(retrieved_doc_ids),
                "conversation_turn": len(self.conversation_history.turns)
            }

        except Exception as e:
            logger.error(f"Failed to get comparative answer: {e}")
            raise

    def get_more_documents(
        self,
        additional_k_per_spa: int = 3,
        use_alternative_method: bool = True
    ) -> Dict:
        """
        Retrieve additional documents for the last question asked.

        :param additional_k_per_spa: Number of additional documents to retrieve per SPA
        :param use_alternative_method: Whether to use an alternative retrieval method
        :return: Dictionary containing additional analysis
        """
        if not self.conversation_history.turns:
            raise ValueError("No previous conversation to expand upon")

        last_turn = self.conversation_history.turns[-1]

        # Use alternative retrieval method if requested
        alt_method = "mmr" if last_turn.retrieval_method == "similarity" else "similarity"
        retrieval_method = alt_method if use_alternative_method else last_turn.retrieval_method

        logger.info(f"Retrieving {additional_k_per_spa} additional documents per SPA using {retrieval_method}")

        return self.answer_comparative_question(
            query=f"Building on our previous discussion: {last_turn.query}",
            spa_names=last_turn.spa_names,
            k_per_spa=additional_k_per_spa,
            retrieval_method=retrieval_method,
            use_conversation_context=True,
            avoid_previous_docs=True
        )

    def continue_conversation(self, follow_up_query: str, spa_names: Optional[List[str]] = None) -> Dict:
        """
        Continue the conversation with a follow-up question.

        :param follow_up_query: The follow-up question
        :param spa_names: Optional list of SPA names (uses previous if None)
        :return: Dictionary containing the response
        """
        return self.answer_comparative_question(
            query=follow_up_query,
            spa_names=spa_names,
            use_conversation_context=True,
            avoid_previous_docs=False  # Don't avoid previous docs for follow-ups
        )

    def get_conversation_summary(self) -> Dict:
        """Get a summary of the current conversation"""
        if not self.conversation_history.turns:
            return {"message": "No conversation history available"}

        total_documents = len(self.conversation_history.all_retrieved_doc_ids)
        spa_names = set()
        for turn in self.conversation_history.turns:
            spa_names.update(turn.spa_names)

        return {
            "total_turns": len(self.conversation_history.turns),
            "total_unique_documents_retrieved": total_documents,
            "spas_discussed": sorted(list(spa_names)),
            "conversation_start": self.conversation_history.turns[0].timestamp.isoformat(),
            "last_update": self.conversation_history.turns[-1].timestamp.isoformat(),
            "recent_questions": [turn.query for turn in self.conversation_history.turns[-3:]]
        }

    def clear_conversation(self):
        """Clear the conversation history"""
        self.conversation_history = ConversationHistory()
        logger.info("Conversation history cleared")

    def get_available_spas(self) -> List[str]:
        """
        Get a list of all available SPA names in the database.

        :return: List of available SPA document names
        """
        try:
            # Query all documents to get unique document names
            all_docs = self.retriever.vector_store.get()

            if 'metadatas' in all_docs and all_docs['metadatas']:
                document_names = set()
                for metadata in all_docs['metadatas']:
                    if 'document_name' in metadata:
                        document_names.add(metadata['document_name'])
                return sorted(list(document_names))
            else:
                logger.warning("No documents found in the database.")
                return []

        except Exception as e:
            logger.error(f"Failed to retrieve available SPAs: {e}")
            return []

    def validate_spa_names(self, spa_names: List[str]) -> Dict[str, bool]:
        """
        Validate that the provided SPA names exist in the database.

        :param spa_names: List of SPA names to validate
        :return: Dictionary mapping SPA names to their existence status
        """
        available_spas = self.get_available_spas()
        return {spa_name: spa_name in available_spas for spa_name in spa_names}
#
# # Example usage:
# """
# # Initialize the enhanced chatbot
# enhanced_chatbot = EnhancedMultiSPAChatbot(retriever)
#
# # Start a conversation
# result1 = enhanced_chatbot.answer_comparative_question(
#     "What are the purchase prices in these SPAs?",
#     spa_names=["SPA_A", "SPA_B", "SPA_C"]
# )
#
# # Ask for more documents on the same topic
# result2 = enhanced_chatbot.get_more_documents(additional_k_per_spa=3)
#
# # Continue with a follow-up question
# result3 = enhanced_chatbot.continue_conversation(
#     "How do the payment terms compare?"
# )
#
# # Get conversation summary
# summary = enhanced_chatbot.get_conversation_summary()
#
# # Clear conversation when done
# enhanced_chatbot.clear_conversation()
# """


################################
################################
###############################
################################
###############################



import json
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SPATopic:
    """Represents an SPA topic with its analysis question"""
    key: str
    name: str
    question: str
    description: str

class SequentialSPAAnalyzer:
    """
    Performs sequential analysis of SPA topics using the existing chatbot infrastructure.
    Goes through all 20 standard SPA topics and generates structured JSON output.
    """

    def __init__(self, chatbot: 'QAChatbot'):
        """
        Initialize the Sequential SPA Analyzer.

        :param chatbot: QAChatbot instance for asking questions
        """
        self.chatbot = chatbot
        self.spa_topics = self._define_spa_topics()

    def _define_spa_topics(self) -> List[SPATopic]:
        """Define the 20 standard SPA topics with their analysis questions"""
        return [
            SPATopic(
                key="de_minimis_basket",
                name="De-minimis & Basket",
                question="What are the de-minimis thresholds and basket amounts for claims? What are the specific monetary amounts and percentages mentioned?",
                description="Small claims filter and aggregate above basket before a claim can be made"
            ),
            SPATopic(
                key="overall_caps",
                name="Overall Caps",
                question="What are the overall caps on warranty liability? What are the monetary limits and percentages of purchase price?",
                description="Monetary limits on warranty liability, split caps for fundamental/title vs business warranties"
            ),
            SPATopic(
                key="time_limitations",
                name="Time Limitations",
                question="What are the time limitations for bringing warranty claims? How long can claims be brought for different types of warranties?",
                description="How long warranty claims can be brought (general, tax, environmental, fraud carve-out)"
            ),
            SPATopic(
                key="specific_indemnities",
                name="Specific Indemnities",
                question="What specific indemnities are provided for pre-sign issues? Are there indemnities for tax, employment, data privacy, litigation, etc.?",
                description="Tailored indemnities for pre-sign issues (tax, employment, data privacy, litigation, etc.)"
            ),
            SPATopic(
                key="tax_covenant_indemnity",
                name="Tax Covenant/Indemnity",
                question="What are the seller's obligations regarding pre-Completion taxes? What procedural controls are mentioned?",
                description="Seller obligation to pay pre-Completion taxes; procedural controls"
            ),
            SPATopic(
                key="locked_box_leakage",
                name="Locked-Box Leakage",
                question="What protection is provided against value leakage between accounts date and Completion? What constitutes leakage?",
                description="Protection against value leakage between accounts date & Completion"
            ),
            SPATopic(
                key="purchase_price_adjustment",
                name="Purchase-Price Adjustment / Completion Accounts",
                question="How is the purchase price adjusted? What are the net-debt and working-capital true-up mechanisms?",
                description="Net-debt & working-capital true-up"
            ),
            SPATopic(
                key="earn_out_mechanics",
                name="Earn-out Mechanics & Definitions",
                question="Are there any earn-out provisions? What are the KPI definitions, measurement criteria, and buyer discretion limitations?",
                description="KPI definitions, measurement, buyer discretion, audit"
            ),
            SPATopic(
                key="material_adverse_change",
                name="Material Adverse Change (MAC)",
                question="How is Material Adverse Change defined? What are the conditions and carve-outs mentioned?",
                description="Condition/termination right if something bad happens pre-Completion"
            ),
            SPATopic(
                key="disclosure_knowledge_qualifiers",
                name="Disclosure & Knowledge Qualifiers",
                question="How are disclosures and knowledge qualifiers defined? What information qualifies warranties?",
                description="What information qualifies warranties"
            ),
            SPATopic(
                key="set_off_withholding",
                name="Set-off / Withholding",
                question="Can the buyer set indemnity claims against consideration? What are the conditions for set-off?",
                description="Ability to set indemnity claims against consideration"
            ),
            SPATopic(
                key="escrow_retention",
                name="Escrow / Retention",
                question="What portion of the price is held back for warranty cover? What are the release conditions and timeframes?",
                description="Portion of price held back for warranty cover"
            ),
            SPATopic(
                key="fundamental_warranties",
                name="Fundamental Warranties",
                question="What fundamental warranties are provided? What aspects of title, capacity, authority, and share capital are covered?",
                description="Title, capacity, authority, share capital, etc."
            ),
            SPATopic(
                key="pre_completion_covenants",
                name="Pre-Completion Covenants",
                question="What are the 'Business as usual' obligations between signing and Completion? What restrictions apply?",
                description="'Business as usual' obligations between signing & Completion"
            ),
            SPATopic(
                key="non_compete_non_solicit",
                name="Non-compete / Non-solicit",
                question="What are the seller restrictions after closing? What geographical and time limitations apply?",
                description="Seller restrictions after closing"
            ),
            SPATopic(
                key="consequential_loss_exclusion",
                name="Consequential-loss Exclusion",
                question="Are lost profits and consequential losses excluded? What specific exclusions and carve-outs are mentioned?",
                description="Whether lost profits, etc., are excluded"
            ),
            SPATopic(
                key="knowledge_scrape_out",
                name="Knowledge Scrape-out",
                question="What is the buyer's right to claim despite knowing about breaches? How is knowledge defined?",
                description="Buyer's right to claim despite knowing about the breach"
            ),
            SPATopic(
                key="third_party_claims_conduct",
                name="Third-party Claims Conduct",
                question="How are claims against the target post-Completion handled? Who controls defense and settlement?",
                description="Handling of claims against the target post-Completion"
            ),
            SPATopic(
                key="dispute_resolution_governing_law",
                name="Dispute Resolution & Governing Law",
                question="What dispute resolution mechanisms are specified? Which courts or arbitration venues are designated?",
                description="Courts vs arbitration, venue, expert determination for accounts"
            ),
            SPATopic(
                key="fraud_wilful_misconduct_carve_out",
                name="Fraud / Wilful Misconduct Carve-out",
                question="Do caps and limits fall away in case of fraud? What constitutes fraud or wilful misconduct?",
                description="Whether caps & limits fall away in case of fraud"
            )
        ]

    def analyze_spa(self, spa_name: str, k_per_question: int = 7) -> Dict:
        """
        Perform sequential analysis of all 20 SPA topics for a given SPA.

        :param spa_name: Name of the SPA to analyze
        :param k_per_question: Number of documents to retrieve per question
        :return: Dictionary containing structured analysis results
        """
        logger.info(f"Starting sequential analysis of SPA: {spa_name}")

        # Validate that the SPA exists
        available_spas = self.chatbot.retriever.vector_store.get()
        spa_exists = False
        if 'metadatas' in available_spas and available_spas['metadatas']:
            for metadata in available_spas['metadatas']:
                if metadata and metadata.get('document_name') == spa_name:
                    spa_exists = True
                    break

        if not spa_exists:
            raise ValueError(f"SPA '{spa_name}' not found in the database")

        # Initialize results structure
        analysis_results = {
            "spa_name": spa_name,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_topics": len(self.spa_topics),
            "topics_analysis": {}
        }

        # Filter for this specific SPA
        spa_filter = {"document_name": spa_name}

        # Analyze each topic sequentially
        for i, topic in enumerate(self.spa_topics, 1):
            logger.info(f"Analyzing topic {i}/20: {topic.name}")

            try:
                # Ask the question for this topic
                result = self.chatbot.answer_question(
                    query=topic.question,
                    k=k_per_question,
                    filter=spa_filter
                )

                # Structure the response for this topic
                topic_analysis = {
                    "topic_number": i,
                    "topic_name": topic.name,
                    "topic_description": topic.description,
                    "question_asked": topic.question,
                    "answer": result["output_text"],
                    "documents_retrieved": len(result["input_documents"]),
                    "source_pages": []
                }

                # Extract source page information
                for doc in result["input_documents"]:
                    page_info = {
                        "page": doc.metadata.get("page", "unknown"),
                        "source": doc.metadata.get("source", "unknown"),
                        "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    }
                    topic_analysis["source_pages"].append(page_info)

                # Add to results using the topic key
                analysis_results["topics_analysis"][topic.key] = topic_analysis

                logger.info(f"Completed analysis for {topic.name}")

            except Exception as e:
                logger.error(f"Error analyzing topic {topic.name}: {str(e)}")

                # Add error information to results
                analysis_results["topics_analysis"][topic.key] = {
                    "topic_number": i,
                    "topic_name": topic.name,
                    "topic_description": topic.description,
                    "question_asked": topic.question,
                    "error": str(e),
                    "answer": f"Error occurred while analyzing this topic: {str(e)}",
                    "documents_retrieved": 0,
                    "source_pages": []
                }

        logger.info(f"Completed sequential analysis of SPA: {spa_name}")
        return analysis_results

    def save_analysis_to_json(self, analysis_results: Dict, filename: Optional[str] = None) -> str:
        """
        Save analysis results to a JSON file.

        :param analysis_results: Analysis results dictionary
        :param filename: Optional filename (auto-generated if None)
        :return: Filename where results were saved
        """
        if filename is None:
            spa_name = analysis_results.get("spa_name", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"spa_analysis_{spa_name}_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Analysis results saved to: {filename}")
        return filename

    def get_topic_summary(self, analysis_results: Dict) -> Dict:
        """
        Generate a summary of the analysis results.

        :param analysis_results: Analysis results dictionary
        :return: Summary dictionary
        """
        topics_analyzed = len(analysis_results["topics_analysis"])
        topics_with_errors = sum(1 for topic in analysis_results["topics_analysis"].values()
                               if "error" in topic)

        total_documents_retrieved = sum(topic.get("documents_retrieved", 0)
                                      for topic in analysis_results["topics_analysis"].values())

        return {
            "spa_name": analysis_results["spa_name"],
            "analysis_timestamp": analysis_results["analysis_timestamp"],
            "topics_analyzed": topics_analyzed,
            "topics_with_errors": topics_with_errors,
            "success_rate": f"{((topics_analyzed - topics_with_errors) / topics_analyzed * 100):.1f}%",
            "total_documents_retrieved": total_documents_retrieved,
            "average_documents_per_topic": f"{(total_documents_retrieved / topics_analyzed):.1f}",
            "topic_keys": list(analysis_results["topics_analysis"].keys())
        }

# Example usage:
"""
# Initialize your existing components
retriever = VectorStoreRetriever()
chatbot = QAChatbot(retriever)

# Create the sequential analyzer
analyzer = SequentialSPAAnalyzer(chatbot)

# Analyze an SPA
results = analyzer.analyze_spa("YourSPAName", k_per_question=5)

# Save to JSON
filename = analyzer.save_analysis_to_json(results)

# Get summary
summary = analyzer.get_topic_summary(results)
print(json.dumps(summary, indent=2))

# Access specific topics
purchase_price_info = results["topics_analysis"]["purchase_price_adjustment"]
warranty_caps = results["topics_analysis"]["overall_caps"]
"""

##################################
###################################
####################################

# Sequential Analyzer Comparison too
import json
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SPATopic:
    """Represents an SPA topic with its analysis question"""
    key: str
    name: str
    question: str
    description: str

class SequentialSPAAnalyzer2:
    """
    Performs sequential analysis of SPA topics using the existing chatbot infrastructure.
    Goes through all 20 standard SPA topics and generates structured JSON output.
    Modified to support comparison across multiple SPAs using MultiSPAChatbot.
    """

    def __init__(self, multi_chatbot: 'MultiSPAChatbot'):
        """
        Initialize the Sequential SPA Analyzer.

        :param single_chatbot: QAChatbot instance for single SPA analysis (kept for backward compatibility if needed)
        :param multi_chatbot: MultiSPAChatbot instance for multi-SPA comparative analysis
        """
        # self.single_chatbot = single_chatbot  # Optional, in case single SPA is still needed
        self.multi_chatbot = multi_chatbot
        self.spa_topics = self._define_spa_topics()

    def _define_spa_topics(self) -> List[SPATopic]:
        """Define the 20 standard SPA topics with their analysis questions"""
        return [
            SPATopic(
                key="de_minimis_basket",
                name="De-minimis & Basket",
                question="What are the de-minimis thresholds and basket amounts for claims? What are the specific monetary amounts and percentages mentioned?",
                description="Small claims filter and aggregate above basket before a claim can be made"
            ),
            SPATopic(
                key="overall_caps",
                name="Overall Caps",
                question="What are the overall caps on warranty liability? What are the monetary limits and percentages of purchase price?",
                description="Monetary limits on warranty liability, split caps for fundamental/title vs business warranties"
            ),
            SPATopic(
                key="time_limitations",
                name="Time Limitations",
                question="What are the time limitations for bringing warranty claims? How long can claims be brought for different types of warranties?",
                description="How long warranty claims can be brought (general, tax, environmental, fraud carve-out)"
            ),
            SPATopic(
                key="specific_indemnities",
                name="Specific Indemnities",
                question="What specific indemnities are provided for pre-sign issues? Are there indemnities for tax, employment, data privacy, litigation, etc.?",
                description="Tailored indemnities for pre-sign issues (tax, employment, data privacy, litigation, etc.)"
            ),
            SPATopic(
                key="tax_covenant_indemnity",
                name="Tax Covenant/Indemnity",
                question="What are the seller's obligations regarding pre-Completion taxes? What procedural controls are mentioned?",
                description="Seller obligation to pay pre-Completion taxes; procedural controls"
            ),
            SPATopic(
                key="locked_box_leakage",
                name="Locked-Box Leakage",
                question="What protection is provided against value leakage between accounts date and Completion? What constitutes leakage?",
                description="Protection against value leakage between accounts date & Completion"
            ),
            SPATopic(
                key="purchase_price_adjustment",
                name="Purchase-Price Adjustment / Completion Accounts",
                question="How is the purchase price adjusted? What are the net-debt and working-capital true-up mechanisms?",
                description="Net-debt & working-capital true-up"
            ),
            SPATopic(
                key="earn_out_mechanics",
                name="Earn-out Mechanics & Definitions",
                question="Are there any earn-out provisions? What are the KPI definitions, measurement criteria, and buyer discretion limitations?",
                description="KPI definitions, measurement, buyer discretion, audit"
            ),
            SPATopic(
                key="material_adverse_change",
                name="Material Adverse Change (MAC)",
                question="How is Material Adverse Change defined? What are the conditions and carve-outs mentioned?",
                description="Condition/termination right if something bad happens pre-Completion"
            ),
            SPATopic(
                key="disclosure_knowledge_qualifiers",
                name="Disclosure & Knowledge Qualifiers",
                question="How are disclosures and knowledge qualifiers defined? What information qualifies warranties?",
                description="What information qualifies warranties"
            ),
            SPATopic(
                key="set_off_withholding",
                name="Set-off / Withholding",
                question="Can the buyer set indemnity claims against consideration? What are the conditions for set-off?",
                description="Ability to set indemnity claims against consideration"
            ),
            SPATopic(
                key="escrow_retention",
                name="Escrow / Retention",
                question="What portion of the price is held back for warranty cover? What are the release conditions and timeframes?",
                description="Portion of price held back for warranty cover"
            ),
            SPATopic(
                key="fundamental_warranties",
                name="Fundamental Warranties",
                question="What fundamental warranties are provided? What aspects of title, capacity, authority, and share capital are covered?",
                description="Title, capacity, authority, share capital, etc."
            ),
            SPATopic(
                key="pre_completion_covenants",
                name="Pre-Completion Covenants",
                question="What are the 'Business as usual' obligations between signing and Completion? What restrictions apply?",
                description="'Business as usual' obligations between signing & Completion"
            ),
            SPATopic(
                key="non_compete_non_solicit",
                name="Non-compete / Non-solicit",
                question="What are the seller restrictions after closing? What geographical and time limitations apply?",
                description="Seller restrictions after closing"
            ),
            SPATopic(
                key="consequential_loss_exclusion",
                name="Consequential-loss Exclusion",
                question="Are lost profits and consequential losses excluded? What specific exclusions and carve-outs are mentioned?",
                description="Whether lost profits, etc., are excluded"
            ),
            SPATopic(
                key="knowledge_scrape_out",
                name="Knowledge Scrape-out",
                question="What is the buyer's right to claim despite knowing about breaches? How is knowledge defined?",
                description="Buyer's right to claim despite knowing about the breach"
            ),
            SPATopic(
                key="third_party_claims_conduct",
                name="Third-party Claims Conduct",
                question="How are claims against the target post-Completion handled? Who controls defense and settlement?",
                description="Handling of claims against the target post-Completion"
            ),
            SPATopic(
                key="dispute_resolution_governing_law",
                name="Dispute Resolution & Governing Law",
                question="What dispute resolution mechanisms are specified? Which courts or arbitration venues are designated?",
                description="Courts vs arbitration, venue, expert determination for accounts"
            ),
            SPATopic(
                key="fraud_wilful_misconduct_carve_out",
                name="Fraud / Wilful Misconduct Carve-out",
                question="Do caps and limits fall away in case of fraud? What constitutes fraud or wilful misconduct?",
                description="Whether caps & limits fall away in case of fraud"
            )
        ]

    def analyze_spas(self, spa_names: List[str], k_per_question: int = 7, retrieval_method: str = "similarity") -> Dict:
        """
        Perform sequential comparative analysis of all 20 SPA topics across multiple SPAs.

        :param spa_names: List of SPA names to compare
        :param k_per_question: Number of documents to retrieve per SPA per question
        :param retrieval_method: Retrieval method for MultiSPAChatbot ("similarity" or "mmr")
        :return: Dictionary containing structured comparative analysis results
        """
        # if len(spa_names) < 2:
        #     raise ValueError("At least two SPA names are required for comparison.")

        logger.info(f"Starting sequential comparative analysis across SPAs: {', '.join(spa_names)}")

        # Validate that all SPAs exist
        available_spas = self.multi_chatbot.get_available_spas()
        missing_spas = [spa for spa in spa_names if spa not in available_spas]
        if missing_spas:
            raise ValueError(f"The following SPAs not found in the database: {', '.join(missing_spas)}")

        # Initialize results structure
        analysis_results = {
            "spa_names": spa_names,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_topics": len(self.spa_topics),
            "topics_analysis": {}
        }

        # Analyze each topic sequentially using MultiSPAChatbot
        for i, topic in enumerate(self.spa_topics, 1):
            logger.info(f"Analyzing topic {i}/20: {topic.name}")

            try:
                # Ask the comparative question for this topic
                result = self.multi_chatbot.answer_comparative_question(
                    query=topic.question,
                    spa_names=spa_names,
                    k_per_spa=k_per_question,
                    retrieval_method=retrieval_method
                )

                # Structure the response for this topic
                topic_analysis = {
                    "topic_number": i,
                    "topic_name": topic.name,
                    "topic_description": topic.description,
                    "question_asked": topic.question,
                    "comparative_answer": result["output_text"],
                    "total_documents_retrieved": result.get("total_documents", 0),
                    "document_counts_per_spa": result.get("document_counts_per_spa", {}),
                    "source_pages": []
                }

                # Extract source page information across all SPAs
                for doc in result.get("input_documents", []):
                    page_info = {
                        "spa_name": doc.metadata.get("document_name", "unknown"),
                        "page": doc.metadata.get("page", "unknown"),
                        "source": doc.metadata.get("source", "unknown"),
                        "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    }
                    topic_analysis["source_pages"].append(page_info)

                # Add to results using the topic key
                analysis_results["topics_analysis"][topic.key] = topic_analysis

                logger.info(f"Completed analysis for {topic.name}")

            except Exception as e:
                logger.error(f"Error analyzing topic {topic.name}: {str(e)}")

                # Add error information to results
                analysis_results["topics_analysis"][topic.key] = {
                    "topic_number": i,
                    "topic_name": topic.name,
                    "topic_description": topic.description,
                    "question_asked": topic.question,
                    "error": str(e),
                    "comparative_answer": f"Error occurred while analyzing this topic: {str(e)}",
                    "total_documents_retrieved": 0,
                    "document_counts_per_spa": {},
                    "source_pages": []
                }

        logger.info(f"Completed sequential comparative analysis across SPAs: {', '.join(spa_names)}")
        return analysis_results

    def save_analysis_to_json(self, analysis_results: Dict, filename: Optional[str] = None) -> str:
        """
        Save analysis results to a JSON file.

        :param analysis_results: Analysis results dictionary
        :param filename: Optional filename (auto-generated if None)
        :return: Filename where results were saved
        """
        if filename is None:
            spa_names_str = "_".join(analysis_results.get("spa_names", ["unknown"]))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"spa_comparative_analysis_{spa_names_str}_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Analysis results saved to: {filename}")
        return filename

    def get_topic_summary(self, analysis_results: Dict) -> Dict:
        """
        Generate a summary of the analysis results.

        :param analysis_results: Analysis results dictionary
        :return: Summary dictionary
        """
        topics_analyzed = len(analysis_results["topics_analysis"])
        topics_with_errors = sum(1 for topic in analysis_results["topics_analysis"].values()
                                 if "error" in topic)

        total_documents_retrieved = sum(topic.get("total_documents_retrieved", 0)
                                        for topic in analysis_results["topics_analysis"].values())

        return {
            "spa_names": analysis_results["spa_names"],
            "analysis_timestamp": analysis_results["analysis_timestamp"],
            "topics_analyzed": topics_analyzed,
            "topics_with_errors": topics_with_errors,
            "success_rate": f"{((topics_analyzed - topics_with_errors) / topics_analyzed * 100):.1f}%" if topics_analyzed > 0 else "0.0%",
            "total_documents_retrieved": total_documents_retrieved,
            "average_documents_per_topic": f"{(total_documents_retrieved / topics_analyzed):.1f}" if topics_analyzed > 0 else "0.0",
            "topic_keys": list(analysis_results["topics_analysis"].keys())
        }


###############################################
#########################################

# =========================
# Unified SPA Orchestrator
# =========================
from typing import List, Optional, Dict, Any

class UnifiedSPAAssistant:
    """
    One class to rule them all:
      - memory=True  -> EnhancedMultiSPAChatbot (works with 1+ SPAs, keeps context)
      - memory=False + 1 SPA -> QAChatbot
      - memory=False + 2+ SPAs -> MultiSPAChatbot

    Also exposes:
      - full_analysis(...) -> runs the 20-topic loop via SequentialSPAAnalyzer2
      - get_available_spas()
      - validate_spa_names(...)
    """

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
    ):
        self.retriever = retriever
        # Under-the-hood bots
        self.qa_bot = QAChatbot(retriever, model=model, api_key=api_key)
        self.multi_bot = MultiSPAChatbot(retriever, model=model, api_key=api_key)
        self.enhanced_bot = EnhancedMultiSPAChatbot(retriever, model=model, api_key=api_key)

    # ---------- Routing helpers ----------
    def _use_enhanced(self) -> EnhancedMultiSPAChatbot:
        return self.enhanced_bot

    def _use_multi(self) -> MultiSPAChatbot:
        return self.multi_bot

    def _use_single(self) -> QAChatbot:
        return self.qa_bot

    # ---------- Public API ----------
    def answer(
        self,
        question: str,
        spa_names: List[str],
        *,
        memory: bool = False,
        k: int = 10,
        k_per_spa: Optional[int] = None,
        retrieval_method: str = "similarity",
        filter: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Route a Q&A/comparison query to the right engine.

        Args:
            question: user question
            spa_names: list of SPA document names (>=1)
            memory: if True, use EnhancedMultiSPAChatbot regardless of count
            k: top-k chunks for single SPA mode
            k_per_spa: top-k per SPA for multi/enhanced modes (defaults to k)
            retrieval_method: "similarity" or "mmr" (for multi/enhanced)
            filter: extra metadata filter (merged with document_name)
        """
        if not spa_names:
            raise ValueError("Provide at least one SPA name")

        # Memory mode overrides everything (works with 1 or many SPAs)
        if memory:
            bot = self._use_enhanced()
            return bot.answer_comparative_question(
                query=question,
                spa_names=spa_names,
                k_per_spa=k_per_spa or max(3, k // max(1, len(spa_names))),
                retrieval_method=retrieval_method,
                use_conversation_context=True,
                avoid_previous_docs=False,
            )

        # Non-memory single SPA -> classic QAChatbot
        if len(spa_names) == 1:
            single = self._use_single()
            # Merge filter with the SPA name constraint
            _filter = {"document_name": spa_names[0]}
            if filter:
                _filter.update(filter)
            return single.answer_question(query=question, k=k, filter=_filter)

        # Non-memory multi SPA -> MultiSPAChatbot
        bot = self._use_multi()
        return bot.answer_comparative_question(
            query=question,
            spa_names=spa_names,
            k_per_spa=k_per_spa or max(3, k // max(1, len(spa_names))),
            retrieval_method=retrieval_method,
        )

    def continue_conversation(
        self,
        follow_up_question: str,
        spa_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Continue an Enhanced (memory) session. If you didn't start with memory,
        this will still work but will use the enhanced bot's history (which may be empty).
        """
        return self._use_enhanced().continue_conversation(
            follow_up_question, spa_names=spa_names
        )

    def full_analysis(
        self,
        spa_names: List[str],
        *,
        memory: bool = False,
        k_per_question: int = 7,
        retrieval_method: str = "similarity",
        save_to_json: bool = False,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the 20-topic SequentialSPAAnalyzer2 loop.
        Uses MultiSPAChatbot by default; if memory=True, uses EnhancedMultiSPAChatbot.
        """
        if not spa_names:
            raise ValueError("Provide at least one SPA name for full analysis")

        # Choose the multi interface for the analyzer (enhanced supports same API)
        multi_iface = self._use_enhanced() if memory else self._use_multi()
        analyzer = SequentialSPAAnalyzer2(multi_iface)

        results = analyzer.analyze_spas(
            spa_names=spa_names,
            k_per_question=k_per_question,
            retrieval_method=retrieval_method,
        )

        if save_to_json:
            analyzer.save_analysis_to_json(results, filename=filename)

        return results

    # ---------- Utilities ----------
    def get_available_spas(self) -> List[str]:
        # Both multi/enhanced expose this; use the regular one
        return self._use_multi().get_available_spas()

    def validate_spa_names(self, spa_names: List[str]) -> Dict[str, bool]:
        return self._use_multi().validate_spa_names(spa_names)
