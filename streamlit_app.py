import streamlit as st
import os
import json
import logging
import asyncio
import threading
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import the UnifiedSPAAssistant and VectorStoreRetriever from the provided file
try:
    from attached_assets.SPA_Tools_Corrected_1754901908313 import UnifiedSPAAssistant, VectorStoreRetriever
except ImportError:
    st.error("Could not import UnifiedSPAAssistant. Please ensure SPA_Tools_Corrected_1754901908313.py is available.")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="SPA Analysis Tool",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_config() -> Dict[str, str]:
    """Get configuration from environment variables with fallbacks."""
    config = {}
    
    # Try to load from environment variables first, then .env, then st.secrets
    def get_env_var(key: str, default: str = "") -> str:
        # First try os.environ (Replit secrets are set as environment variables)
        env_value = os.getenv(key)
        if env_value:
            return env_value
            
        # Try .env (if python-dotenv is available)
        try:
            from dotenv import load_dotenv
            load_dotenv()
            env_value = os.getenv(key)
            if env_value:
                return env_value
        except ImportError:
            pass
        
        # Try st.secrets as fallback
        try:
            if hasattr(st, 'secrets') and key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
        
        return default
    
    config['google_api_key'] = get_env_var('GOOGLE_API_KEY', '')
    config['database_dir'] = get_env_var('DATABASE_DIR', 'DATABASE_SPA')
    config['model_name'] = get_env_var('MODEL_NAME', 'gemini-2.5-flash')
    
    return config

@st.cache_resource
def initialize_assistant():
    """Initialize the UnifiedSPAAssistant with configuration."""
    try:
        config = get_config()
        
        if not config['google_api_key']:
            st.error("Google API Key is required. Please set GOOGLE_API_KEY in environment variables, .env file, or Streamlit secrets.")
            st.stop()
        
        # Set up event loop for async operations if needed
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Initialize the retriever first
        retriever = VectorStoreRetriever(
            persist_directory=config['database_dir'],
            api_key=config['google_api_key']
        )
        
        # Initialize the assistant with the retriever
        assistant = UnifiedSPAAssistant(
            retriever=retriever,
            model=config['model_name'],
            api_key=config['google_api_key']
        )
        return assistant
    except Exception as e:
        st.error(f"Failed to initialize SPA Assistant: {str(e)}")
        st.stop()

def initialize_session_state():
    """Initialize session state variables."""
    if 'memory_enabled' not in st.session_state:
        st.session_state.memory_enabled = False
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = "Single SPA Query"

def display_conversation_history():
    """Display conversation history when memory is enabled."""
    if st.session_state.memory_enabled and st.session_state.conversation_history:
        st.subheader("Conversation History")
        for i, item in enumerate(st.session_state.conversation_history):
            with st.expander(f"Query {i+1}: {item['query'][:50]}..."):
                st.write("**Query:**", item['query'])
                st.write("**SPAs:**", ", ".join(item['spas']) if item['spas'] else "None")
                st.write("**Response:**", item['response'])

def clear_conversation_history():
    """Clear conversation history."""
    st.session_state.conversation_history = []
    st.success("Conversation history cleared!")

def get_available_portfolio_companies(assistant):
    """Get list of unique portfolio companies from the vector database, including 'ALL' option."""
    try:
        # Query the vector store to get all unique portfolio companies
        vector_store = assistant.retriever.vector_store
        collection = vector_store._collection
        
        # Get all documents and extract portfolio companies
        all_docs = collection.get()
        portfolio_companies = set()
        
        for metadata in all_docs['metadatas']:
            if metadata and 'portfolio_company' in metadata:
                portfolio_companies.add(metadata['portfolio_company'])
        
        # Add "ALL" option at the beginning
        portfolio_list = ["ALL"] + sorted(list(portfolio_companies))
        return portfolio_list
    except Exception as e:
        st.error(f"Could not retrieve portfolio companies: {str(e)}")
        return ["ALL"]

def get_spas_for_portfolio_company(assistant, portfolio_company: str):
    """Get list of SPAs for a specific portfolio company, or all SPAs if 'ALL' is selected."""
    try:
        # Query the vector store to get SPAs for the portfolio company
        vector_store = assistant.retriever.vector_store
        collection = vector_store._collection
        
        # If "ALL" is selected, get all documents without filtering
        if portfolio_company == "ALL":
            all_docs = collection.get()
        else:
            # Get documents filtered by portfolio company
            all_docs = collection.get(where={"portfolio_company": portfolio_company})
        
        spa_names = set()
        
        for metadata in all_docs['metadatas']:
            if metadata and 'document_name' in metadata:
                spa_names.add(metadata['document_name'])
        
        return sorted(list(spa_names))
    except Exception as e:
        st.error(f"Could not retrieve SPAs for portfolio company {portfolio_company}: {str(e)}")
        return []

def single_spa_query(assistant, available_spas: List[str]):
    """Handle single SPA query interface with portfolio company selection."""
    st.subheader("Single SPA Query")
    
    # Portfolio company selection
    portfolio_companies = get_available_portfolio_companies(assistant)
    
    if not portfolio_companies:
        st.warning("No portfolio companies found in the database.")
        return
    
    selected_portfolio = st.selectbox(
        "First, select a Portfolio Company:",
        options=[""] + portfolio_companies,
        key="single_portfolio_select"
    )
    
    if not selected_portfolio:
        st.info("Please select a portfolio company to see available SPAs.")
        return
    
    # SPA selection based on portfolio company
    portfolio_spas = get_spas_for_portfolio_company(assistant, selected_portfolio)
    
    if not portfolio_spas:
        st.warning(f"No SPAs found for portfolio company: {selected_portfolio}")
        return
    
    selected_spa = st.selectbox(
        f"Select an SPA for {selected_portfolio}:",
        options=[""] + portfolio_spas,
        key="single_spa_select"
    )
    
    if not selected_spa:
        st.info("Please select an SPA to proceed.")
        return
    
    # Predefined questions
    predefined_questions = [
        ("De-minimis & Basket", "What are the de-minimis thresholds and basket amounts for claims? What are the specific monetary amounts and percentages mentioned?"),
        ("Overall Caps", "What are the overall caps on warranty liability? What are the monetary limits and percentages of purchase price?"),
        ("Time Limitations", "What are the time limitations for bringing warranty claims? How long can claims be brought for different types of warranties?"),
        ("Specific Indemnities", "What specific indemnities are provided for pre-sign issues? Are there indemnities for tax, employment, data privacy, litigation, etc.?"),
        ("Tax Covenant/Indemnity", "What are the seller's obligations regarding pre-Completion taxes? What procedural controls are mentioned?"),
        ("Locked-Box Leakage", "What protection is provided against value leakage between accounts date and Completion? What constitutes leakage?"),
        ("Purchase-Price Adjustment", "How is the purchase price adjusted? What are the net-debt and working-capital true-up mechanisms?"),
        ("Earn-out Mechanics", "Are there any earn-out provisions? What are the KPI definitions, measurement criteria, and buyer discretion limitations?"),
        ("Material Adverse Change", "How is Material Adverse Change defined? What are the conditions and carve-outs mentioned?"),
        ("Disclosure & Knowledge", "How are disclosures and knowledge qualifiers defined? What information qualifies warranties?"),
        ("Set-off / Withholding", "Can the buyer set indemnity claims against consideration? What are the conditions for set-off?"),
        ("Escrow / Retention", "What portion of the price is held back for warranty cover? What are the release conditions and timeframes?"),
        ("Fundamental Warranties", "What fundamental warranties are provided? What aspects of title, capacity, authority, and share capital are covered?"),
        ("Pre-Completion Covenants", "What are the 'Business as usual' obligations between signing and Completion? What restrictions apply?"),
        ("Non-compete / Non-solicit", "What are the seller restrictions after closing? What geographical and time limitations apply?"),
        ("Consequential-loss Exclusion", "Are lost profits and consequential losses excluded? What specific exclusions and carve-outs are mentioned?"),
        ("Knowledge Scrape-out", "What is the buyer's right to claim despite knowing about breaches? How is knowledge defined?"),
        ("Third-party Claims", "How are claims against the target post-Completion handled? Who controls defense and settlement?"),
        ("Dispute Resolution", "What dispute resolution mechanisms are specified? Which courts or arbitration venues are designated?"),
        ("Fraud / Misconduct Carve-out", "Do caps and limits fall away in case of fraud? What constitutes fraud or wilful misconduct?")
    ]
    
    # Predefined question buttons
    st.markdown("**Quick Questions:**")
    cols = st.columns(4)
    for i, (topic_name, question) in enumerate(predefined_questions):
        col_idx = i % 4
        with cols[col_idx]:
            if st.button(topic_name, key=f"single_q_{i}"):
                st.session_state.single_query_input = question
                st.rerun()
    
    # Query input and settings
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "Enter your query about the selected SPA:",
            height=100,
            key="single_query_input"
        )
    
    with col2:
        k_docs = st.number_input(
            "Documents to retrieve:",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            key="single_k_docs",
            help="Number of relevant document chunks to retrieve for analysis"
        )
    
    if st.button("Analyze", key="single_analyze_btn"):
        if not query.strip():
            st.warning("Please enter a query.")
            return
        
        try:
            with st.spinner("Analyzing SPA..."):
                # Use the unified answer method
                result = assistant.answer(
                    question=query,
                    spa_names=[selected_spa],
                    memory=st.session_state.memory_enabled,
                    k=k_docs
                )
                response = result.get('output_text', str(result))
                
                # Add to conversation history if memory is enabled
                if st.session_state.memory_enabled:
                    st.session_state.conversation_history.append({
                        'query': query,
                        'spas': [selected_spa],
                        'response': response
                    })
            
            st.subheader("Analysis Result")
            st.write(response)
            
            # Show retrieved documents if available
            if 'input_documents' in result and result['input_documents']:
                with st.expander(f"📄 View Retrieved Documents ({len(result['input_documents'])} chunks)", expanded=False):
                    for i, doc in enumerate(result['input_documents'], 1):
                        st.markdown(f"**Document Chunk {i}:**")
                        
                        # Show metadata if available
                        if hasattr(doc, 'metadata') and doc.metadata:
                            metadata_info = []
                            if 'page' in doc.metadata:
                                metadata_info.append(f"Page: {doc.metadata['page']}")
                            if 'source' in doc.metadata:
                                metadata_info.append(f"Source: {doc.metadata.get('source', 'unknown')}")
                            if metadata_info:
                                st.caption(" | ".join(metadata_info))
                        
                        # Show document content
                        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                        st.text_area(
                            f"Content {i}:",
                            value=content,
                            height=150,
                            key=f"single_doc_content_{i}",
                            disabled=True
                        )
                        
                        if i < len(result['input_documents']):
                            st.divider()
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

def multi_spa_query(assistant, available_spas: List[str]):
    """Handle multi-SPA comparative query interface with portfolio company selection."""
    st.subheader("Multi-SPA Comparative Analysis")
    
    # Portfolio company selection
    portfolio_companies = get_available_portfolio_companies(assistant)
    
    if not portfolio_companies:
        st.warning("No portfolio companies found in the database.")
        return
    
    selected_portfolio = st.selectbox(
        "First, select a Portfolio Company:",
        options=[""] + portfolio_companies,
        key="multi_portfolio_select"
    )
    
    if not selected_portfolio:
        st.info("Please select a portfolio company to see available SPAs.")
        return
    
    # SPA selection based on portfolio company (minimum 2)
    portfolio_spas = get_spas_for_portfolio_company(assistant, selected_portfolio)
    
    if not portfolio_spas:
        st.warning(f"No SPAs found for portfolio company: {selected_portfolio}")
        return
    
    if len(portfolio_spas) < 2:
        st.warning(f"Only {len(portfolio_spas)} SPA found for {selected_portfolio}. Need at least 2 SPAs for comparative analysis.")
        return
    
    selected_spas = st.multiselect(
        f"Select SPAs from {selected_portfolio} for comparison (minimum 2):",
        options=portfolio_spas,
        key="multi_spa_select"
    )
    
    if len(selected_spas) < 2:
        st.info("Please select at least 2 SPAs for comparative analysis.")
        return
    
    # Predefined comparative questions
    predefined_questions = [
        ("De-minimis & Basket", "What are the de-minimis thresholds and basket amounts for claims? What are the specific monetary amounts and percentages mentioned?"),
        ("Overall Caps", "What are the overall caps on warranty liability? What are the monetary limits and percentages of purchase price?"),
        ("Time Limitations", "What are the time limitations for bringing warranty claims? How long can claims be brought for different types of warranties?"),
        ("Specific Indemnities", "What specific indemnities are provided for pre-sign issues? Are there indemnities for tax, employment, data privacy, litigation, etc.?"),
        ("Tax Covenant/Indemnity", "What are the seller's obligations regarding pre-Completion taxes? What procedural controls are mentioned?"),
        ("Locked-Box Leakage", "What protection is provided against value leakage between accounts date and Completion? What constitutes leakage?"),
        ("Purchase-Price Adjustment", "How is the purchase price adjusted? What are the net-debt and working-capital true-up mechanisms?"),
        ("Earn-out Mechanics", "Are there any earn-out provisions? What are the KPI definitions, measurement criteria, and buyer discretion limitations?"),
        ("Material Adverse Change", "How is Material Adverse Change defined? What are the conditions and carve-outs mentioned?"),
        ("Disclosure & Knowledge", "How are disclosures and knowledge qualifiers defined? What information qualifies warranties?"),
        ("Set-off / Withholding", "Can the buyer set indemnity claims against consideration? What are the conditions for set-off?"),
        ("Escrow / Retention", "What portion of the price is held back for warranty cover? What are the release conditions and timeframes?"),
        ("Fundamental Warranties", "What fundamental warranties are provided? What aspects of title, capacity, authority, and share capital are covered?"),
        ("Pre-Completion Covenants", "What are the 'Business as usual' obligations between signing and Completion? What restrictions apply?"),
        ("Non-compete / Non-solicit", "What are the seller restrictions after closing? What geographical and time limitations apply?"),
        ("Consequential-loss Exclusion", "Are lost profits and consequential losses excluded? What specific exclusions and carve-outs are mentioned?"),
        ("Knowledge Scrape-out", "What is the buyer's right to claim despite knowing about breaches? How is knowledge defined?"),
        ("Third-party Claims", "How are claims against the target post-Completion handled? Who controls defense and settlement?"),
        ("Dispute Resolution", "What dispute resolution mechanisms are specified? Which courts or arbitration venues are designated?"),
        ("Fraud / Misconduct Carve-out", "Do caps and limits fall away in case of fraud? What constitutes fraud or wilful misconduct?")
    ]
    
    # Predefined question buttons  
    st.markdown("**Quick Comparative Questions:**")
    cols = st.columns(4)
    for i, (topic_name, question) in enumerate(predefined_questions):
        col_idx = i % 4
        with cols[col_idx]:
            if st.button(topic_name, key=f"multi_q_{i}"):
                st.session_state.multi_query_input = question
                st.rerun()
    
    # Query input and settings
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "Enter your comparative query:",
            height=100,
            key="multi_query_input"
        )
    
    with col2:
        k_per_spa = st.number_input(
            "Documents per SPA:",
            min_value=1,
            max_value=25,
            value=5,
            step=1,
            key="multi_k_per_spa",
            help="Number of relevant document chunks to retrieve from each SPA"
        )
        
        retrieval_method = st.selectbox(
            "Retrieval method:",
            options=["similarity", "mmr"],
            index=0,
            key="multi_retrieval_method",
            help="similarity: most relevant chunks, mmr: diverse relevant chunks"
        )
    
    if st.button("Analyze", key="multi_analyze_btn"):
        if not query.strip():
            st.warning("Please enter a query.")
            return
        
        try:
            with st.spinner("Performing comparative analysis..."):
                # Use the unified answer method
                result = assistant.answer(
                    question=query,
                    spa_names=selected_spas,
                    memory=st.session_state.memory_enabled,
                    k_per_spa=k_per_spa,
                    retrieval_method=retrieval_method
                )
                response = result.get('output_text', str(result))
                
                # Add to conversation history if memory is enabled
                if st.session_state.memory_enabled:
                    st.session_state.conversation_history.append({
                        'query': query,
                        'spas': selected_spas,
                        'response': response
                    })
            
            st.subheader("Comparative Analysis Result")
            st.write(response)
            
            # Show retrieved documents if available
            if 'input_documents' in result and result['input_documents']:
                with st.expander(f"📄 View Retrieved Documents ({len(result['input_documents'])} chunks)", expanded=False):
                    # Group documents by SPA if document_name is available
                    docs_by_spa = {}
                    ungrouped_docs = []
                    
                    for doc in result['input_documents']:
                        if hasattr(doc, 'metadata') and doc.metadata and 'document_name' in doc.metadata:
                            spa_name = doc.metadata['document_name']
                            if spa_name not in docs_by_spa:
                                docs_by_spa[spa_name] = []
                            docs_by_spa[spa_name].append(doc)
                        else:
                            ungrouped_docs.append(doc)
                    
                    # Display documents grouped by SPA
                    doc_counter = 1
                    for spa_name, docs in docs_by_spa.items():
                        st.markdown(f"**📋 {spa_name} ({len(docs)} chunks)**")
                        for doc in docs:
                            st.markdown(f"**Document Chunk {doc_counter}:**")
                            
                            # Show metadata if available
                            if hasattr(doc, 'metadata') and doc.metadata:
                                metadata_info = []
                                if 'page' in doc.metadata:
                                    metadata_info.append(f"Page: {doc.metadata['page']}")
                                if 'source' in doc.metadata:
                                    metadata_info.append(f"Source: {doc.metadata.get('source', 'unknown')}")
                                if metadata_info:
                                    st.caption(" | ".join(metadata_info))
                            
                            # Show document content
                            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                            st.text_area(
                                f"Content {doc_counter}:",
                                value=content,
                                height=150,
                                key=f"multi_doc_content_{doc_counter}",
                                disabled=True
                            )
                            doc_counter += 1
                            
                            if doc_counter <= len(result['input_documents']):
                                st.divider()
                    
                    # Display any ungrouped documents
                    if ungrouped_docs:
                        st.markdown(f"**📄 Other Documents ({len(ungrouped_docs)} chunks)**")
                        for doc in ungrouped_docs:
                            st.markdown(f"**Document Chunk {doc_counter}:**")
                            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                            st.text_area(
                                f"Content {doc_counter}:",
                                value=content,
                                height=150,
                                key=f"multi_doc_content_{doc_counter}",
                                disabled=True
                            )
                            doc_counter += 1
                            
                            if doc_counter <= len(result['input_documents']):
                                st.divider()
            
        except Exception as e:
            st.error(f"Comparative analysis failed: {str(e)}")

def full_analysis(assistant, available_spas: List[str]):
    """Handle full analysis interface with portfolio company selection."""
    st.subheader("Full Analysis (20 Topics)")
    
    # Portfolio company selection
    portfolio_companies = get_available_portfolio_companies(assistant)
    
    if not portfolio_companies:
        st.warning("No portfolio companies found in the database.")
        return
    
    selected_portfolio = st.selectbox(
        "First, select a Portfolio Company:",
        options=[""] + portfolio_companies,
        key="full_portfolio_select"
    )
    
    if not selected_portfolio:
        st.info("Please select a portfolio company to see available SPAs.")
        return
    
    # SPA selection based on portfolio company
    portfolio_spas = get_spas_for_portfolio_company(assistant, selected_portfolio)
    
    if not portfolio_spas:
        st.warning(f"No SPAs found for portfolio company: {selected_portfolio}")
        return
    
    selected_spas = st.multiselect(
        f"Select SPAs from {selected_portfolio} for comprehensive analysis:",
        options=portfolio_spas,
        key="full_analysis_spa_select"
    )
    
    if not selected_spas:
        st.info("Please select at least one SPA for full analysis.")
        return
    
    # Analysis options
    col1, col2, col3 = st.columns(3)
    with col1:
        use_memory = st.checkbox(
            "Use conversation memory",
            value=st.session_state.memory_enabled,
            key="full_analysis_memory"
        )
    
    with col2:
        k_per_question = st.number_input(
            "Documents per question:",
            min_value=1,
            max_value=25,
            value=7,
            step=1,
            key="full_analysis_k_per_question",
            help="Number of document chunks to retrieve for each analysis topic"
        )
    
    with col3:
        retrieval_method = st.selectbox(
            "Retrieval method:",
            options=["similarity", "mmr"],
            index=0,
            key="full_analysis_retrieval_method",
            help="similarity: most relevant, mmr: diverse relevant"
        )
    
    if st.button("Run Full Analysis", key="full_analysis_btn"):
        try:
            # Create progress tracking containers
            st.info("Starting comprehensive analysis of all 20 SPA topics...")
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            

            
            total_topics = 20
            
            # Define topic names for progress tracking
            topic_names = [
                "Purchase Price & Consideration",
                "Completion Accounts & Adjustment Mechanisms", 
                "Warranties & Representations",
                "Indemnities",
                "Liability Cap & Limits",
                "Time Limits & Survival Periods",
                "Material Adverse Change (MAC)",
                "De Minimis & Basket Thresholds",
                "Escrow & Security Arrangements",
                "Tax Provisions",
                "Employee & Pension Arrangements",
                "Restrictive Covenants & Non-Compete",
                "Completion & Conditions Precedent",
                "Termination Rights & Break Fees",
                "Disclosure Letter & Data Room",
                "Consequential-loss Exclusion",
                "Knowledge Scrape-out",
                "Third-party Claims Conduct",
                "Dispute Resolution & Governing Law",
                "Fraud / Wilful Misconduct Carve-out"
            ]
            
            # Show analysis information
            
            
            # Show all topics that will be analyzed
            with st.expander("📋 Topics to be analyzed (click to expand)", expanded=False):
                for i, topic in enumerate(topic_names, 1):
                    st.write(f"{i}. {topic}")
            

            

            

            
            with st.spinner("Running comprehensive analysis..."):
                start_time = time.time()
                

                
                results = assistant.full_analysis(
                    spa_names=selected_spas,
                    memory=use_memory,
                    k_per_question=k_per_question,
                    retrieval_method=retrieval_method
                )
                
                end_time = time.time()
                elapsed_time = int(end_time - start_time)
                
                # Show completion
                if 'topics_analysis' in results:
                    completed_count = len(results['topics_analysis'])
                    progress_bar.progress(1.0)

                    status_placeholder.success(f"Full analysis finished successfully!")
                    
                    # Show completion summary
                    success_count = sum(1 for topic_data in results['topics_analysis'].values() 
                                      if 'error' not in topic_data)
                    error_count = completed_count - success_count
                    
                    summary_text = f"**Final Analysis Summary:**\n"
                    summary_text += f"- ✅ Successfully analyzed: {success_count} topics\n"
                    if error_count > 0:
                        summary_text += f"- ❌ Topics with errors: {error_count}\n"
                    summary_text += f"- ⏱️ Total time: {elapsed_time//60}min {elapsed_time%60}sec\n"
                    summary_text += f"- 📊 Average time per topic: {elapsed_time//completed_count if completed_count > 0 else 0}sec\n"
                    summary_text += f"- 📄 Documents retrieved per topic: {k_per_question}\n"
                    summary_text += f"- 🔍 Retrieval method: {retrieval_method}"
                    
                    st.markdown(summary_text)
            
            st.subheader("Full Analysis Results")
            
            # Display results using the topics_analysis structure
            if 'topics_analysis' in results and results['topics_analysis']:
                for topic_key, topic_data in results['topics_analysis'].items():
                    # Create a clean topic title
                    topic_title = f"TOPIC {topic_data.get('topic_number', '?')}: {topic_data.get('topic_name', topic_key)}"
                    
                    with st.expander(topic_title, expanded=False):
                        # Show topic description
                        if 'topic_description' in topic_data:
                            st.markdown(f"**Description:** {topic_data['topic_description']}")
                            st.divider()
                        
                        # Show the question asked
                        if 'question_asked' in topic_data:
                            st.markdown(f"**Question:** {topic_data['question_asked']}")
                            st.divider()
                        
                        # Show the analysis result
                        if 'comparative_answer' in topic_data:
                            st.markdown("**Analysis:**")
                            st.write(topic_data['comparative_answer'])
                            st.divider()
                        
                        # Show document retrieval stats
                        total_docs = topic_data.get('total_documents_retrieved', 0)
                        doc_counts = topic_data.get('document_counts_per_spa', {})
                        if total_docs > 0:
                            st.markdown(f"**Documents Retrieved:** {total_docs} total")
                            if doc_counts:
                                counts_text = " | ".join([f"{spa}: {count}" for spa, count in doc_counts.items()])
                                st.caption(counts_text)
                        
                        # Show retrieved documents if available
                        source_pages = topic_data.get('source_pages', [])
                        if source_pages:
                            with st.expander(f"📄 View Source Documents ({len(source_pages)} chunks)", expanded=False):
                                # Group documents by SPA
                                docs_by_spa = {}
                                for page_info in source_pages:
                                    spa_name = page_info.get('spa_name', 'Unknown')
                                    if spa_name not in docs_by_spa:
                                        docs_by_spa[spa_name] = []
                                    docs_by_spa[spa_name].append(page_info)
                                
                                # Display documents grouped by SPA
                                doc_counter = 1
                                for spa_name, docs in docs_by_spa.items():
                                    st.markdown(f"**📋 {spa_name} ({len(docs)} chunks)**")
                                    for doc_info in docs:
                                        st.markdown(f"**Document Chunk {doc_counter}:**")
                                        
                                        # Show metadata
                                        metadata_info = []
                                        if 'page' in doc_info and doc_info['page'] != 'unknown':
                                            metadata_info.append(f"Page: {doc_info['page']}")
                                        if 'source' in doc_info and doc_info['source'] != 'unknown':
                                            metadata_info.append(f"Source: {doc_info['source']}")
                                        if metadata_info:
                                            st.caption(" | ".join(metadata_info))
                                        
                                        # Show document content
                                        content = doc_info.get('content_preview', '')
                                        if content:
                                            st.text_area(
                                                f"Content {doc_counter}:",
                                                value=content,
                                                height=150,
                                                key=f"full_analysis_{topic_key}_doc_{doc_counter}",
                                                disabled=True
                                            )
                                        doc_counter += 1
                                        
                                        if doc_counter <= len(source_pages):
                                            st.divider()
                        
                        # Show error if any
                        if 'error' in topic_data:
                            st.error(f"Error in analysis: {topic_data['error']}")
            else:
                # Fallback for old format
                for topic, analysis in results.items():
                    if topic != 'topics_analysis':
                        with st.expander(f"📊 {topic.replace('_', ' ').title()}", expanded=False):
                            st.write(analysis)
            
            # Provide JSON download
            st.subheader("Download Results")
            results_json = json.dumps(results, indent=2, ensure_ascii=False)
            
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                label="📥 Download Full Analysis (JSON)",
                data=results_json,
                file_name=f"spa_full_analysis_{'-'.join(selected_spas)}_{timestamp}.json",
                mime="application/json",
                key="download_full_analysis"
            )
            
            # Add to conversation history if memory is enabled
            if use_memory:
                st.session_state.conversation_history.append({
                    'query': f"Full analysis of {len(selected_spas)} SPAs",
                    'spas': selected_spas,
                    'response': "Full analysis completed. Results available for download."
                })
            
        except Exception as e:
            st.error(f"Full analysis failed: {str(e)}")

def document_upload_interface(assistant):
    """Interface for uploading new SPA documents to the database and managing existing ones."""
    st.subheader("Document Management")
    st.write("Upload new SPA documents or manage existing ones in the vector database.")
    
    # Create tabs for Upload and Delete
    upload_tab, delete_tab = st.tabs(["📤 Upload SPA", "🗑️ Delete SPA"])
    
    with upload_tab:
        st.subheader("Upload New SPA Document")
        upload_spa_interface(assistant)
    
    with delete_tab:
        st.subheader("Delete SPA Document")
        delete_spa_interface(assistant)

def upload_spa_interface(assistant):
    """Interface for uploading new SPA documents."""
    
    # Import the required classes
    try:
        from attached_assets.SPA_Tools_Corrected_1754901908313 import DocumentChunker, VectorStoreEmbedder
    except ImportError:
        st.error("Could not import DocumentChunker and VectorStoreEmbedder classes.")
        return
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        spa_name = st.text_input(
            "SPA Name:",
            placeholder="Enter the name of the SPA document",
            help="This will be used as the document identifier in the database"
        )
    
    with col2:
        portfolio_company = st.text_input(
            "Portfolio Company:",
            placeholder="Enter the portfolio company name",
            help="This will be used to group SPAs by portfolio company"
        )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a PDF file:",
        type=['pdf'],
        help="Upload the SPA document in PDF format"
    )
    
    # Advanced settings (optional)
    with st.expander("Advanced Settings (Optional)", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            min_chunk_size = st.number_input(
                "Minimum Chunk Size:",
                min_value=500,
                max_value=5000,
                value=1100,
                step=100,
                help="Minimum size for document chunks in characters"
            )
            
            breakpoint_threshold = st.slider(
                "Breakpoint Threshold:",
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                step=0.1,
                help="Threshold for semantic chunking (higher = fewer, larger chunks)"
            )
        
        with col2:
            recursive_chunk_size = st.number_input(
                "Recursive Chunk Size:",
                min_value=1000,
                max_value=5000,
                value=2000,
                step=100,
                help="Chunk size for recursive text splitting"
            )
            
            recursive_chunk_overlap = st.number_input(
                "Chunk Overlap:",
                min_value=50,
                max_value=500,
                value=200,
                step=50,
                help="Overlap between chunks for better context"
            )
    
    # Upload button
    if st.button("📤 Upload and Process Document", type="primary"):
        # Validation
        if not spa_name.strip():
            st.error("Please enter a SPA name.")
            return
        
        if not portfolio_company.strip():
            st.error("Please enter a portfolio company name.")
            return
        
        if uploaded_file is None:
            st.error("Please upload a PDF file.")
            return
        
        try:
            # Save uploaded file temporarily
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Get configuration
            config = get_config()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Initialize DocumentChunker
                status_text.text("Initializing document chunker...")
                progress_bar.progress(10)
                
                chunker = DocumentChunker(
                    pdf_path=tmp_file_path,
                    SPA_name=spa_name.strip(),
                    min_chunk_size=min_chunk_size,
                    breakpoint_threshold_amount=breakpoint_threshold,
                    recursive_chunk_size=recursive_chunk_size,
                    recursive_chunk_overlap=recursive_chunk_overlap,
                    api_key=config['google_api_key'],
                    default_portfolio_company=portfolio_company.strip()
                )
                
                # Step 2: Chunk the document
                status_text.text("Processing and chunking document...")
                progress_bar.progress(30)
                
                chunks = chunker.chunk_documents()
                
                if not chunks:
                    st.error("No chunks were generated from the document. Please check the PDF file.")
                    return
                
                st.success(f"Generated {len(chunks)} chunks from the document.")
                
                # Step 3: Initialize VectorStoreEmbedder
                status_text.text("Initializing vector store embedder...")
                progress_bar.progress(50)
                
                # Step 4: Embed and store chunks
                status_text.text("Embedding and storing chunks in vector database...")
                progress_bar.progress(70)
                
                embedder = VectorStoreEmbedder(
                    persist_directory=config['database_dir'],
                    api_key=config['google_api_key']
                )
                
                embedder.embed_and_store(chunks)
                
                # Step 5: Complete
                status_text.text("Document processing completed!")
                progress_bar.progress(100)
                
                st.success(f"✅ Successfully added '{spa_name}' from '{portfolio_company}' to the database!")
                st.info(f"Document: {spa_name}")
                st.info(f"Portfolio Company: {portfolio_company}")
                st.info(f"Chunks Generated: {len(chunks)}")
                
                # Show chunk preview
                with st.expander("📄 Preview Document Chunks", expanded=False):
                    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                        st.markdown(f"**Chunk {i+1}:**")
                        st.text_area(
                            f"Content Preview {i+1}:",
                            value=chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content,
                            height=100,
                            key=f"chunk_preview_{i}",
                            disabled=True
                        )
                        if i < 2:  # Don't add divider after last chunk
                            st.divider()
                    
                    if len(chunks) > 3:
                        st.caption(f"... and {len(chunks) - 3} more chunks")
                
                # Clear the cache so new SPA appears in other tabs
                if hasattr(assistant, 'get_available_spas') and hasattr(assistant.get_available_spas, 'clear'):
                    assistant.get_available_spas.clear()
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            
            # Clean up on error
            try:
                if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
            except:
                pass
    
    # Show current database status
    st.divider()
    st.subheader("Database Status")
    
    try:
        # Get current portfolio companies and SPAs
        portfolio_companies = get_available_portfolio_companies(assistant)
        if portfolio_companies and len(portfolio_companies) > 1:  # More than just "ALL"
            portfolio_companies = [pc for pc in portfolio_companies if pc != "ALL"]  # Remove ALL for display
            
            st.write(f"**Portfolio Companies in Database:** {len(portfolio_companies)}")
            
            for pc in portfolio_companies:
                spas = get_spas_for_portfolio_company(assistant, pc)
                with st.expander(f"{pc} ({len(spas)} SPAs)", expanded=False):
                    for spa in spas:
                        st.write(f"• {spa}")
        else:
            st.info("No documents found in the database.")
            
    except Exception as e:
        st.warning(f"Could not retrieve database status: {str(e)}")

def delete_spa_interface(assistant):
    """Interface for deleting SPA documents from the database."""
    st.write("Select SPAs to remove from the vector database.")
    
    # Get available portfolio companies and SPAs
    try:
        portfolio_companies = get_available_portfolio_companies(assistant)
        if not portfolio_companies or (len(portfolio_companies) == 1 and portfolio_companies[0] == "ALL"):
            st.info("No SPAs found in the database to delete.")
            return
        
        # Portfolio company selection for filtering
        selected_portfolio = st.selectbox(
            "Select Portfolio Company:",
            options=portfolio_companies,
            key="delete_portfolio_select",
            help="Choose a portfolio company to filter SPAs, or select 'ALL' to see all SPAs"
        )
        
        # Get SPAs for the selected portfolio company
        available_spas = get_spas_for_portfolio_company(assistant, selected_portfolio)
        
        if not available_spas:
            st.warning(f"No SPAs found for portfolio company: {selected_portfolio}")
            return
        
        st.write(f"**Available SPAs for {selected_portfolio}:** {len(available_spas)}")
        
        # SPA selection for deletion
        spas_to_delete = st.multiselect(
            "Select SPAs to delete:",
            options=available_spas,
            key="delete_spa_select",
            help="Choose one or more SPAs to permanently remove from the database"
        )
        
        if spas_to_delete:
            st.warning(f"⚠️ You are about to delete {len(spas_to_delete)} SPA(s): {', '.join(spas_to_delete)}")
            st.error("**This action cannot be undone!**")
            
            # Confirmation checkbox
            confirm_delete = st.checkbox(
                f"I understand that deleting these {len(spas_to_delete)} SPA(s) will permanently remove all associated document chunks from the database",
                key="confirm_delete_checkbox"
            )
            
            if confirm_delete:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🗑️ Delete Selected SPAs", type="primary", key="delete_spas_btn"):
                        delete_spas_from_database(assistant, spas_to_delete)
                
                with col2:
                    if st.button("❌ Cancel", key="cancel_delete_btn"):
                        st.rerun()
    
    except Exception as e:
        st.error(f"Error loading SPAs for deletion: {str(e)}")

def delete_spas_from_database(assistant, spa_names):
    """Delete specified SPAs from the vector database."""
    try:
        # Get the vector store
        vector_store = assistant.retriever.vector_store
        collection = vector_store._collection
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_deleted = 0
        
        for i, spa_name in enumerate(spa_names):
            status_text.text(f"Deleting {spa_name}...")
            progress = int((i / len(spa_names)) * 100)
            progress_bar.progress(progress)
            
            try:
                # Get all document IDs for this SPA
                docs = collection.get(where={"document_name": spa_name})
                
                if docs['ids']:
                    # Delete all chunks for this SPA
                    collection.delete(ids=docs['ids'])
                    deleted_count = len(docs['ids'])
                    total_deleted += deleted_count
                    st.success(f"✅ Deleted {spa_name} ({deleted_count} chunks)")
                else:
                    st.warning(f"⚠️ No chunks found for {spa_name}")
                    
            except Exception as e:
                st.error(f"❌ Failed to delete {spa_name}: {str(e)}")
        
        # Complete
        status_text.text("Deletion completed!")
        progress_bar.progress(100)
        
        if total_deleted > 0:
            st.success(f"🎉 Successfully deleted {len(spa_names)} SPA(s) with {total_deleted} total chunks!")
            
            # Clear any cached data
            if hasattr(assistant, 'get_available_spas') and hasattr(assistant.get_available_spas, 'clear'):
                assistant.get_available_spas.clear()
            
            # Refresh the page to update the interface
            st.balloons()
            time.sleep(2)
            st.rerun()
        else:
            st.warning("No documents were deleted.")
            
    except Exception as e:
        st.error(f"Error during deletion process: {str(e)}")

def main():
    """Main application function."""
    st.title("📄 SPA Analysis Tool")
    st.write("Review and analyze SPA files with comprehensive querying capabilities.")
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize assistant
    assistant = initialize_assistant()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Memory toggle
        memory_enabled = st.toggle(
            "Enable Memory (Enhanced Routes)",
            value=st.session_state.memory_enabled,
            help="When enabled, conversation context persists within the session for enhanced analysis."
        )
        st.session_state.memory_enabled = memory_enabled
        
        if memory_enabled:
            st.info("Memory is enabled. Context will persist across queries in this session.")
            if st.session_state.conversation_history:
                if st.button("🗑️ Clear History", key="clear_history_btn"):
                    clear_conversation_history()
        
        st.divider()
        
        # Analysis mode selection
        st.header("Analysis Mode")
        analysis_mode = st.radio(
            "Select analysis type:",
            options=[
                "Single SPA Query",
                "Multi-SPA Comparative", 
                "Full Analysis",
                "Document Upload"
            ],
            key="analysis_mode_radio"
        )
        st.session_state.analysis_mode = analysis_mode
    
    # Get available SPAs
    try:
        with st.spinner("Loading available SPAs..."):
            available_spas = assistant.get_available_spas()
        
        if not available_spas:
            st.error("No SPAs found in the database. Please ensure the DATABASE_SPA directory contains embedded documents.")
            st.stop()
        
        st.success(f"Found {len(available_spas)} available SPAs")
        
        # Display available SPAs in an expander
        with st.expander(f"📋 Available SPAs ({len(available_spas)})", expanded=False):
            for i, spa in enumerate(available_spas, 1):
                st.write(f"{i}. {spa}")
    
    except Exception as e:
        st.error(f"Failed to load available SPAs: {str(e)}")
        st.stop()
    
    # Main content area
    st.divider()
    
    # Route to appropriate interface based on selected mode
    if st.session_state.analysis_mode == "Single SPA Query":
        single_spa_query(assistant, available_spas)
    elif st.session_state.analysis_mode == "Multi-SPA Comparative":
        multi_spa_query(assistant, available_spas)
    elif st.session_state.analysis_mode == "Full Analysis":
        full_analysis(assistant, available_spas)
    elif st.session_state.analysis_mode == "Document Upload":
        document_upload_interface(assistant)
    
    # Display conversation history at the bottom
    if st.session_state.memory_enabled:
        st.divider()
        display_conversation_history()

if __name__ == "__main__":
    main()
