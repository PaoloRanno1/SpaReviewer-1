import streamlit as st
import os
import json
import logging
import asyncio
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
        try:
            loop = asyncio.get_event_loop()
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

def single_spa_query(assistant, available_spas: List[str]):
    """Handle single SPA query interface."""
    st.subheader("Single SPA Query")
    
    # SPA selection
    selected_spa = st.selectbox(
        "Select an SPA for analysis:",
        options=[""] + available_spas,
        key="single_spa_select"
    )
    
    if not selected_spa:
        st.info("Please select an SPA to proceed.")
        return
    
    # Query input
    query = st.text_area(
        "Enter your query about the selected SPA:",
        height=100,
        key="single_query_input"
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
                    memory=st.session_state.memory_enabled
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
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

def multi_spa_query(assistant, available_spas: List[str]):
    """Handle multi-SPA comparative query interface."""
    st.subheader("Multi-SPA Comparative Analysis")
    
    # SPA selection (minimum 2)
    selected_spas = st.multiselect(
        "Select SPAs for comparative analysis (minimum 2):",
        options=available_spas,
        key="multi_spa_select"
    )
    
    if len(selected_spas) < 2:
        st.info("Please select at least 2 SPAs for comparative analysis.")
        return
    
    # Query input
    query = st.text_area(
        "Enter your comparative query:",
        height=100,
        key="multi_query_input"
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
                    memory=st.session_state.memory_enabled
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
            
        except Exception as e:
            st.error(f"Comparative analysis failed: {str(e)}")

def full_analysis(assistant, available_spas: List[str]):
    """Handle full analysis interface."""
    st.subheader("Full Analysis")
    st.write("Perform comprehensive multi-topic analysis across selected SPAs.")
    
    # SPA selection
    selected_spas = st.multiselect(
        "Select SPAs for full analysis:",
        options=available_spas,
        key="full_analysis_spa_select"
    )
    
    if not selected_spas:
        st.info("Please select at least one SPA for full analysis.")
        return
    
    # Analysis options
    col1, col2 = st.columns(2)
    with col1:
        include_comparative = st.checkbox(
            "Include comparative analysis",
            value=len(selected_spas) > 1,
            disabled=len(selected_spas) <= 1,
            key="include_comparative"
        )
    
    with col2:
        use_memory = st.checkbox(
            "Use conversation memory",
            value=st.session_state.memory_enabled,
            key="full_analysis_memory"
        )
    
    if st.button("Run Full Analysis", key="full_analysis_btn"):
        try:
            with st.spinner("Running comprehensive analysis... This may take a few minutes."):
                conversation_history = st.session_state.conversation_history if use_memory else []
                
                results = assistant.full_analysis(
                    spa_names=selected_spas,
                    memory=use_memory
                )
            
            st.subheader("Full Analysis Results")
            
            # Display results in organized sections
            for topic, analysis in results.items():
                with st.expander(f"📊 {topic.replace('_', ' ').title()}", expanded=True):
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
                "Full Analysis"
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
    
    # Display conversation history at the bottom
    if st.session_state.memory_enabled:
        st.divider()
        display_conversation_history()

if __name__ == "__main__":
    main()
