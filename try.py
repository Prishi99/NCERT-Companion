import os
import streamlit as st
from pathlib import Path
import time
import faiss
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import base64

# Load environment variables
load_dotenv()

# Setup for OpenAI API
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY is not set in the environment variables.")
    st.stop()
os.environ["OPENAI_API_KEY"] = openai_api_key

# Define subject folders
subject_folders = {
    "Biology": Path("/teamspace/studios/this_studio/ncert_dataset/Biology"),
    "Chemistry": Path("/teamspace/studios/this_studio/ncert_dataset/Chemistry")
}

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}
if "is_initialized" not in st.session_state:
    st.session_state.is_initialized = {}
if "question" not in st.session_state:
    st.session_state.question = ""
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

# Function to add custom CSS for styling
def add_custom_css():
    st.markdown("""
    <style>
    /* Main page background with gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: black; /* Set default text color to black */
    }
    
    /* Custom title */
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 42px;
        font-weight: 700;
        color: black; /* Set title color to black */
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        padding: 20px;
        background: rgba(255, 255, 255, 0.7);
        border-radius: 10px;
    }
    
    /* Chat message styling */
    .user-message {
        background-color: rgba(64, 115, 255, 0.1);
        padding: 15px 20px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 5px solid #4073ff;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        color: black; /* Set user message color to black */
    }
    
    .assistant-message {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 15px 20px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 5px solid #34c759;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        color: black; /* Set assistant message color to black */
    }
    
    /* Suggested questions styling */
    .suggested-questions {
        background-color: rgba(255, 255, 255, 0.7);
        padding: 10px 15px;
        border-radius: 10px;
        margin-top: 10px;
    }
    
    .suggested-title {
        font-weight: bold;
        color: black; /* Set suggested title color to black */
        margin-bottom: 8px;
    }
    
    /* Custom button styling */
    .stButton > button {
        background-color: #4073ff;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #2d5ae0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    .suggestion-button > button {
        background-color: #f7f9fc;
        color: #4073ff;
        border: 1px solid #4073ff;
        font-size: 0.9em;
    }
    
    .suggestion-button > button:hover {
        background-color: #e8eeff;
        color: #2d5ae0;
    }
    
    /* Input box styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 15px 20px;
        background-color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        color: black; /* Set input text color to black */
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4073ff;
        box-shadow: 0 4px 10px rgba(64, 115, 255, 0.2);
    }
    
    /* Subject sidebar styling */
    .subject-header {
        font-size: 1.2em;
        font-weight: 600;
        color: black; /* Set subject header color to black */
        margin-bottom: 15px;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: #4073ff;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 10px;
        color: black; /* Set selectbox text color to black */
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(0,0,0,0.1), transparent);
        margin: 20px 0;
    }
    
    /* Source citation styling */
    .source-citation {
        font-size: 0.9em;
        font-style: italic;
        color: black; /* Set source citation color to black */
        border-top: 1px solid #eee;
        padding-top: 8px;
        margin-top: 10px;
    }
    
    /* Card container for the sidebar */
    .sidebar-card {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        color: black; /* Set card text color to black */
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        font-size: 0.9em;
        color: black; /* Set footer color to black */
        border-top: 1px solid #eee;
    }
    
    /* Animations for messages */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animated {
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Function to handle asking questions
def ask_question():
    if st.session_state.question:
        process_question(st.session_state.question)
        # Clear the question input after submission
        st.session_state.question = ""

# Function to handle clicking on suggested questions
def ask_suggested_question(question):
    st.session_state.question = question
    process_question(question)
    # Clear the question input after submission
    st.session_state.question = ""

# Function to generate suggested follow-up questions
def generate_suggested_questions(question, answer, subject):
    try:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
        )
        
        # Create a prompt for generating related questions
        prompt = f"""
        Based on the following question and answer about {subject}, suggest 3 follow-up questions that would help the user explore this topic further:
        
        QUESTION: {question}
        
        ANSWER: {answer}
        
        Generate 3 engaging, concise follow-up questions that are directly related to the topic and would help deepen understanding. 
        Format the response as a simple list of questions, one per line. Do not include any explanations, numbering, or extra text.
        """
        
        # Get response from LLM
        response = llm.invoke(prompt)
        
        # Parse the response into a list of questions
        suggested_questions = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
        
        # Limit to 3 questions maximum
        return suggested_questions[:3]
    
    except Exception as e:
        st.error(f"Error generating suggested questions: {str(e)}")
        return []

# Function to process the question and get an answer
def process_question(question):
    # Get the selected subject
    subject = st.session_state.get("subject", list(subject_folders.keys())[0])
    
    # Load vector store if not already loaded
    vector_store = load_vector_store(subject)
    
    if vector_store is None:
        st.error(f"Failed to load vector store for {subject}.")
        return
    
    # Add user question to history
    st.session_state.chat_history.append({"role": "user", "content": question})
    
    # Get answer
    with st.spinner("Generating answer..."):
        # Create conversation chain
        chain = get_conversation_chain(vector_store)
        
        try:
            # Get response from chain
            response = chain({"question": question})
            answer = response.get("answer", "I couldn't find an answer to that question.")
            sources = response.get("source_documents", [])
            
            # Add sources citation to answer if available
            if sources:
                source_text = "\n\n**Sources:**\n"
                for i, doc in enumerate(sources[:2]):  # Limit to top 2 sources
                    source = doc.metadata.get('source', '').split('/')[-1] if 'source' in doc.metadata else 'Unknown'
                    source_text += f"- {source}\n"
                answer += source_text
            
            # Add bot response to history
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            
            # Generate suggested follow-up questions
            st.session_state.suggested_questions = generate_suggested_questions(question, answer, subject)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            st.session_state.suggested_questions = []

# Function to clear the chat history
def clear_chat():
    st.session_state.chat_history = []
    st.session_state.suggested_questions = []

# Configuration for faster loading
@st.cache_resource
def load_vector_store(subject):
    """Load or create FAISS vector store for a subject with caching"""
    start_time = time.time()
    
    # Skip if already initialized
    if subject in st.session_state.is_initialized and st.session_state.is_initialized[subject]:
        return st.session_state.vector_stores[subject]
    
    # Load all PDFs from the subject folder
    pdf_files = list(subject_folders[subject].glob("**/*.pdf"))
    
    if not pdf_files:
        st.error(f"No PDF files found in {subject_folders[subject]}")
        return None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Loading {len(pdf_files)} PDF files for {subject}...")
    
    # Load documents with progress reporting
    documents = []
    for i, pdf in enumerate(pdf_files):
        try:
            docs = PyPDFLoader(str(pdf)).load()
            documents.extend(docs)
        except Exception as e:
            pass  # Silently skip problematic files for faster loading
        
        # Update progress
        progress = int((i + 1) / len(pdf_files) * 50)  # First half of progress bar
        progress_bar.progress(progress)
        status_text.text(f"Processed {i+1}/{len(pdf_files)} files...")
    
    # Split documents into chunks
    status_text.text("Creating text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Larger chunks for fewer embeddings to compute
        chunk_overlap=50,  # Reduced overlap for faster processing
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and FAISS index
    status_text.text(f"Creating vector embeddings for {len(chunks)} chunks...")
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        chunk_size=1000  # Process in larger batches
    )
    
    # Update progress for embedding creation
    for i in range(51, 90):
        time.sleep(0.01)  # Simulate progress
        progress_bar.progress(i)
    
    # Build vector store with fewer dimensions for faster loading
    vector_store = FAISS.from_documents(chunks, embedding_model)
    
    # Try to move to GPU if available
    try:
        gpu_resources = faiss.StandardGpuResources()
        vector_store.index = faiss.index_cpu_to_gpu(gpu_resources, 0, vector_store.index)
        status_text.text("FAISS index moved to GPU for faster search!")
    except Exception:
        status_text.text("Using CPU for FAISS (GPU acceleration unavailable)")
    
    # Complete progress bar
    progress_bar.progress(100)
    
    # Store in session state
    st.session_state.vector_stores[subject] = vector_store
    st.session_state.is_initialized[subject] = True
    
    elapsed = time.time() - start_time
    status_text.text(f"Loaded {len(chunks)} chunks in {elapsed:.2f} seconds")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    return vector_store

def get_conversation_chain(vector_store):
    """Create a conversational chain with the vector store"""
    # Initialize language model
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.2,
    )
    
    # Initialize memory with explicit output key
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",  # Specify the output key explicitly to fix the error
        return_messages=True
    )
    
    # Create retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Reduced k for faster retrieval
        ),
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    
    return chain

def main():
    st.set_page_config(
        page_title="NCERT Learning Assistant", 
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    add_custom_css()
    
    # Title with custom styling
    st.markdown("<h1 class='main-title'>📚 QuestNCERT</h1>", unsafe_allow_html=True)
    
    # Two column layout with adjusted ratio
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Sidebar card for subject selection
        st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
        st.markdown("<p class='subject-header'>Select Your Subject</p>", unsafe_allow_html=True)
        
        # Subject selection with icons
        subject_options = {
            "Biology": "🧬 Biology",
            "Chemistry": "⚗️ Chemistry"
        }
        
        subject = st.selectbox(
            "Choose a subject to explore",
            options=list(subject_folders.keys()),
            format_func=lambda x: subject_options.get(x, x),
            key="subject"
        )
        
        # Descriptive text
        if subject == "Biology":
            st.markdown("Explore living organisms, their structure, function, growth, and evolution.")
        elif subject == "Chemistry":
            st.markdown("Discover the properties, composition, and behavior of matter.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Action buttons in a card
        st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
        st.markdown("<p class='subject-header'>Actions</p>", unsafe_allow_html=True)
        
        # Preload button with icon
        if st.button("🔄 Load Knowledge Base"):
            with st.spinner(f"Loading {subject} knowledge base..."):
                load_vector_store(subject)

        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            clear_chat()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Display chat history
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                role = message["role"]
                content = message["content"]
                
                if role == "user":
                    st.markdown(f"<div class='user-message animated'>{content}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='assistant-message animated'>{content}</div>", unsafe_allow_html=True)
        
        # Input box for the question
        st.text_input("Ask me anything about the topic:", key="question", on_change=ask_question)
        
        # Display suggested questions
        if st.session_state.suggested_questions:
            st.markdown("<div class='suggested-questions'>", unsafe_allow_html=True)
            st.markdown("<p class='suggested-title'>Explore Further</p>", unsafe_allow_html=True)
            
            for q in st.session_state.suggested_questions:
                if st.button(q, key=f"suggested_{q}", on_click=ask_suggested_question, args=(q,), type="primary", use_container_width=True, disabled=False):
                    pass
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("<footer class='footer'>Made with ❤️ by AI Enthusiasts</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
