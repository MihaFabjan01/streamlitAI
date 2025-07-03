import streamlit as st
import chromadb
from transformers import pipeline
from pathlib import Path
import tempfile
from datetime import datetime
import time


# Document conversion imports (copy from your converter app)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice

# TODO: Copy your convert_to_markdown function here
def convert_to_markdown(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        pdf_opts = PdfPipelineOptions(do_ocr=False)
        pdf_opts.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CPU
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_opts,
                    backend=DoclingParseV2DocumentBackend
                )
            }
        )
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext in [".doc", ".docx"]:
        converter = DocumentConverter()
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext == ".txt":
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="replace")

    raise ValueError(f"Unsupported extension: {ext}")


# Simple Q&A App using Streamlit
# Students: Replace the documents below with your own!


# Fix SQLite version issue for ChromaDB on Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# IMPORTS - These are the libraries we need
import streamlit as st          # Creates web interface components
import chromadb                # Stores and searches through documents  
from transformers import pipeline  # AI model for generating answers


def get_answer(collection, question):
    """
    This function searches documents and generates answers while minimizing hallucination
    """
    
    # STEP 1: Search for relevant documents in the database
    # We get 3 documents instead of 2 for better context coverage
    results = collection.query(
        query_texts=[question],    # The user's question
        n_results=3               # Get 3 most similar documents
    )
    
    # STEP 2: Extract search results
    # docs = the actual document text content
    # distances = how similar each document is to the question (lower = more similar)
    docs = results["documents"][0]
    distances = results["distances"][0]
    
    # STEP 3: Check if documents are actually relevant to the question
    # If no documents found OR all documents are too different from question
    # Return early to avoid hallucination
    if not docs or min(distances) > 1.5:  # 1.5 is similarity threshold - adjust as needed
        return "🧠💭 Hmm... I don’t have info on that topic in my science vault just yet. Try asking me something about space, animals, Earth, or amazing discoveries!"
    
    # STEP 4: Create structured context for the AI model
    # Format each document clearly with labels
    # This helps the AI understand document boundaries
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    
    # STEP 5: Build improved prompt to reduce hallucination
    # Key changes from original:
    # - Separate context from instructions
    # - More explicit instructions about staying within context
    # - Clear format structure
    prompt = f"""Context information:
{context}

Question: {question}

Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with "I don't know." Do not add information from outside the context.

Answer:"""
    
    # STEP 6: Generate answer with anti-hallucination parameters
    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(
        prompt, 
        max_length=150
    )
    
    # STEP 7: Extract and clean the generated answer
    answer = response[0]['generated_text'].strip()
    

    
    # STEP 8: Return the final answer
    return answer

# MAIN APP STARTS HERE - This is where we build the user interface

# STREAMLIT BUILDING BLOCK 1: PAGE TITLE
# st.title() creates a large heading at the top of your web page
# The emoji 🤖 makes it more visually appealing
# This appears as the biggest text on your page

st.markdown("""
    <div style="display: flex; justify-content: center;">
        <img src="https://i.imgur.com/39dL9gG.png" alt="ExplAIniac Logo" width="500">
    </div>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='text-align: center; font-size: 27px; color: #FF00FF;'><b>Your AI for Wild Wonders 🧠 & Real Facts🔬</b></div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    @keyframes coin-spin {
        0% {
            transform: perspective(1000px) rotateY(0deg);
        }
        100% {
            transform: perspective(1000px) rotateY(360deg);
        }
    }
    .logo-spinning {
        animation: coin-spin 5s linear infinite;
        border-radius: 50%; 
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.8);
    }
    </style>
    <div style="display: flex; justify-content: center; margin-bottom: 10px;">
        <img src="https://i.imgur.com/yFddIhW.png"
             class="logo-spinning"
             width="160">
    </div>
    """,
    unsafe_allow_html=True
)

# Random motivational science quote section
def get_random_science_quote():
    """Return a random motivational science quote from Hello Bio's collection"""
    quotes = [
        "Everything is theoretically impossible, until it is done. – Robert A. Heinlein",
        "The reward of the young scientist is the emotional thrill of being the first person in the history of the world to see something or to understand something. Nothing can compare with that experience. – Cecilia Payne‑Gaposchkin",
        "What you learn from a life in science is the vastness of our ignorance. – David Eagleman",
        "If I have seen further it is by standing on the shoulders of Giants. – Isaac Newton",
        "If a cluttered desk is a sign of a cluttered mind, of what, then, is an empty desk a sign? – Albert Einstein",
        "Our virtues and our failures are inseparable, like force and matter. When they separate, man is no more. – Nikola Tesla",
        "Impossible only means that you haven't found the solution yet. – Anonymous",
        "In science the credit goes to the man who convinces the world, not to the man to whom the idea first occurs. – Sir William Osler",
        "Every brilliant experiment, like every great work of art, starts with an act of imagination. – Jonah Lehrer",
        "The good thing about science is that it's true whether or not you believe in it. – Neil deGrasse Tyson",
        "Science is not only a disciple of reason but also one of romance and passion. – Stephen Hawking",
        "Science and everyday life cannot and should not be separated. – Rosalind Franklin",
        "All outstanding work, in art as well as in science, results from immense zeal applied to a great idea. – Santiago Ramón y Cajal",
        "If you know you are on the right track, if you have this inner knowledge, then nobody can turn you off... no matter what they say. – Barbara McClintock",
        "Above all, don't fear difficult moments. The best comes from them. – Rita Levi‑Montalcini",
        "Research is to see what everybody else has seen, and to think what nobody else has thought. – Albert Szent‑Györgyi",
        "If you want to have good ideas, you must have many ideas. – Linus Pauling",
        "We are just an advanced breed of monkeys on a minor planet of a very average star. But we can understand the Universe. That makes us something very special. – Stephen Hawking",
        "Nothing in life is to be feared, it is only to be understood. Now is the time to understand more, so that we may fear less. – Marie Curie",
        "Science means constantly walking a tightrope between blind faith and curiosity; between expertise and creativity; between bias and openness; between experience and epiphany; between ambition and passion; and between arrogance and conviction – in short, between an old today and a new tomorrow. – Heinrich Rohrer",
        "Science knows no country, because knowledge belongs to humanity, and is the torch which illuminates the world. – Louis Pasteur",
        "When kids look up to great scientists the way they do musicians, actors [and sports figures], civilization will jump to the next level. – Brian Greene",
        "The important thing is to never stop questioning [or learning]. – Albert Einstein",
        "We cannot solve problems with the same thinking we used to create them. – Albert Einstein",
        "I am among those who think that science has great beauty. – Marie Curie"
    ]
    import random
    return random.choice(quotes)

# Display random quote
quote = get_random_science_quote()
st.markdown(
    f"<div style='text-align: center; font-size: 16px; font-style: italic; color: #FFD700; margin: 20px 0; padding: 15px; background: rgba(0,0,0,0.3); border-radius: 10px; border-left: 4px solid #FFD700;'>💫 {quote}</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://i.imgur.com/zpbdrBH.gif");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Enhanced Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        padding: 15px 0;
        justify-content: center;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 70px;
        padding: 10px 20px;
        border-radius: 15px;
        font-size: 14px;
        font-weight: bold;
        border: 2px solid transparent;
        background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("https://www.chromethemer.com/wallpapers/chromebook-wallpapers/images/960/galaxy-chromebook-wallpaper.jpg");
        background-size: cover;
        background-position: center;
        backdrop-filter: blur(10px);
        margin: 0 6px;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        min-width: 120px;
        line-height: 1.2;
        white-space: pre-line;
    }
    
    /* Individual tab colors */
    .stTabs [data-baseweb="tab"]:nth-child(1) {
        color: #FFFFFF !important;
        border-color: #FF6B6B;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    
    .stTabs [data-baseweb="tab"]:nth-child(2) {
        color: #FFFFFF !important;
        border-color: #4ECDC4;
        box-shadow: 0 4px 15px rgba(78, 205, 196, 0.3);
    }
    
    .stTabs [data-baseweb="tab"]:nth-child(3) {
        color: #FFFFFF !important;
        border-color: #45B7D1;
        box-shadow: 0 4px 15px rgba(69, 183, 209, 0.3);
    }
    
    .stTabs [data-baseweb="tab"]:nth-child(4) {
        color: #FFFFFF !important;
        border-color: #96CEB4;
        box-shadow: 0 4px 15px rgba(150, 206, 180, 0.3);
    }
    
    /* Active tab styling */
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.4) !important;
        transform: scale(1.08);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        border-width: 4px !important;
    }
    
    /* Hover effects */
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.25);
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
    }
    
    /* Tab content area */
    .stTabs [data-baseweb="tab-panel"] {
        padding: 20px;
        background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url("https://www.chromethemer.com/wallpapers/chromebook-wallpapers/images/960/galaxy-chromebook-wallpaper.jpg");
        background-size: cover;
        background-position: center;
        border-radius: 10px;
    }

    .stExpander {
        background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url("https://www.chromethemer.com/wallpapers/chromebook-wallpapers/images/960/galaxy-chromebook-wallpaper.jpg") !important;
        background-size: cover !important;
        border-radius: 10px !important;
    }

    .styled-expander .stExpander {
        background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url("https://www.chromethemer.com/wallpapers/chromebook-wallpapers/images/960/galaxy-chromebook-wallpaper.jpg");
        background-size: cover;
        background-position: center;
        border-radius: 10px;
        border: none;
    }
    
    /* Center buttons */
    .center-button {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# STREAMLIT BUILDING BLOCK 2: DESCRIPTIVE TEXT  
# st.write() displays regular text on the page
# Use this for instructions, descriptions, or any text content
# It automatically formats the text nicely
st.markdown(
    """
    <div style='
        text-align: center; 
        font-size: 25px; 
        background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("https://www.chromethemer.com/wallpapers/chromebook-wallpapers/images/960/galaxy-chromebook-wallpaper.jpg"); 
        background-size: cover; 
        background-position: center;
        color: #FFFFFF; 
        padding: 20px; 
        border-radius: 10px; 
        margin-top: 20px; 
        margin-bottom: 20px;
    '>
    Welcome to my personal science & nature vault!🦉 Ask me anything about space 🚀, animals 🐘, discoveries 💡, and the wonders of our world🌿... From the smallest DNA 🧬 to the farthest galaxy🌌 — ask away!
    </div>
    """,
    unsafe_allow_html=True
)




# TO RUN: Save as app.py, then type: streamlit run app.py



# NEW: Function to handle uploaded files
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def reset_collection(client, collection_name: str):
    """Delete existing collection and create a new empty one"""
    
    try:
        # Delete existing collection
        client.delete_collection(name=collection_name)
        print(f"Deleted collection '{collection_name}'")
    except Exception as e:
        print(f"Collection '{collection_name}' doesn't exist or already deleted")
    
    # Create new empty collection
    new_collection = client.create_collection(name=collection_name)
    print(f"Created new empty collection '{collection_name}'")
    
    return new_collection




def add_text_to_chromadb(text: str, filename: str, collection_name: str = "documents"):
    """
    Add text to existing or new ChromaDB collection.
    Safe to call multiple times with same collection_name.
    """
    
    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,       
        chunk_overlap=100,     
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    
    # Initialize components (reuse if possible)
    if not hasattr(add_text_to_chromadb, 'client'):
        add_text_to_chromadb.client = chromadb.Client()
        add_text_to_chromadb.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        add_text_to_chromadb.collections = {}
    
    # Get or create collection
    if collection_name not in add_text_to_chromadb.collections:
        try:
            collection = add_text_to_chromadb.client.get_collection(name=collection_name)
        except:
            collection = add_text_to_chromadb.client.create_collection(name=collection_name)
        add_text_to_chromadb.collections[collection_name] = collection
    
    collection = add_text_to_chromadb.collections[collection_name]
    
    # Process chunks
    for i, chunk in enumerate(chunks):
        embedding = add_text_to_chromadb.embedding_model.encode(chunk).tolist()
        
        metadata = {
            "filename": filename,
            "chunk_index": i,
            "chunk_size": len(chunk)
        }
        
        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[metadata],
            ids=[f"{filename}_chunk_{i}"]
        )
    
    print(f"Added {len(chunks)} chunks from {filename}")
    return collection


# MAIN APP
def main():
    if "client" not in st.session_state:
        st.session_state.client = chromadb.Client()

    if "collection" not in st.session_state:
        try:
            st.session_state.collection = st.session_state.client.get_collection(name="documents")
        except:
            st.session_state.collection = st.session_state.client.create_collection(name="documents")
    
    # Initialize stored documents list
    if "stored_documents" not in st.session_state:
        st.session_state.stored_documents = []

    # Use the new tabbed interface
    create_tabbed_interface()

# ENHANCED FEATURES FROM STEP2


# FEATURE 1: Show which document answered the question
def get_answer_with_source(collection, question):
    """Enhanced answer function that shows source document"""
    results = collection.query(
        query_texts=[question],
        n_results=3
    )
    
    docs = results["documents"][0]
    distances = results["distances"][0]
    ids = results["ids"][0] if "ids" in results else []  # This tells us which document
    
    if not docs or min(distances) > 1.5:
        return "🤷‍♂️ I don't have info on that topic in your uploaded documents.", "No source"
    
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    
    prompt = f"""Context information:
{context}

Question: {question}

Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with "I don't know."

Answer:"""
    
    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(prompt, max_length=150)
    
    answer = response[0]['generated_text'].strip()
    
    # Extract source from best matching document
    if ids:
        best_source = ids[0].split('_chunk_')[0] if '_chunk_' in ids[0] else ids[0]
    else:
        best_source = "Unknown"
    
    return answer, best_source

def store_document_content(doc_name, content):
    if 'document_contents' not in st.session_state:
        st.session_state.document_contents = {}
    st.session_state.document_contents[doc_name] = content

# FEATURE 2: Document manager with delete option
def show_document_manager():
        """Display document manager interface"""
        st.subheader("🧬 Organize Your Discoveries")
        
        if "stored_documents" not in st.session_state or not st.session_state.stored_documents:
            st.info("No documents uploaded yet.")
            return
        
        # Show each document with delete button
        for i, doc_name in enumerate(st.session_state.stored_documents):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.write(f"📄 {doc_name}")
            
            with col2:
                # Preview button
                if st.button("Preview", key=f"preview_{i}", use_container_width=True):
                    st.session_state[f'show_preview_{i}'] = not st.session_state.get(f'show_preview_{i}', False)
            
            with col3:
                # Delete button
                if st.button("🗑️ Delete", key=f"delete_{i}", use_container_width=True):
                    # Remove from session state
                    st.session_state.stored_documents.pop(i)
                    # Also remove content to save memory
                    if 'document_contents' in st.session_state and doc_name in st.session_state.document_contents:
                        del st.session_state.document_contents[doc_name]
                    st.success(f"Deleted {doc_name}")
                    st.rerun()

            with col4:
                # Save as Markdown button
                markdown_content = st.session_state.get('document_contents', {}).get(doc_name, "")
                st.download_button(
                    label="📥 Save as MD",
                    data=markdown_content,
                    file_name=f"{Path(doc_name).stem}.md",
                    mime="text/markdown",
                    key=f"download_{i}",
                    use_container_width=True
                )
            
            # Show preview if requested
            if st.session_state.get(f'show_preview_{i}', False):
                with st.expander(f"Preview: {doc_name}", expanded=True):
                    st.write(st.session_state.get('document_contents', {}).get(doc_name, "Preview not available."))
                    if st.button("Hide Preview", key=f"hide_{i}"):
                        st.session_state[f'show_preview_{i}'] = False
                        st.rerun()

    # FEATURE 3: Search history
def add_to_search_history(question, answer, source):
        """Add search to history"""
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        # Add new search to beginning of list
        st.session_state.search_history.insert(0, {
            'question': question,
            'answer': answer,
            'source': source,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        # Keep only last 10 searches
        if len(st.session_state.search_history) > 10:
            st.session_state.search_history = st.session_state.search_history[:10]

def show_search_history():
        """Display search history"""
        st.subheader("🕒 Recent Searches")
        
        if 'search_history' not in st.session_state or not st.session_state.search_history:
            st.info("No searches yet.")
            return
        
        for i, search in enumerate(st.session_state.search_history):
            with st.expander(f"Q: {search['question'][:50]}... ({search['timestamp']})"):
                st.write("**Question:**", search['question'])
                st.write("**Answer:**", search['answer'])
                st.write("**Source:**", search['source'])

    # FEATURE 4: Document statistics
def show_document_stats():
        """Show statistics about uploaded documents"""
        st.subheader("📊 Document Statistics")
        
        if "stored_documents" not in st.session_state or not st.session_state.stored_documents:
            st.info("No documents to analyze.")
            return
        
        # Calculate basic stats
        total_docs = len(st.session_state.stored_documents)
        
        # Display in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Documents", total_docs)
        
        with col2:
            st.metric("Collection Status", "Active" if st.session_state.collection else "Empty")
        
        with col3:
            st.metric("Ready for Q&A", "Yes" if total_docs > 0 else "No")
        
        # Show breakdown by file type
        file_types = {}
        for doc_name in st.session_state.stored_documents:
            ext = Path(doc_name).suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
        
        if file_types:
            st.write("**File Types:**")
            for ext, count in file_types.items():
                st.write(f"• {ext}: {count} files")

    # FEATURE 5: Enhanced UI with tabs
def create_tabbed_interface():
        """Create a tabbed interface for better organization"""
        
        tab1, tab2, tab3, tab4 = st.tabs(["🚀 Launch\nNew Knowledge", "🔍 Ask ExplAIniac\nAnything!", "📋 Manage\nDocuments", "📡 Explore System\nInsights"])
        
        with tab1:
            st.header("🧠 ExplAIniac's Cosmic Library 📚✨")
            
            uploaded_files = st.file_uploader(
                "Upload documents",
                type=["pdf", "doc", "docx", "txt"],
                accept_multiple_files=True
            )

            if st.button("Chunk and Store Documents"):
                if uploaded_files:
                    # Reset Chroma collection and session state
                    st.session_state.collection = reset_collection(st.session_state.client, "documents")
                    st.session_state.stored_documents = []  # Reset stored documents list

                    for file in uploaded_files:
                        suffix = Path(file.name).suffix
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                                temp_file.write(file.getvalue())
                                temp_file_path = temp_file.name

                            text = convert_to_markdown(temp_file_path)
                            store_document_content(file.name, text)
                            st.session_state.collection = add_text_to_chromadb(text, file.name, collection_name="documents")
                            st.session_state.stored_documents.append(file.name)  # Add filename to stored documents
                            st.success(f"✅ Stored {file.name} successfully!")

                        except Exception as e:
                            st.error(f"❌ Failed to process {file.name}: {str(e)}")
                else:
                    st.warning("⚠️ Please upload at least one document first.")

            # Display stored documents
            if st.session_state.get('stored_documents', []):
                st.markdown("### 🌟 ExplAIniac's Stored Knowledge 📄")
                for i, doc_name in enumerate(st.session_state.stored_documents, 1):
                    st.info(f"📋 {i}. {doc_name}")
                st.markdown(f"**Total documents stored: {len(st.session_state.stored_documents)}**")
            else:
                st.info("📝 No documents currently stored. Upload some documents to get started!")
        
        with tab2:
            st.markdown("<h2 style='color: magenta;'>🔍 What mystery of the universe can ExplAIniac solve today?</h2>", unsafe_allow_html=True)
            
            if st.session_state.get('stored_documents', []):
                question = st.text_input("What mystery of the universe can ExplAIniac solve today?")
                
                if st.button("**Reveal the Wonders! 🌟**", type="primary"):
                    if not question.strip():
                        st.warning("⚠️ Please enter a question.")
                    else:
                        with st.spinner("Thinking... 🧠"):
                            try:
                                answer, source = get_answer_with_source(st.session_state.collection, question)
                                
                                st.success("✅ Answer found!")
                                st.markdown(f"**🧠 Answer:** {answer}")
                                st.markdown(f"**📄 Source:** {source}")
                                
                                # Add to history
                                add_to_search_history(question, answer, source)
                                
                            except Exception as e:
                                st.error(f"❌ Error during answering: {str(e)}")
            else:
                st.info("📝 Upload documents first to start asking questions!")
            
            # Show recent searches
            show_search_history()
        
        with tab3:
            show_document_manager()
        
        with tab4:
            show_document_stats()

# FINAL INSTRUCTION FOR STUDENTS:
# Replace your main() function with enhanced_main()

# Custom CSS for better appearance
def add_custom_css():
    """Add custom styling to make app look professional"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        border-radius: 10px;
    }
    
    .stButton > button, .stDownloadButton > button {
        width: 100%;
        height: 3rem; /* Set a fixed height */
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .success-box {
        padding: 1rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        margin: 1rem 0;
        color: #721c24;
    }
    
    .info-box {
        padding: 1rem;
        background-color: #cce7ff;
        border: 1px solid #99d6ff;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        font-weight: bold;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Enhanced error handling
def safe_convert_files(uploaded_files):
    """Convert files with comprehensive error handling"""
    converted_docs = []
    errors = []
    
    if not uploaded_files:
        st.warning("Please upload at least one document.")
        return [], []

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Logic from create_tabbed_interface
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            text = convert_to_markdown(temp_file_path)
            store_document_content(uploaded_file.name, text)
            st.session_state.collection = add_text_to_chromadb(text, uploaded_file.name, collection_name="documents")
            st.session_state.stored_documents.append(uploaded_file.name)
            converted_docs.append(uploaded_file.name)
            status_text.text(f"✅ Converted {uploaded_file.name}")

        except Exception as e:
            errors.append((uploaded_file.name, str(e)))
            status_text.text(f"❌ Error converting {uploaded_file.name}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("Conversion complete!")
    return converted_docs, errors

# Better user feedback
def show_conversion_results(converted_docs, errors):
    """Display conversion results with good UX"""
    
    if converted_docs:
        st.markdown('<div class="success-box"><b>🛰️ Files Beamed In Successfully!</b></div>', unsafe_allow_html=True)
        for doc_name in converted_docs:
            st.write(f"📄 {doc_name}")
    
    if errors:
        st.markdown('<div class="info-box"><b>Errors Encountered:</b></div>', unsafe_allow_html=True)
        for doc_name, error_msg in errors:
            st.error(f"**{doc_name}:** {error_msg}")

# Better question interface
def enhanced_question_interface():
    """Professional question asking interface"""
    
    st.subheader("💬 Ask Your Question")
    
    # Provide example questions
    with st.expander("💡 Example questions you can ask"):
        st.write("- What are the main findings in the research paper?")
        st.write("- Summarize the key points of the meeting notes.")
        st.write("- What is the conclusion of the technical report?")
    
    # Question input with suggestions
    question = st.text_input(
        "Type your question here:",
        placeholder="e.g., What are the main findings in the research paper?"
    )
    
    # Two-column layout for buttons
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("✨ Get Answer", type="primary"):
            if question:
                with st.spinner("Finding answer..."):
                    answer, source = get_answer_with_source(st.session_state.collection, question)
                    st.success("Answer found!")
                    st.markdown(f"**Answer:** {answer}")
                    st.markdown(f"**Source:** {source}")
                    add_to_search_history(question, answer, source)
            else:
                st.warning("Please enter a question.")
    
    with col2:
        if st.button("Clear"):
            # This could clear the question or results, for now it does nothing
            pass

# App health check
def check_app_health():
    """Check if all components are working"""
    issues = []
    
    # Check session state
    required_keys = ['converted_docs', 'collection']
    for key in required_keys:
        if key not in st.session_state:
            issues.append(f"⚠️ Session state key '{key}' is missing.")
    
    # Check ChromaDB
    try:
        st.session_state.client.heartbeat()
    except Exception as e:
        issues.append(f"❌ ChromaDB connection failed: {e}")
    
    # Check AI model
    try:
        # A simple check for the pipeline
        pipeline("text2text-generation", model="google/flan-t5-small")
    except Exception as e:
        issues.append(f"❌ AI model (Hugging Face pipeline) failed to load: {e}")
    
    return issues

# Loading animations
def show_loading_animation(text="Processing..."):
    """Show professional loading animation"""
    with st.spinner(text):
        time.sleep(2) # Simulate a loading process

# Enhanced main function template
def enhanced_main():
    """Professional main function with all features"""
    
    # Apply custom CSS
    add_custom_css()
    
    # Header
    st.markdown(
    """
    <div style='
        text-align: center; 
        padding: 15px; 
        background: rgba(0,0,0,0.3); 
        border-radius: 10px; 
        border: 2px solid #FFD700; 
        margin: 20px 0;
    '>
        <p style='color:#FFD700; font-weight:bold; font-size: 20px; margin: 0;'>🚀 Beam up your files, let ExplAIniac decode their wonders, and ask mind-blowing questions about space, nature, and beyond!</p>
    </div>
    """,
    unsafe_allow_html=True
)
    
    # Initialize session state
    if 'converted_docs' not in st.session_state:
        st.session_state.converted_docs = []
    if 'collection' not in st.session_state:
        client = chromadb.Client()
        st.session_state.collection = client.get_or_create_collection(name="documents")
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if "stored_documents" not in st.session_state:
        st.session_state.stored_documents = []
    if "client" not in st.session_state:
        st.session_state.client = chromadb.Client()

    # Health check (optional - show only if there are issues)
    health_issues = check_app_health()
    if health_issues:
        with st.expander("🚨 App Health Issues Detected"):
            for issue in health_issues:
                st.warning(issue)
    
    # Main interface using tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚀 Launch New Knowledge", "🔍 Ask ExplAIniac Anything!", "🧬 Organize Your Discoveries", "📡 Explore System Insights", "🧪 System Diagnostics"])

    with tab1:
        st.header("📡 Beam In Your Knowledge")
        uploaded_files = st.file_uploader(
            "Choose documents to upload",
            type=["pdf", "doc", "docx", "txt"],
            accept_multiple_files=True
        )
        if st.button("Begin Conversion"):
            converted_docs, errors = safe_convert_files(uploaded_files)
            st.session_state.converted_docs = converted_docs
            show_conversion_results(converted_docs, errors)

    with tab2:
        st.header("🔍 What mystery of the universe (or your backyard) can ExplAIniac solve today?")
        if not st.session_state.get('stored_documents'):
            st.info("Please upload and convert documents in the first tab to begin asking questions.")
        else:
            enhanced_question_interface()

    with tab3:
        st.header("📈 Knowledge Metrics & Search Timeline")
        show_document_stats()
        st.divider()
        show_search_history()

    with tab4:
        st.header("📂 Organize Scientific Discoveries")
        show_document_manager()

    with tab5:
        st.header("🔋 AI Engine Health Monitor")
        issues = check_app_health()
        if issues:
            for issue in issues:
                st.error(issue)
        else:
            st.success("✅ All systems are running smoothly!")
        if st.button("Re-check Health"):
            st.rerun()


if __name__ == "__main__":
    enhanced_main()

# Add instructions expander with matching background
with st.expander("🌌 How to Use ExplAIniac – Your AI Science Navigator"):
    st.markdown("""
    <div style='
        background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("https://www.chromethemer.com/wallpapers/chromebook-wallpapers/images/960/galaxy-chromebook-wallpaper.jpg"); 
        background-size: cover; 
        background-position: center;
        color: #FFFFFF; 
        padding: 20px; 
        border-radius: 10px; 
        margin: 10px 0;
    '>
    
    
    🚀 <strong>1. Launch Your Knowledge</strong><br>
    Start your journey by uploading files (PDF, DOCX, TXT). Just drag and drop them into the "📡 Beam In Your Knowledge" tab. ExplAIniac will scan, split, and store their contents in your science vault.<br><br>
    
    🧠 <strong>2. Automatic Conversion</strong><br>
    Click "Begin Conversion" — ExplAIniac will extract the knowledge, convert it to an AI-readable format, and store it safely for future exploration.<br><br>
    
    🧾 <strong>3. Check Document Stats</strong><br>
    Visit the "📈 Knowledge Metrics & Search Timeline" tab to see how many files are stored, what types they are, and whether everything is ready for Q&A. You'll also find a timeline of your past questions here!<br><br>
    
    💬 <strong>4. Ask Anything!</strong><br>
    Head over to "🔍 Ask ExplAIniac Anything!"<br>
    Type your question — about black holes, biodiversity, AI ethics, climate change, or anything your uploaded documents contain.<br><br>
    
    ✨ Click "Get Answer" — ExplAIniac will search through all stored knowledge and give you a clear, sourced response. No fluff. No hallucination. Just real answers.<br><br>
    
    🗃️ <strong>5. Manage Your Discoveries</strong><br>
    Use the "📂 Organize Scientific Discoveries" tab to preview uploaded documents, delete ones you no longer need, and keep your vault clean and sharp.<br><br>
    
    🧪 <strong>6. Monitor the AI Engine</strong><br>
    In the "🔋 AI Engine Health Monitor", run real-time diagnostics to ensure ExplAIniac's brain is fully operational — including ChromaDB connection, AI model health, and document storage.<br><br>
    
    🌠 <strong>7. Stay Inspired</strong><br>
    Keep an eye out for random quotes from famous scientists throughout the app — your daily boost of curiosity and cosmic wisdom.<br><br>
    
    📫 <strong>Need help or want to suggest new features?</strong><br>
    You can reach mission control at support@explAIniac.si.
    
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<video autoplay loop muted playsinline style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; z-index: -1;">
        <source src="https://i.imgur.com/ZaDqA7A.mp4" type="video/mp4">
    </video>
    <style>
    .stApp {
        background: transparent;
    }
    </style>
""", unsafe_allow_html=True)
