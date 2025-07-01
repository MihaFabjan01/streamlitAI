# Fix SQLite version issue for ChromaDB on Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass


import streamlit as st
import chromadb
from transformers import pipeline
from pathlib import Path
import tempfile
from pathlib import Path


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

st.markdown(
    """
    <div style="display: flex; justify-content: center; margin-bottom: 10px;">
        <img src="https://i.imgur.com/yFddIhW.png"
             width="160" style="border-radius: 50%; box-shadow: 0 0 10px rgba(255,255,255,0.4);">
    </div>
    """,
    unsafe_allow_html=True
)



st.markdown("<h1 style='text-align: center;'>🧪 ExplAIniac 🪐</h1>", unsafe_allow_html=True)


st.markdown(
    "<div style='text-align: center; font-size: 27px'>Your AI for Wild Wonders 🧠 & Real Facts🔬</span>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://c4.wallpaperflare.com/wallpaper/816/251/630/space-universe-planets-dark-background-stars-uncountable-abstract-wallpaper-preview.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
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
    <div style='text-align: center; font-size: 25px'>
    Welcome to my personal science & nature vault!🦉 Ask me anything about space 🚀, animals 🐘, discoveries 💡, and the wonders of our world🌿... From the smallest DNA 🧬 to the farthest galaxy🌌 — ask away!
    </div>
    """,
    unsafe_allow_html=True
)




# STREAMLIT BUILDING BLOCK 10: EXPANDABLE SECTION
# st.expander() creates a collapsible section
# - Users can click to show/hide the content inside
# - Great for help text, instructions, or extra information
# - Keeps the main interface clean
with st.expander(" **🧭 How to Use ExplAIniac**"):
    st.write("""
    🧬🔍 Ask anything about space, animals, science, or nature — and I’ll explain it!

    🖊️ **1. Type your question**
   
    Curious about black holes, biodiversity, or DNA? 
    Just type your question into the box. 
    
    Example:        
    “When will humans go back to the Moon?”
    “What causes climate change?”
    “Are tigers still endangered?”
             
    🚀 **2. Click “Reveal the Wonders! 🌟”**
             
    Hit the big button and I’ll start searching through the science vault for you.

    ⏳ **3. Watch the spinner**
             
    While I explore the galaxies and dig through the data,
    you’ll see a spinner saying I’m working on it.

    ✅ **4. See your answer appear**
             
    I’ll return a clear, fact-based explanation based on what I know — fast, friendly, and fascinating.

    🦉 **5. Need help?**
             
    You can contact us on support@explAIniac.si.

    ✨ That’s it! Explore the universe of facts with ExplAIniac — where curiosity gets real answers.""")

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
    st.title("📚 Smart Document Knowledge Base")

    if "client" not in st.session_state:
        st.session_state.client = chromadb.Client()

    if "collection" not in st.session_state:
        try:
            st.session_state.collection = st.session_state.client.get_collection(name="documents")
        except:
            st.session_state.collection = st.session_state.client.create_collection(name="documents")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "doc", "docx", "txt"],
        accept_multiple_files=True
    )

    if st.button("Chunk and Store Documents"):
        if uploaded_files:
            # Reset Chroma collection and session state
            st.session_state.collection = reset_collection(st.session_state.client, "documents")

            for file in uploaded_files:
                suffix = Path(file.name).suffix
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                        temp_file.write(file.getvalue())
                        temp_file_path = temp_file.name

                    text = convert_to_markdown(temp_file_path)
                    st.session_state.collection = add_text_to_chromadb(text, file.name, collection_name="documents")
                    st.success(f"✅ Stored {file.name} successfully!")

                except Exception as e:
                    st.error(f"❌ Failed to process {file.name}: {str(e)}")
        else:
            st.warning("⚠️ Please upload at least one document first.")

    # --- Q&A SECTION ---
    st.markdown("---")
    st.subheader("❓ Ask a question about your uploaded documents:")
    question = st.text_input("Ask a question:")

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("⚠️ Please enter a question.")
        elif not st.session_state.collection:
            st.error("❌ No documents found. Please upload and store documents first.")
        else:
            with st.spinner("Thinking... 🧠"):
                try:
                    results = st.session_state.collection.query(query_texts=[question], n_results=3)
                    docs = results["documents"][0]
                    distances = results["distances"][0]

                    if not docs or min(distances) > 1.5:
                        st.info("🤷‍♂️ I don’t have info on that topic in your uploaded documents.")
                    else:
                        context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
                        prompt = f"""Context information:
{context}

Question: {question}

Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with "I don't know."

Answer:"""

                        ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
                        response = ai_model(prompt, max_length=150)
                        answer = response[0]['generated_text'].strip()

                        st.success("✅ Answer found!")
                        st.markdown(f"**🧠 Answer:** {answer}")

                except Exception as e:
                    st.error(f"❌ Error during answering: {str(e)}")

if __name__ == "__main__":
    main()
