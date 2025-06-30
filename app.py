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

def setup_documents():
    """
    This function creates our document database
    NOTE: This runs every time someone uses the app
    In a real app, you'd want to save this data permanently
    """
    client = chromadb.Client()
    try:
        collection = client.get_collection(name="docs")
    except Exception:
        collection = client.create_collection(name="docs")
  
    
    # STUDENT TASK: Replace these 5 documents with your own!
    # Pick ONE topic: movies, sports, cooking, travel, technology
    # Each document should be 150-200 words
    # IMPORTANT: The quality of your documents affects answer quality!
    
    my_documents = [
        """This document is about space exploration and recent developments in space science. One major effort is NASA’s Artemis program, which is preparing to return humans to the Moon for the first time since Apollo. Artemis I (2022) was an uncrewed lunar orbital test, and Artemis II will carry astronauts around the Moon (now slated for 2026). The subsequent Artemis III intends to land astronauts on the lunar surface – the first human moon landing since 1972. In parallel, private companies are playing a big role: SpaceX, for example, is developing its Starship rocket to serve as the lunar lander for Artemis and advance reusable launch technology. Another leap in space science has come from the James Webb Space Telescope (JWST). Since becoming operational in 2022, JWST has peered deeper into the cosmos than ever before. It discovered galaxies dating to just 300–500 million years after the Big Bang – one galaxy formed only around 320 million years post-Big Bang – revealing that massive galaxies and stars appeared earlier than expected. On Mars, NASA’s Perseverance rover (landed 2021) is exploring an ancient lakebed and caching samples for a future return mission. These advances, from Moon missions to powerful space telescopes, mark a new era of cosmic discovery.""",
        
        """This document is about climate change and the environment, focusing on current trends and actions. Global warming continues to accelerate: 2024 was confirmed as the hottest year on record, with global temperature about 1.55 °C above the pre-industrial average. Such record warmth is not an outlier – it follows a string of extremely hot years and is pushing the world beyond the aspirational 1.5°C limit. The impacts of this warming are increasingly evident. Every additional fraction of a degree is driving more frequent and intense heatwaves, droughts, heavy rainfall, and rising sea levels as ice sheets and glaciers melt. In 2023, many regions saw unprecedented heat and wildfires, while others experienced severe floods – highlighting the urgent need for action. The global response has been centered on international agreements. Under the 2015 Paris Agreement, countries pledged to limit warming to 1.5–2°C by cutting greenhouse emissions. However, current policies are still falling short, prompting stronger calls at recent summits. At COP28 in 2023, nearly 200 nations agreed to signal the beginning of the end of the fossil fuel era and accelerate the transition to clean energy. The outcome urged a roughly 43% cut in global emissions by 2030 and tripling of renewable energy capacity.""",
        
        """This document is about wildlife, biodiversity, and conservation efforts in nature. Earth is currently facing a biodiversity crisis, with human activities driving habitat loss, pollution, and climate shifts that threaten countless species. Scientists warn that we are experiencing the largest mass extinction event since the dinosaur age. A United Nations report estimated that one million species are at risk of extinction, many within the coming decades. Wildlife populations have plummeted globally – for instance, the WWF Living Planet Index has documented large declines in vertebrate populations since 1970. Yet, there are some encouraging signs due to conservation. A notable example is wild tigers: after a century of decline, global tiger numbers have stabilized and even increased by about 40% in the last seven years. This marks the first potential rise in tiger population in decades, credited to improved monitoring and protected areas. Similarly, mountain gorillas and giant pandas have slowly recovered. In late 2022, nearly 200 countries signed a biodiversity pact to protect 30% of Earth’s land and oceans by 2030. In 2023 alone, scientists discovered over 800 new species, reminding us that even as life disappears, nature still holds incredible secrets to uncover and protect.""",
        
        """This document is about recent scientific discoveries and breakthroughs across various fields. In the last few years, we have witnessed advances that were once science fiction. In energy, scientists achieved a milestone in nuclear fusion. In 2022, the U.S. National Ignition Facility conducted the first fusion experiment that produced more energy than it consumed. This “net energy gain” was repeated with even greater output in 2023 and is a major step toward clean, limitless power. In biotechnology, CRISPR gene-editing technology has led to powerful therapies. In 2023, a gene therapy for sickle-cell anemia called casgevy became the first CRISPR-based treatment nearing full approval, offering a functional cure. In medicine, a new malaria vaccine approved in 2023 showed 75% efficacy, the first to meet the WHO’s benchmark and offering hope to millions of children. Beyond health, scientists are developing blood tests to detect Alzheimer’s early, astronomers are capturing images of black holes and discovering new exoplanets, and AI is helping model complex systems. These innovations demonstrate how science is rapidly evolving, improving health and deepening our understanding of the universe, biology, and energy – laying the groundwork for transformative technologies in the years ahead.""",
        
        """This document is about the future outlook of science and nature – looking ahead to goals and emerging trends by 2030 and beyond. In space, humanity is preparing for long-term exploration. NASA aims to establish a lunar outpost by the end of the decade and launch astronauts to Mars by the 2030s. Private and international missions are planning advanced telescopes, lunar bases, and tourism infrastructure. Environmentally, urgent climate goals include peaking global emissions by 2025 and cutting them by 43% by 2030. Most countries have committed to net-zero emissions by 2050, which scientists say is essential to stabilizing the climate. Simultaneously, nations are working toward the “30 by 30” goal of protecting 30% of Earth’s land and oceans by 2030 to halt biodiversity loss. Future innovations will be key. Advancements in clean energy, from battery tech to fusion, may transform power systems. Gene therapy and new vaccines could eliminate diseases like HIV and malaria. Artificial intelligence is expected to accelerate discovery in fields from medicine to environmental modeling. Some scientists even believe that by 2050, we may have the technology to detect extraterrestrial life or build self-sustaining, eco-friendly cities. The next decades could redefine science, health, and our place in the cosmos."""
    ]
    
    # Add documents to database with unique IDs
    # ChromaDB needs unique identifiers for each document
    collection.add(
        documents=my_documents,
        ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
    )
    
    return collection

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

# STREAMLIT BUILDING BLOCK 3: FUNCTION CALLS
# We call our function to set up the document database
# This happens every time someone uses the app
collection = setup_documents()

# STREAMLIT BUILDING BLOCK 4: TEXT INPUT BOX
# st.text_input() creates a box where users can type
# - First parameter: Label that appears above the box
# - The text users type gets stored in the 'question' variable
# - Users can click in this box and type their question
# Custom-styled label

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    "<span style='font-size:21px; color:magenta;'>🔍 What mystery of the universe (or your backyard) can ExplAIniac solve today?</span>",
    unsafe_allow_html=True
)

# Text input box with no visible label
question = st.text_input(label="")

# STREAMLIT BUILDING BLOCK 5: BUTTON
# st.button() creates a clickable button
# - When clicked, all code inside the 'if' block runs
# - type="primary" makes the button blue and prominent
# - The button text appears on the button itself
if st.button("**Reveal the Wonders! 🌟**", type="primary"):
    
    # STREAMLIT BUILDING BLOCK 6: CONDITIONAL LOGIC
    # Check if user actually typed something (not empty)
    if question:
        
        # STREAMLIT BUILDING BLOCK 7: SPINNER (LOADING ANIMATION)
        # st.spinner() shows a rotating animation while code runs
        # - Text inside quotes appears next to the spinner
        # - Everything inside the 'with' block runs while spinner shows
        # - Spinner disappears when the code finishes
        with st.spinner("Launching satellites, scanning DNA strands, and decoding nature’s secrets..."):
            answer = get_answer(collection, question)
        
        # STREAMLIT BUILDING BLOCK 8: FORMATTED TEXT OUTPUT
        # st.write() can display different types of content
        # - **text** makes text bold (markdown formatting)
        # - First st.write() shows "Answer:" in bold
        # - Second st.write() shows the actual answer
        st.write("**🌍 Fact Uncovered:**")
        st.write(answer)
    
    else:
        # STREAMLIT BUILDING BLOCK 9: SIMPLE MESSAGE
        # This runs if user didn't type a question
        # Reminds them to enter something before clicking
        st.markdown(
    "<span style='color:gold; font-size:18px'>⚠️ Oops! You forgot to ask a question. Try typing something fun. I'm ready to explain the wonders of the universe!</span>",
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

