import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Hamm Institute AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR HAMM INSTITUTE BRANDING ---
st.markdown("""
    <style>
    /* MAIN BACKGROUND & FONT */
    .stApp {
        background-color: #FFFFFF;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* SIDEBAR STYLING */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #E0E0E0;
    }
    
    /* HEADERS (Black/Dark Grey) */
    h1, h2, h3 {
        color: #000000 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
    }
    
    /* ACCENT COLORS (Hamm Orange) */
    a {
        color: #FF782D !important;
        text-decoration: none;
    }
    .stButton button {
        background-color: #FF782D !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
    }
    .stButton button:hover {
        background-color: #E06010 !important; /* Darker Orange on Hover */
    }
    
    /* INPUT FIELDS */
    .stTextInput input {
        border-radius: 4px;
        border: 1px solid #CCCCCC;
    }
    
    /* CHAT MESSAGES */
    [data-testid="stChatMessage"] {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        line-height: 1.6;
    }
    
    /* LOGO & HEADER ALIGNMENT */
    .header-container {
        display: flex;
        align-items: center;
        padding-bottom: 2rem;
        border-bottom: 2px solid #FF782D;
        margin-bottom: 2rem;
    }
    .logo-img {
        height: 80px;
        margin-right: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOGO & HEADER ---
col1, col2 = st.columns([1, 4])
with col1:
    # Display Logo from the link provided
    st.image("https://github.com/srikarananand/state-energy-rag/blob/main/logo.png?raw=true", width=150)
with col2:
    st.title("Energy Policy Intelligence")
    st.markdown("### Hamm Institute for American Energy")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("Configuration")
    
    # API Keys handling
    if "PINECONE_API_KEY" in st.secrets:
        pinecone_key = st.secrets["PINECONE_API_KEY"]
    else:
        pinecone_key = st.text_input("Pinecone API Key", type="password")

    if "GROQ_API_KEY" in st.secrets:
        groq_key = st.secrets["GROQ_API_KEY"]
    else:
        groq_key = st.text_input("Groq API Key", type="password")
        
    st.markdown("---")
    st.markdown("**System Status:** 🟢 Online")
    st.caption("v1.0.0 | Production Build")

# --- MAIN APP LOGIC ---

# Stop if keys are missing
if not pinecone_key or not groq_key:
    st.warning("⚠️ Access Restricted: Please enter API credentials.")
    st.stop()

@st.cache_resource
def load_chain(pinecone_api_key, groq_api_key):
    """
    Initializes the RAG chain using modern LCEL.
    """
    # 1. Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Vector Store
    vector_store = PineconeVectorStore(
        index_name="energy-policy",
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # 3. LLM (Llama-3.3-70b)
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=groq_api_key
    )
    
    # 4. Professional Policy Analyst Prompt
    template = """
    You are a senior energy policy analyst at the Hamm Institute. 
    Your tone is professional, objective, and precise.
    Answer the question based ONLY on the following context.
    Cite the document name for every fact you state. 
    If the context contains tables, analyze the data rows carefully.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # 5. Build Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    
    return rag_chain_with_source

try:
    chain = load_chain(pinecone_key, groq_key)
except Exception as e:
    st.error(f"Connection Error: {e}")
    st.stop()

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome. I am ready to analyze the State Energy Plans. Please enter your query."}]

# Display History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle Input
if query := st.chat_input("Enter your policy question here..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Processing inquiry..."):
            try:
                response = chain.invoke(query)
                answer = response["answer"]
                sources = response["context"]
                
                st.write(answer)
                
                # Professional Source Display
                with st.expander("REFERENCE DOCUMENTS"):
                    for i, doc in enumerate(sources):
                        source_name = doc.metadata.get('source', 'Unknown Document')
                        
                        # Clean up filename for display
                        display_name = source_name.split("/")[-1] if "/" in source_name else source_name
                        
                        st.markdown(f"**{i+1}. {display_name}**")
                        if source_name.startswith("http"):
                            st.markdown(f"[Open Document]({source_name})")
                        st.caption(doc.page_content[:250] + "...")
                        st.markdown("---")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"System Error: {e}")
