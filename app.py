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
    page_title="Policy Intelligence Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- BRANDING & CSS ---
st.markdown("""
    <style>
        /* 1. Main Background and Font */
        .stApp {
            background-color: #FFFFFF;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }
        
        /* 2. TOP DECORATION BAR (The very top line) - Set to Navy Blue */
        div[data-testid="stDecoration"] {
            background-image: none;
            background-color: #1E2B3C !important; 
            height: 5px;
        }

        /* 3. HEADER BORDER (The Orange Accent Line) */
        header[data-testid="stHeader"] {
            background-color: rgba(255, 255, 255, 0.95) !important; /* Ensure header is not transparent */
            border-bottom: 5px solid #FF7300 !important; /* OSU Orange */
        }

        /* 4. Titles and Headers */
        h1, h2, h3 {
            color: #1E2B3C !important; /* Dark Navy Blue */
            font-weight: 700;
            padding-bottom: 10px;
        }
        
        /* 5. Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #F4F4F4;
            border-right: 1px solid #DDDDDD;
        }
        
        /* 6. Custom Button Styling */
        div.stButton > button {
            background-color: #FF7300 !important;
            color: white !important;
            border: none;
            border-radius: 4px;
            font-weight: bold;
        }
        div.stButton > button:hover {
            background-color: #E06000 !important;
            color: white !important;
        }
        
        /* 7. Link Styling (Optional - for source links) */
        a {
            color: #FF7300 !important;
            text-decoration: none;
        }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR & SETTINGS ---
with st.sidebar:
    # LOGO: Using the RAW GitHub link so it renders correctly
    st.image("https://raw.githubusercontent.com/srikarananand/state-energy-rag/main/logo.png", use_container_width=True)
    
    st.markdown("### Settings")
    st.markdown("---")
    
    # API Keys Handling
    if "PINECONE_API_KEY" in st.secrets:
        pinecone_key = st.secrets["PINECONE_API_KEY"]
    else:
        pinecone_key = st.text_input("Pinecone API Key", type="password")

    if "GROQ_API_KEY" in st.secrets:
        groq_key = st.secrets["GROQ_API_KEY"]
    else:
        groq_key = st.text_input("Groq API Key", type="password")
    


# --- MAIN APP LOGIC ---

# Custom Title with the Orange Accent Line underneath implied by CSS
st.title("Energy Policy Intelligence")
st.markdown("Welcome to the policy analysis tool. Search across indexed state energy plans.")
st.markdown("**This is a prototype and only NY is available so far - more states to be updated**")

# Stop if keys are missing
if not pinecone_key or not groq_key:
    st.warning("Please enter your API keys in the sidebar to continue.")
    st.stop()

@st.cache_resource
def load_chain(pinecone_api_key, groq_api_key):
    """
    Initializes the RAG chain using LCEL.
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
    
    # 3. LLM (Using the versatile model)
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=groq_api_key
    )
    
    # 4. Prompt Template (Clean, professional tone)
    template = """
    You are a senior policy analyst for the Hamm Institute. 
    Answer the question based ONLY on the following context.
    Cite the document name for every fact you state. 
    If the context contains tables, interpret the rows and columns accurately.

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
    st.session_state.messages = [{"role": "assistant", "content": "Hello. Accessing the National Energy Policy database. How can I assist you today?"}]

# Display History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle Input
if query := st.chat_input("Enter your policy question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Processing request..."):
            try:
                # Invoke the chain
                response = chain.invoke(query)
                
                answer = response["answer"]
                sources = response["context"]
                
                st.write(answer)
                
                # Show Sources (Clean Format)
                with st.expander("Reference Documents"):
                    for i, doc in enumerate(sources):
                        source_name = doc.metadata.get('source', 'Unknown PDF')
                        # Clean link formatting
                        if source_name.startswith("http"):
                            st.markdown(f"**{i+1}.** [{source_name}]({source_name})")
                        else:
                            st.markdown(f"**{i+1}.** {source_name}")
                            
                        st.caption(doc.page_content[:200] + "...")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"System Error: {e}")
