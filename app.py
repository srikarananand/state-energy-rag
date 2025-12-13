import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- PAGE CONFIG ---
st.set_page_config(page_title="NY Energy Policy AI", layout="wide")

# --- SIDEBAR & KEYS ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/usa.png", width=50)
    st.title("Settings")
    
    # In Streamlit Cloud, these come from 'st.secrets'. 
    # Locally, it will look for them in environment variables or you can paste them here.
    if "PINECONE_API_KEY" in st.secrets:
        pinecone_key = st.secrets["PINECONE_API_KEY"]
    else:
        pinecone_key = st.text_input("Pinecone API Key", type="password")

    if "GROQ_API_KEY" in st.secrets:
        groq_key = st.secrets["GROQ_API_KEY"]
    else:
        groq_key = st.text_input("Groq API Key", type="password")
    
    st.markdown("---")
    st.markdown("**Status:** 🟢 System Online")
    st.markdown(f"**Index:** energy-policy")

# --- MAIN APP LOGIC ---
st.title("⚡ New York Energy Policy Intelligence")
st.markdown("Ask questions about the **230+ indexed documents** from the NY State Energy Plan.")

if not pinecone_key or not groq_key:
    st.warning("⚠️ Please enter your API keys in the sidebar to continue.")
    st.stop()

@st.cache_resource
def load_chain(pinecone_api_key, groq_api_key):
    """
    Initializes the RAG chain. Cached so it doesn't reload on every click.
    """
    # 1. Embeddings (Must match what you used for Ingestion!)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Vector Store (The 18,922 records)
    vector_store = PineconeVectorStore(
        index_name="energy-policy",
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key
    )
    
    # 3. LLM (The Brain)
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0,
        groq_api_key=groq_api_key
    )
    
    # 4. Prompt
    template = """
    You are an expert energy policy analyst.
    Answer the user's question based ONLY on the context below.
    If the context contains tables, carefully read the rows and columns.
    ALWAYS cite the filename/source for your facts.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    
    # 5. Chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

try:
    chain = load_chain(pinecone_key, groq_key)
except Exception as e:
    st.error(f"Connection Error: {e}")
    st.stop()

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I have read the NY Energy Plans. Ask me about targets, capacity, or regulations."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ex: What is the 2030 target for solar capacity?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing 18,000+ policy records..."):
            try:
                response = chain.invoke({"query": prompt})
                answer = response["result"]
                sources = response["source_documents"]
                
                st.write(answer)
                
                # Show Sources in a clean dropdown
                with st.expander("📚 View Source Documents"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1}:** [{doc.metadata.get('source', 'PDF')}]({doc.metadata.get('source', '#')})")
                        st.caption(doc.page_content[:200] + "...")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error generating response: {e}")
