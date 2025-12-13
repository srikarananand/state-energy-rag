import streamlit as st
import os

# --- YOUR CONFIRMED IMPORTS ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# --- PAGE CONFIG ---
st.set_page_config(page_title="NY Energy Policy AI", layout="wide")

# --- SIDEBAR & KEYS ---
with st.sidebar:
    st.title("Settings")
    
    # Check for secrets first, otherwise ask user
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
def get_rag_components(pinecone_api_key, groq_api_key):
    """
    Initializes the RAG components. 
    Returns the retriever and the chain separately so we can handle sources easily.
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
    
    # 3. LLM (Groq)
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0,
        groq_api_key=groq_api_key
    )
    
    # 4. Prompt Template (Modern Chat Format)
    template = """You are an expert energy policy analyst.
    Answer the user's question based ONLY on the context below.
    If the context contains tables, carefully read the rows and columns.
    ALWAYS cite the filename/source for your facts.

    Context:
    {context}

    Question: 
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 5. Build the Generation Chain (LCEL)
    # We do NOT include the retriever here. We will retrieve manually 
    # so we can display the sources in the UI easily.
    chain = (
        prompt 
        | llm 
        | StrOutputParser()
    )
    
    return retriever, chain

try:
    retriever, chain = get_rag_components(pinecone_key, groq_key)
except Exception as e:
    st.error(f"Connection Error: {e}")
    st.stop()

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I have read the NY Energy Plans. Ask me about targets, capacity, or regulations."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_query := st.chat_input("Ex: What is the 2030 target for solar capacity?"):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing 18,000+ policy records..."):
            try:
                # 2. Explicit Retrieval Step (So we can show sources)
                docs = retriever.invoke(user_query)
                
                # Format context string
                context_text = "\n\n".join([d.page_content for d in docs])
                
                # 3. Generate Answer
                answer = chain.invoke({"context": context_text, "question": user_query})
                
                st.write(answer)
                
                # 4. Show Sources
                with st.expander("📚 View Source Documents"):
                    for i, doc in enumerate(docs):
                        source_name = doc.metadata.get('source', 'Unknown PDF')
                        # Make the source clickable if it's a URL
                        st.markdown(f"**Source {i+1}:** [{source_name}]({source_name})")
                        st.caption(doc.page_content[:200].replace("\n", " ") + "...")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error generating response: {e}")
