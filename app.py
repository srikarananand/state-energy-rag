import streamlit as st
import os
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
    
    # Check for secrets first, then fall back to manual entry
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

# --- MAIN APP LOGIC ---
st.title("⚡ New York Energy Policy Intelligence")
st.markdown("Ask questions about the **indexed NY State Energy Plan documents**.")

# Stop if keys are missing
if not pinecone_key or not groq_key:
    st.warning("⚠️ Please enter your API keys in the sidebar to continue.")
    st.stop()

@st.cache_resource
def load_chain(pinecone_api_key, groq_api_key):
    """
    Initializes the RAG chain using modern LCEL (LangChain Expression Language).
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
    
    # 3. LLM (Using the model you verified works)
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=groq_api_key
    )
    
    # 4. Prompt Template
    template = """
    You are a policy analyst. Answer the question based ONLY on the following context.
    Cite the document name for every fact you state. If the context contains tables, read them carefully.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # 5. Build Chain (LCEL Style)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # The chain that generates the answer
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    # The chain that retrieves docs AND generates answer
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
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I have access to the NY Energy plans. Ask me about targets or regulations."}]

# Display History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle Input
if query := st.chat_input("Ex: What are the renewable targets for 2030?"):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            try:
                # Invoke the chain
                response = chain.invoke(query)
                
                answer = response["answer"]
                sources = response["context"]
                
                st.write(answer)
                
                # Show Sources
                with st.expander("📚 View Source Documents"):
                    for i, doc in enumerate(sources):
                        # Safely get metadata
                        source_name = doc.metadata.get('source', 'Unknown PDF')
                        # Create a clickable link if it looks like a URL
                        if source_name.startswith("http"):
                            st.markdown(f"**Source {i+1}:** [{source_name}]({source_name})")
                        else:
                            st.markdown(f"**Source {i+1}:** {source_name}")
                            
                        st.caption(doc.page_content[:200] + "...")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
