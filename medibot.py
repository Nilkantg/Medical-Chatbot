import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment variables. Please set it in .env.")
    st.stop()

# Cache the vector store
@st.cache_resource
def get_vectorstore():
    """Load FAISS vector store with HuggingFace embeddings."""
    try:
        logger.info("Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        logger.info("Loading FAISS vector store...")
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        logger.info("Vector store loaded successfully.")
        return db
    except Exception as e:
        logger.error(f"Failed to load vector store: {str(e)}", exc_info=True)
        st.error(f"Failed to load vector store: {str(e)}")
        return None

# Cache the LLM
@st.cache_resource
def load_llm(groq_api_key: str):
    """Load ChatGroq LLM."""
    try:
        logger.info("Loading ChatGroq LLM...")
        llm = ChatGroq(
            model="Gemma2-9b-It",
            groq_api_key=groq_api_key,
            temperature=0.3
        )
        logger.info("LLM loaded successfully.")
        return llm
    except Exception as e:
        logger.error(f"Failed to load LLM: {str(e)}", exc_info=True)
        st.error(f"Failed to load LLM: {str(e)}")
        return None

# Define custom prompt template
def set_custom_prompt():
    """Create prompt template with conversation history."""
    custom_prompt_template = """
    You are a medical assistant. Use the provided context and conversation history to answer the user's question.
    If you don't know the answer, say so clearly and do not make up information.
    Only use information from the given context and history.

    Conversation History (if any):
    {chat_history}

    Context:
    {context}

    Question:
    {question}

    Answer directly and concisely.
    """
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["chat_history", "context", "question"]
    )

# Initialize session state and memory
def initialize_session():
    """Initialize session state and conversation memory."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=5  # Keep last 5 interactions
        )

def main():
    st.title("I'm Your MediBot! 🩺")

    # Initialize session state and memory
    initialize_session()

    # Sidebar for settings and controls
    with st.sidebar:
        st.header("Settings")
        st.button("Clear Chat History", on_click=lambda: clear_chat_history())

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Get user input
    prompt = st.chat_input("Ask your medical question here (e.g., 'Explain me about cancer')")

    if prompt:
        # Validate input
        if len(prompt.strip()) == 0 or len(prompt) > 1000:
            st.warning("Please enter a valid prompt (1-1000 characters).")
            return

        # Display user message
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            # Load vector store and LLM
            vectorstore = get_vectorstore()
            llm = load_llm(GROQ_API_KEY)

            if vectorstore is None or llm is None:
                st.error("Failed to initialize vector store or LLM.")
                st.stop()

            # Create ConversationalRetrievalChain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                memory=st.session_state.memory,
                combine_docs_chain_kwargs={'prompt': set_custom_prompt()},
                return_source_documents=True
            )

            # Invoke the chain
            with st.spinner("Thinking..."):
                logger.info(f"Invoking chain with prompt: {prompt}")
                response = qa_chain.invoke({"question": prompt})

            # Extract answer and source documents
            result = response["answer"]
            source_documents = response["source_documents"]

            # Display assistant response
            with st.chat_message('assistant'):
                st.markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

            # Display source documents
            with st.expander("Source Documents"):
                for idx, doc in enumerate(source_documents):
                    st.markdown(f"**Document {idx + 1}**: {doc.page_content}")

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            st.error(f"Error processing your request: {str(e)}")

def clear_chat_history():
    """Clear chat history and memory."""
    st.session_state.messages = []
    st.session_state.memory.clear()

if __name__ == "__main__":
    main()