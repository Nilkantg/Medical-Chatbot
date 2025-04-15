# Run this command before executing this script in the terminal (optional if set in script):
# set TF_ENABLE_ONEDNN_OPTS=0

# Suppress TensorFlow logs and warnings
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO and WARNING logs
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Step 1: Setup LLM 
from langchain_huggingface import HuggingFaceEndpoint  #type: ignore
from langchain_core.prompts import PromptTemplate #type: ignore
from langchain.chains import RetrievalQA #type: ignore
from langchain_huggingface import HuggingFaceEmbeddings #type: ignore
from langchain_community.vectorstores import FAISS #type: ignore
from langchain_groq import ChatGroq #type: ignore
from dotenv import load_dotenv #type: ignore
load_dotenv()

# from huggingface_hub import InferenceClient
# client = InferenceClient(token=token)
# response = client.text_generation("Test query", model="mistralai/Mistral-7B-Instruct-v0.3")
# print(response)

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
GROQ_API_KEY = os.getenv("GROK_API_KEY")

def load_llm(hf_repo_id, GROK_API_KEY):
    print("Using HF_TOKEN:", HF_TOKEN)
    print("Using repo_id:", hf_repo_id)
    # llm = HuggingFaceEndpoint(
    #     repo_id=hf_repo_id,
    #     temperature=0.5,
    #     model_kwargs={
    #         "token": HF_TOKEN,
    #         "max_length": 512
    #     }
    # )
    llm = ChatGroq(
        model="Gemma2-9b-It", 
        groq_api_key=GROQ_API_KEY
    )
    return llm

# Step 2: Connect LLM with FAISS db and create chain
DB_FAISS_PATH = "vectorstore/db_faiss"
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID, GROQ_API_KEY),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Invoke with a single query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({"query": user_query})

print("Result: ", response["result"])
print("Source Documents: ", response["source_documents"])