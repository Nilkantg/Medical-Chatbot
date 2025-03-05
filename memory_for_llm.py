from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import transformers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# laoding the raw pdf's
data_path = "data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents

documents=load_pdf_files(data=data_path)
print(f"Length of PDF pages: {len(documents)}")

def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

text_chunks = create_chunks(documents)

def embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    return embedding_model

embed_model = embedding_model()

Vector_db_path = "vector_db/"
db = FAISS.from_documents(text_chunks, embed_model)
db.save_local(Vector_db_path)


