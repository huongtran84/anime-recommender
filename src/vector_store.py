from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader

from dotenv import load_dotenv

load_dotenv()

class VectorStoreBuilder:
    def __init__(self,csv_path: str, persist_dir: str="chroma_db"):
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    def build_and_save_vectorstore(self):
        # Load documents from CSV
        loader = CSVLoader(file_path=self.csv_path,
                            encoding="utf-8",
                            metadata_columns=[])
        documents = loader.load()

        # Split documents into smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        
        db = Chroma.from_documents(documents=texts,
                                    embedding=self.embeddings,
                                    persist_directory=self.persist_dir)
        db.persist()
    
    def load_vector_store(self):
        db = Chroma(persist_directory=self.persist_dir,
                    embedding_function=self.embeddings)
        return db