from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path



def build_vectorstore(docs_dir: Path):

    #Loading the files 
    documents = [] #empty list to collect all the loaded documents
    for rst_path in sorted(docs_dir.glob("*.rst")):  # sorting the files alphabetically everytime

        #Creates a LangChain loader for that one file and loading the data and the source
        loader= TextLoader(str(rst_path),encoding='"utf-8')
        documents.extend(loader.load())   

    #Splitting the documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) #lanchain chunks using characters
    chunks = splitter.split_documents(documents)

    #Embedding 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #Vector
    faiss_vector= FAISS.from_documents(chunks, embeddings)

    return faiss_vector




