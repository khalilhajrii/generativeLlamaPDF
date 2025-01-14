from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader  

import os
from pinecone import Pinecone
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
# Load API key from .env file
load_dotenv()
# Load sentence transformer model for embedding
# Use a model that produces embeddings with dimension 768
embeddings = SentenceTransformerEmbeddings(model_name='all-mpnet-base-v2') 

# Initialize Pinecone with API key
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
# Define Pinecone index
index_name = "test2"
index = pc.Index(index_name)
# Print index statistics
index.describe_index_stats()

# Load the PDF document
file = "./data/test.pdf"
loader = PyPDFLoader(file)
data = loader.load()
# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
fragments = text_splitter.split_documents(data)
# Print a sample of the text fragments
print("Fragments sample:", fragments[:3])

# Convert fragments into embeddings and store in Pinecone
pinecone = PineconeVectorStore.from_documents(
    fragments, embeddings, index_name=index_name
)