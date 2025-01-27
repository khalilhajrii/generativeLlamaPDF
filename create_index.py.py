import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
# Initialize Pinecone with API key
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
# Define index name
index_name = "test"
# Check if index already exists, and delete if necessary
if index_name in pc.list_indexes():
    pc.delete_index(index_name)

# Create a new index with dimension 384 using cosine similarity
pc.create_index(name=index_name, dimension=384, metric="cosine", spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ))

# Connect to the index
index = pc.Index(index_name)

# Print index statistics
print(index.describe_index_stats())