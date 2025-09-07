import os
from dotenv import load_dotenv
import pandas as pd
from langchain_community.chat_models import ChatOpenAI
from llama_index.core import VectorStoreIndex, ServiceContext, Document,Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
# from llama_index import VectorStoreIndex, Document, Settings
from llama_index.core import Settings

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings


# โหลดค่าใน .env
load_dotenv()

# ดึงค่า API Key จาก environment variable ชื่อ OPENAI_API_KEY
api_key = os.getenv("OPENAI_API_KEY")

# # ตรวจสอบ key
if not api_key:
    raise ValueError("กรุณาใส่ OPENAI_API_KEY ในไฟล์ .env")

data = pd.read_csv("data/resturant.csv")
documents = [Document(text=f"{row['dish']}: {row['description']}, ราคา {row['price']} บาท") for _, row in data.iterrows()]

# สร้าง embeddings local
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
# print(documents)




Settings.llm = OpenAI(model="gpt-3.5-turbo")
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900

query_engine = index.as_query_engine(llm=Settings.llm)
response = query_engine.query("อยากกินไก่ทอด")
print(response)
