from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

# Initialize Flask app
app = Flask(__name__)

# Load environment variables from .env
load_dotenv()

# Get API keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set API keys in environment (for LangChain / Pinecone / OpenAI)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load embeddings (HuggingFace)
embeddings = download_hugging_face_embeddings()

# Use existing Pinecone index
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Load LLM (OpenAI)
llm = OpenAI(temperature=0.4, max_tokens=500)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    user_msg = request.form["msg"]
    print("User:", user_msg)

    response = rag_chain.invoke({"input": user_msg})
    bot_answer = response["answer"]
    print("Bot:", bot_answer)

    return str(bot_answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
