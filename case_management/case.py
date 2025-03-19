import os
from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from groq import Groq

app = Flask(__name__)

# Groq setup
GROQ_API_KEY = "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"
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize SentenceTransformer embeddings
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Function to load and process the PDF
def load_and_process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    return texts

# Check if the embeddings folder exists and is not empty
if os.path.exists("embeddings") and os.listdir("embeddings"):
    # If the folder exists and is not empty, load the vectorstore from the embeddings
    vectorstore = Chroma(persist_directory="embeddings", embedding_function=embedding_function)
else:
    # If the folder is empty or does not exist, create a new vectorstore
    pdf_path = "C:/Users/somia.kumari/Downloads/vilpdf.pdf" # Make sure this PDF file exists in your directory
    texts = load_and_process_pdf(pdf_path)
    vectorstore = Chroma.from_documents(documents=texts, embedding=embedding_function, persist_directory="embeddings")
    vectorstore.persist()

print(f"Number of documents in vectorstore: {vectorstore._collection.count()}")

async def rag(query: str, contexts: list) -> str:
    print("> RAG Called")
    context_str = "\n".join(contexts)
    prompt = f"""You are a helpful assistant, below is a query from a user and
    some relevant contexts. Answer the question given the information in those
    contexts. If you cannot find the answer to the question, say "I don't know".
    There are sentences saying refer the following screen, please don't say any reference to images or screen.
    Avoid "Refer to the following screen" text in the message.
    If the response can be explained in details then explain it.
    If the user greets (e.g., "hi", "hello", "hey", "good morning", "good evening"), respond strictly with:  
    "Hello! How may I assist you?"  
    If the user asks "How are you?" or similar, respond strictly with:  
    "I'm good! How can I assist you??  
   
    
    Contexts:
    {context_str}
    Query: {query}
    Answer: """

    # Generate answer using Groq
    completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-70b-8192",
        temperature=0.2,
        max_tokens=1000
    )
    return completion.choices[0].message.content

async def retrieve(query: str) -> list:
    print("Retrieving relevant contexts")
    results = vectorstore.similarity_search(query, k=5)
    contexts = [result.page_content for result in results]
    return contexts



async def execute(prompt):
    print("in execute")
    data = await retrieve(prompt)
    response = await rag(prompt, data)
    return response

@app.route('/smsbot', methods=['POST'])
async def send_sms():
    received_message = request.json.get('payload', '')
    print("The received message is", received_message)
    s = await execute(prompt=received_message)
    return jsonify({"message": s})

if __name__ == "__main__":
    print("in main fn")
    app.run(debug=True, host = '0.0.0.0',port=8082)