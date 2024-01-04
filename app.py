from flask import Flask, request, jsonify, render_template
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import textract
import requests
import json
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

# Initialize Flask application
app = Flask(__name__)

# Initialize other components
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = lambda text: len(tokenizer.encode(text)),
)

# Get a list of all files in the input directory
input_dir = './input'
files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

# Loop through each file
for file in files:
    # Full path to the current file
    file_path = os.path.join(input_dir, file)

    # Check if the file is a PDF
    if not file_path.lower().endswith('.pdf'):
        print(f"Skipping non-PDF file: {file_path}")
        continue

    try:
        # Step 1: Convert PDF to text
        doc = textract.process(file_path)
    except textract.exceptions.ShellError as e:
        print(f"Failed to process {file_path}. Reason: {str(e)}")
        continue

    # Step 2: Save to .txt and reopen (helps prevent issues)
    file_name, _ = os.path.splitext(file)
    output_dir = './processed'
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    txt_file = os.path.join(output_dir, file_name + '.txt')
    with open(txt_file, 'wb') as f:
        f.write(doc)

    with open(txt_file, 'r') as f:
        text = f.read()

    # Step 3: Split text into chunks
    chunks = text_splitter.create_documents([text])

    # Get embedding model
    embeddings = OpenAIEmbeddings()

    # Create vector database
    db = FAISS.from_documents(chunks, embeddings)

    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")


@app.route('/', methods=['GET'])
def home():
    # If it's a GET request, just render the template without a response
    return render_template('index.html')

@app.route('/', methods=['POST'])
def handle_question():
    # Get the question from form data
    question = request.form.get('question')
    if question is None:
        return jsonify({'error': 'No question provided'})

    prompt = (
    "All questions are related to Capytech. "
    "You will only respond with answers related to Capytech. "
    "If a question is outside of this, you will respond with 'Sorry, I cannot help you'."
    )   

    merged_text = prompt + " " + question

    # Perform a similarity search in the database
    print("Before similarity search")  # Debug print
    docs = db.similarity_search(merged_text)
    print("After similarity search")  # Debug print
    
    print("Before chain.run")  # Debug print
    response = chain.run(input_documents=docs, question=merged_text)
    print("After chain.run")
    print (response)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)