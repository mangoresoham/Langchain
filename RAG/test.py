import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Step 2: Generate embeddings using 'all-MiniLM-L6-v2'
def generate_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings

# Step 3: Build FAISS index
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Step 4: Retrieve similar documents
def retrieve_similar_documents(query, index, texts, k=5):
    query_embedding = SentenceTransformer('all-MiniLM-L6-v2').encode([query])
    distances, indices = index.search(query_embedding, k)
    return [texts[i] for i in indices[0]]

# Step 5: Answer question using a QA model
def answer_question(context, question, model_name='distilbert-base-uncased-distilled-squad'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

    result = nlp(question=question, context=context)
    return result['answer']

# Example usage
pdf_path = 'D:\Programming\Projects\Langchain\RAG\TEST_PDF.pdf'  # Replace with your PDF file path
text = extract_text_from_pdf(pdf_path)
texts = text.split('\n')  # Split text into smaller chunks
embeddings = generate_embeddings(texts)
index = build_faiss_index(embeddings)

# Query the system
query_text = 'What is attention ?'  # Replace with your query text
similar_texts = retrieve_similar_documents(query_text, index, texts)

# Combine similar texts to form a context
context = " ".join(similar_texts)
answer = answer_question(context, query_text)
print(answer)
