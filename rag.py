import logging
import ebooklib
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from mistralai import Mistral
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from ebooklib import epub
from bs4 import BeautifulSoup
import os

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Токен Telegram бота
TELEGRAM_BOT_TOKEN = '7466857543:AAHWnxfKJ4hpS46zoI48CaRIYMP7aCir4BY'

# API endpoint Mistral
MISTRAL_API_ENDPOINT = 'BbzDhJzmYPuASFVeZC7C6pLuHuC6qb6m'
MISTRAL_MODEL = 'mistral-medium'  # Replace with the name of your Mistral model
MISTRAL_CLIENT = Mistral(api_key=MISTRAL_API_ENDPOINT)
# API endpoint Pinecone
PINECONE_API_KEY = 'pcsk_3jKr3Z_SrBc6DCo3JRmssPRaJ6yhRHkK4DK2VmdZyrmwSZu6ypEUuPYhFWyY743QiDadmZ'
PINECONE_ENVIRONMENT = 'us-east-1'
INDEX_NAME = 'multilingual-e5-large'

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Hi! I am your Mistral RAG bot. Send me a message and I will retrieve and generate a response for you.')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    logger.info(f"Received message: {user_message}")

    # Step 1: Retrieve relevant documents from the vector database
    retrieved_docs = retrieve_documents(user_message)

    # Step 2: Generate a response using Mistral API
    response = generate_response(user_message, retrieved_docs)

    await update.message.reply_text(response)

def retrieve_documents(query: str):
    # Convert the query to a vector
    query_vector = model.encode(query).tolist()

    # Query the Pinecone index with keyword arguments
    index = pc.Index(INDEX_NAME)
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)

    # Extract the document IDs and scores
    retrieved_docs = [{'id': match['id'], 'score': match['score']} for match in results['matches']]

    return retrieved_docs


def generate_response(query: str, documents: list):
    """Generate a response using Mistral API."""
    # Construct the prompt
    prompt = f"""
        Based on using {documents} answer to the question.
    """

    # Send the prompt to Mistral API
    try:
        response = MISTRAL_CLIENT.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{'role': "user", 'content': prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response at the moment."

def extract_text_from_epub(file_path):
    book = epub.read_epub(file_path)
    text = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_body_content(), 'html.parser')
        text.append(soup.get_text())
    return ' '.join(text)

def generate_embeddings(documents):
    embeddings = []
    for doc in documents:
        embedding = model.encode(doc['text'])
        embeddings.append({'id': doc['id'], 'values': embedding.tolist()})
    return embeddings

def main() -> None:
    # Extract text from EPUB files and generate embeddings
    epub_folder = '/home/bullat/projects/rag/data'
    documents = []
    for filename in os.listdir(epub_folder):
        if filename.endswith('.epub'):
            file_path = os.path.join(epub_folder, filename)
            text = extract_text_from_epub(file_path)
            documents.append({'id': filename, 'text': text})

    embeddings = generate_embeddings(documents)

    # Create the index if it doesn't exist
    if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=len(embeddings[0]['values']),
                metric='euclidean',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )

    # Upsert embeddings to the Pinecone index
    index = pc.Index(INDEX_NAME)
    index.upsert(embeddings)

    # Create the ApplicationBuilder and pass it your bot's token.
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))

    # on noncommand i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the Bot
    application.run_polling()

if __name__ == '__main__':
    main()
