import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def build_rag_app(pdf_path: str):
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to a .env file or set it in your environment."
        )

    print(f"Loading PDF: {pdf_path}")
    docs = PyPDFLoader(pdf_path).load()
    if not docs:
        raise RuntimeError("The PDF contains no readable text.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    chunks = text_splitter.split_documents(docs)

    embeddings = FastEmbedEmbeddings()
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="rag_collection",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="rag_collection",
        embedding=embeddings,
    )
    vectorstore.add_documents(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0.2,
        max_tokens=2500,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Use the PDF context to answer questions about "
            "the PDF, and use the conversation history to understand follow-up questions. "
            "If a question is unrelated to the PDF, answer it normally using your general "
            "knowledge. If a PDF-related answer is not supported by the context, say so "
            "clearly and do not invent PDF facts.\n\n"
            "PDF context:\n{context}\n\nConversation history:\n{history}",
        ),
        ("human", "{question}"),
    ])

    qa_chain = prompt | llm | StrOutputParser()
    print(f"Indexed {len(chunks)} chunks from {len(docs)} PDF pages.")
    print(f"Using Groq model: {model_name}")
    return retriever, qa_chain


def query_rag(question: str, retriever, qa_chain, history: str = ""):
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    response = qa_chain.invoke({
        "context": context,
        "history": history,
        "question": question,
    })
    print("\nAnswer:")
    print(response)
    print("-" * 60)
    return response


def interactive_chat():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(project_dir, "Introduction_To_Java.pdf")
    retriever, qa_chain = build_rag_app(pdf_path)
    conversation = []

    print("\nPDF question assistant ready.")
    print("Ask about the PDF or any other topic. Type 'exit' or 'quit' to stop.\n")
    while True:
        question = input("Your question: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break
        history = "\n".join(
            f"User: {user_question}\nAssistant: {answer}"
            for user_question, answer in conversation[-5:]
        )
        answer = query_rag(question, retriever, qa_chain, history)
        conversation.append((question, answer))


if __name__ == "__main__":
    interactive_chat()
