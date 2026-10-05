import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "bank_documents")


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="BankWise AI",
    page_icon="🏦",
    layout="wide"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        color: #666;
        margin-bottom: 25px;
    }

    .service-card {
        padding: 15px;
        border-radius: 12px;
        background: #f5f7fb;
        margin-bottom: 10px;
    }

    .source-box {
        padding: 10px;
        border-radius: 8px;
        background: #f1f5f9;
        margin-top: 10px;
        font-size: 14px;
    }

    .warning-box {
        padding: 14px;
        border-radius: 10px;
        background: #fff3cd;
        border: 1px solid #ffe69c;
        color: #664d03;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# FOLDERS
# ---------------------------------------------------------

Path(DOCUMENTS_PATH).mkdir(exist_ok=True)


# ---------------------------------------------------------
# EMBEDDINGS
# ---------------------------------------------------------

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# ---------------------------------------------------------
# LOAD PDF DOCUMENTS
# ---------------------------------------------------------

def load_bank_documents():

    pdf_files = list(Path(DOCUMENTS_PATH).glob("*.pdf"))

    documents = []

    for pdf_file in pdf_files:

        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()

            for doc in docs:
                doc.metadata["source_file"] = pdf_file.name

            documents.extend(docs)

        except Exception as e:
            st.warning(f"Could not read {pdf_file.name}: {e}")

    return documents


# ---------------------------------------------------------
# CREATE VECTOR DATABASE
# ---------------------------------------------------------

def create_vector_database():

    documents = load_bank_documents()

    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    if Path(CHROMA_PATH).exists():
        shutil.rmtree(CHROMA_PATH)

    embeddings = get_embeddings()

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="bankwise_documents",
        persist_directory=CHROMA_PATH
    )

    return vector_db


# ---------------------------------------------------------
# LOAD EXISTING VECTOR DATABASE
# ---------------------------------------------------------

@st.cache_resource
def load_vector_database():

    embeddings = get_embeddings()

    if Path(CHROMA_PATH).exists():

        return Chroma(
            collection_name="bankwise_documents",
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH
        )

    return None


# ---------------------------------------------------------
# OLLAMA
# ---------------------------------------------------------

@st.cache_resource
def get_llm():

    return OllamaLLM(
        model=OLLAMA_MODEL,
        temperature=0.2
    )


# ---------------------------------------------------------
# SENSITIVE INFORMATION CHECK
# ---------------------------------------------------------

def contains_sensitive_information(question):

    sensitive_words = [
        "otp",
        "pin",
        "cvv",
        "password",
        "passcode",
        "card number",
        "account number",
        "upi pin",
        "security code"
    ]

    question_lower = question.lower()

    return any(word in question_lower for word in sensitive_words)


# ---------------------------------------------------------
# DETECT BANK-SPECIFIC QUESTIONS
# ---------------------------------------------------------

def is_bank_document_question(question):

    banking_keywords = [
        "bank",
        "account",
        "atm",
        "debit card",
        "credit card",
        "upi",
        "kyc",
        "loan",
        "internet banking",
        "net banking",
        "transaction",
        "interest",
        "withdraw",
        "deposit",
        "cheque",
        "card",
        "emi",
        "fund transfer",
        "mobile banking",
        "cash",
        "branch",
        "security",
        "fraud"
    ]

    question_lower = question.lower()

    return any(word in question_lower for word in banking_keywords)


# ---------------------------------------------------------
# GENERAL AI PROMPT
# ---------------------------------------------------------

def general_ai_answer(question):

    llm = get_llm()

    prompt = f"""
You are BankWise AI, a friendly banking information assistant.

User question:
{question}

Answer the user's question clearly and naturally.

Rules:
1. You can answer general banking questions using your general knowledge.
2. Do not pretend to have access to a user's real bank account.
3. Do not claim to perform transactions.
4. Never ask for OTP, PIN, CVV, password, full card number, or full account number.
5. Do not provide instructions for bypassing banking security.
6. For account-specific problems, advise the user to contact their official bank.
7. If the question involves a financial decision, provide general educational information and mention that the user should verify details with their bank or a qualified professional.
8. Keep the answer simple and useful.

Answer:
"""

    return llm.invoke(prompt)


# ---------------------------------------------------------
# RAG ANSWER
# ---------------------------------------------------------

def rag_answer(question, vector_db):

    if vector_db is None:
        return general_ai_answer(question), []

    results = vector_db.similarity_search(
        question,
        k=4
    )

    if not results:
        return general_ai_answer(question), []

    context_parts = []

    for doc in results:

        source = doc.metadata.get(
            "source_file",
            "Unknown document"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"Source: {source}, Page: {page + 1 if isinstance(page, int) else page}\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    llm = get_llm()

    prompt = f"""
You are BankWise AI, an intelligent banking assistant.

Use the following banking documents as your primary source when they contain
information relevant to the user's question.

BANKING DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Instructions:

1. If the documents contain the answer, answer using the documents.
2. If the documents do not contain the answer, use your general banking knowledge.
3. Never invent bank-specific policies, fees, limits, interest rates, or rules.
4. If information is unavailable, clearly say that the provided documents do not specify it.
5. Never request OTP, PIN, CVV, password, full card number, or full account number.
6. Do not claim that you can access or modify a real bank account.
7. For urgent fraud, unauthorized transactions, or account-specific problems,
   advise the user to contact their official bank immediately.
8. Give a clear and beginner-friendly answer.
9. Do not mention "RAG", "vector database", "embedding", or internal system details
   unless the user specifically asks about the technology.

Answer:
"""

    answer = llm.invoke(prompt)

    return answer, results


# ---------------------------------------------------------
# INITIALIZE VECTOR DATABASE
# ---------------------------------------------------------

vector_db = load_vector_database()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.title("🏦 BankWise AI")

    st.markdown("### Banking Services")

    services = [
        "💳 Debit Cards",
        "💰 Credit Cards",
        "🏧 ATM Services",
        "📱 UPI Services",
        "🔐 KYC Guidelines",
        "🏦 Bank Accounts",
        "💵 Loan Services",
        "🌐 Internet Banking",
        "🛡️ Banking Security"
    ]

    for service in services:
        st.markdown(
            f'<div class="service-card">{service}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown("### 📚 Knowledge Base")

    pdf_files = list(Path(DOCUMENTS_PATH).glob("*.pdf"))

    if pdf_files:

        st.success(
            f"{len(pdf_files)} banking documents available"
        )

        for pdf in pdf_files:
            st.caption(f"📄 {pdf.name}")

    else:

        st.warning(
            "No banking PDF documents found."
        )

    st.markdown("---")

    if st.button(
        "🔄 Rebuild Knowledge Base",
        use_container_width=True
    ):

        with st.spinner(
            "Reading banking documents and creating knowledge base..."
        ):

            new_db = create_vector_database()

            if new_db:

                st.cache_resource.clear()

                st.success(
                    "Knowledge base rebuilt successfully!"
                )

                st.rerun()

            else:

                st.error(
                    "No PDF documents found."
                )

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ---------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🏦 BankWise AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your intelligent banking information assistant powered by RAG, '
    'Hugging Face, ChromaDB and Ollama.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# IMPORTANT DISCLAIMER
# ---------------------------------------------------------

st.info(
    "ℹ️ BankWise AI provides general banking information. "
    "It cannot access real bank accounts or perform transactions. "
    "Never share OTPs, PINs, CVVs, passwords, or complete card/account numbers."
)


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📚 Sources"):

                for source in message["sources"]:
                    st.write(source)


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

question = st.chat_input(
    "Ask me anything about banking..."
)


if question:

    # Display user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Sensitive information warning

    if contains_sensitive_information(question):

        with st.chat_message("assistant"):

            st.markdown(
                """
                <div class="warning-box">
                🔐 <b>Security reminder:</b><br><br>
                Never share OTPs, PINs, CVVs, passwords, or complete
                card/account numbers with anyone, including BankWise AI.
                </div>
                """,
                unsafe_allow_html=True
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "Security reminder: Never share OTPs, PINs, CVVs, "
                    "passwords, or complete card/account numbers."
                )
            }
        )

    else:

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    if is_bank_document_question(question):

                        answer, results = rag_answer(
                            question,
                            vector_db
                        )

                    else:

                        answer = general_ai_answer(
                            question
                        )

                        results = []

                    st.markdown(answer)

                    sources = []

                    for doc in results:

                        source_file = doc.metadata.get(
                            "source_file",
                            "Unknown"
                        )

                        page = doc.metadata.get(
                            "page",
                            "Unknown"
                        )

                        if isinstance(page, int):
                            page = page + 1

                        source_text = (
                            f"{source_file} - Page {page}"
                        )

                        if source_text not in sources:
                            sources.append(source_text)

                    if sources:

                        with st.expander("📚 Sources"):

                            for source in sources:
                                st.write(f"📄 {source}")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        }
                    )

                except Exception as e:

                    error_message = (
                        "Sorry, I couldn't process that request. "
                        "Please make sure Ollama is running and the "
                        "knowledge base is available."
                    )

                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message
                        }
                    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "BankWise AI • RAG + Hugging Face + ChromaDB + Ollama + Streamlit"
)

