from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

def create_vector_store(texts):
    if not texts or texts.strip() == "":
        raise ValueError("Resume is empty or could not be read. Please upload a valid PDF or DOCX file.")

    # Use smaller chunks for speed
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(texts)

    if not chunks:
        raise ValueError("Could not split resume into chunks. Please check the file content.")

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_texts(chunks, embeddings)
    return vectordb


def create_conversational_rag(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})  # Limit top 3 relevant chunks for speed
    llm = ChatOpenAI(temperature=0)
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    return conv_chain

def answer_question(conv_chain, query, chat_history):
    result = conv_chain({"question": query, "chat_history": chat_history})
    return result["answer"]
