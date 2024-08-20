from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough

import warnings
warnings.filterwarnings("ignore")


def load_model_Ollama(model_name: str):
    # Load the ChatOllama model with the specified model name
    model = ChatOllama(model=model_name)
    return model

def embed_text_into_vector_db(file_path: str, embedding_name='sentence-transformers/all-mpnet-base-v2', 
                                                                          chunk_size=100, chunk_overlap=0):
  # Load the PDF file
  loader = PyPDFLoader(file_path)  
  docs = loader.load()
  
  # Transform HTML content to plain text
  html2text = Html2TextTransformer()
  docs_transformed = html2text.transform_documents(docs)
  
  # Split the documents into chunks
  text_splitter = CharacterTextSplitter(chunk_size=chunk_size,
                                        chunk_overlap=chunk_overlap)
  chunked_documents = text_splitter.split_documents(docs_transformed)


  # Create a FAISS vector store from the chunked documents
  db = FAISS.from_documents(chunked_documents,
                            HuggingFaceEmbeddings(model_name=embedding_name))

  # Create a retriever from the vector store
  retriever = db.as_retriever()

  return retriever


def create_prompt(model, retriever):

  # Create prompt template
  prompt_template = """
  ### [INST] Instruction: Please answer the following question based on the provided `context` that follows the question.". Here is context to help:

  {context}

  ### QUESTION:
  {question} [/INST]
  """

  # Create prompt from prompt template
  prompt = PromptTemplate(
      input_variables=["context", "question"],
      template=prompt_template,
  )

  # Create LLM chain
  llm_chain = LLMChain(llm=model, prompt=prompt)

# Create a RAG (Retrieval-Augmented Generation) chain
  rag_chain = (
  {"context": retriever, "question": RunnablePassthrough()}
      | llm_chain
  )

  return rag_chain


