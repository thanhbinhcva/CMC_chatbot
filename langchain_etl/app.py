import os
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv(find_dotenv())  
openai.api_key = os.environ['OPENAI_API_KEY']

if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please check your .env file.")

persist_directory = "docs/chroma"

embedding = OpenAIEmbeddings()

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

template = """Sử dụng các phần ngữ cảnh sau đây để trả lời câu hỏi ở cuối. Nếu không biết câu trả lời, bạn chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời. Sử dụng tối đa ba câu. Giữ câu trả lời ngắn gọn nhất có thể. Luôn nói "cảm ơn vì đã hỏi thăm!" ở cuối câu trả lời.{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

question = "tổng doanh thu là gì?"

result = qa_chain({"query": question})

print(f"Result: {result['result']}")

