import os
from fastapi import FastAPI
from pydantic import BaseModel

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

app = FastAPI()
class QueryRequest(BaseModel):
    query: str

def sllm_agent(query: str) -> str:
    # 환경 변수 설정
    os.environ["TAVILY_API_KEY"] = "your_tavily_api_key"
    os.environ["LANGCHAIN_API_KEY"] = "your_langchain_api_key"
    MY_ACCECE_KEY = "your_openai_api_key"
    GPT4_KEY = "your_openai_api_key"

    search = TavilySearchResults(k=5)
    search.invoke(query)

    loader = PyPDFLoader("/path/to/your/pdf/file.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = loader.load_and_split(text_splitter)

    # OpenAI 임베딩 및 FAISS 설정
    embeddings = OpenAIEmbeddings(openai_api_key=MY_ACCECE_KEY)
    vector = FAISS.from_documents(split_docs, embeddings)

    # Retriever 생성
    retriever = vector.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="증권시장 관련 정보를 PDF 문서에서 검색합니다. '코스닥, 증권, 장외시장' 관련 질문은 이 도구를 사용합니다.",
    )

    tools = [search, retriever_tool]
    gpt = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0, openai_api_key=GPT4_KEY)
    prompt = hub.pull("hwchase17/openai-functions-agent",)
    agent = create_openai_functions_agent(gpt, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke({"input": query})
    return response["output"]

@app.post("/ask")
async def ask_sllm(query_request: QueryRequest):
    answer = sllm_agent(query_request.query)
    return {"question": query_request.query, "answer": answer}