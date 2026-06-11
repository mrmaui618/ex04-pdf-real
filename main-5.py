# pip install --upgrade langchain langchain-community langchain-text-splitters langchain-openai langchain-chroma pypdf python-dotenv
# pip install  langchain 
import os
# [!!] protobuf 버전 충돌 에러 해결을 위한 임시방편 코드입니다.
# [!!] 이 방법은 성능 저하를 유발할 수 있으므로, 터미널에서 protobuf 버전을 다운그레이드하는 '방법 1'을 권장합니다.
# [!!] (권장 해결책) 터미널 실행: pip uninstall -y protobuf && pip install protobuf==3.20.3
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

loader = PyPDFLoader("unsu.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,           # 하나의 청크가 가질 최대 글자 수
    chunk_overlap  = 20,        # 청크 간 문맥 연결을 위해 겹칠 글자 수
    length_function = len,      # 길이 측정 기준 (기본 문자열 길이)
    is_separator_regex = False, # 구분 기호의 정규표현식 해석 여부
)
texts = text_splitter.split_documents(pages)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(texts, embeddings_model)

# 멀티 쿼리 리트리버 생성 & LLM 설정
# 모델 명시적으로 지정할 수 있다. 예: model="gpt-4o-mini"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 사용자의 질문을 다양한 각도에서 재해석하여 검색 확률을 높이는 MultiQueryRetriever를 생성함
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(), 
    llm=llm
)

# RAG 체인 구성
# LLM에게 전달할 프롬프트(지시문)을 정의함
system_prompt = (
    "너는 질문-답변을 돕는 유능한 비서야. "
    "아래 제공된 맥락(context)만을 사용하여 질문에 답해줘. "
    "답을 모르면 모른다고 하고, 절대 답변을 지어내지 마.\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 검색된 문서들을 활용하여 질문에 답변하는 체인 생성
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Chroma DB를 검색기(Retriever)로 전환합니다.
retriever = db.as_retriever()  

# RAG 체인 생성: 검색과 질문-답변 체인을 결합하여 최종 RAG 체인을 만듦
rag_chain = create_retrieval_chain(retriever_from_llm, question_answer_chain)

# 질문 실행 및 결과 출력
question = "아내가 사달라고 했던 음식은 무엇인가요?"

response = rag_chain.invoke({"input": question})

# 결과 출력 수정
print("--- [최종 답변] ---")
print(response["answer"])