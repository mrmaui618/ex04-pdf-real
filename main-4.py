# pip install --upgrade langchain langchain-community langchain-text-splitters langchain-openai langchain-chroma pypdf python-dotenv

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

loader = PyPDFLoader("unsu.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,           # 하나의 청크가 가질 최대 글자 수
    chunk_overlap = 20,         # 청크 간에 겹칠 글자 수 (문맥 단절 방지)
    length_function = len,      # 길이를 측정할 함수 (기본 문자열 길이)
    is_separator_regex = False, # 구분 기호(separator)를 정규표현식으로 해석할지 여부
)

texts = text_splitter.split_documents(pages)

embeddings_model = OpenAIEmbeddings()
# 임베딩 모델 생성
# 분할된 텍스트 청크들을 임베딩 모델을 통해 벡터로 변환하고, Chroma 데이터베이스에 저장합니다.
# from_documents는 임베딩과 저장 과정을 한 번에 처리합니다.
# 현재 코드는 메모리 상에 임시 저장하는 방식입니다.

db = Chroma.from_documents(texts, embeddings_model)