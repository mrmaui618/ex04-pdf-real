# pip install -U langchain-community langchain-text-splitters pypdf

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = PyPDFLoader("unsu.pdf")
pages = loader.load_and_split()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
  
    chunk_overlap = 20,

    length_function = len,

    is_separator_regex = False,
)


texts = text_splitter.split_documents(pages)


if texts:
    print("--- [첫 번째 텍스트 조각(Chunk) 객체 출력] ---")
    print(texts[0])
    
    print("\n--- [첫 번째 조각의 실제 텍스트 내용만 출력] ---")
    print(texts[0].page_content)
else:
    print("분할된 텍스트 조각이 없습니다. PDF 파일 내용을 확인해 주세요.")