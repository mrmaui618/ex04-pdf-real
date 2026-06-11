import os
# 중요: langchain, chromadb 등 다른 모든 import보다 무조건 위에 있어야 합니다!
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# --- 기존 코드 시작 ---
from langchain_chroma import Chroma
import streamlit as st
# ... 나머지 기존 코드들


# import os
# import tempfile
# import streamlit as st
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader


st.set_page_config(page_title="PDF Q&A", page_icon="📄")
st.title("📄 PDF Q&A 앱")
st.write("PDF 파일을 업로드하고 질문을 입력해보세요!")

# PDF 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])
loader = PDFPlumberLoader("insa.pdf")
pages = loader.load_and_split()


if uploaded_file is not None:
    # 임시 파일로 저장 (PyPDFLoader는 파일 경로가 필요함)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.success("파일이 성공적으로 업로드되었습니다. 문서를 분석 중입니다...")

    try:
        # 문서 로드 및 분할
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.split_documents(pages)

        # 임베딩 모델 및 벡터 DB 설정
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma.from_documents(texts, embeddings_model)

        # LLM 및 RAG 체인 설정
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=db.as_retriever(), 
            llm=llm
        )

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

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever_from_llm, question_answer_chain)

        st.info("문서 분석이 완료되었습니다. 질문을 입력해주세요.")

        # 사용자 질문 입력
        question = st.text_input("질문:")

        if st.button("답변 생성"):
            if question:
                with st.spinner("답변을 생성하는 중입니다..."):
                    response = rag_chain.invoke({"input": question})
                    
                    st.subheader("답변")
                    st.write(response["answer"])
                    
                    with st.expander("참조 문서 확인"):
                        for i, doc in enumerate(response.get("context", [])):
                            st.markdown(f"**[{i+1}]** {doc.page_content}")
            else:
                st.warning("질문을 입력해주세요.")
    finally:
        # 임시 파일 삭제
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

else:
    st.info("위의 업로드 창을 통해 PDF 파일을 선택해주세요.")
