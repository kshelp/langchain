# 패키지 임포트
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langsmith import Client
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import tempfile
import os


# 제목
st.title("PDF 기반 챗봇")
st.markdown("---")

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])
st.markdown("---")


# 업로드한 PDF 파일을 Document 객체로 변환하는 함수
def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(temp_filepath)
        pages = loader.load_and_split()
        return pages
    

# 업로드 파일 처리 
if uploaded_file is not None:
    load_dotenv()

    # PDF 로드
    pages = pdf_to_document(uploaded_file)

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len,
        is_separator_regex=False
    )
    texts = text_splitter.split_documents(pages)

    # 임베딩 생성
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    # Chroma DB에 저장
    db = Chroma.from_documents(documents=texts, embedding=embedding)

    # 사용자 입력을 처리
    st.header("PDF에 질문해 보세요!!!")
    question = st.text_input("질문을 입력하세요.")

    if st.button("질문하기"):
        # 아래쪽 모든 코드를 한 칸 들여쓰기

        # 문서 검색기 생성
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        retriever = MultiQueryRetriever.from_llm(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={"k": 3})
        )

        # 사용자 질의를 LLM에 전달해 답변 생성
        client = Client()
        prompt = client.pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            { "context": retriever | format_docs, "question": RunnablePassthrough() }
            | prompt
            | llm
            | StrOutputParser()
        )

        #result = rag_chain.invoke("아내가 먹고 싶어하는 음식은 무엇인가요?")
        #print(result)

        result = rag_chain.invoke(question)
        st.write(result)

