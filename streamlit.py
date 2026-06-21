import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# 웹 페이지 제목 및 안내 문구 설정
st.set_page_config(page_title="LangChain 챗봇", page_icon="🤖")
st.title("🤖 LangChain & Streamlit 챗봇")
st.write("스트림릿을 이용해 구현한 실시간 대화형 웹 애플리케이션입니다.")

# 대화 이력을 저장할 변수를 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 대화 이력을 출력
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 입력 처리
if user_input := st.chat_input("메시지를 입력하세요..."):
    # 사용자 입력을 화면에 출력하고 대화 이력에 저장
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # LLM 호출 및 답변 생성
    with st.chat_message("assistant"):
        # 답변이 생성되는 동안 로딩 애니메이션 표시
        with st.spinner("생각 중..."):
            # 프롬프트 템플릿 | 모델 | 출력 파서 체인 구성
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "당신은 사용자 질문에 정확하게 답하는 친절한 AI 조수입니다.",
                    ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ]
            )
            model = ChatOpenAI(model="gpt-4o-mini")
            chain = prompt | model | StrOutputParser()

            # session_state에 저장된 대화 이력을 LangChain 프롬프트 형식으로 변환해 주입
            chat_history = [
                (msg["role"], msg["content"]) for msg in st.session_state["messages"]
            ]

            # 체인 실행 및 응답 출력
            response = chain.invoke({"chat_history": chat_history, "input": user_input})
            st.write(response)

    # AI의 응답을 대화 이력에 저장
    st.session_state["messages"].append({"role": "assistant", "content": response})

