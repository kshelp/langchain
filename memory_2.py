from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

# 대화 이력이 들어갈 자리를 MessagesPlaceholder로 정의
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 친절하고 전문적인 AI 어시스턴트입니다. 이전 대화 내용을 기억해 맥락에 맞는 답변을 하세요.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
# 모델 및 출력 파서 생성
model = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

# 체인 구성
chain = prompt | model | output_parser

# 대화 이력을 저장할 리스트를 생성
# chat_history = []

# 대화 이력을 저장할 InMemoryChatMessageHistory 객체를 생성
chat_history = InMemoryChatMessageHistory()

while True:
    # 사용자 입력 처리
    user_input = input("\n사용자 >>> ").strip()

    if user_input.lower() == "exit":
        print("챗봇을 종료합니다!!!")
        break

    if not user_input:
        continue

    # 체인 실행 및 대화 이력 주입
    response = chain.invoke({"chat_history": chat_history.messages, "input": user_input})
    print(f"AI >>> {response}")

    # 사용자 입력과 LLM 응답을 대화 이력에 추가
    #chat_history.append(HumanMessage(content=user_input))
    #chat_history.append(AIMessage(content=response))
    chat_history.add_user_message(user_input)
    chat_history.add_ai_message(response)
