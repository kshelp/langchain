from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
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

# 대화 이력을 저장할 InMemoryChatMessageHistory 객체를 생성
#chat_history = InMemoryChatMessageHistory()

#def get_chat_history():
#    return chat_history

# 사용자별 대화 내용을 저장할 저장소
user_chat_history = {}

def get_user_chat_history(session_id):
    if session_id not in user_chat_history: 
        user_chat_history[session_id] = InMemoryChatMessageHistory()
    return user_chat_history[session_id]

# 기본 체인에 대화 내용 저장 기능을 래핑
chain_with_history = RunnableWithMessageHistory(
    chain, 
    # get_chat_history, 
    get_user_chat_history, 
    input_messages_key="input", 
    history_messages_key="chat_history"
)

if __name__ == "__main__":
    while True:
        user_input = input("\n>>> 입장할 대화방 번호: ")
        if user_input == "exit":
            print("대화를 종료합니다.")
            break

        session_id = user_input
        print(f"{session_id}번 대화방에 입장했습니다.")
        print("-" * 50)

        while True:
            user_input = input("사용자: ")
            if user_input == "exit":
                print(f"{session_id}번 대화방을 나갑니다.")
                break

            response = chain_with_history.invoke(
                {"input": user_input}, config={"session_id": session_id}
            )
            print(f"AI: {response}\n")
