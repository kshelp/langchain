from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from datetime import datetime
import pytz
from dotenv import load_dotenv

load_dotenv()


# 현재 시간을 반환하는 함수를 정의하고 도구로 등록
@tool
def get_current_time(timezone: str, location: str) -> str:
    """현재 시간을 YYYY-MM-DD HH:MI:SS 형식으로 반환하는 함수

    Args:
        timezone (str): 타임존(예: "Asia/Seoul"). 실제 존재해야 함
        location (str): 지역명. 타임존은 모든 지역에 대응되지 않으므로 이후 llm 답변 생성에 사용됨
    """
    tz = pytz.timezone(timezone)
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    location_and_local_time = f"{timezone} ({location}) 현재 시각 {now}"
    return location_and_local_time


# 도구 목록 정의
tools = [get_current_time]
tool_dict = {"get_current_time": get_current_time}

# 모델 정의
llm = ChatOpenAI(model="gpt-4o-mini")
# 도구를 모델에 바인딩 => 모델에 도구를 바인딩하면, 도구를 사용해 답변을 생성할 수 있음
llm_with_tools = llm.bind_tools(tools)

# 도구를 사용해 답변 생성
messages = [
    SystemMessage("너는 사용자의 질문에 답변을 하기 위해 tools를 사용할 수 있다."),
    HumanMessage("부산은 지금 몇 시야??"),
]
response = llm_with_tools.invoke(messages)
messages.append(response)

# print(response)
# print("-" * 50)
# print(messages)

for tool_call in response.tool_calls:
    selected_tool = tool_dict[tool_call["name"]]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)

response = llm_with_tools.invoke(messages)
messages.append(response)

print(messages)

