from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

@tool
def calculator(a: int, b: int) -> int:
    """
    두 정수를 곱한 결과를 반환합니다.
    단순 산술 곱셈을 수행해야 할 때만 이 도구를 사용하세요.
    """
    return a * b


@tool
def get_weather(city: str) -> str:
    """
    특정 도시의 현재 날씨 정보를 반환합니다.
    특정 지역 또는 장소의 날씨는 그 곳이 속한 도시를 기반으로 답변합니다.
    예) 불국사의 현재 날씨 -> 경주시의 현재 날씨
    """

    # 실제 환경에서는 외부 날씨 API를 호출
    if "서울" in city:
        weather = "서울의 현재 날씨는 섭씨 24도이며 맑습니다."
    elif "부산" in city:
        weather = "부산의 현재 날씨는 섭씨 30도이며 흐립니다."
    else:
        weather = f"{city}의 날씨 정보를 찾을 수 없습니다."

    return weather


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 주어진 도구들을 활용할 수 있는 AI 에이전트입니다.\n",
            "중요: 사용자의 질문이 제공된 도구의 기능과 정확히 일치하지 않거나, ",
            "도구가 해결할 수 없는 질문이라면 도구를 호출하지 말고 직접 일반 텍스트 답변을 작성해 응답하세요."""
        ),
        ("human", "{input}"),
    ]
)
model = ChatOpenAI(model="gpt-4o-mini")

# 모델이 사용할 수 있는 도구 목록을 리스트로 묶어 바인딩
tools = [calculator, get_weather]
model_with_tools = model.bind_tools(tools)
chain = prompt | model_with_tools


# 테스트 실행
# 모델은 질문을 분석 후 자신이 직접 답하지 않고 multiply_numbers 도구 호출을 요청
def test(question):
    response = chain.invoke({"input": question})
    print(question)
    print("-" * 50)
    if response.tool_calls:
        print(f"도구 호출 >>> {response.tool_calls}")
    else:
        print(f"일반 답변 >>> {response.content}")
    print()


test("123 곱하기 456은 뭐야?")
test("해운대에 놀러가려고 하는데 날씨가 좋은가?")

test("123 더하기 456은 뭐야?")
test("부산은 지금 몇 시야?")
