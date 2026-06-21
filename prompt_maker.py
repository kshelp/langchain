from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# 모델 및 출력 파서 설정
model = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

# 게으른 사용자 프롬프트
instruction = """반드시 한글로 작성되어야 합니다.
사용자의 질문을 읽고, 핵심 키워드를 파악해 전문 지식이 있는 사람의 질문으로 변경해 주세요.
더 체계적이고, 단계적인 질문이 될 수 있도록 변경하세요.
사용자의 질문에서 벗어나서는 안됩니다."""
question = "강아지 눈이 좀 이상해요."
print("\n사용자 프롬프트 >>>")
print("-" * 50)
print(instruction)
print(question)

# 첫번째 체인 -> 개선된 프롬프트
client = Client()
prompt1 = client.pull_prompt("hardkothari/prompt-maker", dangerously_pull_public_prompt=True)
chain1 = prompt1 | model | output_parser

improved_promt = chain1.invoke({"task": instruction, "lazy_prompt": question})
print("\n개선된 프롬프트 >>>")
print("-" * 50)
print(improved_promt)

# 두번째 체인 -> 최종 답변
chain2 = model | output_parser

final_answer = chain2.invoke(improved_promt)

print("\n최종 답변 >>>")
print("-" * 50)
print(final_answer)

# hardkothari/prompt-maker 프롬프트 정보
print("\n프롬프트 정보 >>>")
print("-" * 50)
print(f"프롬프트 타입: {type(prompt1).__name__}")
print(f"입력 변수: {prompt1.input_variables}")
print(f"메시지 구성:")
for msg in prompt1.messages:
    print(f"- {type(msg).__name__}: {str(msg)[:100]}")

