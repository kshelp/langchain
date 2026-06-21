# 프롬프트 템플릿을 정의하고, 프롬프트를 생성
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template("{topic}에 대해 쉽게 설명해 주세요.")
prompt = template.invoke({"topic": "랭체인"})
print(type(prompt))
print(prompt)


# LLM 모델 객체를 생성하고, 모델에 질의
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
response = model.invoke(prompt)
print(type(response))
print(response)


# 출력 파서를 정의하고, 모델 응답에서 텍스트 데이트만 추출
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
output = parser.invoke(response)
print(type(output))
print(output)
