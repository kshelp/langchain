# 프롬프트 템플릿, 모델, 출력 파서를 정의
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

template = PromptTemplate.from_template("{topic}에 대해 쉽게 설명해 주세요.")
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()


# 체인을 정의하고, 실행
chain = template | model | parser
# output = chain.invoke({"topic": "랭체인"})
# print(type(output))
# print(output)

output = chain.stream({"topic": "랭체인"})
for chunk in output:
    print(chunk, end="|", flush=True)
