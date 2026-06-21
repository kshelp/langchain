from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# 프롬프트 정의
prompt = PromptTemplate.from_template("{topic}에 대해 {level} 수준으로 설명해 주세요.")

# LLM 모델 객체 생성
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 출력 파서 정의
output_parser = StrOutputParser()

# 체인 정의
chain = prompt | model | output_parser

# 체인 실행
response = chain.stream({"topic": "랭체인", "level": "초등학생"})
for chunk in response:
    print(chunk, end="|", flush=True)
