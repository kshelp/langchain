from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# 공통 요소
model = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

# 체인 1: 이야기 요약기
summarize_prompt = PromptTemplate.from_template(
    "다음 제목의 이야기를 3줄로 요약해줘: {title}"
)
summarize_chain = summarize_prompt | model | output_parser

# 체인 2: 영문 번역기
translate_prompt = PromptTemplate.from_template(
    "다음 한국어 문장을 영어로 번역해줘: {summary}"
)
translate_chain = translate_prompt | model | output_parser

# 두 체인을 결합
# summarize_chain의 결과(요약된 텍스트)가 translate_chain의 입력 변수({summary})로 전달
composed_chain = {"summary": summarize_chain} | translate_chain

# 결합된 체인을 실행
final_result = composed_chain.invoke({"title": "홍길동전"})
print(final_result)
