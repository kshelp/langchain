from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

# 체인 1: 업적 검색
achievement_prompt = PromptTemplate.from_template("'{person}'의 주요 업적은?")
achievement_chain = achievement_prompt | model | output_parser

# 체인 2: 생애 검색
bio_prompt = PromptTemplate.from_template("'{person}'의 출생 연도와 사망 연도는?")
bio_chain = bio_prompt | model | output_parser

# 병렬 결합 (두 체인을 동시에 실행)
parallel_step = RunnableParallel(achievements=achievement_chain, biography=bio_chain)

# 최종 보고서 작성 체인
final_report_prompt = PromptTemplate.from_template(
    "다음 정보를 바탕으로 보고서를 써줘.\n업적: {achievements}\n생애: {biography}"
)
final_report_chain = final_report_prompt | model | output_parser

# 결합된 체인 (병렬 처리 -> 보고서 작성)
full_workflow_chain = parallel_step | final_report_chain

# 결합된 체인 실행
result = full_workflow_chain.invoke({"person": "알베르트 아인슈타인"})

print(result)
