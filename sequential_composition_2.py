
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

# 첫 번째 체인의 프롬프트
prompt1 = PromptTemplate.from_template("{topic}에 대해 {level} 수준으로 설명해 주세요.")

# 첫 번재 체인 생성
chain1 = prompt1 | model | output_parser

# 새로운 딕셔너리 생성
chain1_result_dict = {"chain1_result": chain1}


# 두 번째 체인의 프롬프트
prompt2 = PromptTemplate.from_template("이 정보가 유익한가요? 정보: {chain1_result}")

# 결합된 체인 생성
combined_chain = chain1_result_dict | prompt2 | model | output_parser


# 결합된 체인 실행 
final_result = combined_chain.invoke({"topic": "인공지능", "level": "초등학생"})
print(final_result)

