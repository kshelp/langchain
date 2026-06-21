from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import (
    StrOutputParser,
    CommaSeparatedListOutputParser,
)
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
output_parser = CommaSeparatedListOutputParser()
format_instruction = output_parser.get_format_instructions()

# 사용자 질문에 대한 답변 생성
question = "인셉션 감독이 누구인가요?"
template = """
당신은 영화를 추천해 주는 AI 챗봇입니다. 사용자 질문에 답변하세요.
사용자 질문: {question}
"""
prompt = PromptTemplate(template=template)
chain = prompt | llm | StrOutputParser()
answer = chain.invoke({"question": question})

# 추천 질문
template_recommend = """
당신은 영화를 추천해 주는 AI 챗봇입니다. 
사용자의 질문과 AI의 답변을 바탕으로 사용자가 새롭게 질문할 만한 추천 프롬프트 3개를 생성하시오.
사용자 질문: {question}
AI 답변: {answer}

FORMAT: {format}
"""
prompt_recommend = PromptTemplate(
    template=template_recommend, partial_variables={"format": format_instruction}
)

# 질문 증강
template_augmented = """
당신은 사용자 질문을 새롭게 생성하는 AI 비서입니다. 
사용자의 질문과 유사하지만 사용된 단어가 다른 새로운 질문 3개를 생성하시오.
사용자 질문: {question}

FORMAT: {format}
"""
prompt_augmented = PromptTemplate(
    template=template_augmented, partial_variables={"format": format_instruction}
)

# 추천 질문 및 질문 증강 체인
chain_recommend = prompt_recommend | llm | output_parser
chain_augmented = prompt_augmented | llm | output_parser

# 추천 질문과 질문 증강 체인을 동시에 실행
parallel_chain = RunnableParallel(recommend=chain_recommend, augmented=chain_augmented)
result = parallel_chain.invoke({"question": question, "answer": answer})

# 추천 질문과 질문 증강 체인 실행 결과 출력
print("\n추천 질문")
print("-" * 50)
for x in result["recommend"]:
    print("-", x)

print("\n증강 질문")
print("-" * 50)
for x in result["augmented"]:
    print("-", x)
