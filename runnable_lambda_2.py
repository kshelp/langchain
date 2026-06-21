from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()


# 사용자 질문에 대한 주제를 파악해 한 단어로 답변하는 체인
template_classifier = """
주어진 사용자 질문을 `수학`, `과학`, `기타` 중 하나로 분류하세요.
이외의 답변은 허용하지 않습니다.

{question}
"""
prompt_classifier = PromptTemplate.from_template(template_classifier)
chain_classifier = prompt_classifier | model | output_parser

question = "2 + 2는 무엇입니까?"
# result = chain_classifier.invoke({"question": question})
# print(result)


# 각 주제에 특화된 체인을 정의
# 수학
template_math = """
당신은 수학 전문가 입니다. 
항상 다음과 같이 답변을 시작합니다. "피타고라스께서 말씀하시기를 ..."

다음 질문에 답하세요.
질문: {question}
답변: 
"""
prompt_math = PromptTemplate.from_template(template_math)
chain_math = prompt_math | model | output_parser

# 과학
template_science = """
당신은 과학 전문가 입니다. 
항상 다음과 같이 답변을 시작합니다. "아인슈타인께서 말씀하시기를 ..."

다음 질문에 답하세요.
질문: {question}
답변: 
"""
prompt_science = PromptTemplate.from_template(template_science)
chain_science = prompt_science | model | output_parser

# 일반 상식
template_general = """
당신은 일반 상식 전문가 입니다. 
항상 다음과 같이 답변을 시작합니다. "어른들이 말씀하시기를 ..."

다음 질문에 답하세요.
질문: {question}
답변: 
"""
prompt_general = PromptTemplate.from_template(template_general)
chain_general = prompt_general | model | output_parser


# 라우터 함수 정의
def route(info):
    if "수학" in info["topic"]:
        return chain_math
    elif "과학" in info["topic"]:
        return chain_science
    else:
        return chain_general


# 전체 체인 연결
full_chain = (
    {"topic": chain_classifier, "question": lambda x: x["question"]}
    | RunnableLambda(route)
    | output_parser
)
answer = full_chain.invoke({"question": question})
print(f"질문: {question}")
print(f"답변: {answer}\n")


question = "물이 끓는 온도는?"
answer = full_chain.invoke({"question": question})
print(f"질문: {question}")
print(f"답변: {answer}\n")
