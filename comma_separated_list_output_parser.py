from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from dotenv import load_dotenv

load_dotenv()

# 파서 객체 생성
parser = CommaSeparatedListOutputParser()

# 파서가 요구하는 형식 지시사항
format_instructions = parser.get_format_instructions()

# 프롬프트 작성
prompt = PromptTemplate(
    template="{subject} 3가지만 추천해줘.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

# 모델 생성
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 체인 생성 및 실행
chain = prompt | model | parser
result = chain.invoke({"subject": "여름 과일"})

print(result)
print(type(result))
print("-" * 50)
print(format_instructions)

for x in result:
    print(x)
