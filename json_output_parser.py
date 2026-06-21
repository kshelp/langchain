from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
output_parser = JsonOutputParser()

template = """
당신은 K-Pop 정보를 제공하는 AI 입니다.  그룹의 멤버 이름을 한국어로 말해주세요.
K-pop group ; {name}

FORMAT:
{format}
"""
prompt = PromptTemplate(
    template=template,
    partial_variables={"format": output_parser.get_format_instructions()},
)

chain = prompt | model | output_parser

result = chain.invoke({"name": "BTS"})
print(result)
print(type(result))
print("-" * 50)
print(output_parser.get_format_instructions())
