from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

# 템플릿 문자열 정의
template_str = "{country}의 수도는 어디인가요? {language}로 대답해 주세요."

prompt = PromptTemplate(
    input_variables=["country", "language"],
    template=template_str
)

# 값 주입 (invoke 또는 format 사용)
result = prompt.invoke({"country": "프랑스", "language": "영어"})
print(type(result))
print(result)

result = prompt.format(country="프랑스", language="영어")
print(type(result))
print(result)
