from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

# 함수 정의
def length_function(word):
    return len(word)


# 기본 프롬프트, 모델, 출력 파서 정의
prompt = PromptTemplate.from_template("{first_word_length} + {second_word_length}는 무엇인가요?")
model = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

# 체인 구성 및 실행
chain = (
    {
        "first_word_length": RunnableLambda(lambda x: length_function(x["first_word"])),
        "second_word_length": RunnableLambda(lambda x: length_function(x["second_word"])),
    }
    | prompt
    | model
    | output_parser
)
result = chain.invoke({"first_word": "Hello", "second_word": "RunnableLambda"})
print(result)
