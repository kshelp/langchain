from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()
parser = StrOutputParser()

chain = RunnableParallel(
    capital=ChatPromptTemplate.from_template("{country}의 수도는?") | llm | parser,
    area=ChatPromptTemplate.from_template("{country}의 면적은?") | llm | parser,
)
result = chain.invoke({"country": "대한민국"})
print(result)
