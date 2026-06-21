from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()
joke_prompt = PromptTemplate.from_template("{topic}을 주제로 농담 하나를 만들어 줘.")
poem_prompt = PromptTemplate.from_template("{topic}을 주제로 시(poem) 하나를 만들어 줘.")

joke_chain = joke_prompt | model | parser
poem_chain = poem_prompt | model | parser

chain = RunnableParallel(joke=joke_chain, peom=poem_chain)
result = chain.invoke({"topic": "아울렛"})
print(result)
