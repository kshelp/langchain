# from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
# from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# 모델 객체 생성
# model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.7)
# model_id = "timHan/llama3.2korean3B4QKM"
# model = OllamaLLM(
#     model = model_id,  
#     #num_predict = 100,
#     top_p = 0.9,
#     temperature = 0.7    
# )

# 메시지 생성
message = HumanMessage(content="파이썬의 장점 3가지만 말해줘.")

# 모델 실행
response = model.invoke([message])

print(response)
print(response.content)
