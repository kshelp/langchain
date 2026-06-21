from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# 1. 간단한 함수 정의
def add_five(x):
    return x + 5

def multiply_by_two(x):
    return x * 2

# 2. Runnable 객체로 변환
runnable1 = RunnableLambda(add_five)
runnable2 = RunnableLambda(multiply_by_two)
runnablex = RunnablePassthrough(lambda x: print(f"Pass throught ... {x}"))

# 3. 체인 생성 (5를 더한 뒤 2를 곱함)
chain = runnable1 | runnablex | runnable2

# 4. 실행
print(chain.invoke(10)) # (10 + 5) * 2 = 30
