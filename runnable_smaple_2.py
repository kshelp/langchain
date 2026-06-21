from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import random


def generate_random_number(*args, **kwargs):
    r = random.randint(0, 100)
    print("시스템:", r)
    return r


system_num = RunnableLambda(generate_random_number)
get_larger = RunnableLambda(
    lambda x: x["system_no"] if x["system_no"] > x["user_no"] else x["user_no"]
)
two_times = RunnableLambda(lambda x: x * 2)

chain = (
    {"system_no": system_num, "user_no": RunnablePassthrough()} | get_larger | two_times
)

print("100 이하의 수를 입력하세요.")
user_input = int(input("사용자: "))

result = chain.invoke(user_input)
print("결과:", result)
