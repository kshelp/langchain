from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()


@tool
def calculator1(expression):
    """
    간단한 수학 표현식을 계산합니다.
    Args:
        expression (str): 계산할 수학 표현식 (예: "2 + 3 * 4")
    Returns:
        str: 계산 결과
    """
    return str(eval(expression))


print("\ncalculator1")
print("-" * 50)
print(calculator1.name)
print(calculator1.description)
print(calculator1.invoke({"expression": "10 + 20 * 2"}))


@tool("math_calculator")
def calculator2(expression):
    """
    간단한 수학 표현식을 계산합니다.
    Args:
        expression (str): 계산할 수학 표현식 (예: "2 + 3 * 4")
    Returns:
        str: 계산 결과
    """
    return str(eval(expression))


print("\ncalculator2")
print("-" * 50)
print(calculator2.name)
print(calculator2.description)
print(calculator2.invoke({"expression": "10 + 20 * 2"}))


@tool("math_calculator", description="표현식(문자열)을 계산해 반환")
def calculator3(expression):
    """
    간단한 수학 표현식을 계산합니다.
    Args:
        expression (str): 계산할 수학 표현식 (예: "2 + 3 * 4")
    Returns:
        str: 계산 결과
    """
    return str(eval(expression))


print("\ncalculator3")
print("-" * 50)
print(calculator3.name)
print(calculator3.description)
print(calculator3.invoke({"expression": "10 + 20 * 2"}))

