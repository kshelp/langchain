from langchain_core.prompts import PromptTemplate

# 템플릿 생성 시 format 변수를 미리 설정
prompt = PromptTemplate(
    template="{subject}에 대해 {format} 형식으로 요약해 주세요.",
    input_variables=["subject"], 					# 나중에 받을 변수만 명시
    partial_variables={"format": "글머리 기호(Bullet points)"} 	# 미리 채워둘 변수
)

# 실행할 때는 subject 변수만 전달
result = prompt.format(subject="파이썬의 주요 특징")
print(result)
