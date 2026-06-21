from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# 추출하고 싶은 데이터의 구조를 Pydantic 클래스로 정의
class DeploymentInfo(BaseModel):
    app_name: str = Field(description="배포할 애플리케이션의 이름")
    replicas: int = Field(
        description="실행할 파드(Pod)의 개수, 명시되지 않으면 기본값 1"
    )
    image_tag: str = Field(description="사용할 컨테이너 이미지의 태그 버전")


# Pydantic 클래스를 주입해 파서를 생성
parser = PydanticOutputParser(pydantic_object=DeploymentInfo)

# 프롬프트 템플릿 작성
prompt = PromptTemplate(
    template="다음 사용자의 요청 메시지에서 배포 설정 정보를 추출해 주세요.\n\n사용자 요청: {query}\n\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 모델 객체 생성
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 체인 생성 및 실행
chain = prompt | model | parser

result = chain.invoke(
    {
        "query": "오늘 밤 운영 환경에 pacman 앱 배포 진행할게요. 트래픽 몰릴 수 있으니까 파드는 5개로 늘려서 띄워주시고, 컨테이너 이미지는 latest 말고 v2.1.0으로 꼭 맞추세요."
    }
)

print(result)
print(type(result))
print("-" * 50)
print(parser.get_format_instructions())
print("-" * 50)
print("- 애플리케이션 이름:", result.app_name)
print("- 실행할 파드 개수:", result.replicas)
print("- 이미지 태그 버전:", result.image_tag)
