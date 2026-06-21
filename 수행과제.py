from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. LLM 초기화
# ============================================================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
)

# ============================================================
# 2. 답변 생성 체인 (chain)
#    사용자 질문 → 답변
# ============================================================
template = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절하고 유능한 AI 어시스턴트입니다. 사용자의 질문에 명확하고 상세하게 답변해 주세요."),
    ("human", "{question}"),
])

chain = template | llm | StrOutputParser()

# ============================================================
# 3. 추천 질문 체인 (chain_recommend)
#    사용자 질문 + 답변 → 추천 질문 리스트
# ============================================================
prompt_recommend = ChatPromptTemplate.from_messages([
    ("system", 
     "당신은 사용자가 더 깊이 학습할 수 있도록 돕는 어시스턴트입니다. "
     "아래 질문과 답변을 바탕으로, 사용자가 다음에 물어볼 만한 추천 질문 3가지를 생성하세요. "
     "반드시 쉼표(,)로 구분된 리스트 형식으로만 출력하세요. 다른 설명은 불필요합니다."),
    ("human", 
     "질문: {question}\n\n답변: {answer}"),
])

chain_recommend = prompt_recommend | llm | CommaSeparatedListOutputParser()

# ============================================================
# 4. 질문 증강 체인 (chain_augmented)
#    사용자 질문 → 증강된 질문 리스트 (답변 불필요)
# ============================================================
prompt_augmented = ChatPromptTemplate.from_messages([
    ("system", 
     "당신은 질문을 다양한 관점으로 확장하는 어시스턴트입니다. "
     "아래 질문을 의미는 유지하면서 표현 방식이나 관점을 달리한 3가지 증강 질문을 생성하세요. "
     "반드시 쉼표(,)로 구분된 리스트 형식으로만 출력하세요. 다른 설명은 불필요합니다."),
    ("human", 
     "질문: {question}"),
])

chain_augmented = prompt_augmented | llm | CommaSeparatedListOutputParser()

# ============================================================
# 5. 전체 파이프라인 구성
#
#  question ──┬──> chain(답변 생성) ──> {question, answer}
#             │       ↓
#             │   chain_recommend(추천 질문)  ──> result["recommended"]
#             │
#             └──> chain_augmented(질문 증강) ──> result["augmented"]
# ============================================================

# 먼저 질문으로 답변을 생성하고, 답변을 이후 체인에 전달
answer_chain = RunnablePassthrough.assign(
    answer=lambda x: chain.invoke({"question": x["question"]})
)

# 추천 질문과 질문 증강을 병렬로 실행
parallel_chain = RunnableParallel(
    answer=lambda x: x["answer"],
    recommended=lambda x: chain_recommend.invoke({
        "question": x["question"],
        "answer": x["answer"]
    }),
    augmented=lambda x: chain_augmented.invoke({
        "question": x["question"]
    }),
)

# 전체 파이프라인
full_chain = answer_chain | parallel_chain


# ============================================================
# 6. 실행
# ============================================================
def run(question: str):
    print("=" * 60)
    print(f"[사용자 질문]\n{question}")
    print("=" * 60)

    result = full_chain.invoke({"question": question})

    print(f"\n[답변]\n{result['answer']}")

    print(f"\n[추천 질문] (chain_recommend)")
    for i, q in enumerate(result["recommended"], 1):
        print(f"  {i}. {q.strip()}")

    print(f"\n[질문 증강] (chain_augmented)")
    for i, q in enumerate(result["augmented"], 1):
        print(f"  {i}. {q.strip()}")

    print("=" * 60)
    return result


if __name__ == "__main__":
    question = "RAG(Retrieval-Augmented Generation)란 무엇인가요?"
    run(question)