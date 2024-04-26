# 토큰 정보로드를 위한 라이브러리

from dotenv import load_dotenv

# 토큰 정보 로드
load_dotenv()

from langchain_openai import ChatOpenAI

# 객체 생성
llm = ChatOpenAI(
    temperature=0,
    max_tokens=2048,
    model_name="gpt-3.5-turbo",
)

# 질의내용
question = "what is the capital city of Korea"

# 질의
print(f"[Answer]: {llm.predict(question)}")