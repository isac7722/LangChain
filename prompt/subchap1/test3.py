from datetime import datetime
from langchain.prompts import PromptTemplate

# 월 일 형식으로 오늘 날짜를 반환하는 함수
def get_today():
    now = datetime.now()
    return now.strftime("%B %d")


prompt_template = PromptTemplate(
    template = "오늘의 날짜는 {today} 입니다. 오늘이 생일인 유명인 {n}명을 나열해 주세요.",
    input_variables=["n"],
    partial_variables={"today":get_today}, #partial_variables에 함수 전달
)


# prompt = prompt_template.format(n=10)

from langchain_core.runnables import RunnablePassthrough

runnable_template = { "n": RunnablePassthrough()} | prompt_template

print(runnable_template.invoke(5))



