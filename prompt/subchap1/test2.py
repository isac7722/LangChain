from langchain.prompts import PromptTemplate

## PromptTemplate 객체를 활용하여 prompt_template 생성하는 방법

# template 정의
template = "{country}의 수도는 어디인가요?"

# PromptTemplate 객체를 활용하여 prompt_template 생성
prompt_template = PromptTemplate(
    template = template,
    input_variables=["country"],
)

# prompt 생성
prompt = prompt_template.format(country="중국")

print(prompt)