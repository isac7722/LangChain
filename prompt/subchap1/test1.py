from langchain.prompts import PromptTemplate

## from_template 메소드를 이용하여 PromptTempalate 객체 생성하는 방법

# template 정의
template = "{country}의 수도는 어디인가요?"

# from_template 메소드를 이용하여 PromptTemplate 객체 생성
prompt_template = PromptTemplate.from_template(template)

# prompt 생성
prompt = prompt_template.format(country="대한민국")

print(prompt)


