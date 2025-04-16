from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from learning_langchain.config import settings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import chain

# Imperative Composition basically just means composing langchain calls into functions and classes
# imperative composition means telling the systme HOW to get what you want, step by step

# basic completion stuff
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=settings.OPENAI_API_KEY
)

template = ChatPromptTemplate.from_messages([
        ("system", "you are a helpful assistant"),
        ("human", "{question}")
    ]
)

@chain
def invoke_chatbot(values):
    prompt = template.invoke(values)
    return model.invoke(prompt)


invoke_res = invoke_chatbot.invoke({"question": "which model providers offer LLMs?"})
print(f'\ninvoke_res: {invoke_res}')


@chain
def stream_chatbot(values):
    prompt = template.invoke(values)
    for token in model.stream(prompt):
        yield token


stream_res = stream_chatbot.stream({"question": "which US state has the highest unemployment rate?"})
for token in stream_res:
    print(f'\ntoken: {token}')