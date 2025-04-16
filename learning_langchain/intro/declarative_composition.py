from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from learning_langchain.config import settings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import chain

# Declarative Composition means telling the system WHAT you want
# D

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

# this uses LCEL, which is a declarative language.
# so rather than imperatively say how to complete each step, we allow the framework to handle it,
# and just tell the framework the pieces that should fit together, but not how each of them should work at each step
chatbot = template | model

chatbot_output = chatbot.invoke({"question": "what is the capital of Vietnam?"})
print(f'\nchatbot_output: {chatbot_output}')

# one big difference here is that we don't have to add a new function for streaming,
# like we did with imperative composition
stream_output = chatbot.stream({"question": "what is the capital of Germany?"})
for res in stream_output:
    print(f'res: {res}')
