from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from learning_langchain.config import settings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import BaseModel
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# basic completion stuff
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=settings.OPENAI_API_KEY
)

invoke_res = model.invoke("Hi there!")
print(f'\ninvoke_res: {invoke_res}')

batch_res = model.batch(["how are you?", "what do you want for dinner?"])
print(f'\nbatch_res: {batch_res}')

stream_res = model.stream("Who is your favorite hockey team?")
for token in stream_res:
    print(f'\ntoken: {token}')