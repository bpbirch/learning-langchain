from langchain_openai import ChatOpenAI
from learning_langchain.config import settings
from pydantic import BaseModel
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# basic completion stuff
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=settings.OPENAI_API_KEY
)


class AnswerWithJustification(BaseModel):
    """An answer to the user's question, along with a justification"""
    answer: str
    """The answer to the user's question"""
    justification: str
    """The justification for the answer"""


structured = model.with_structured_output(AnswerWithJustification)
res = structured.invoke("What weights more? a pound of bricks, or a pound of feathers?")
print(f'\nres: {res}')
print(f'\nres.answer: {res.answer}')
print(f'\nres.justification: {res.justification}')

# output parsing
parser = CommaSeparatedListOutputParser()
res_list = parser.invoke("banana, apple, potato")
print(f'\nres_list: {res_list}')