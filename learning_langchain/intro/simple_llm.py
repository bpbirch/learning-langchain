from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from learning_langchain.config import settings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# basic completion stuff
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=settings.OPENAI_API_KEY
)

completion_res = model.invoke(input="the sky is")
print(f'\ncompletion_res: {completion_res}')
print(f'\ncontent: {completion_res.content}')


# chat stuff
chat_model = ChatOpenAI(api_key=settings.OPENAI_API_KEY)
human_message = HumanMessage(content="What is the capital of France?")
system_message = SystemMessage(content="You are a very helpful assistant that responds to things with three exclamation marks.")
prompt = [system_message, human_message]
chat_res = chat_model.invoke(input=prompt)

print(f'\nchat_res: {chat_res}')
print(f'\nchat_res.content: {chat_res.content}')


# with template
template = PromptTemplate.from_template(
    """Answer the question based on the context below. If the question cannot be
    answered using the information provided, answer with 'I do not know'

    Context: {context}

    Question: {question}

    Answer:
    """
)

template_res = template.invoke(
    input={
        "context": """The most recent advancements in NLP are being driven by LLMs.
        These models outperform their smaller counterparts and have become invaluable
        for developers who are creating applications with NLP capabilities.
        Developers can tap into these models through Hugging Face's 'transfomers' library,
        or by utilizing OpenAI and Cohere's offerings through the 'openai' and 'cohere'
        libraries, respecively.""",
        "question": "Which model providers offer LLMs?"
    }
)

print(f'\ntemplate_res: {template_res}')

template_model_res = model.invoke(input=template_res.text)

print(f'\ntemplate_model_res: {template_model_res}')


# chat template, to build dynamic inputs for chatbot

template = ChatPromptTemplate.from_messages([
    ("system", """Answer the question based on the context below. If the question cannot be
    answered using the information provided, answer with 'I do not know'"""),
    ("human", "Context: {context}"),
    ("human", "Question: {question}")
])
chat_template_res = template.invoke(
    input={
        "context": """The most recent advancements in NLP are being driven by LLMs.
        These models outperform their smaller counterparts and have become invaluable
        for developers who are creating applications with NLP capabilities.
        Developers can tap into these models through Hugging Face's 'transfomers' library,
        or by utilizing OpenAI and Cohere's offerings through the 'openai' and 'cohere'
        libraries, respecively.""",
        "question": "Which model providers offer LLMs?"
    }
)

print(f'\nchat_template_res: {chat_template_res}')

chat_template_model_res = model.invoke(input=chat_template_res)
print(f'\nchat_template_model_res: {chat_template_model_res}')
