import chainlit as cl
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer, TextIteratorStreamer
import torch
from huggingface_hub import login

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.base import Runnable
import transformers

template ="""
user : {question} \n bot :\n
"""

@cl.on_chat_start
async def on_chat_start():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cl.user_session.set("device", device)

    model_id = "Qwen/Qwen1.5-0.5B" #stabilityai/stablelm-2-1_6b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device_map="auto",
        max_length=1000,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )

    llm = HuggingFacePipeline(
        pipeline=pipeline,
        model_kwargs={"temperature": 0.8}
    )

    prompt = PromptTemplate(template=template, input_variables=["GPT4 Correct User"])
    runnable = prompt | llm | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

