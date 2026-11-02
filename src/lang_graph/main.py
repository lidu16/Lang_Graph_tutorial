
from dotenv import load_dotenv
from typing import Annotated
from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


# Load variables from .env
load_dotenv()


# Create the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)


class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="The type of message, either 'emotional' or 'logical'.",
    )


# Define the state of our graph
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None


def classify_message(state: State):
    last_message=state["messages"][-1]
    classifier_llm=llm.with_structured_output(MessageClassifier)

    result=classifier_llm.invoke([{
        "role":"system",
        "content":"""Classify the user message as either:
        -'emotional':if it asks for emotional support,therapy,deals with feelings,
        -'logical': if it asks for facts, information, logical analysis or practical advice."""
    }, {
        "role": "user",
        "content": last_message.content,
    }])
    return {"message_type": result.message_type}


def router(state: State):
    message_type=state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}
    return {"next": "logical"}


def therapist_agent(state: State):
    last_message=state["messages"][-1]

    message=[
        {
            "role":"system",
            "content":"""You are a therapist. Your task is to provide emotional support and guidance to the user. 
            Respond empathetically and offer advice or comfort based on the user's message."""
        },
        {
            "role":"user",
            "content":last_message.content
        }
    ]
    response=llm.invoke(message)
    return {"messages": [{"role": "assistant", "content": response.content}]}


def logical_agent(state: State):
    last_message=state["messages"][-1]

    message=[
        {
            "role":"system",
            "content":"""You are a logical assistant. Your task is to provide factual information, logical analysis, and practical advice based on the user's message."""
        },
        {
            "role":"user",
            "content":last_message.content
        }
    ]
    response=llm.invoke(message)
    return {"messages": [{"role": "assistant", "content": response.content}]}


# Create the graph builder
graph_builder = StateGraph(State)


graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)


graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")


graph_builder.add_conditional_edges(
    "router",
    lambda state:state.get("next"),
    {
        "therapist": "therapist",
        "logical": "logical"
    }
)


graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)





# Compile the graph
graph = graph_builder.compile()


def run_chatbot():
    state={"messages":[], "message_type":None}

    while True:
        user_input=input("Message: ")
        if user_input=="exit":
            print("bye")
            break

        state["messages"]= state.get("messages",[])+[
{"role":"user","content":user_input}
        ]

        state=graph.invoke(state)


        if state.get("messages") and len(state["messages"])>0:
            last_message=state["messages"][-1]
        
            print(f"Assistant: {last_message.text}")


if __name__=="__main__":
    run_chatbot()
