from app.graph import create_chat_graph
from langgraph.checkpoint.mongodb import MongoDBSaver

MONGODB_URI = "mongodb://localhost:27017"

config = {"configurable": {"thread_id": "1"}} # Every Graph invoktion is a Thread. Thread id will be different for different users so that no two or more chats collide with each other

def init():
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph_with_mongo = create_chat_graph(checkpointer=checkpointer)
        while True:
            user_input = input("> ")
            for event in graph_with_mongo.stream({"messages":[user_input]}, config, stream_mode="values"):
                if "messages" in event:
                    event["messages"][-1].pretty_print()

init()