These python scripts are my attempt to get a client going with some memory and functionality that made it useful for me.
I asked a few AI services to make it, CoPilot ended up getting this after a fiew hours of my tinkering and it works.

Right now I am:
  python3 lamrag.py lamrag.py lamchat2.py --db chat_history.db            (this feeds the two scripts into a chat history)
Then:
  python3 lamchat2.py --model "ollama:model" --model_a "gemma:model" --model_b "coder:model" --logfile chat1.log
From here you are placed in a chat, type and hit enter to talk with model(default) type @A: <prompt> hit enter to talk with model_a, same for b.
Everything said is logged in the log file and entered in the chat history database for all members of the group to see.
You can have them act as a developer team, writers, whatever you want. [default: outline some project] [@A: implement defaults project] [@B: Ensure the project is fully functioning, clear, concise and documented.]
 I do not have a clue what I am doing, so feel free to.. do all that stuff or some stuff or not. BYE >:)  oh and thanks to all the AI platforms letting me dink around with stuff.

 Everything else in this file and the RoadMap.txt are CoPilots outlines for the future of the application and its documentation.
 Im working on some variations. Share yours.

Creative Names
Main Project (Multi‑Model Streaming Chat Client): ConvoMosaic A vibrant think tank where multiple model "voices" combine to form a dynamic, shared conversation—a mosaic of ideas.

Secondary Tool (Knowledge Ingestion/RaG Tool): DataSprout A tool that reads text files and “plants” knowledge into a persistent SQLite database, letting your chatbots grow smarter with every seed of data.

README.md
You can use the following README for ConvoMosaic. It includes my "CoPilot, ThinkDeep." signature and a note granting permission to share the work on GitHub.

markdown
# ConvoMosaic: Multi-Model Streaming Chat Client

(I let a few sets of assistants run on this "personas" idea, ill upload it later)

Welcome to **ConvoMosaic** – a multi-model streaming chat client that brings together different AI model personas into a shared conversation space. With persistent logging via SQLite and support for real-time token streaming, ConvoMosaic is designed for creative synergy, making it easy to collaborate with multiple “developer bots” (or any models you choose).

## Features

- **Multi-Model Chatroom:**  
  - Query different models by prefixing messages (e.g., using `@A:` or `@B:`).  
  - If no prefix is specified, the default model is used.
- **Streaming Text Responses:**  
  - Real-time token streaming that lets you see responses as they are generated.
- **Persistent Logging & Shared Memory:**  
  - All conversation history is stored in an SQLite database.
  - Responses are annotated by model identifier.
- **Integration with DataSprout:**  
  - Use the companion tool, **DataSprout**, to ingest text files into a knowledge base for retrieval-augmented generation (RaG).

## Getting Started

### Prerequisites

- Python 3.8+
- [httpx](https://www.python-httpx.org/)
- [LangChain](https://github.com/hwchase17/langchain)
- [Pydantic](https://pydantic-docs.helpmanual.io/)

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/YourUsername/ConvoMosaic.git
   cd ConvoMosaic
Install Dependencies:

bash
pip install -r requirements.txt
(Alternatively, install them manually: pip install httpx langchain pydantic)

Running the Chat Client
Run the client with a default model and (optionally) additional models by specifying command-line arguments:

bash
python3 multi_model_streaming_chat.py --model "default-model" --model_a "model-A" --model_b "model-B" --base_url "http://localhost:11434" --completion_endpoint "/api/chat" --sqlite_db "chat_history.db" --logfile "conversation.log" --debug
Message Prefixing:

Start your message with @A: to query model A, @B: for model B.

If no prefix is provided, the default model is used.

Logging & Debugging:

All conversations are saved in chat_history.db and optionally logged to conversation.log.

Use the --debug flag to print full raw JSON tokens for troubleshooting.

Using DataSprout
To ingest text files into the knowledge base, use the companion tool DataSprout:

bash
python3 lam-rag.py new-data.txt additional-data.txt --db knowledge.db
This loads your data into a separate SQLite database (knowledge.db) so that your chat clients can later retrieve background documents as needed.

Contributing
Contributions are welcome! Feel free to fork, improve, and share this project.

CoPilot, ThinkDeep.

Sharing Permission: I hereby grant permission to share and distribute this work on GitHub for everyone’s use. Enjoy, adapt, and transform it as you see fit!

License
This project is licensed under the MIT License – see the LICENSE file for details.


---

This README outlines the project’s intent, features, installation instructions, and usage guidance. It also includes the signature “CoPilot, ThinkDeep.” along with permission to share the awesome work on GitHub.

Feel free to modify the names, details, and sections as you wish for your project!
