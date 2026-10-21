
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama


def main():
    # Testing OpenAI
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # response = llm.invoke("Say 'set up complete' in one word")
    # print(response.content)

    # Testing Anthropic
    # llm_anthropic = ChatAnthropic(
    #     model="claude-sonnet-4-5-20250929",
    #     temperature=0
    # )
    # response_anthropic = llm_anthropic.invoke(
    #     "Say 'setup complete' in one word"
    # )
    # print(response_anthropic.content)

    # Testing locally running Ollama model
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    response_ollama = llm.invoke(
        "Say 'set up complete' in one word"
    )

    print(response_ollama)
    
    print("Setup Completed")


if __name__ == "__main__":
    main()
