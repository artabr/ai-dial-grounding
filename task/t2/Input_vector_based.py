import asyncio
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `vector_based_grounding.png` to see the flow of app

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

INSTRUCTIONS:
1. Use ONLY the information from the retrieved context to answer the question
2. If the context contains relevant information, provide a clear and comprehensive answer
3. If the context doesn't contain enough information to answer the question, clearly state that
4. Do not make up information that is not present in the context
5. Present the information in a clear and organized manner"""

USER_PROMPT = """## RETRIEVED CONTEXT:
{context}

## USER QUESTION:
{query}"""


def format_user_document(user: dict[str, Any]) -> str:
    # Prepare context from users JSONs in the same way as in `no_grounding.py` `join_context` method
    result = ["User:"]
    for key, value in user.items():
        result.append(f"  {key}: {value}")
    return "\n".join(result)


class UserRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = None

    async def __aenter__(self):
        print("🔎 Loading all users...")
        # 1. Get all users (use UserClient)
        user_client = UserClient()
        all_users = user_client.get_all_users()
        
        # 2. Prepare array of Documents where page_content is `format_user_document(user)`
        documents = [Document(page_content=format_user_document(user)) for user in all_users]
        
        # 3. call `_create_vectorstore_with_batching` and setup it as obj var `vectorstore`
        self.vectorstore = await self._create_vectorstore_with_batching(documents)
        print("✅ Vectorstore is ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _create_vectorstore_with_batching(self, documents: list[Document], batch_size: int = 100):
        # 1. Split all `documents` on batches (100 documents in 1 batch)
        document_batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        print(f"Split {len(documents)} documents into {len(document_batches)} batches")
        
        # 2. Iterate through document batches and create array with tasks
        tasks = []
        for batch in document_batches:
            task = FAISS.afrom_documents(batch, self.embeddings)
            tasks.append(task)
        
        # 3. Gather tasks with asyncio
        print(f"Creating {len(tasks)} FAISS vectorstores...")
        vectorstores = await asyncio.gather(*tasks)
        
        # 4. Create `final_vectorstore` via merge of all vector stores
        final_vectorstore = vectorstores[0]
        for vs in vectorstores[1:]:
            final_vectorstore.merge_from(vs)
        
        # 5. Return `final_vectorstore`
        return final_vectorstore

    async def retrieve_context(self, query: str, k: int = 10, score: float = 0.1) -> str:
        # 1. Make similarity search
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k, score_threshold=score)
        
        # 2. Create `context_parts` empty array
        context_parts = []
        
        # 3. Iterate through retrieved relevant docs
        print(f"\n🔍 Retrieved {len(results)} relevant documents:")
        for doc, relevance_score in results:
            context_parts.append(doc.page_content)
            print(f"  Score: {relevance_score:.4f} | Content preview: {doc.page_content[:100]}...")
        
        # 4. Return joined context from `context_parts`
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        # Make augmentation for USER_PROMPT via `format` method
        return USER_PROMPT.format(context=context, query=query)

    def generate_answer(self, augmented_prompt: str) -> str:
        # 1. Create messages array
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt)
        ]
        
        # 2. Generate response
        response = self.llm_client.invoke(messages)
        
        # 3. Return response content
        return response.content


async def main():
    # 1. Create AzureOpenAIEmbeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="",
        model="text-embedding-3-small-1",
        dimensions=384
    )
    
    # 2. Create AzureChatOpenAI
    llm_client = AzureChatOpenAI(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="",
        model="gpt-4o"
    )

    async with UserRAG(embeddings, llm_client) as rag:
        print("\nQuery samples:")
        print(" - I need user emails that filled with hiking and psychology")
        print(" - Who is John?")
        while True:
            user_question = input("\n> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break
            
            # 1. Retrieve context
            context = await rag.retrieve_context(user_question)
            
            # 2. Make augmentation
            augmented_prompt = rag.augment_prompt(user_question, context)
            
            # 3. Generate answer and print it
            print("\n💭 Generating answer...")
            answer = rag.generate_answer(augmented_prompt)
            print("\n--- ANSWER ---")
            print(answer)
            print("---------------")


asyncio.run(main())

# The problems with Vector based Grounding approach are:
#   - In current solution we fetched all users once, prepared Vector store (Embed takes money) but we didn't play
#     around the point that new users added and deleted every 5 minutes. (Actually, it can be fixed, we can create once
#     Vector store and with new request we will fetch all the users, compare new and deleted with version in Vector
#     store and delete the data about deleted users and add new users).
#   - Limit with top_k (we can set up to 100, but what if the real number of similarity search 100+?)
#   - With some requests works not so perfectly. (Here we can play and add extra chain with LLM that will refactor the
#     user question in a way that will help for Vector search, but it is also not okay in the point that we have
#     changed original user question).
#   - Need to play with balance between top_k and score_threshold
# Benefits are:
#   - Similarity search by context
#   - Any input can be used for search
#   - Costs reduce