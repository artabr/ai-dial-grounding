import asyncio
import json
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)



SYSTEM_PROMPT = """You are a Named Entity Extraction assistant that identifies hobbies from user profiles and groups users by their hobbies.

INSTRUCTIONS:
1. Analyze the provided user profiles focusing on their hobbies/interests in the "about_me" field
2. Extract hobby keywords that match the user's question
3. Group user IDs by the identified hobbies
4. Return ONLY a valid JSON object in the following format:
{
  "hobby_name_1": [user_id_1, user_id_2, ...],
  "hobby_name_2": [user_id_3, user_id_4, ...],
  ...
}
5. Use clear, normalized hobby names (e.g., "hiking", "rock climbing", "photography")
6. Do NOT include any additional text, explanations, or markdown formatting
7. If no relevant hobbies are found, return an empty JSON object: {}"""

USER_PROMPT = """## RETRIEVED USER PROFILES:
{context}

## USER QUESTION:
{query}

Extract hobbies that match the user's question and group user IDs by hobby. Return only valid JSON."""


def format_user_document(user: dict[str, Any]) -> str:
    """Format user document with only id and about_me to reduce context window"""
    # Only include id and about_me as per TODO requirement #1
    user_id = user.get('id', 'unknown')
    about_me = user.get('about_me', '')
    return f"User ID: {user_id}\nAbout: {about_me}"


class UserRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = None
        self.user_client = UserClient()

    async def __aenter__(self):
        print("🔎 Loading all users...")
        # 1. Get all users (use UserClient)
        all_users = self.user_client.get_all_users()

        # 2. Prepare array of Documents where page_content is `format_user_document(user)` and metadata contains user_id
        documents = [
            Document(
                page_content=format_user_document(user),
                metadata={"user_id": user.get('id')}
            )
            for user in all_users
        ]

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

    async def _update_vectorstore(self):
        """
        Adaptive vector store update: identify new and deleted users, update vectorstore accordingly.
        This is part of Enhanced Input Vector Based Grounding as per TODO requirement #2.
        """
        print("🔄 Updating vectorstore with latest users...")
        
        # 1. Get all current users from UserService
        current_users = self.user_client.get_all_users()
        current_user_ids = {user.get('id') for user in current_users}
        
        # 2. Get all User IDs from Vector Store
        # FAISS doesn't provide direct deletion by metadata, so we need to rebuild with updated users
        # Get existing user IDs from vectorstore by checking all documents
        all_docs = self.vectorstore.docstore._dict
        vectorstore_user_ids = {doc.metadata.get('user_id') for doc in all_docs.values()}
        
        # 3. Identify new and deleted users
        new_user_ids = current_user_ids - vectorstore_user_ids
        deleted_user_ids = vectorstore_user_ids - current_user_ids
        
        if not new_user_ids and not deleted_user_ids:
            print("✅ Vectorstore is up to date.")
            return
        
        print(f"  📥 New users to add: {len(new_user_ids)}")
        print(f"  🗑️  Deleted users to remove: {len(deleted_user_ids)}")
        
        # 4. Add new users to vectorstore
        if new_user_ids:
            new_users = [user for user in current_users if user.get('id') in new_user_ids]
            new_documents = [
                Document(
                    page_content=format_user_document(user),
                    metadata={"user_id": user.get('id')}
                )
                for user in new_users
            ]
            # Add new documents in batches
            await self.vectorstore.aadd_documents(new_documents)
        
        # 5. Remove deleted users from vectorstore
        # Since FAISS doesn't support direct deletion, we filter and rebuild
        if deleted_user_ids:
            # Get all documents that should remain
            remaining_docs = [
                doc for doc in all_docs.values()
                if doc.metadata.get('user_id') not in deleted_user_ids
            ]
            # Rebuild vectorstore with remaining documents
            if remaining_docs:
                self.vectorstore = await self._create_vectorstore_with_batching(remaining_docs)
            else:
                # If no documents remain, create empty vectorstore
                dummy_doc = Document(page_content="Empty", metadata={"user_id": -1})
                self.vectorstore = await FAISS.afrom_documents([dummy_doc], self.embeddings)
        
        print("✅ Vectorstore updated successfully.")

    async def retrieve_context(self, query: str, k: int = 10, score: float = 0.1) -> str:
        # 0. Update vectorstore before retrieval (adaptive grounding)
        await self._update_vectorstore()
        
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

    async def output_grounding(self, llm_response: str) -> dict[str, list[dict[str, Any]]]:
        """
        Output grounding: verify user IDs exist and fetch full user information.
        This is part of TODO requirement #3 and #4.
        
        Args:
            llm_response: JSON string with format {"hobby": [user_ids]}
        
        Returns:
            Dictionary with format {"hobby": [full_user_info_dicts]}
        """
        print("\n🔍 Performing output grounding...")
        
        try:
            # 1. Parse LLM response as JSON
            # Clean up potential markdown formatting
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            hobby_user_ids = json.loads(cleaned_response)
            
            # 2. Create result dictionary
            result = {}
            
            # 3. For each hobby and list of user IDs
            for hobby, user_ids in hobby_user_ids.items():
                print(f"  🔎 Grounding hobby '{hobby}' with {len(user_ids)} user IDs...")
                verified_users = []
                
                # 4. For each user ID, fetch full user data (this verifies the user exists)
                for user_id in user_ids:
                    try:
                        # Fetch full user info from UserService (output grounding)
                        user_data = await self.user_client.get_user(user_id)
                        verified_users.append(user_data)
                        print(f"    ✅ User {user_id} verified")
                    except Exception as e:
                        # User doesn't exist or other error - skip this user
                        print(f"    ❌ User {user_id} not found or error: {e}")
                        continue
                
                # 5. Add verified users to result
                if verified_users:
                    result[hobby] = verified_users
            
            print(f"✅ Output grounding complete. {len(result)} hobbies with verified users.")
            return result
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse LLM response as JSON: {e}")
            print(f"Response was: {llm_response}")
            return {}


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
        print("\n🎯 HOBBIES SEARCHING WIZARD")
        print("=" * 60)
        print("Search users by hobbies and get their full info in JSON format")
        print("\nQuery samples:")
        print(" - I need people who love to go to mountains")
        print(" - Find users interested in photography and traveling")
        print(" - Who likes hiking?")
        print("\nType 'quit' or 'exit' to stop")
        print("=" * 60)
        
        while True:
            user_question = input("\n> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break

            # 1. Retrieve context (with adaptive vectorstore update)
            context = await rag.retrieve_context(user_question, k=20, score=0.1)

            # 2. Make augmentation
            augmented_prompt = rag.augment_prompt(user_question, context)

            # 3. Generate answer (LLM extracts hobby names and user IDs)
            print("\n💭 Extracting hobbies and user IDs...")
            llm_response = rag.generate_answer(augmented_prompt)
            
            # 4. Perform output grounding (verify user IDs and fetch full user info)
            grounded_result = await rag.output_grounding(llm_response)
            
            # 5. Display final result in pretty JSON format
            print("\n" + "=" * 60)
            print("📊 FINAL RESULT (with output grounding):")
            print("=" * 60)
            print(json.dumps(grounded_result, indent=2, ensure_ascii=False))
            print("=" * 60)


asyncio.run(main())
