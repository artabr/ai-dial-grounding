import asyncio
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `flow_diagram.png` to see the flow of app

BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY: 
{query}"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens
        }

# 1. Create AzureChatOpenAI client
llm = AzureChatOpenAI(
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    api_version="",
    model="gpt-4o"
)

# 2. Create TokenTracker
token_tracker = TokenTracker()

def join_context(context: list[dict[str, Any]]) -> str:
    # Collect user data in a readable string format
    result = []
    for user in context:
        result.append("User:")
        for key, value in user.items():
            result.append(f"  {key}: {value}")
        result.append("")  # Empty line between users
    return "\n".join(result)


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")
    # 1. Create messages array with system prompt and user message
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    # 2. Generate response
    response = await llm.ainvoke(messages)
    
    # 3. Get usage from response metadata
    token_usage = response.response_metadata.get('token_usage', {})
    total_tokens = token_usage.get('total_tokens', 0)
    
    # 4. Add tokens to token_tracker
    token_tracker.add_tokens(total_tokens)
    
    # 5. Print response content and total_tokens
    print(f"Response tokens: {total_tokens}")
    
    # 6. Return response content
    return response.content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()
    if user_question:
        print("\n--- Searching user database ---")

        # 1. Get all users (use UserClient)
        user_client = UserClient()
        all_users = user_client.get_all_users()
        
        # 2. Split all users on batches (100 users in 1 batch)
        batch_size = 100
        user_batches = [all_users[i:i + batch_size] for i in range(0, len(all_users), batch_size)]
        print(f"Split {len(all_users)} users into {len(user_batches)} batches")
        
        # 3. Prepare tasks for async run of response generation for users batches
        tasks = []
        for batch in user_batches:
            context = join_context(batch)
            user_prompt = USER_PROMPT.format(context=context, query=user_question)
            task = generate_response(BATCH_SYSTEM_PROMPT, user_prompt)
            tasks.append(task)
        
        # 4. Run tasks asynchronously
        print(f"\nProcessing {len(tasks)} batches...")
        batch_results = await asyncio.gather(*tasks)
        
        # 5. Filter results on 'NO_MATCHES_FOUND'
        filtered_results = [result for result in batch_results if "NO_MATCHES_FOUND" not in result]
        
        # 6. If results after filtration are present
        if filtered_results:
            print(f"\nFound matches in {len(filtered_results)} batches. Generating final response...")
            
            # Combine filtered results
            combined_results = "\n\n".join(filtered_results)
            
            # Generate final response
            final_prompt = f"## SEARCH RESULTS:\n{combined_results}\n\n## ORIGINAL QUERY:\n{user_question}"
            final_response = await generate_response(FINAL_SYSTEM_PROMPT, final_prompt)
            
            print("\n--- FINAL ANSWER ---")
            print(final_response)
        else:
            print(f"\nNo users found matching: {user_question}")
        
        # 7. Print info about usage
        print("\n--- TOKEN USAGE SUMMARY ---")
        summary = token_tracker.get_summary()
        print(f"Total tokens used: {summary['total_tokens']}")
        print(f"Number of API calls: {summary['batch_count']}")
        print(f"Tokens per call: {summary['batch_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())
