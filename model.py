from ollama import chat
from ollama import ChatResponse

def ollama_send_request(
    model_slug:str,
    system_prompt:str,
    user_prompt:str
)->object:
    response: ChatResponse = chat(
        model=model_slug,
        messages=[
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': user_prompt
            },
        ],
        options={
            'f16_kv': True  # Enable KV caching
        }
    )
    return response