from ollama import chat, generate, ChatResponse


def ollama_preload_model(model_slug: str):
    generate(model=model_slug, prompt='Hi')


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