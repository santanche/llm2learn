import requests

def call_local_llm(prompt, model="phi3", model_url="http://localhost:11434/v1/completions"):
    """Call local LM API (Ollama)"""
    response = requests.post(model_url, json={
        "model": model,
        "prompt": prompt,
        "max_tokens": 256,
        "temperature": 0.1
    })
    
    # Check for errors
    if response.status_code != 200:
        print(f"Error: {response.json()}")
        return None
        
    return response.json()['choices'][0]['text']

# Agent 1: Generator
def generate_sql(question, schema, model="phi3"):
    prompt = f"Schema: {schema}\n\nQuestion: {question}\n\nSQL:"
    return call_local_llm(prompt, model=model)

# Agent 2: Validator
def validate_sql(question, sql, model="phi3"):
    prompt = f"Does this SQL answer the question?\nQ: {question}\nSQL: {sql}\nAnswer YES or NO:"
    return call_local_llm(prompt, model=model)

# Simple orchestration
question = "Find top 5 products"
schema = "products(id, name, sales)"

print("Generating SQL with llama3.2:3b...")
sql = generate_sql(question, schema, model="llama3.2:3b")
print(f"Generated: {sql}\n")

print("Validating SQL with phi3...")
valid = validate_sql(question, sql, model="phi3")
print(f"Valid: {valid}")