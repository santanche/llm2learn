from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import sqlparse

# Initialize small local LM
llm = Ollama(
    model="phi3",  # Change to your installed model
    temperature=0.1
)

# Agent 1: Schema Analyzer
schema_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template="""Given this database schema:
{schema}

For the question: "{question}"

List only the relevant tables and columns needed. Be concise.

Relevant schema:"""
)

schema_chain = LLMChain(llm=llm, prompt=schema_prompt)

# Agent 2: Query Generator
sql_prompt = PromptTemplate(
    input_variables=["question", "relevant_schema"],
    template="""Convert this question to SQL using only these tables/columns:
{relevant_schema}

Question: {question}

SQL query (return only the query, no explanation):"""
)

sql_chain = LLMChain(llm=llm, prompt=sql_prompt)

# Agent 3: Syntax Validator
def validate_syntax(sql_query):
    try:
        parsed = sqlparse.parse(sql_query)
        if not parsed:
            return False, "Empty or invalid SQL"
        return True, "Syntax valid"
    except Exception as e:
        return False, f"Syntax error: {str(e)}"

# Agent 4: Semantic Verifier
verify_prompt = PromptTemplate(
    input_variables=["question", "sql_query"],
    template="""Does this SQL query correctly answer the question?

Question: {question}
SQL: {sql_query}

Answer with "YES" or "NO" and briefly explain why.

Answer:"""
)

verify_chain = LLMChain(llm=llm, prompt=verify_prompt)

# Orchestration
def nl_to_sql(question, full_schema, max_iterations=3):
    print(f"Question: {question}\n")
    
    # Step 1: Analyze schema - UPDATED to use invoke()
    relevant_schema_result = schema_chain.invoke({
        "schema": full_schema,
        "question": question
    })
    
    # Extract text from result
    relevant_schema = relevant_schema_result.get('text', relevant_schema_result) if isinstance(relevant_schema_result, dict) else relevant_schema_result
    print(f"Relevant schema:\n{relevant_schema}\n")
    
    for iteration in range(max_iterations):
        print(f"--- Iteration {iteration + 1} ---")
        
        # Step 2: Generate SQL - UPDATED to use invoke()
        sql_result = sql_chain.invoke({
            "question": question,
            "relevant_schema": relevant_schema
        })
        
        # Extract text from result
        sql_query = sql_result.get('text', sql_result) if isinstance(sql_result, dict) else sql_result
        sql_query = sql_query.strip()
        print(f"Generated SQL:\n{sql_query}\n")
        
        # Step 3: Validate syntax
        is_valid, syntax_msg = validate_syntax(sql_query)
        print(f"Syntax validation: {syntax_msg}")
        
        if not is_valid:
            print("Retrying with syntax error feedback...\n")
            continue
        
        # Step 4: Verify semantics - UPDATED to use invoke()
        verification_result = verify_chain.invoke({
            "question": question,
            "sql_query": sql_query
        })
        
        # Extract text from result
        verification = verification_result.get('text', verification_result) if isinstance(verification_result, dict) else verification_result
        print(f"Semantic verification:\n{verification}\n")
        
        if "YES" in verification.upper():
            return sql_query
        else:
            print("Semantic issues found. Retrying...\n")
    
    return sql_query  # Return best attempt

# Example usage
schema = """
Tables:
- products (id, name, category, price)
- orders (id, customer_id, order_date, total)
- order_items (id, order_id, product_id, quantity, price)
"""

result = nl_to_sql(
    "What were the top 5 products by revenue in 2024?",
    schema
)

print(f"\n=== Final SQL ===\n{result}")
