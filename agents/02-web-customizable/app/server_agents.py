from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import sqlparse
import json
import asyncio
from typing import AsyncGenerator
from pathlib import Path

app = FastAPI(title="NL to SQL Multi-Agent System")

# Enable CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Request model
class ExecutionRequest(BaseModel):
    schema: str
    schema_prompt: str
    query_prompt: str
    syntax_prompt: str
    semantic_prompt: str
    question: str
    model: str = "phi3"
    max_iterations: int = 3

# Initialize LLM
def get_llm(model_name: str):
    return Ollama(model=model_name, temperature=0.1)

# Syntax validator (deterministic)
def validate_syntax(sql_query: str):
    try:
        parsed = sqlparse.parse(sql_query)
        if not parsed:
            return False, "Empty or invalid SQL"
        return True, "Syntax valid"
    except Exception as e:
        return False, f"Syntax error: {str(e)}"

# Stream event helper
def create_event(event_type: str, **kwargs):
    data = {"type": event_type, **kwargs}
    return f"data: {json.dumps(data)}\n\n"

# Main agent pipeline
async def execute_pipeline(request: ExecutionRequest) -> AsyncGenerator[str, None]:
    try:
        # Initialize LLM
        llm = get_llm(request.model)
        
        yield create_event("agent_start", agent="Schema Analyzer")
        yield create_event("agent_input", content=f"Schema: {request.schema[:100]}... | Question: {request.question}")
        
        # Agent 1: Schema Analyzer
        schema_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template=request.schema_prompt
        )
        schema_chain = LLMChain(llm=llm, prompt=schema_prompt)
        
        relevant_schema_result = schema_chain.invoke({
            "schema": request.schema,
            "question": request.question
        })
        relevant_schema = relevant_schema_result.get('text', relevant_schema_result) if isinstance(relevant_schema_result, dict) else relevant_schema_result
        
        yield create_event("agent_output", content=relevant_schema.strip())
        
        # Iteration loop
        sql_query = None
        for iteration in range(request.max_iterations):
            yield create_event("iteration", iteration=iteration + 1)
            
            # Agent 2: Query Generator
            yield create_event("agent_start", agent="Query Generator")
            yield create_event("agent_input", content=f"Relevant schema: {relevant_schema[:100]}...")
            
            query_prompt = PromptTemplate(
                input_variables=["question", "relevant_schema"],
                template=request.query_prompt
            )
            sql_chain = LLMChain(llm=llm, prompt=query_prompt)
            
            sql_result = sql_chain.invoke({
                "question": request.question,
                "relevant_schema": relevant_schema
            })
            sql_query = sql_result.get('text', sql_result) if isinstance(sql_result, dict) else sql_result
            sql_query = sql_query.strip()
            
            yield create_event("agent_output", content=sql_query)
            
            # Agent 3: Syntax Validator
            yield create_event("agent_start", agent="Syntax Validator")
            is_valid, syntax_msg = validate_syntax(sql_query)
            
            if is_valid:
                yield create_event("validation", content=syntax_msg, status="pass")
            else:
                yield create_event("validation", content=syntax_msg, status="fail")
                continue
            
            # Agent 4: Semantic Verifier
            yield create_event("agent_start", agent="Semantic Verifier")
            yield create_event("agent_input", content=f"Checking if SQL answers: {request.question}")
            
            verify_prompt = PromptTemplate(
                input_variables=["question", "sql_query"],
                template=request.semantic_prompt
            )
            verify_chain = LLMChain(llm=llm, prompt=verify_prompt)
            
            verification_result = verify_chain.invoke({
                "question": request.question,
                "sql_query": sql_query
            })
            verification = verification_result.get('text', verification_result) if isinstance(verification_result, dict) else verification_result
            
            yield create_event("agent_output", content=verification.strip())
            
            if "YES" in verification.upper():
                yield create_event("validation", content="Query is semantically correct", status="pass")
                break
            else:
                yield create_event("validation", content="Query has semantic issues", status="fail")
        
        # Final result
        yield create_event("final_result", sql=sql_query if sql_query else "No valid SQL generated")
        
    except Exception as e:
        yield create_event("error", message=str(e))

@app.get("/")
async def root():
    """Serve the main web interface"""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Place index.html in the static/ directory"}

@app.post("/execute")
async def execute(request: ExecutionRequest):
    """Execute the multi-agent NL to SQL pipeline with streaming logs"""
    return StreamingResponse(
        execute_pipeline(request),
        media_type="text/event-stream"
    )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

@app.get("/models")
async def list_models():
    """List available Ollama models"""
    # This would require calling ollama CLI or API
    # For now, return common models
    return {
        "models": ["phi3", "llama3.2:3b", "gemma2:2b", "mistral"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
