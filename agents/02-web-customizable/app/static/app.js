const { useState } = React;

const NLtoSQLApp = () => {
  const [isExecuting, setIsExecuting] = useState(false);
  const [logs, setLogs] = useState('');
  
  const [schema, setSchema] = useState(`Tables:
- products (id, name, category, price)
- orders (id, customer_id, order_date, total)
- order_items (id, order_id, product_id, quantity, price)`);

  const [schemaPrompt, setSchemaPrompt] = useState(`Given this database schema:
{schema}

For the question: "{question}"

List only the relevant tables and columns needed. Be concise.

Relevant schema:`);

  const [queryPrompt, setQueryPrompt] = useState(`Convert this question to SQL using only these tables/columns:
{relevant_schema}

Question: {question}

SQL query (return only the query, no explanation):`);

  const [syntaxPrompt, setSyntaxPrompt] = useState(`Check if this SQL query is syntactically valid:
{sql_query}

Answer with "VALID" or "INVALID" and explain any issues.

Answer:`);

  const [semanticPrompt, setSemanticPrompt] = useState(`Does this SQL query correctly answer the question?

Question: {question}
SQL: {sql_query}

Answer with "YES" or "NO" and briefly explain why.

Answer:`);

  const [question, setQuestion] = useState("What were the top 5 products by revenue in 2024?");

  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    const prefix = type === 'error' ? '❌' : type === 'success' ? '✅' : type === 'agent' ? '🤖' : 'ℹ️';
    setLogs(prev => `${prev}[${timestamp}] ${prefix} ${message}\n`);
  };

  const executeAgentPipeline = async () => {
    setIsExecuting(true);
    setLogs('');
    
    addLog('Starting NL to SQL conversion...', 'info');
    addLog(`Question: "${question}"`, 'info');
    
    try {
      const response = await fetch('/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          schema,
          schema_prompt: schemaPrompt,
          query_prompt: queryPrompt,
          syntax_prompt: syntaxPrompt,
          semantic_prompt: semanticPrompt,
          question
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'agent_start') {
                addLog(`Agent: ${data.agent}`, 'agent');
              } else if (data.type === 'agent_input') {
                addLog(`Input: ${data.content}`, 'info');
              } else if (data.type === 'agent_output') {
                addLog(`Output: ${data.content}`, 'info');
              } else if (data.type === 'validation') {
                addLog(`Validation: ${data.content}`, data.status === 'pass' ? 'success' : 'error');
              } else if (data.type === 'iteration') {
                addLog(`--- Iteration ${data.iteration} ---`, 'info');
              } else if (data.type === 'final_result') {
                addLog(`\n=== FINAL SQL QUERY ===\n${data.sql}`, 'success');
              } else if (data.type === 'error') {
                addLog(`Error: ${data.message}`, 'error');
              }
            } catch (e) {
              // Skip malformed JSON
            }
          }
        }
      }
      
    } catch (error) {
      addLog(`Error: ${error.message}`, 'error');
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-lg shadow-xl p-6 mb-6">
          <div className="flex items-center gap-3 mb-6">
            <svg className="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
            </svg>
            <h1 className="text-3xl font-bold text-gray-800">Natural Language to SQL</h1>
            <svg className="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <p className="text-gray-600 mb-4">Multi-Agent System with Small Language Models</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column - Configuration */}
          <div className="space-y-4">
            {/* Database Schema */}
            <div className="bg-white rounded-lg shadow p-4">
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                📊 Database Schema
              </label>
              <textarea
                value={schema}
                onChange={(e) => setSchema(e.target.value)}
                className="w-full h-32 p-3 border border-gray-300 rounded-lg font-mono text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                placeholder="Enter database schema..."
              />
            </div>

            {/* Schema Analyzer Prompt */}
            <div className="bg-white rounded-lg shadow p-4">
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                🔍 Schema Analyzer Prompt
              </label>
              <textarea
                value={schemaPrompt}
                onChange={(e) => setSchemaPrompt(e.target.value)}
                className="w-full h-32 p-3 border border-gray-300 rounded-lg font-mono text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>

            {/* Query Generator Prompt */}
            <div className="bg-white rounded-lg shadow p-4">
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                ⚙️ Query Generator Prompt
              </label>
              <textarea
                value={queryPrompt}
                onChange={(e) => setQueryPrompt(e.target.value)}
                className="w-full h-32 p-3 border border-gray-300 rounded-lg font-mono text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>

            {/* Syntax Validator Prompt */}
            <div className="bg-white rounded-lg shadow p-4">
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                ✓ Syntax Validator Prompt
              </label>
              <textarea
                value={syntaxPrompt}
                onChange={(e) => setSyntaxPrompt(e.target.value)}
                className="w-full h-32 p-3 border border-gray-300 rounded-lg font-mono text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>

            {/* Semantic Verifier Prompt */}
            <div className="bg-white rounded-lg shadow p-4">
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                🎯 Semantic Verifier Prompt
              </label>
              <textarea
                value={semanticPrompt}
                onChange={(e) => setSemanticPrompt(e.target.value)}
                className="w-full h-32 p-3 border border-gray-300 rounded-lg font-mono text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>

            {/* User Question */}
            <div className="bg-white rounded-lg shadow p-4">
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                💬 User Question
              </label>
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                className="w-full h-24 p-3 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                placeholder="Enter your question in natural language..."
              />
            </div>

            {/* Execute Button */}
            <button
              onClick={executeAgentPipeline}
              disabled={isExecuting}
              className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 text-white font-semibold py-3 px-6 rounded-lg shadow-lg transition-colors duration-200 flex items-center justify-center gap-2"
            >
              {isExecuting ? (
                <>
                  <svg className="w-5 h-5 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Executing...
                </>
              ) : (
                <>
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Execute Pipeline
                </>
              )}
            </button>
          </div>

          {/* Right Column - Logs */}
          <div className="bg-white rounded-lg shadow p-4">
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              📋 Execution Log
            </label>
            <textarea
              value={logs}
              readOnly
              className="w-full h-[calc(100vh-200px)] p-3 border border-gray-300 rounded-lg font-mono text-xs bg-gray-50 focus:outline-none overflow-auto"
              placeholder="Logs will appear here when you execute the pipeline..."
            />
          </div>
        </div>
      </div>
    </div>
  );
};

ReactDOM.render(<NLtoSQLApp />, document.getElementById('root'));