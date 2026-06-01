import google.generativeai as genai
from src.utils.helpers import load_env_vars
from src.talk_to_data.prompt_templates import SQL_PROMPT, BUSINESS_PROMPT
from src.talk_to_data.query_runner import execute_sql

# Initialize LLM API
try:
    api_key = load_env_vars()
    genai.configure(api_key=api_key)
    # Using the newer, faster 1.5 Flash model
    model = genai.GenerativeModel('gemini-1.5-flash') 
except Exception as e:
    model = None

def ask_data(question):
    if not model:
        return "API Key not configured properly."
        
    # 1. Generate SQL
    prompt = SQL_PROMPT.format(question=question)
    response = model.generate_content(prompt)
    sql_query = response.text.strip().replace("```sql", "").replace("```", "")
    
    # 2. Execute SQL
    data_result = execute_sql(sql_query)
    
    if isinstance(data_result, str) and "Error" in data_result:
        return f"Could not process query. {data_result}"
        
    # 3. Generate Business Summary
    summary_prompt = BUSINESS_PROMPT.format(question=question, data=data_result)
    final_response = model.generate_content(summary_prompt)
    
    return {
        "sql": sql_query,
        "raw_data": data_result[:5], # Show top 5
        "insight": final_response.text
    }