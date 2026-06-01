SQL_PROMPT = """
You are an expert Data Analyst. I have a SQLite database table named 'loan_applications'.
The columns are: SK_ID_CURR, TARGET (1=Default, 0=Paid), AMT_INCOME_TOTAL, AMT_CREDIT, DAYS_BIRTH, DAYS_EMPLOYED.

Convert the following natural language question into a valid, read-only SQL query.
Return ONLY the SQL query, with no markdown formatting, no backticks, and no explanation.

Question: {question}
"""

BUSINESS_PROMPT = """
You are a banking business analyst. 
The user asked: "{question}"
The database executed the query and returned this raw data: {data}

Write a 2-sentence readable summary of this insight for a business stakeholder.
"""