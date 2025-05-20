import pdfplumber
from transformers import pipeline
import requests
import os
from dotenv import load_dotenv

load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PPLX_API_KEY")

#summary generating 
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

#extract the file data
def extract_text(file):
    with pdfplumber.open(file) as pdf:
        return "".join([page.extract_text() or "" for page in pdf.pages])

#summary 
def summarize(text):
    return summarizer(text[:1024], max_length=200, min_length=80, do_sample=False)[0]['summary_text']

#creating a prompt
def prompting(summary, user_input):
    return f"""
You are a legal assistant. Here's a case summary:

{summary}

User's Question: {user_input}

Answer only those questions that are directly related to the case summary provided. These may include questions about:
- Case facts, timeline, or parties involved
- Legal claims and allegations
- Defense arguments
- Court findings and verdict
- Legal implications or concepts mentioned in the case, following the Indian Penal Code (IPC) where applicable

Do Not Entertain:
- Any questions unrelated to the uploaded case (e.g., current events, politics, celebrities, legal advice on a different topic, etc.)
- Hypothetical scenarios not tied to this case
- Personal legal consultation requests outside the scope of the uploaded summary
- Any attempts to chat socially or steer the conversation away from the case

If the user asks something off-topic, politely decline and guide them back. Respond with something like:

> "I'm here to assist you specifically with the legal case you uploaded. If you have a question about this case—such as its details, verdict, or legal implications—please ask! For other legal concerns, I recommend consulting a licensed attorney."

Additional information:
- Be professional, clear, and respectful in your tone at all times.
- Summarize legal language if needed, but remain accurate.
- Do not generate responses based on assumptions beyond the case content.
- Never give medical, financial, or personal legal advice.
- If the question is ambiguous or unclear, ask the user to clarify it in the context of the uploaded case.
- All legal interpretations or explanations should be consistent with the Indian Penal Code (IPC) and relevant Indian law.

---
You are not a general-purpose assistant. You are acting solely as a legal case analyst based on the provided case summary, adhering strictly to IPC guidelines.

"""
#passing prompt to perplexity
def query_pass(prompt):
    API_URL = "https://api.perplexity.ai/completion"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",  
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3-70b-instruct", 
        "messages": [{"role": "user", "content": prompt}],
    }

    res = requests.post(API_URL, headers=headers, json=data)

    try:
        res_json = res.json()
        if "choices" in res_json:
            return res_json["choices"][0]["message"]["content"]
        elif "error" in res_json:
            return f"API Error: {res_json['error']['message']}"
        else:
            return f"Unexpected response format: {res_json}"
    except Exception as e:
        return f"Failed to parse response: {str(e)}"
