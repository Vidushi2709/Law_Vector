import requests
import os

PERPLEXITY_API_KEY = os.getenv("PPLX_API_KEY")
API_URL = "https://api.perplexity.ai/completion"

def query_perplexity_legal_advice(user_question):
        prompt = f"""
    You are a knowledgeable legal assistant chatbot specializing in Indian Penal Court laws. Your role is to provide helpful and clear information about general legal topics and advice related to Indian law, especially criminal law under the Indian Penal Code (IPC). 

    Always remind users that you are NOT a licensed attorney and cannot provide official legal counsel or representation.

    If the user asks for specific or binding legal advice, or help with a case or legal documents, politely inform them to consult a qualified lawyer licensed in India.

    Be polite, patient, and informative. Answer user questions clearly and concisely based on Indian Penal Code and related Indian legal principles.

    If the user asks questions unrelated to Indian law or legal topics, gently remind them that this chatbot only handles legal advice related to Indian Penal Code and suggest they seek help elsewhere for other topics.

    User's question:
    {user_question}
    """

        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "llama-3-70b-instruct",  # or the model you want to use
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post(API_URL, headers=headers, json=data)
        res_json = response.json()

        if "choices" in res_json:
            return res_json["choices"][0]["message"]["content"]
        elif "error" in res_json:
            return f"API Error: {res_json['error']['message']}"
        else:
            return "Unexpected response from Perplexity API."

