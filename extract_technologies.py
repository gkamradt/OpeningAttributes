from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
from dotenv import load_dotenv()
import os

load_dotenv()

def extract_technologies(text):
    chat = ChatOpenAI(model="gpt-3.5-turbo-1106", openai_api_key = os.getenv('OPENAI_API_KEY', 'yourapikey'))
    # chat = ChatOpenAI(model="gpt-4-1106-preview", os.getenv('OPENAI_API_KEY', 'yourapikey'))

    functions = [
        {
            "name": "extract_application_companies",
            "description": "Extract a list of the companies of applications listed on a job description",
            "parameters": {
                "type": "object",
                "properties": {
                    "applications": {
                        "type": "array",
                        "description": "List of applications listed on a job description",
                        "items": {
                            "type": "string"
                        },
                    }
                },
                "required": ["applications"],
            },
        }
    ]

    system_prompt = f"""
    You are bot that is very good at extracting other companies from job descriptions.
    The user will give you a job description and you should pull the names of other companies.
    The goal is to identify which applications a company is using based off their job descriptions.
    Example 1: "You must know Netsuite" > Netsuite
    Example 2: "We use Salesforce and Oracle" > Salesforce, Oracle

    They must be specific companies. Do not list industries or workflows or departments.
    """

    output = chat(
        messages=[
            SystemMessage(content=system_prompt),
            HumanMessage(content=text),
        ],
        functions=functions,
        function_call={"name": "extract_application_companies"}
    )


    tech_found = json.loads(
        output.additional_kwargs["function_call"]["arguments"]
    )['applications']

    return list(set(tech_found))