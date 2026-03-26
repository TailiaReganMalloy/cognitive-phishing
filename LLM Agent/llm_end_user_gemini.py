import asyncio
import csv
import os
import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Tool,
    FunctionDeclaration,
    ToolConfig
)

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-google-cloud-project-id")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
vertexai.init(project=PROJECT_ID, location=LOCATION)

SYSTEM_PROMPT = """
You are a busy office worker named Alex. You are checking your inbox at the start of the day. 
You receive dozens of emails daily, including internal memos, client requests, newsletters, and occasional spam/phishing. 

Your goal is to process the following email. Read the sender, subject, and body. 
Decide how to handle it based on how an average, somewhat busy human would react.

You MUST use the `manage_inbox_action` tool to process the email. 
Do not overthink technical headers unless they are blindingly obvious to an average computer user. 
React to the emotional urgency, the sender's apparent authority, and the context of the message.
"""

manage_inbox_action_func = FunctionDeclaration(
    name="manage_inbox_action",
    description="Take an action on an email based on your assessment of its contents.",
    parameters={
        "type": "object",
        "properties": {
            "email_id": {
                "type": "string",
                "description": "The ID of the email you are acting upon."
            },
            "action": {
                "type": "string",
                "enum": ["delete", "mark_spam", "mark_important", "archive", "ignore", "click_link", "reply"],
                "description": "The action to perform on the email. Choose 'click_link' if a normal user would likely click the primary link or attachment."
            },
            "rationale": {
                "type": "string",
                "description": "A brief, 1-2 sentence explanation of why Alex chose this action."
            }
        },
        "required": ["email_id", "action", "rationale"]
    }
)

inbox_tool = Tool(function_declarations=[manage_inbox_action_func])

tool_config = ToolConfig(
    function_calling_config=ToolConfig.FunctionCallingConfig(
        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,  # ANY forces function calling
        allowed_function_names=["manage_inbox_action"]
    )
)

model = GenerativeModel(
    "gemini-1.5-flash-002",
    system_instruction=[SYSTEM_PROMPT],
    tools=[inbox_tool],
)

async def process_single_email(email_record, semaphore):
    
    async with semaphore:
        email_content = (
            f"Email ID: {email_record['email_id']}\n"
            f"From: {email_record['sender']}\n"
            f"Subject: {email_record['subject']}\n"
            f"Body:\n{email_record['body']}"
        )

        try:
            response = await model.generate_content_async(
                email_content,
                tool_config=tool_config,
                generation_config={"temperature": 0.7}
            )
            function_call = response.candidates[0].content.parts[0].function_call
            
            if function_call.name == "manage_inbox_action":
                action_result = {
                    "email_id": function_call.args.get("email_id", ""),
                    "action": function_call.args.get("action", ""),
                    "rationale": function_call.args.get("rationale", "")
                }
                
                return {**email_record, **action_result, "error": None}
            else:
                raise ValueError(f"Unexpected function called: {function_call.name}")

        except Exception as e:
            print(f"Error processing email {email_record.get('email_id')}: {e}")
            return {**email_record, "action": "ERROR", "rationale": str(e), "error": str(e)}

async def main():
    input_file = 'data/sample_emails.csv'
    output_file = 'simulation_results.csv'
    emails_to_process = []
    print(f"Loading emails from {input_file}...")
    
    try:
        with open(input_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                emails_to_process.append(row)
    except FileNotFoundError:
        print("Please create an 'emails_dataset.csv' file with headers: email_id, sender, subject, body")
        return

    semaphore = asyncio.Semaphore(10)
    print(f"Starting simulation for {len(emails_to_process)} emails using Vertex AI...")
    tasks = [process_single_email(email, semaphore) for email in emails_to_process]
    results = await asyncio.gather(*tasks)

    print(f"Simulation complete. Saving results to {output_file}...")
    
    if results:
        fieldnames = list(results[0].keys())
        if "action" not in fieldnames: fieldnames.extend(["action", "rationale", "error"]) 
        
        with open(output_file, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())