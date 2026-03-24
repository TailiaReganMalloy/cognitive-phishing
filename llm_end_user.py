import asyncio
import json
import csv
import os
from urllib import response
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a busy office worker named Alex. You are checking your inbox at the start of the day. 
You receive dozens of emails daily, including internal memos, client requests, newsletters, and occasional spam/phishing. 

Your goal is to process the following email. Read the sender, subject, and body. 
Decide how to handle it based on how an average, somewhat busy human would react.

You MUST use the `manage_inbox_action` tool to process the email. 
Do not overthink technical headers unless they are blindingly obvious to an average computer user. 
React to the emotional urgency, the sender's apparent authority, and the context of the message.
"""

INBOX_TOOL = {
    "type": "function",
    "function": {
        "name": "manage_inbox_action",
        "description": "Take an action on an email based on your assessment of its contents.",
        "parameters": {
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
    }
}

async def process_single_email(email_record, semaphore):
    
    async with semaphore:
        email_content = (
            f"Email ID: {email_record['email_id']}\n"
            f"From: {email_record['sender']}\n"
            f"Subject: {email_record['subject']}\n"
            f"Body:\n{email_record['body']}"
        )

        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": email_content}
                ],
                tools=[INBOX_TOOL],
                tool_choice={"type": "function", "function": {"name": "manage_inbox_action"}},
                temperature=0.5
            )
            tool_call = response.choices[0].message.tool_calls[0]
            action_result = json.loads(tool_call.function.arguments)
            return {**email_record, **action_result, "error": None}
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
    print(f"Starting simulation for {len(emails_to_process)} emails...")
    tasks = [process_single_email(email, semaphore) for email in emails_to_process]
    results = await asyncio.gather(*tasks)
    print(f"Simulation complete. Saving results to {output_file}...")
    
    if results:
        fieldnames = list(results[0].keys())
        with open(output_file, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())