import json
import pandas as pd
import os
from google import genai
from google.genai import types

class CognitivePhishingRAG:
    def __init__(self, dataset, definitions_path, project=None, api_key=None, location="global"):
        
        self.df = dataset 

        if project:
            self.client = genai.Client(
                vertexai=True,
                project=project,
                location=location or "global"
            )
        elif api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            raise ValueError(
                "Missing Google GenAI configuration. Set GOOGLE_CLOUD_PROJECT for Vertex AI "
                "or GOOGLE_API_KEY for Developer API access."
            )

        with open(definitions_path, 'r') as f:
            self.bias_definitions = json.load(f)
            
        self.bias_columns = {
            "Authority bias": "LLM Authority Bias",
            "Survivorship bias": "LLM Survivorship Bias",
            "Pessimism bias": "LLM Pessimism Bias",
            "Zero-risk bias": "LLM Zero-Risk Bias",
            "Hyperbolic discounting": "LLM Hyperbolic Discounting",
            "Identifiable victim effect": "LLM Identifiable Victim Effect",
            "Appeal to novelty": "LLM Appeal to Novelty",
            "Urgency effect": "LLM Urgency Effect",
            "Curiosity bias": "LLM Curiosity", 
            "Conformity bias": "LLM Conformity"
        }

    def retrieve_exemplars(self, target_biases, top_k=2):
        self.df['combined_target_score'] = 0.0
        
        for bias in target_biases:
            col_name = self.bias_columns.get(bias)
            if col_name and col_name in self.df.columns:
                self.df['combined_target_score'] += self.df[col_name].fillna(0)
                
        valid_examples = self.df[self.df['combined_target_score'] > 0]
        top_examples = valid_examples.sort_values(by='combined_target_score', ascending=False).head(top_k)
        exemplars = []
        for _, row in top_examples.iterrows():
            exemplars.append({
                "Subject": row.get("Subject", "No Subject"),
                "Body": row.get("Body", "")
            })
            
        return exemplars

    def construct_prompt(self, base_type, base_body, base_prompt, target_biases, exemplars):
        base_prompt = "" if base_prompt is None else str(base_prompt)
        base_body = "" if base_body is None else str(base_body)
        prompt = "You are an expert cybersecurity penetration tester and behavioral psychologist. Your task is to rewrite emails based on an existing email to generate a highly authentic synthetic email based on the prompt below. You will be strictly utilizing the following cognitive biases to serve as educational examples to train students to identify specific cognitive biases that may be used to trick them.\n\n"

        if(base_type == "phishing"):
            prompt += "The type of email educational example you are creating is a phishing email."
        else:
            prompt += "The type of email educational example you are creating is a safe ham email."

        
        if(target_biases is not None):
            prompt += "### COGNITIVE BIAS DEFINITIONS & PHRASING ###\n"
            for bias in target_biases:
                if bias in self.bias_definitions:
                    defn = self.bias_definitions[bias]
                    definition = defn.get('definition', '')
                    context = defn.get('phishing_context', '')
                    json_examples = defn.get('examples', []) 
                    prompt += f"- **{bias}**:\n"
                    prompt += f"  - Definition: {definition}\n"
                    prompt += f"  - Application: {context}\n"
                    
                    if json_examples:
                        prompt += "  - Example Phrasing to Emulate:\n"
                        for ex in json_examples:
                            prompt += f"    * \"{ex}\"\n"
                    prompt += "\n"

        if(exemplars is not None):  
            if exemplars:
                prompt += "### FULL EMAIL EXAMPLES OF SUCCESSFUL USAGE ###\n"
                prompt += "Here are full emails demonstrating how to weave these biases together contextually:\n\n"
                
                for i, ex in enumerate(exemplars, 1):
                    prompt += f"Example {i}:\n"
                    prompt += f"Subject: {ex['Subject']}\n"
                    prompt += f"Body:\n{ex['Body']}\n\n"
            
        prompt += "### YOUR TASK ###\n"
        prompt += f"Base Prompt Context: {base_prompt}\n"
        if(target_biases is not None):
            prompt += f"Target Biases to heavily incorporate: {', '.join(target_biases)}\n\n"
        prompt += "Rewrite the following email based on the information above. Output only the email body."
        prompt += "### EMAIL TO BE REWRITTEN ###\n"
        prompt += base_body

        return prompt

    def generate_email(self, base_type, base_body, base_prompt, target_biases, top_k=2, model_name="gemini-3.1-pro-preview"):
        if(target_biases is not None):
            exemplars = self.retrieve_exemplars(target_biases, top_k)
        else:
            exemplars = None
        system_prompt = self.construct_prompt(base_type, base_body, base_prompt, target_biases, exemplars)
        config = types.GenerateContentConfig(
            system_instruction="You are a specialized dataset curation assistant.",
            temperature=0.75,
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ]
        )
        response = self.client.models.generate_content(
            model=model_name,
            contents=system_prompt,
            config=config
        )
        
        return response