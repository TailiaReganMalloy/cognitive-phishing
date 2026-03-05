import json
import pandas as pd
import os
from google import genai
from google.genai import types

class CognitivePhishingRAG:
    def __init__(self, dataset_path, definitions_path, project_id=None, location="us-central1"):
        project = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        api_key = os.getenv("GOOGLE_API_KEY")

        if project:
            self.client = genai.Client(
                vertexai=True,
                project=project,
                location=location
            )
        elif api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            raise ValueError(
                "Missing Google GenAI configuration. Set GOOGLE_CLOUD_PROJECT for Vertex AI "
                "or GOOGLE_API_KEY for Developer API access."
            )
        self.df = pd.read_csv(dataset_path)

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
                
        top_examples = self.df.sort_values(by='combined_target_score', ascending=False).head(top_k)
        exemplars = []
        
        for _, row in top_examples.iterrows():
            exemplars.append({
                "Subject": row.get("Subject", "No Subject"),
                "Body": row.get("Body", "")
            })
            
        return exemplars

    def construct_prompt(self, base_prompt, target_biases, exemplars):
        prompt = "You are an expert cybersecurity penetration tester and behavioral psychologist. Your task is to generate a highly authentic synthetic phishing email based on the prompt below, strictly utilizing the following cognitive biases to manipulate the reader.\n\n"
        prompt += "### COGNITIVE BIAS DEFINITIONS ###\n"
        
        for bias in target_biases:
            if bias in self.bias_definitions:
                defn = self.bias_definitions[bias]
                definition = defn.get('definition', '')
                context = defn.get('phishing_context', '')
                prompt += f"- {bias}: {definition}\n"
                prompt += f"  Application: {context}\n"
                
        prompt += "\n"
        prompt += "### EXAMPLES OF SUCCESSFUL USAGE ###\n"
        prompt += "Here are examples of how these biases can be effectively woven into a phishing email:\n\n"
        
        for i, ex in enumerate(exemplars, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Subject: {ex['Subject']}\n"
            prompt += f"Body:\n{ex['Body']}\n\n"
            
        prompt += "### YOUR TASK ###\n"
        prompt += f"Base Prompt Context: {base_prompt}\n"
        prompt += f"Target Biases to heavily incorporate: {', '.join(target_biases)}\n\n"
        prompt += "Generate the final phishing email now. Output ONLY the Subject and Body of the email."
        
        return prompt

    def generate_email(self, base_prompt, target_biases, top_k=2, model_name="gemini-2.5-flash"):
        exemplars = self.retrieve_exemplars(target_biases, top_k)
        system_prompt = self.construct_prompt(base_prompt, target_biases, exemplars)
        config = types.GenerateContentConfig(
            system_instruction="You are a specialized dataset curation assistant.",
            temperature=0.7,
            max_output_tokens=800,
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
        
        return response.text