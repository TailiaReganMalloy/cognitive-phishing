import json
import pandas as pd
import os
from openai import OpenAI

from llm_email_gen.keys.openai import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
class CognitivePhishingRAG:
    def __init__(self, dataset_path, definitions_path):
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
                
        valid_examples = self.df[self.df['combined_target_score'] > 0]
        top_examples = valid_examples.sort_values(by='combined_target_score', ascending=False).head(top_k)
        exemplars = []
        for _, row in top_examples.iterrows():
            exemplars.append({
                "Subject": row.get("Subject", "No Subject"),
                "Body": row.get("Body", "")
            })
            
        return exemplars

    def construct_prompt(self, base_prompt, target_biases, exemplars):
        prompt = "You are an expert cybersecurity penetration tester and behavioral psychologist. Your task is to generate a highly authentic synthetic phishing email based on the prompt below, strictly utilizing the following cognitive biases to manipulate the reader.\n\n"
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
                
        if exemplars:
            prompt += "### FULL EMAIL EXAMPLES OF SUCCESSFUL USAGE ###\n"
            prompt += "Here are full emails demonstrating how to weave these biases together contextually:\n\n"
            
            for i, ex in enumerate(exemplars, 1):
                prompt += f"Example {i}:\n"
                prompt += f"Subject: {ex['Subject']}\n"
                prompt += f"Body:\n{ex['Body']}\n\n"
            
        prompt += "### YOUR TASK ###\n"
        prompt += f"Base Prompt Context: {base_prompt}\n"
        prompt += f"Target Biases to heavily incorporate: {', '.join(target_biases)}\n\n"
        prompt += "Generate the final phishing email now. Output ONLY the Subject and Body of the email."
        
        return prompt

    def generate_email(self, base_prompt, target_biases, top_k=2, model="gpt-4o"):
        exemplars = self.retrieve_exemplars(target_biases, top_k)
        system_prompt = self.construct_prompt(base_prompt, target_biases, exemplars)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a specialized dataset curation assistant."},
                {"role": "user", "content": system_prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        return response.choices[0].message.content