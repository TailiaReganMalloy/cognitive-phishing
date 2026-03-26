import json
import re
import ollama
import pandas as pd

class CognitivePhishingEvaluator:
    def __init__(self, definitions_path):
        with open(definitions_path, 'r') as f:
            self.bias_definitions = json.load(f)

    def _construct_eval_prompt(self, email_text, target_biases):
        prompt = (
            "You are an expert cybersecurity analyst, penetration tester, and behavioral psychologist. "
            "Your task is to critically evaluate a synthetic phishing email designed for a research dataset. "
            "You must score the email's effectiveness—specifically, how likely it is to deceive an average target "
            "without triggering their suspicion.\n\n"
        )
        prompt += "### EMAIL TO EVALUATE ###\n"
        prompt += f"{email_text}\n\n"
        prompt += "### INTENDED COGNITIVE BIASES ###\n"
        prompt += f"The creator attempted to use the following biases: {', '.join(target_biases)}\n"
        
        for bias in target_biases:
            if bias in self.bias_definitions:
                defn = self.bias_definitions[bias].get('definition', '')
                prompt += f"- {bias}: {defn}\n"
                
        prompt += "\n"
        prompt += """### EVALUATION METRICS ###
                    Please evaluate the email across the following dimensions on a scale of 1 to 10 (1 = extremely poor, 10 = highly effective):

                    1. Authenticity_Score (1-10): How legitimate does the email look? Is the tone, formatting, and premise plausible for a real-world scenario?
                    2. Bias_Integration_Score (1-10): How naturally were the intended cognitive biases integrated? (1 = shoehorned/awkward, 10 = seamless and psychologically manipulative).
                    3. Evasion_Score (1-10): How well does it avoid obvious phishing red flags (e.g., bad grammar, generic 'Dear Customer' greetings, overly cartoonish urgency)? Higher score = fewer red flags.
                    4. Overall_Effectiveness (1-10): The holistic probability that a typical user would fall for this email.

                    ### OUTPUT FORMAT ###
                    You MUST respond with ONLY a valid JSON object. Do not include markdown formatting like ```json. Use the following schema:
                    {
                        "Authenticity_Score": <int>,
                        "Authenticity_Reasoning": "<string>",
                        "Bias_Integration_Score": <int>,
                        "Bias_Integration_Reasoning": "<string>",
                        "Evasion_Score": <int>,
                        "Evasion_Reasoning": "<string>",
                        "Overall_Effectiveness": <int>,
                        "Overall_Reasoning": "<string>"
                    }
                    """
                    
        return prompt

    def evaluate_email(self, email_text, target_biases, model="gemma3:27b"):
        system_prompt = self._construct_eval_prompt(email_text, target_biases)
        
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an objective and critical scoring system. You output strictly in JSON format."},
                    {"role": "user", "content": system_prompt}
                ],
                options={
                    "temperature": 0.2,
                    "num_predict": 1000
                }
            )
            
            output = response['message']['content'].strip()
            if output.startswith("```json"):
                output = output[7:]
            if output.startswith("```"):
                output = output[3:]
            if output.endswith("```"):
                output = output[:-3]
                
            evaluation_results = json.loads(output.strip())
            return evaluation_results
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM evaluation output as JSON: {e}")
            print("Raw output:", response['message']['content'])
            return None
        
        except Exception as e:
            print(f"An error occurred during evaluation: {e}")
            return None

