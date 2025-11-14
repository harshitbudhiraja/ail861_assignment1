import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import evaluate
from tokenizer import prepare_data_pipeline
from model_architecture import DecoderOnlyTransformer
import os
import json
from tqdm import tqdm
from collections import Counter
import math

class ModelEvaluator:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.bleu_metric = evaluate.load("bleu")
    def calculate_perplexity(self, text: str) -> float:
        self.model.eval()
        token_ids = self.processor.text_to_ids(text, add_special_tokens=True)
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        with torch.no_grad():
            logits = self.model(input_tensor)
            targets = input_tensor[:, 1:] 
            logits = logits[:, :-1, :]   
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, 
                                  ignore_index=self.processor.special_tokens['<pad>'])
            perplexity = torch.exp(loss).item()
        return perplexity
    def calculate_bleu_score(self, predictions: List[str], references: List[str]) -> dict:
        filtered_pairs = [(pred, ref) for pred, ref in zip(predictions, references) if pred.strip() and ref.strip()]
        if not filtered_pairs:
            return {'bleu': 0.0, 'precisions': [0.0, 0.0, 0.0, 0.0], 'brevity_penalty': 0.0}
        preds, refs = zip(*filtered_pairs)
        pred_tokens = [pred.split() for pred in preds]
        ref_tokens = [[ref.split()] for ref in refs]
        bleu_scores = self.bleu_metric.compute(
            predictions=pred_tokens,
            references=ref_tokens
        )
        return bleu_scores
    def calculate_bleu_score_simple(self, predictions: List[str], references: List[str]) -> dict:
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        def calculate_bleu_1(pred_tokens, ref_tokens):
            pred_1grams = Counter(get_ngrams(pred_tokens, 1))
            ref_1grams = Counter(get_ngrams(ref_tokens, 1))
            overlap = sum(min(pred_1grams[gram], ref_1grams[gram]) for gram in pred_1grams)
            total = sum(pred_1grams.values())
            return overlap / total if total > 0 else 0.0
        filtered_pairs = [(pred, ref) for pred, ref in zip(predictions, references) if pred.strip() and ref.strip()]
        if not filtered_pairs:
            return {'bleu': 0.0, 'precisions': [0.0, 0.0, 0.0, 0.0], 'brevity_penalty': 0.0}
        preds, refs = zip(*filtered_pairs)
        bleu_1_scores = []
        for pred, ref in zip(preds, refs):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            bleu_1 = calculate_bleu_1(pred_tokens, ref_tokens)
            bleu_1_scores.append(bleu_1)
        avg_bleu_1 = np.mean(bleu_1_scores)
        return {
            'bleu': avg_bleu_1,
            'precisions': [avg_bleu_1, 0.0, 0.0, 0.0],
            'brevity_penalty': 1.0
        }
    def generate_continuations(self, prompts: List[str], max_length: int = 50,
                             temperature: float = 0.8, top_k: int = 50, beam_search: bool = False, beam_size: int = 5) -> List[str]:
        """
        Generate continuations from text prompts.
        
        Args:
            prompts: List of text prompts (already tokenized text)
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            List of generated text continuations
        """
        self.model.eval()
        generated_texts = []
        for prompt in tqdm(prompts, desc="Generating continuations"):
            # Convert prompt text to token IDs (without adding special tokens since prompt is already tokenized)
            prompt_token_ids = self.processor.text_to_ids(prompt, add_special_tokens=False)
            
            # Add SOS token at the beginning
            start_tokens = [self.processor.special_tokens['<sos>']] + prompt_token_ids
            start_tokens = torch.tensor([start_tokens], dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                generated = self.model.generate(
                    start_tokens,
                    max_length=max_length + len(prompt_token_ids),  # Account for prompt length
                    temperature=temperature,
                    top_k=top_k,
                    eos_token_id=self.processor.special_tokens['<eos>'],
                    beam_search=beam_search,
                    beam_size=beam_size
                )
            # Convert generated tokens to text
            generated_text = self.processor.ids_to_text(generated[0].cpu().tolist())
            generated_texts.append(generated_text)
        return generated_texts

    def evaluate_on_dataset(self, test_file: str, num_samples: int = 100,
                          max_length: int = 50, prompt_length: int = 5, beam_search: bool = False, beam_size: int = 5) -> dict:
        print(f"Loading test data from {test_file}")
        with open(test_file, 'r', encoding='utf-8') as f:
            stories = [line.strip() for line in f if line.strip()]
        if len(stories) > num_samples:
            import random
            stories = random.sample(stories, num_samples)
        print(f"Evaluating on {len(stories)} samples")
        prompts = []
        references = []
        for story in stories:
            token_ids = self.processor.text_to_ids(story, add_special_tokens=False)
            if len(token_ids) > prompt_length:
                prompt_ids = token_ids[:prompt_length]
                reference_ids = token_ids[prompt_length:]
                    
                prompt = self.processor.ids_to_text(prompt_ids)
                reference = self.processor.ids_to_text(reference_ids)
                
                prompts.append(prompt)
                references.append(reference)
        print(f"Generated {len(prompts)} prompt-reference pairs (using first {prompt_length} tokens as prompt)")
        generated_texts = self.generate_continuations(
            prompts, max_length=max_length, beam_search=beam_search, beam_size=beam_size
        )
        print("Calculating perplexity scores...")
        perplexities = []
        for text in tqdm(generated_texts, desc="Computing perplexity"):
            try:
                perplexity = self.calculate_perplexity(text)
                perplexities.append(perplexity)
            except Exception as e:
                print(f"Error calculating perplexity: {e}")
                perplexities.append(float('inf'))
        print("Calculating BLEU scores...")
        print(f"Generated texts sample: {generated_texts[:2]}")
        print(f"References sample: {references[:2]}")
        try:
            bleu_scores = self.calculate_bleu_score(generated_texts, references)
        except Exception as e:
            print(f"Error calculating BLEU score with Hugging Face evaluate: {e}")
            print("Falling back to simple BLEU calculation...")
            bleu_scores = self.calculate_bleu_score_simple(generated_texts, references)
        avg_perplexity = np.mean([p for p in perplexities if p != float('inf')])
        avg_bleu = bleu_scores['bleu']
        results = {
            'num_samples': len(prompts),
            'avg_perplexity': avg_perplexity,
            'bleu_score': avg_bleu,
            'bleu_scores': bleu_scores,
            'perplexities': perplexities,
            'generated_texts': generated_texts,  # Include all generated texts
            'references': references,  # Include all references
            'prompts': prompts  # Include prompts for reference
        }
        return results
    def save_results(self, results: dict, output_file: str):
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_results[key] = float(value)
            else:
                serializable_results[key] = value
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")
