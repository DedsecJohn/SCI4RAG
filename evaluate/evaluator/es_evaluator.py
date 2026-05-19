import json
from typing import List, Dict, Any
from src.LLM_model.chat.api.llm_model import get_chat_model


class ESEvaluator:
    """
    Entity Specificity (ES) Evaluator
    Based on the reference-free (No Ground Truth) evaluation method in the paper,
    this uses an LLM to extract entities from the generated text and score their
    domain relevance (specificity).
    """
  
    def __init__(
        self, 
        temperature: float = 0.1,
        max_tokens: int = 4096
        ) -> None:
        """
        Initialize the evaluator llm model.
        
        Args:
            temperature: The temperature parameter for the LLM model.
            max_tokens: The maximum number of tokens for the LLM model.
        """
        
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.llm = get_chat_model(
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

    @staticmethod
    def _parse_entity_json(content: str) -> List[str]:
        """Helper method to parse and sanitize JSON output for entity extraction."""
        data = None
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            content_fixed = content.replace('\\', '\\\\')
            try:
                data = json.loads(content_fixed)
            except json.JSONDecodeError:
                pass

        if data is None and "[" in content and "]" in content:
            start = content.find("[")
            end = content.rfind("]") + 1
            try:
                data = json.loads(content[start:end])
            except json.JSONDecodeError:
                pass

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for val in data.values():
                if isinstance(val, list):
                    return val
        return []

    @staticmethod
    def _parse_scoring_json(content: str) -> Dict[str, int]:
        """Helper method to parse LLM scoring output into a dictionary."""
        data = None
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            if "[" in content and "]" in content:
                start = content.find("[")
                end = content.rfind("]") + 1
                try:
                    data = json.loads(content[start:end])
                except json.JSONDecodeError:
                    pass

        scores = {}
        # Expected format from prompt: [{"entity": "Name", "specificity": 1}, ...]
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "entity" in item and "specificity" in item:
                    scores[item["entity"]] = item["specificity"]
        elif isinstance(data, dict):
            # Fallback if LLM returns {"EntityName": 1}
            for k, v in data.items():
                if isinstance(v, int):
                    scores[k] = v
        return scores

    def extract_entities(
        self, 
        text: str
        ) -> List[str]:
        """
        Extract a list of entities from the raw text using the LLM.
        
        Args:
            text: The raw text to extract entities from.

        Returns:
            A list of extracted entities.
        """
        if not text or len(str(text).strip()) < 2:
            return []
            
        prompt = f"""
        Task: Extract all specific entities, concepts, and technical terms from the text below.
        Output strictly a JSON list of strings (e.g., ["Entity1", "Entity2"]).
        
        Text:
        "{text}"
        
        Constraints:
        1. Extract noun phrases representing specific concepts.
        2. Do not explain, just output the JSON list.
        """
        
        try:
            response = self.llm.invoke(
                [
                    {"role": "system", "content": "You are an entity extractor. Output JSON only."},
                    {"role": "user", "content": prompt}
                ]
            )
            content = response.content if hasattr(response, "content") else response
            return self._parse_entity_json(content)
        except Exception as e:
            print(f"[Extract Error] {e}")
            return []

    def score_entities_specificity(
        self, 
        entities: List[str], 
        domain_context: str
        ) -> Dict[str, int]:
        """
        Use a Large Language Model to score the "Specificity" of the extracted entities.
        Based on Appendix B.5.1 of the paper: 0 = irrelevant, 1 = highly relevant.
        
        Args:
            entities: A list of extracted entities.
            domain_context: The domain context (question/topic) for the entities.

        Returns:
            A dictionary mapping entities to their specificity scores.
        """
        if not entities:
            return {}

        entities_str = "\n".join([f"- {e}" for e in entities])
        
        prompt = f"""
        Role: You are an educational knowledge graph expert skilled in evaluating entity extraction quality.
        
        Task: Given a knowledge domain context (question/topic) and extracted entities, score each entity's specificity.
        
        Domain Context:
        "{domain_context}"
        
        Extracted Entities:
        {entities_str}
        
        Constraints:
        - Specificity score must be 0 or 1.
        - 0 = irrelevant/general term, 1 = highly relevant/domain-specific.
        - Output strictly a JSON list of objects.
        
        Output Template:
        [
            {{"entity": "Entity1", "specificity": 1}},
            {{"entity": "Entity2", "specificity": 0}}
        ]
        """
        
        try:
            response = self.llm.invoke(
                [
                    {"role": "system", "content": "You are an entity extractor. Output JSON only."},
                    {"role": "user", "content": prompt}
                ]
            )
            content = response.content if hasattr(response, "content") else response
            return self._parse_scoring_json(content)
        except Exception as e:
            print(f"[Score Error] {e}")
            return {}

    def evaluate_instance(
        self, 
        pred_text: str, 
        domain_context: str
        ) -> Dict[str, Any]:
        """
        Perform a full ES evaluation on a single sample.

        Args:
            pred_text (str): The predicted text generated by the model.
            domain_context (str): The domain context for evaluation (e.g., the user's original Instruction, 
                                  used to judge entity relevance).

        Returns:
            Dict[str, Any]: Contains the ES score, extracted entities, and specific scores for each entity.
        """
        # print("  [ES] Extracting entities from prediction...")
        pred_entities = self.extract_entities(pred_text)
        
        # Remove duplicates
        pred_entities = list(set(pred_entities))
        # print(f"  [ES] Extracted {len(pred_entities)} entities. Scoring specificity...")
        
        if not pred_entities:
            return {
                "es_score": 0.0,
                "entity_count": 0,
                "entities": [],
                "scores_detail": {}
            }

        # Let LLM score the entities
        scores_dict = self.score_entities_specificity(pred_entities, domain_context)
        
        # Calculate ES (average specificity of all entities)
        total_score = 0
        valid_count = len(pred_entities)
        
        for entity in pred_entities:
            # If parsing fails or the LLM does not return the entity, adopt a conservative strategy and score as 0
            total_score += scores_dict.get(entity, 0)
            
        es_score = total_score / valid_count if valid_count > 0 else 0.0
        
        return {
            "es_score": round(es_score, 4),
            "entity_count": valid_count,
            "entities": pred_entities,
            "scores_detail": scores_dict
        }


if __name__ == "__main__":
    # Test code (this module can be run independently for testing)
    # For usage, simply initialize evaluator = ESEvaluator() and call evaluate_instance in your pipeline loop.
    
    # Simulate a question (domain context) from the test set and the pipeline's answer (prediction)
    test_question = "What is the relationship between a proton and an electron in electromagnetism?"
    test_prediction = "A proton has a positive charge, while an electron has a negative charge. Quantum entanglement and eating an apple pie are not directly related."
    
    # try:
    es_eval = ESEvaluator()
    metrics = es_eval.evaluate_instance(pred_text=test_prediction, domain_context=test_question)
    
    print("\n=== ES Evaluation Results ===")
    print(f"ES Score: {metrics['es_score']:.4f}")
    print(f"Extracted Entities: {metrics['entity_count']}")
    print("Scoring Details:")
    for entity, score in metrics['scores_detail'].items():
        print(f"  - {entity}: {score}")
        
    # except Exception as e:
    #     print(f"Initialization failed (ensure config.py and api_key exist): {e}")