import json
from typing import List, Dict, Any, Tuple
from src.llm.chat.api.chat_model import get_chat_model

class RSEvaluator:
    """
    Relation Strength (RS) Evaluator
    Based on the paper, in the absence of Ground Truth, this utilizes a large language model 
    to score the "strength and tightness" (0-10 scale) of the relations (edges) extracted 
    from the predicted text.
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
    def _parse_graph_json(content: str) -> List[Tuple[str, str]]:
        """Extract the list of relations returned by the model [["A", "B"], ...]"""
        data = None
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            content_fixed = content.replace('\\', '\\\\')
            try:
                data = json.loads(content_fixed)
            except json.JSONDecodeError:
                pass

        relations = []
        if isinstance(data, dict):
            raw_rels = data.get("relations", [])
            for r in raw_rels:
                if isinstance(r, list) and len(r) >= 2:
                    relations.append((str(r[0]), str(r[1])))
                elif isinstance(r, dict) and "source" in r and "target" in r:
                    relations.append((str(r["source"]), str(r["target"])))
        return relations

    @staticmethod
    def _parse_scoring_json(content: str) -> Dict[str, int]:
        """Extract the relation scoring results"""
        data = None
        try:
            data = json.loads(content)
        except Exception:
            pass

        scores = {}
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "relation" in item and "strength" in item:
                    # Uniformly convert to string as the key for easy retrieval
                    scores[str(item["relation"])] = item["strength"]
        return scores

    def extract_relations(self, text: str) -> List[Tuple[str, str]]:
        """Extract relation pairs from the generated text"""
        if not text or len(str(text).strip()) < 2:
            return []
            
        prompt = f"""
        Task: Extract specific entity relationships from the text below.
        Output strictly a JSON object with a 'relations' key containing a list of [source_entity, target_entity] pairs.
        
        Text:
        "{text}"
        
        Output Template:
        {{
            "relations": [
                ["Proton", "Positive Charge"],
                ["Gravity", "Mass"]
            ]
        }}
        """
        
        try:            
            response = self.llm.invoke(
                [
                    {"role": "system", "content": "You are a graph relation extractor. Output JSON only."},
                    {"role": "user", "content": prompt}
                ]
            )
            content = response.content if hasattr(response, "content") else response
            return self._parse_graph_json(content)
        except Exception as e:
            print(f"[Extract Error] {e}")
            return []

    def score_relation_strength(self, relations: List[Tuple[str, str]], domain_context: str) -> Dict[str, int]:
        """
        Prompt the LLM to score the strength of each extracted relation pair (0-10).
        """
        if not relations:
            return {}

        relations_str = "\n".join([f"- {r[0]} <---> {r[1]}" for r in relations])
        
        prompt = f"""
        Role: You are a scientific knowledge graph expert evaluating the strength of relationships.
        
        Task: Given a knowledge domain context and extracted relations, score the logical strength and closeness of each relationship from 0 to 10.
        
        Domain Context:
        "{domain_context}"
        
        Extracted Relations:
        {relations_str}
        
        Constraints:
        - Score must be an integer between 0 and 10.
        - 0 = no logical connection / hallucination.
        - 10 = extremely strong, direct, and well-supported scientific relationship.
        - Output strictly a JSON list of objects.
        
        Output Template:
        [
            {{"relation": "['Entity A', 'Entity B']", "strength": 9}},
            {{"relation": "['Entity C', 'Entity D']", "strength": 2}}
        ]
        """
        
        try:
            response = self.llm.invoke(
                [
                    {"role": "system", "content": "You must output valid JSON following the template."},
                    {"role": "user", "content": prompt}
                ]
            )
            content = response.content if hasattr(response, "content") else response
            return self._parse_scoring_json(content)
        except Exception as e:
            print(f"[Score Error] {e}")
            return {}

    def evaluate_instance(self, pred_text: str, domain_context: str) -> Dict[str, Any]:
        """Perform RS evaluation on a single text instance"""
        # print("  [RS] Extracting relations from prediction...")
        relations = self.extract_relations(pred_text)
        
        # Remove duplicates
        relations = list(set(relations))
        # print(f"  [RS] Extracted {len(relations)} relations. Scoring strength...")
        
        if not relations:
            return {
                "rs_score": 0.0,
                "relation_count": 0,
                "scores_detail": {}
            }

        scores_dict = self.score_relation_strength(relations, domain_context)
        
        total_score = 0
        valid_count = len(relations)
        
        for rel in relations:
            # Format the key to match the LLM's response
            rel_key = str(list(rel))
            score = scores_dict.get(rel_key, 0)
            total_score += score
            
        rs_score = total_score / valid_count if valid_count > 0 else 0.0
        
        return {
            "rs_score": round(rs_score, 4),
            "relation_count": valid_count,
            "relations_list": relations,
            "scores_detail": scores_dict
        }

if __name__ == "__main__":
    test_question = "Explain the relationship between Gravity and Mass."
    test_prediction = "Mass generates gravity. Also, eating apples makes you heavy."
    
    evaluator = RSEvaluator()
    metrics = evaluator.evaluate_instance(test_prediction, test_question)
    
    print("\n=== RS Evaluation Results ===")
    print(f"Average RS Score: {metrics['rs_score']:.4f} / 10.0")
    print(f"Total Relations: {metrics['relation_count']}")