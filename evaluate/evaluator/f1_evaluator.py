import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from src.llm.chat.api.chat_model import get_chat_model
from src.llm.embed.api.embed_model import get_embed_model

def cosine_similarity(
    vec1: np.ndarray, 
    vec2: np.ndarray
    ) -> np.ndarray:
    """
    Calculate the cosine similarity between two vectors.

    Args:
        vec1 (np.ndarray): The first vector.
        vec2 (np.ndarray): The second vector.

    Returns:
        np.ndarray: The cosine similarity between the two vectors.
    """
    A_norm = vec1 / np.linalg.norm(vec1, axis=1, keepdims=True)
    B_norm = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)
    return np.dot(A_norm, B_norm.T) 


class F1Evaluator:
    """
    F1 Evaluator class for evaluating the performance of the F1 model.
    """
    def __init__(
        self, 
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_k: int = 5,
        similarity_threshold: float = 0.99
        ) -> None:
        """
        Initialize the f1 evaluator llm model and embedding model.
        
        Args:
            temperature: The temperature parameter for the LLM model.
            max_tokens: The maximum number of tokens for the LLM model.
            top_k: The number of top-k candidates to consider.
            similarity_threshold: The threshold for considering two vectors as similar.
        """
    
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        self.llm = get_chat_model(
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        self.embed = get_embed_model()

    def embed_documents(
        self,
        text_list: List[str]
        ) -> np.ndarray:
        """
        Get vector representations for a batch of texts.

        Args:
            text_list (List[str]): List of texts to be vectorized.

        Returns:
            np.ndarray: An array of embeddings.
        """
        return self.embed.embed_documents(text_list)
    
    @staticmethod
    def _parse_entity_json(content: str) -> List[str]:
        """
        Helper method to parse and sanitize JSON output from the LLM.
        
        Args:
            content (str): The raw string content returned by the LLM.
            
        Returns:
            List[str]: A parsed list of entities.
        """
        data = None
        
        # 1. Standard JSON parsing
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # 2. Fix unescaped backslashes
            content_fixed = content.replace('\\', '\\\\')
            try:
                data = json.loads(content_fixed)
            except json.JSONDecodeError:
                pass

        # 3. Fallback: try to extract a JSON array using string matching if standard parsing fails
        if data is None and "[" in content and "]" in content:
            start = content.find("[")
            end = content.rfind("]") + 1
            try:
                data = json.loads(content[start:end])
            except json.JSONDecodeError:
                pass

        # 4. Validate and extract the final list
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # If a dict is returned, look for the first list value
            for val in data.values():
                if isinstance(val, list):
                    return val
                    
        return []

    def extract_entities(
        self, 
        text: str
        ) -> List[str]:
        """
        Extract a list of entities from the raw text using the LLM.

        Args:
            text (str): The raw text from which entities will be extracted.

        Returns:
            List[str]: A list of extracted entities.
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
            error_msg = str(e)
            if "401" in error_msg or "authentication" in error_msg.lower():
                raise PermissionError(f"[Fatal Error] Invalid API Key: {error_msg}")
            print(f"[Extract Error] {e}")
            return []

    def llm_judge_match(
        self, 
        target: str, 
        candidates: List[str]
        ) -> Optional[str]:
        """
        Use the LLM to judge whether the target entity matches any of the candidate entities.

        Args:
            target (str): The target entity to match.
            candidates (List[str]): The list of candidate entities for reference.

        Returns:
            Optional[str]: The matched candidate string, or None if no match is found.
        """
        # Quick exact match (case-insensitive)
        target_lower = target.lower().strip()
        for cand in candidates:
            if cand.lower().strip() == target_lower:
                return cand

        candidates_str = "\n".join([f"- {c}" for c in candidates])
        prompt = f"""
        Role: Knowledge Graph alignment judge.
        Task: Does 'Target' match any 'Candidate' (synonyms, aliases, singular/plural)?
        Target: "{target}"
        Candidates:
        {candidates_str}
        
        Return ONLY the matched candidate text or "None".
        """

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else response
            # Clean up potential quotes returned by the LLM
            result = content.replace('"', '').replace("'", "")
            
            if result == "None" or result not in candidates:
                return None
            return result
        except Exception as e:
            print(f"[Judge Warning] {e}")
            return None

    def calculate_one_way_match(
        self, 
        src_entities: List[str], 
        tgt_entities: List[str], 
        src_vecs: np.ndarray, 
        tgt_vecs: np.ndarray
        ) -> Tuple[float, List[str]]:
        """
        Calculate the one-way match rate from source entities to target entities and record details.

        Args:
            src_entities (List[str]): Source entity list.
            tgt_entities (List[str]): Target entity list.
            src_vecs (np.ndarray): Vector matrix corresponding to the source entities.
            tgt_vecs (np.ndarray): Vector matrix corresponding to the target entities.

        Returns:
            Tuple[float, List[str]]: A tuple containing the match rate and a list of match details.
        """
        if len(src_entities) == 0:
            return 0.0, []

        match_count = 0
        match_details = []
        sim_matrix = cosine_similarity(src_vecs, tgt_vecs)

        for i, src_ent in enumerate(src_entities):
            # Retrieve top-k most similar target entities
            top_k_indices = np.argsort(sim_matrix[i])[-self.top_k:][::-1]
            candidates = [tgt_entities[idx] for idx in top_k_indices]
            best_sim_score = sim_matrix[i][top_k_indices[0]] if len(top_k_indices) > 0 else 0
            
            matched_cand = None
            if best_sim_score > self.similarity_threshold:
                # Direct match if similarity exceeds the threshold
                matched_cand = candidates[0] 
            else:
                # Fallback to LLM judgment for ambiguous cases
                matched_cand = self.llm_judge_match(src_ent, candidates)
            
            if matched_cand:
                match_count += 1
                match_details.append(f"{src_ent} -> {matched_cand}")
        
        return match_count / len(src_entities), match_details

    def evaluate_instance(
        self, 
        gt_text: str, 
        pred_text: str
        ) -> Dict[str, Any]:
        """
        Perform a full evaluation on a single sample, including entity extraction and metric calculation.

        Args:
            gt_text (str): Ground Truth text.
            pred_text (str): Predicted text to be evaluated.

        Returns:
            Dict[str, Any]: A dictionary containing precision, recall, f1, and detailed match lists.
        """
        # print("Extracting entities from text...")
        gt_list = self.extract_entities(gt_text)
        pred_list = self.extract_entities(pred_text)
        
        # Remove duplicates
        gt_list = list(set(gt_list))
        pred_list = list(set(pred_list))
        # print(f"Entity Count - GT: {len(gt_list)}, Pred: {len(pred_list)}")

        if not gt_list or not pred_list:
            return {
                "precision": 0.0, "recall": 0.0, "f1": 0.0, 
                "gt_list": gt_list, "pred_list": pred_list, "recall_details": [] 
            }

        # Vectorize entities
        gt_vecs = self.embed_documents(gt_list)
        pred_vecs = self.embed_documents(pred_list) 

        # Calculate Recall (GT -> Pred) and Precision (Pred -> GT)
        recall, recall_details = self.calculate_one_way_match(gt_list, pred_list, gt_vecs, pred_vecs)
        precision, _ = self.calculate_one_way_match(pred_list, gt_list, pred_vecs, gt_vecs)

        f1 = 0.0
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "gt_list": gt_list,
            "pred_list": pred_list,
            "recall_details": recall_details
        }
        
if __name__ == "__main__":
    gt_text = """
    A proton is a subatomic particle with a positive electric charge.
    It is composed of quarks and is found in the nucleus of an atom.
    """

    pred_text = """
    Protons carry positive electric charges and are made of quarks.
    They exist inside atomic nuclei.
    """

    evaluator = F1Evaluator(
        temperature=0.0,
        top_k=5,
        similarity_threshold=0.9
    )

    result = evaluator.evaluate_instance(gt_text, pred_text)

    print("\n===== Evaluation Result =====")
    print(json.dumps(result, indent=2, ensure_ascii=False))