import json
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any
from src.LLM_model.chat.api.llm_model import get_chat_model
from src.LLM_model.embed.api.embed_model import get_embed_model

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

class MEDEvaluator:
    """
    Mapping-based Edge Distance (MED) Evaluator
    Calculates the edge distance after extracting and mapping the knowledge graph, as defined in the paper.
    Lower values are better, as it measures whether logically adjacent entities remain 'close' in the generated answer.
    """
    def __init__(
        self, 
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_k: int = 5,
        similarity_threshold: float = 0.99
        ) -> None:
        """
        Initialize the MED evaluator llm model and embedding model.
        
        Args:
            temperature: The temperature parameter for the LLM model.
            max_tokens: The maximum number of tokens for the LLM model.
            top_k: The number of top candidates to consider.
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
    def _parse_graph_json(content: str) -> Dict[str, Any]:
        data = None
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            content_fixed = content.replace('\\', '\\\\')
            try:
                data = json.loads(content_fixed)
            except json.JSONDecodeError:
                pass

        result = {"entities": [], "relations": []}
        if isinstance(data, dict):
            result["entities"] = data.get("entities", [])
            relations = data.get("relations", [])
            for r in relations:
                if isinstance(r, list) and len(r) >= 2:
                    result["relations"].append((str(r[0]), str(r[1])))
                elif isinstance(r, dict) and "source" in r and "target" in r:
                    result["relations"].append((str(r["source"]), str(r["target"])))
                    
        return result

    def extract_graph(
        self, 
        text: str
        ) -> Dict[str, Any]:
        """
        Extract entities and relations from a text using a LLM model.

        Args:
            text (str): The text to extract entities and relations from.

        Returns:
            Dict[str, Any]: A dictionary containing extracted entities and relations.
        """
        if not text or len(str(text).strip()) < 2:
            return {"entities": [], "relations": []}
            
        prompt = f"""
        Task: Extract specific entities and their relationships from the text below.
        Output strictly a JSON object with two keys: 'entities' (list of strings) and 'relations' (list of [source, target] pairs).
        
        Text:
        "{text}"
        
        Output Template:
        {{
            "entities": ["Entity1", "Entity2", "Entity3"],
            "relations": [
                ["Entity1", "Entity2"],
                ["Entity2", "Entity3"]
            ]
        }}
        """
        
        try:
            response = self.llm.invoke(
                    [
                        {"role": "system", "content": "You are a graph extractor. Output JSON only."},
                        {"role": "user", "content": prompt}
                    ]
                )
            content = response.content if hasattr(response, "content") else response
            return self._parse_graph_json(content)
        except Exception as e:
            print(f"[Graph Extract Error] {e}")
            return {"entities": [], "relations": []}

    def llm_judge_match(
        self, 
        target: str, 
        candidates: List[str]
        ) -> Optional[str]:
        """
        Judge whether a target string matches any of the given candidates using a LLM model.

        Args:
            target (str): The target string to be judged.
            candidates (List[str]): A list of candidate strings.

        Returns:
            Optional[str]: The matched candidate string or None if no match is found.
        """
        target_lower = target.lower().strip()
        for cand in candidates:
            if cand.lower().strip() == target_lower:
                return cand

        candidates_str = "\n".join([f"- {c}" for c in candidates])
        prompt = f"""
        Role: Knowledge Graph alignment judge.
        Task: Does 'Target' match any 'Candidate'?
        Target: "{target}"
        Candidates:
        {candidates_str}
        
        Return ONLY the matched candidate text or "None".
        """
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else response
            result = content.strip().replace('"', '').replace("'", "")
            if result == "None" or result not in candidates:
                return None
            return result
        except Exception as e:
            return None

    def build_entity_mapping(
        self, 
        gt_nodes: List[str], 
        pred_nodes: List[str]
        ) -> Dict[str, str]:
        """
        Build a mapping between ground truth entities and predicted entities using embeddings and a LLM model.

        Args:
            gt_nodes (List[str]): A list of ground truth entities.
            pred_nodes (List[str]): A list of predicted entities.

        Returns:
            Dict[str, str]: A dictionary mapping ground truth entities to predicted entities.
        """
        mapping = {}
        if not gt_nodes or not pred_nodes:
            return mapping

        gt_vecs = self.embed_documents(gt_nodes)
        pred_vecs = self.embed_documents(pred_nodes)
        sim_matrix = cosine_similarity(gt_vecs, pred_vecs)

        for i, gt_node in enumerate(gt_nodes):
            top_k_indices = np.argsort(sim_matrix[i])[-self.top_k:][::-1]
            candidates = [pred_nodes[idx] for idx in top_k_indices]
            best_sim = sim_matrix[i][top_k_indices[0]] if len(top_k_indices) > 0 else 0
            
            if best_sim > self.similarity_threshold:
                mapping[gt_node] = candidates[0]
            else:
                matched_cand = self.llm_judge_match(gt_node, candidates)
                if matched_cand:
                    mapping[gt_node] = matched_cand
                    
        return mapping

    def evaluate_instance(
        self, 
        gt_text: str, 
        pred_text: str
        ) -> Dict[str, Any]:
        """
        Evaluate Mapping-based Edge Distance (MED).
        
        Args:
            gt_text (str): The ground truth text.
            pred_text (str): The predicted text.

        Returns:
            Dict[str, Any]: A dictionary containing the MED score and other evaluation metrics.
        """
        # print("  [MED] Extracting graphs from text...")
        gt_data = self.extract_graph(gt_text)
        pred_data = self.extract_graph(pred_text)
        
        G_gt = nx.Graph()
        G_gt.add_nodes_from(gt_data["entities"])
        G_gt.add_edges_from(gt_data["relations"])
        
        G_pred = nx.Graph()
        G_pred.add_nodes_from(pred_data["entities"])
        G_pred.add_edges_from(pred_data["relations"])

        if len(G_gt.edges()) == 0:
            return {"med_score": 0.0, "gt_total_edges": 0}
            
        mapping = self.build_entity_mapping(list(G_gt.nodes()), list(G_pred.nodes()))
        
        # 1. Calculate the normalization denominator (average shortest path length d_G_avg)
        d_G_avg = 1.0
        if len(G_pred.nodes()) > 1:
            try:
                if nx.is_connected(G_pred):
                    d_G_avg = nx.average_shortest_path_length(G_pred)
                else:
                    # Use the largest connected component to represent the connectivity of the predicted graph
                    components = max(nx.connected_components(G_pred), key=len)
                    subgraph = G_pred.subgraph(components)
                    if len(subgraph.nodes()) > 1:
                        d_G_avg = nx.average_shortest_path_length(subgraph)
            except ZeroDivisionError:
                pass
        if d_G_avg == 0: d_G_avg = 1.0

        penalty_distance = max(len(G_pred.nodes()), 2) # Apply maximum distance penalty when disconnected or mapping is not found

        total_distance = 0.0
        
        # 2. Calculate the number of hops for each ground truth relation in the predicted graph
        for u_gt, v_gt in G_gt.edges():
            u_pred = mapping.get(u_gt)
            v_pred = mapping.get(v_gt)
            
            if u_pred and v_pred and u_pred in G_pred and v_pred in G_pred:
                if nx.has_path(G_pred, u_pred, v_pred):
                    dist = nx.shortest_path_length(G_pred, u_pred, v_pred)
                    total_distance += (dist / d_G_avg)
                else:
                    total_distance += (penalty_distance / d_G_avg)
            else:
                total_distance += (penalty_distance / d_G_avg)
                
        med_score = total_distance / len(G_gt.edges())
        
        return {
            "med_score": round(med_score, 4),
            "d_G_avg": round(d_G_avg, 4),
            "penalty_distance": penalty_distance,
            "mapping_details": mapping
        }

if __name__ == "__main__":
    test_gt = "A proton has a positive charge."
    test_pred = "Protons carry positive electric charges. Quarks make up protons."
    
    evaluator = MEDEvaluator(similarity_threshold=0.95)
    res = evaluator.evaluate_instance(test_gt, test_pred)
    print(json.dumps(res, indent=2))