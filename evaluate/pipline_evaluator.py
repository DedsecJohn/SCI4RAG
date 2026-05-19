from tqdm import tqdm
from typing import List
from pathlib import Path
from collections import defaultdict
from src.service.document.load_document import load_json

from evaluate.evaluator.f1_evaluator import F1Evaluator
from evaluate.evaluator.rs_evaluator import RSEvaluator
from evaluate.evaluator.es_evaluator import ESEvaluator
from evaluate.evaluator.mec_evaluator import MECEvaluator
from evaluate.evaluator.med_evaluator import MEDEvaluator

EVALUATOR_NAME_MAP = {
    F1Evaluator: "F1",
    RSEvaluator: "RS",
    ESEvaluator: "ES",
    MECEvaluator: "MEC",
    MEDEvaluator: "MED",
}

from src.pipline.simple_pipeline import SimplePipeline
from src.pipline.rag_pipeline import RAGPipeline
from src.pipline.web_pipeline import WebPipeline

PIPELINE_NAME_MAP = {
    SimplePipeline: "Simple",
    RAGPipeline: "RAG",
    WebPipeline: "Web",
}

def get_evaluator_name(evaluator) -> str:
    """
    Get evaluator name from evaluator instance or class.
    """
    evaluator_cls = evaluator if isinstance(evaluator, type) else type(evaluator)

    if evaluator_cls in EVALUATOR_NAME_MAP:
        return EVALUATOR_NAME_MAP[evaluator_cls]

    return "UnknownEvaluator"

def get_pipeline_name(pipeline) -> str:
    """
    Get pipeline name from pipeline instance or class.
    """
    pipeline_cls = pipeline if isinstance(pipeline, type) else type(pipeline)
    if pipeline_cls in PIPELINE_NAME_MAP:
        return PIPELINE_NAME_MAP[pipeline_cls]

    return "UnknownPipeline"

def check_benchmark() -> List[str]:
    """
    Check which benchmark datasets exist in the benchmark folder.

    Returns:
        List[str]: List of benchmark dataset names (without suffix).
    """
    benchmark_folder = Path("evaluate/benchmark")

    # 1. Check the benchmark folder
    if not benchmark_folder.exists():
        print(f"[Warning] Benchmark folder not found: {benchmark_folder}")
        return []

    # 2. Srearch for JSON files
    json_files = list(benchmark_folder.glob("*.json"))

    if not json_files:
        print(f"[Info] No benchmark datasets found in {benchmark_folder}")
        return []

    # 3. Get the benchmark names
    benchmark_names = [f.stem for f in json_files]

    # 4. Print the benchmark names
    print(f"Found {len(benchmark_names)} benchmark dataset(s):")
    for name in benchmark_names:
        print(f" - {name}")

    return benchmark_names

def check_benchmark_exists(benchmark_name: str) -> str:
    """
    Check if a specific benchmark dataset exists in the benchmark folder.

    Args:
        benchmark_name (str): Name of the benchmark dataset (without suffix).

    Returns:
        
    """
    benchmark_folder = Path("evaluate/benchmark")
    benchmark_file = benchmark_folder / f"{benchmark_name}.json"
    if benchmark_file.exists():
        return benchmark_file
    else:
        print(f"[Error] Benchmark dataset not found: {benchmark_file}")
        check_benchmark()
        return None
    


class Evaluator_Pipeline:
    def __init__(
        self, 
        pipeline, 
        evaluator,
        benchmark: str,
    ):
        """
        Args:
            pipeline: RAG pipeline instance (must have .query())
            evaluator: evaluator instance (must have .evaluate_instance())
            benchmark (str): benchmark dataset name
        """
        self.pipeline = pipeline
        self.evaluator = evaluator
        self.benchmark_name = benchmark
        self.pipeline_name  = get_pipeline_name(pipeline)
        self.evaluator_name = get_evaluator_name(evaluator)
        print(f"[Info] Evaluator: {self.evaluator_name} | Pipeline: {self.pipeline_name} | Benchmark: {self.benchmark_name}")
        
    def evaluate(self) -> float:
        """
        Run evaluation on the benchmark dataset.

        Returns:
            float: average precision score
        """
        scores = defaultdict(float)
        # 1. Load benchmark data
        benchmark_data = check_benchmark_exists(self.benchmark_name)
        if benchmark_data is None:  
            print(f"[Error] Benchmark dataset not found: {self.benchmark_name}")
            return
        benchmark_data = load_json(benchmark_data)
        if not benchmark_data:
            print(f"[Error] Empty benchmark: {self.benchmark_name}")
            return
        total_questions = len(benchmark_data)
        
        # 2. Initialize evaluator Index
        EVAL_KEYS = {
            "F1": ["precision", "recall", "f1"],
            "RS": ["rs_score"],
            "ES": ["es_score"],
            "MEC": ["mec_score"],
            "MED": ["med_score"]
        }
        
        keys = EVAL_KEYS[self.evaluator_name]
        
        # 3. Evaluate each question
        with tqdm(
            total=total_questions,
            desc= f"Evaluator: {self.evaluator_name}",
            unit="sample",
            ncols=100,
            position=0,
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}]"
        ) as pbar:
            for data in benchmark_data:
                gt_text = data["output"]
                # print(f"[Info] Evaluating: {data['instruction']}")  
                pred_text = self.pipeline.query(data["instruction"])
                metrics = self.evaluator.evaluate_instance(gt_text, pred_text)
                for k in keys:
                    scores[k] += metrics.get(k, 0.0)
                pbar.update(1)
        
        # 4. Calculate average precision
        results = {k: v / total_questions for k, v in scores.items()}
        for k, v in results.items():
            print(f"  Avg {k}: {v:.4f}") 
        return results


    
if __name__ == "__main__":
    check_benchmark_exists("test")
    pipline = SimplePipeline()
    evaluator = F1Evaluator()
    evaluator_pipeline = Evaluator_Pipeline(pipline, evaluator, "test")
    score = evaluator_pipeline.evaluate()