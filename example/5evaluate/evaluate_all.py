from src.core.paths import *
from src.core.utils import load_json, save_json
from evaluate.pipline_evaluator import Evaluator_Pipeline, EVALUATOR_NAME_MAP, get_evaluator_name, get_pipeline_name
from src.pipline.simple_pipeline import SimplePipeline
from src.pipline.rag_pipeline import RAGPipeline
from src.pipline.web_pipeline import WebPipeline

# RUN:  python -m example.5evaluate.evaluate_all

username = "administrator"
dataset_name = "test"
benchmark = 'test'

# load evaluator result
evaluator_path = evaluate_result(benchmark)
evaluator_result = load_json(evaluator_path)

# 1. SimplePipeline
pipeline = SimplePipeline(username=username)
pipeline_name = get_pipeline_name(pipeline)
if pipeline_name not in evaluator_result:
    evaluator_result[pipeline_name] = {}
    for evaluator_cls in EVALUATOR_NAME_MAP:
        evaluator = evaluator_cls()
        evaluator_name = get_evaluator_name(evaluator)
        result = Evaluator_Pipeline(pipeline, evaluator, benchmark).evaluate()
        evaluator_result[pipeline_name] |= result
        save_json(evaluator_result, evaluator_path)

# 2. RAGPipeline
pipeline = RAGPipeline(username=username, dataset_name=dataset_name)
pipeline_name = get_pipeline_name(pipeline)
if pipeline_name not in evaluator_result:
    evaluator_result[pipeline_name] = {}
    for evaluator_cls in EVALUATOR_NAME_MAP:
        evaluator = evaluator_cls()
        evaluator_name = get_evaluator_name(evaluator)
        result = Evaluator_Pipeline(pipeline, evaluator, benchmark).evaluate()
        evaluator_result[pipeline_name] |= result
        save_json(evaluator_result, evaluator_path)
        
# 3. WebPipeline
pipeline = WebPipeline(username=username)
pipeline_name = get_pipeline_name(pipeline)
if pipeline_name not in evaluator_result:
    evaluator_result[pipeline_name] = {}
    for evaluator_cls in EVALUATOR_NAME_MAP:
        evaluator = evaluator_cls()
        evaluator_name = get_evaluator_name(evaluator)
        result = Evaluator_Pipeline(pipeline, evaluator, benchmark).evaluate()
        evaluator_result[pipeline_name] |= result
        save_json(evaluator_result, evaluator_path)