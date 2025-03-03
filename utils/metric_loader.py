import config
from metrics import llm_accuracy, vqa_mqm, meteor

class MetricLoader:
    @staticmethod
    def load_metric(metric_name):
        if metric_name == config.LLM_ACCURACY_METRIC:
            return llm_accuracy.LLMAccuracy()
        if metric_name == config.VQA_MQM_METRIC:
            return vqa_mqm.VQA_MQM()
        if metric_name == config.METEOR_METRIC:
            return meteor.Meteor()
        raise ValueError(f"Undefined metric name received: {metric_name}. Please check the metric list to identify the correct metric name.")