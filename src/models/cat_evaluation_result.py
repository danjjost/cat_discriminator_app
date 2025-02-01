import json
from numpy import number


class CatEvaluationResult:
    def __init__(self):
        self.actual_label: str = ''
        self.predicted_label: str = ''

        self.captain_score: number = 0.0
        self.control_score: number = 0.0
        self.bathrooom_cat_score: number = 0.0

        self.captain_percent: number = 0.0
        self.control_percent: number = 0.0
        self.bathrooom_cat_percent: number = 0.0
        
    def to_json(self) -> str:
        return json.dumps({
            'actual_label': self.actual_label,
            'predicted_label': float(self.predicted_label),

            'captain_score': float(self.captain_score),
            'control_score': float(self.control_score),
            'bathrooom_cat_score': float(self.bathrooom_cat_score),

            'control_percent': float(self.control_percent),
            'captain_percent': float(self.captain_percent),
            'bathroom_cat_percent': float(self.bathrooom_cat_percent)
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'CatEvaluationResult':
        data = json.loads(json_str)
        
        result = cls()

        result.actual_label = data['actual_label']
        result.predicted_label = data['predicted_label']

        result.captain_score = data['captain_score']
        result.control_score = data['control_score']
        result.bathrooom_cat_score = data['bathrooom_cat_score']

        result.captain_percent = data['captain_percent']
        result.control_percent = data['control_percent']
        result.bathrooom_cat_percent = data['bathroom_cat_percent']

        return result

    def print_result(self):
        print(f'Actual: {self.actual_label}, Predicted: {self.predicted_label}')
        print(f'bathroom-cat: {self.bathrooom_cat_score:.4f}, captain: {self.captain_score:.4f}, control: {self.control_score:.4f}')
        print(f'bathroom-cat: {self.bathrooom_cat_percent:.4f}%, captain: {self.captain_percent:.4f}%, control: {self.control_percent:.4f}%')
        print("")
