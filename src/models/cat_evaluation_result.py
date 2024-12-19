from numpy import number


class CatEvaluationResult:
    def __init__(self):
        self.actual_label: str = ''
        self.predicted_label: str = ''

        self.captain_score: number = 0.0
        self.control_score: number = 0.0
        self.bathrooom_cat_score: number = 0.0

    def print_result(self):
        print(f'Actual: {self.actual_label}, Predicted: {self.predicted_label}')
        print(f'Bathroom-cat: {self.bathrooom_cat_score}, Captain: {self.captain_score}, Control: {self.control_score}')
        print("")