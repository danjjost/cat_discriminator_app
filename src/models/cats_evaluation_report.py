from src.models.cat_evaluation_result import CatEvaluationResult


class CatsEvaluationReport:
    def __init__(self):
        self.results: list[CatEvaluationResult] = []
        self.number_correct: int = 0
        self.total_evaluations: int = 0 

    def add_result(self, result: CatEvaluationResult):
        self.results.append(result)

    def finalize(self) -> 'CatsEvaluationReport':
        self.__set_number_correct()
        self.total_evaluations = len(self.results)
        self.percent_correct = self.number_correct / len(self.results) * 100
        return self

    def __set_number_correct(self):
        self.number_correct = 0
        for result in self.results:
            if result.actual_label == result.predicted_label:
                self.number_correct += 1

    def print_results(self):
        print(f'Number correct: {self.number_correct}')
        print(f'Total evaluations: {self.total_evaluations}')
        print(f'Percent correct: {self.percent_correct}%')
        print("")
        print("")

    def print_verbose_results(self):
        self.print_results()

        for result in self.results:
            result.print_result()
