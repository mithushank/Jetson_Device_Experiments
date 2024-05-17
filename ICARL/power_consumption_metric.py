from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name
from tegrastats_module import start_tegrastats, stop_tegrastats, parse_tegrastats_output

class PowerConsumptionMetric(PluginMetric[float]):
    """
    A metric to measure power consumption using tegrastats.
    """

    def __init__(self, output_file='tegrastats_output.txt', interval=100):
        """
        Initialize the metric.
        :param output_file: path to the output file for tegrastats
        :param interval: interval for tegrastats measurements (in ms)
        """
        super().__init__()
        self.output_file = output_file
        self.interval = interval
        self.tegrastats_process = None
        self.total_gpu_soc_power = 0
        self.total_cpu_cv_power = 0
        self.total_power = 0

    def before_training(self, strategy: 'PluggableStrategy') -> None:
        """
        Start tegrastats before training begins.
        """
        self.tegrastats_process = start_tegrastats(self.interval, self.output_file)

    def after_training(self, strategy: 'PluggableStrategy') -> None:
        """
        Stop tegrastats and parse the output after training ends.
        """
        stop_tegrastats(self.tegrastats_process)
        self.total_gpu_soc_power, self.total_cpu_cv_power, self.total_power = parse_tegrastats_output(self.output_file, self.interval)

    def result(self, **kwargs) -> float:
        """
        Emit the result.
        """
        return self.total_power

    def reset(self, **kwargs) -> None:
        """
        Reset the metric.
        """
        self.total_gpu_soc_power = 0
        self.total_cpu_cv_power = 0
        self.total_power = 0

    def _package_result(self, strategy):
        metric_value = self.result()
        metric_name = get_metric_name(self, strategy)
        return [MetricValue(self, metric_name, metric_value, strategy.clock.train_iterations)]

    def __str__(self):
        return "PowerConsumption"