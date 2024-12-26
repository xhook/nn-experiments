import csv
import os
from typing import List, Dict


import numpy as np
from typing import Optional

class MovingAverage:
    def __init__(self, window_size: int) -> None:
        """
        Initialize the MovingAverage with a specified window size.

        Parameters:
        window_size (int): The number of recent metrics to include in the average.

        Raises:
        ValueError: If window_size is not a positive integer.
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        self.window_size: int = window_size
        self.buffer: np.ndarray = np.zeros(self.window_size, dtype=float)
        self.index: int = 0
        self.count: int = 0
        self.current_sum: float = 0.0

    def update(self, metric: float) -> None:
        """
        Update the moving average with a new metric.

        Parameters:
        metric (float): The new metric to add.
        """
        if self.count < self.window_size:
            # Buffer not yet full
            self.buffer[self.index] = metric
            self.current_sum += metric
            self.count += 1
        else:
            # Buffer is full, subtract the oldest metric and add the new one
            old_metric: float = self.buffer[self.index]
            self.current_sum = self.current_sum - old_metric + metric
            self.buffer[self.index] = metric
        # Move the index forward in a circular manner
        self.index = (self.index + 1) % self.window_size

    def get(self) -> float:
        """
        Get the current moving average.

        Returns:
        float: The current moving average.

        Raises:
        ValueError: If no metrics have been added yet.
        """
        if self.count == 0:
            raise ValueError("No metrics have been added yet.")
        return self.current_sum / self.count


class MetricsLogger:
    def log(self, metrics: Dict[str, float]) -> None:
        """
        Log a dictionary of metrics.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class CSVMetricsLogger(MetricsLogger):
    def __init__(self, filename: str, fieldnames: List[str]) -> None:
        """
        Initialize the CSV logger.

        Parameters:
        - filename (str): The path to the CSV file where metrics will be saved.
        - fieldnames (List[str]): List of metric names to log (e.g., ['epoch', 'loss', 'accuracy']).
        """
        self.filename = filename
        self.fieldnames = fieldnames

        # If file doesn't exist, write the header
        if not os.path.exists(filename):
            with open(self.filename, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, metrics: Dict[str, float]) -> None:
        """
        Log a dictionary of data to the CSV file after validating the field names.

        Parameters:
        - data (Dict[str, float]): A dictionary where the keys are the fieldnames and the values are the metrics to log.
                       Example: {'epoch': 1, 'loss': 0.5, 'accuracy': 0.8}
        
        Raises:
        - ValueError: If any of the keys in the data do not match the fieldnames.
        """
        # Validate that data keys match the expected fieldnames
        if set(metrics.keys()) != set(self.fieldnames):
            raise ValueError(f"Invalid field names: Expected {self.fieldnames}, but got {list(metrics.keys())}")

        # Append the row to the CSV file
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(metrics)

