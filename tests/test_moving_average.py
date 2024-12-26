import pytest
from nnutils.metrics import MovingAverage

# 1. Initialization Tests
def test_initialization_positive_window():
    ma = MovingAverage(window_size=5)
    assert ma.window_size == 5
    assert ma.count == 0
    assert ma.current_sum == 0.0
    assert ma.buffer.tolist() == [0.0] * 5
    assert ma.index == 0

def test_initialization_zero_window():
    with pytest.raises(ValueError, match="window_size must be positive."):
        MovingAverage(window_size=0)

def test_initialization_negative_window():
    with pytest.raises(ValueError, match="window_size must be positive."):
        MovingAverage(window_size=-3)

# 2. Update and Get Tests
def test_get_before_updates():
    ma = MovingAverage(window_size=3)
    with pytest.raises(ValueError, match="No metrics have been added yet."):
        ma.get()

def test_update_single_metric():
    ma = MovingAverage(window_size=3)
    ma.update(10.0)
    assert ma.get() == 10.0

def test_update_multiple_metrics_less_than_window():
    ma = MovingAverage(window_size=5)
    metrics = [10, 20, 30]
    for metric in metrics:
        ma.update(metric)
    expected_average = sum(metrics) / len(metrics)
    assert ma.get() == expected_average

# 3. Circular Buffer Behavior
def test_update_exact_window_size():
    ma = MovingAverage(window_size=4)
    metrics = [10, 20, 30, 40]
    for metric in metrics:
        ma.update(metric)
    expected_average = sum(metrics) / 4
    assert ma.get() == expected_average

def test_update_exceeding_window_size():
    ma = MovingAverage(window_size=3)
    metrics = [10, 20, 30, 40, 50]
    for metric in metrics:
        ma.update(metric)
    # After adding 10, 20, 30, buffer is [10,20,30], average=20
    # Adding 40: buffer becomes [40,20,30], sum=90, average=30
    # Adding 50: buffer becomes [40,50,30], sum=120, average=40
    expected_average = 40.0
    assert ma.get() == expected_average

def test_circular_buffer_overwrites_old_metrics():
    ma = MovingAverage(window_size=3)
    ma.update(1)
    ma.update(2)
    ma.update(3)
    assert ma.get() == 2.0  # (1+2+3)/3 = 2.0
    ma.update(4)
    assert ma.get() == (2 + 3 + 4) / 3  # 3.0
    ma.update(5)
    assert ma.get() == (3 + 4 + 5) / 3  # 4.0

# 4. Edge Case Tests
def test_all_metrics_same():
    ma = MovingAverage(window_size=4)
    for _ in range(4):
        ma.update(5.0)
    assert ma.get() == 5.0

def test_all_metrics_zero():
    ma = MovingAverage(window_size=3)
    for _ in range(3):
        ma.update(0.0)
    assert ma.get() == 0.0

def test_mixed_integer_and_float_metrics():
    ma = MovingAverage(window_size=5)
    metrics = [1, 2.5, 3, 4.5, 5]
    for metric in metrics:
        ma.update(metric)
    expected_average = sum(metrics) / 5
    assert ma.get() == expected_average

def test_very_large_metrics():
    ma = MovingAverage(window_size=2)
    ma.update(1e12)
    ma.update(1e12)
    assert ma.get() == 1e12
    ma.update(1e12)
    assert ma.get() == 1e12

# 5. Additional Tests (Optional)
def test_internal_buffer_after_updates():
    ma = MovingAverage(window_size=3)
    ma.update(10)
    ma.update(20)
    ma.update(30)
    ma.update(40)  # Overwrites the first metric (10)
    assert ma.buffer.tolist() == [40.0, 20.0, 30.0]

def test_multiple_get_calls():
    ma = MovingAverage(window_size=3)
    ma.update(10)
    ma.update(20)
    ma.update(30)
    assert ma.get() == 20.0
    assert ma.get() == 20.0  # Should consistently return the same average
