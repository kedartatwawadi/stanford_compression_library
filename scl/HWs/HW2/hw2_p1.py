"""
The Boy Scout Final Task Puzzle
"""

from typing import List


def compute_minimal_effort(rope_lengths_arr: List[float]) -> float:
    """
    lengths_arr -> list of rope lengths (positive floating points)
    output -> the value of the minimum effort in joining all the ropes together 
    """
    effort = None
    ############################################################
    # ToDo: add code here to compute the minimal effort
    ############################################################
    raise NotImplementedError("Please implement compute_minimal_effort function")
    ############################################################
    return effort


def test_compute_minimal_effort_func():
    """
    Sample test function.
    please feel free to add more test cases of your choice
    """
    rope_lengths_to_test = [[5, 5], [1, 1, 1, 1], [20, 30, 40]]
    expected_minimal_effort = [10, 8, 140]

    for rope_lengths_arr, expected_effort in zip(rope_lengths_to_test, expected_minimal_effort):
        min_effort = compute_minimal_effort(rope_lengths_arr)
        assert (abs(min_effort - expected_effort) < 1e-6), "incorrect minimum effort"
