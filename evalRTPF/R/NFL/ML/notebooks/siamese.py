import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import process_data


parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)


interpolated_dir = os.path.join(parent_dir, "dataset_interpolated_with_overtime")

features = ["score_difference", "relative_strength", "type.id", "home_has_possession", "end.down", "end.yardsToEndzone", "end.distance", "field_position_shift", "home_timeouts_left", "away_timeouts_left"]

interpolated_dir = os.path.join(parent_dir, "dataset_interpolated_with_overtime")
training_data = process_data.load_data(interpolated_dir, 
                                       years = [2016, 2017, 2018, 2019, 2020, 2021, 2022], 
                                       history_length = 6, 
                                       features = features, 
                                       label_feature = "final_score_difference")

test_data = process_data.load_data(interpolated_dir, 
                                       years = [2023, 2024], 
                                       history_length = 6, 
                                       features = features, 
                                       label_feature = "final_score_difference")


import kernel_methods.kernel_knn
kernel_methods.kernel_knn.setup_models(training_data, test_data, num_models=50, epochs=50)