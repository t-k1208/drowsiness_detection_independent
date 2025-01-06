import glob
import sys
import pyedflib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from statsmodels import robust
from scipy.signal import welch
import scipy.signal as signal  # 信号処理ライブラリをインポート
import seaborn as sns

from need_utils import *
from preprocessing import *


filenames = glob.glob("../../DROZY/psg/*.edf")
with open("../../DROZY/KSS.txt", "r") as f:
    original_labels = f.read().split("\n")

subject = 6
# label = 1
label = 3
low_cut = 0.3
high_cut = 70.0
selected_filename = None
# method = "robust"
# method = "iqr"
method = None
start = 0
end = 300
downsampling = True

for filename in filenames:
    num_subject, num_experiment = get_subject_experiment_numbers(filename)
    if num_subject == subject - 1 and num_experiment == label - 1:
        selected_filename = filename
        break

if selected_filename is not None:
    plot_data(
        selected_filename,
        original_labels[subject - 1].split(" ")[label - 1],
        cutoff_freq=[low_cut, high_cut],
        method=method,
        start=start,
        end=end,
        downsampling=downsampling,
    )
else:
    print(f"指定された Subject: {subject}, Label: {label} に一致するファイルが見つかりませんでした。")


