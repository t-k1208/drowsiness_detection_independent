
from tqdm import tqdm
import sys
import pyedflib
import copy
import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from statsmodels import robust
from scipy.signal import welch
import scipy.signal as signal  # 信号処理ライブラリをインポート
import seaborn as sns
from scipy.stats import skew, kurtosis


# ダウンサンプリングの関数
def downsample(data, fs, new_fs):
    # ダウンサンプリング後のデータ数を計算
    new_data_length = int(len(data) * new_fs / fs)
    # ダウンサンプリング
    new_data = signal.resample(data, new_data_length)
    return new_data


def apply_bandpass_filter(data, sampling_rate, low_cut, high_cut, order=5):
    """
    脳波データにバンドパスフィルタを適用する

    Parameters:
    data (array-like): 入力脳波データ (trials, samples)
    sampling_rate (float): サンプリングレート (Hz)
    low_cut (float): 低域遮断周波数 (Hz)
    high_cut (float): 高域遮断周波数 (Hz)
    order (int): フィルタの次数 (デフォルト: 5)

    Returns:
    array-like: フィルタ後の脳波データ (trials, samples)
    """
    # num_trials, num_samples = data.shape
    nyquist_freq = 0.5 * sampling_rate
    low = low_cut / nyquist_freq
    high = high_cut / nyquist_freq
    
    b, a = butter(order, [low, high], btype='band')
    
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data



def replace_outliers_with_iqr(data):
    """
    眼電図データの外れ値を90パーセンタイル値で置き換える。

    Args:
        data: 1次元NumPy配列 (眼電図データ)

    Returns:
        data_with_outliers_replaced: 外れ値を置き換えた1次元NumPy配列
    """
    q99, q01 = np.percentile(data, [99, 1])
    iqr = q99 - q01
    lower_bound = q01 - 1.5 * iqr
    upper_bound = q99 + 1.5 * iqr

    # 外れ値を90パーセンタイル値で置き換える
    data_with_outliers_replaced = np.where(
        data < q01, q01,
        np.where(data > q99, q99, data)
    )

    return data_with_outliers_replaced


""" ロバスト統計量で外れ値処理 """
def replace_outliers_with_robust_statistics(data):
    median = np.median(data)
    mad = robust.mad(data)
    modified_z_scores = 0.6745 * (data - median) / mad

    threshold = 3.5
    data_with_outliers_replaced = np.where(
        np.abs(modified_z_scores) > threshold,
        median,
        data
    )
    return data_with_outliers_replaced



""" セグメンテーション """
def segmentation(signal, fs=512, sec=13, slide=1):
    segment_length = int(fs * sec)  # セグメント長
    step = int(fs * slide)  # オーバーラップ長
    # step = segment_length - overlap_length  # ステップサイズ

    segments = []
    for i in range(0, len(signal) - segment_length + 1, step):
        segment = signal[i:i + segment_length]  # セグメントを抽出
        segments.append(segment)  # リストに追加

    return np.array(segments)


""" 正規化 """
def normalize(signal):
    """一次元信号を正規化する関数"""
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    normalized_signal = (signal - signal_min) / (signal_max - signal_min)
    return normalized_signal


""" 標準化 """
def standardize(signal):
    """一次元信号を標準化する関数"""
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    standardized_signal = (signal - signal_mean) / signal_std
    return standardized_signal



# セグメントからα, β, α/βを抽出
def get_psd_features(data):
    # 統計量
    mean = np.mean(data)
    std = np.std(data)
    maximum = np.max(data)
    minimum = np.min(data)

    # αパワー、βパワー、α/β
    freq, psd = welch(data, fs=512)
    alpha_mask = (freq >= 8) & (freq <= 12)
    beta_mask = (freq >= 13) & (freq <= 30)
    alpha_power = sum(psd[alpha_mask])
    beta_power = sum(psd[beta_mask])
    alpha_beta_ratio = alpha_power / beta_power

    return mean, std, maximum, minimum, alpha_power, beta_power, alpha_beta_ratio
    # return alpha_power, beta_power, alpha_beta_ratio


# セグメントリストから特徴量とα, β, α/βを抽出
def get_psd_features_list(data):
    mean_list = []
    std_list = []
    max_list = []
    min_list = []
    alpha_list = []
    beta_list = []
    alpha_beta_ratio_list = []

    for segment in data:
        mean, std, maximum, minimum, alpha, beta, ratio = get_psd_features(segment)
        mean_list.append(mean)
        std_list.append(std)
        max_list.append(maximum)
        min_list.append(minimum)
        alpha_list.append(alpha)
        beta_list.append(beta)
        alpha_beta_ratio_list.append(ratio)

    return np.array(mean_list), np.array(std_list), np.array(max_list), np.array(min_list), np.array(alpha_list), np.array(beta_list), np.array(alpha_beta_ratio_list)
    # return np.array(alpha_list), np.array(beta_list), np.array(alpha_beta_ratio_list)




""" 再構成相空間 """
# def reconstruct_phase_space(data, delay, embedding_dim):
#     X = np.zeros((len(data) - (embedding_dim - 1) * delay, embedding_dim))
#     for i in range(embedding_dim):
#         X[:, i] = data[i * delay:i * delay + X.shape[0]]
#     return X

def reconstruct_phase_space(data, delay, embedding_dim):
    if len(data.shape) == 2:
        # バッチデータの場合
        reconstructed_data = []
        for single_data in data:
            X = np.zeros((len(single_data) - (embedding_dim - 1) * delay, embedding_dim))
            for i in range(embedding_dim):
                X[:, i] = single_data[i * delay:i * delay + X.shape[0]]
            reconstructed_data.append(X)
        return np.array(reconstructed_data)
    else:
        # 単一データの場合
        X = np.zeros((len(data) - (embedding_dim - 1) * delay, embedding_dim))
        for i in range(embedding_dim):
            X[:, i] = data[i * delay:i * delay + X.shape[0]]
        return X

""" 再構成相空間からの特徴量抽出関数 """
def extract_features_from_phase_space(reconstructed):
    """
    再構成相空間データから特徴量を抽出する。
    :param reconstructed: 再構成相空間データ (N x embedding_dim)
    :return: 特徴量のリスト
    """
    features = []
    # 各次元ごとの統計特徴量
    for dim in range(reconstructed.shape[1]):
        dim_data = reconstructed[:, dim]
        features.append(np.mean(dim_data))      # 平均
        features.append(np.std(dim_data))       # 標準偏差
        features.append(np.min(dim_data))       # 最小値
        features.append(np.max(dim_data))       # 最大値
        features.append(skew(dim_data))         # 歪度
        features.append(kurtosis(dim_data))     # 尖度

    # 相空間全体の特徴量
    distances = np.sqrt(np.sum(np.diff(reconstructed, axis=0) ** 2, axis=1))  # 距離
    features.append(np.mean(distances))      # 平均距離
    features.append(np.std(distances))       # 距離の標準偏差
    features.append(np.min(distances))       # 最小距離
    features.append(np.max(distances))       # 最大距離
    features.append(skew(distances))         # 距離の歪度
    features.append(kurtosis(distances))     # 距離の尖度
    return features

""" 再構成位相空間特徴量生成 """
def generate_phase_space_features(X, delay_time, embedding_dim):
    all_features = []
    for segment in X:
        reconstructed = reconstruct_phase_space(segment, delay_time, embedding_dim)
        features = extract_features_from_phase_space(reconstructed)
        all_features.append(features)
    return np.array(all_features)



""" 被験者ごとにデータをセグメンテーション """
def split_data_by_subject(data_dic, WINDOW_SEC, SLIDE_SEC, fs=512):
    segmented_data_dic = {}  # 分割されたデータを格納する辞書
    for subj_num in data_dic:  # 被験者ごとに処理
        data = data_dic[subj_num]  # 被験者のデータ
        segmented_data = {}  # 分割されたデータを格納する辞書
        for label in data:  # ラベルごとに処理
            signal = data[label]  # ラベルに対応するデータ
            segmented_signal = []  # 分割されたデータを格納するリスト
            for i in range(0, int(len(signal)-fs*WINDOW_SEC+1), int(fs*SLIDE_SEC)):  # スライドさせながら分割
                segmented_signal.append(signal[i:i+fs*WINDOW_SEC])  # 分割したデータをリストに追加
            segmented_data[label] = segmented_signal  # ラベルごとに分割されたデータを格納
        segmented_data_dic[subj_num] = segmented_data  # 被験者ごとに分割されたデータを格納
    print("----------------------------------------")
    print(f"ウィンドウ: {WINDOW_SEC}秒, スライド: {SLIDE_SEC}秒でデータのセグメンテーションが完了しました。")

    return segmented_data_dic