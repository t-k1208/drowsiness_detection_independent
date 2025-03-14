

import os
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
from sklearn.preprocessing import StandardScaler

# from need_utils import *
from preprocessing import *


def get_data(filename, label, cutoff_freq=[0.5, 40.0], method=None):
    """
    引数: ファイル名, KSSラベル, カットオフ周波数, 外れ値除去の手法(iqr, robust, None), 
            プロットするスタート時間, プロットする終わり時間, ダウンサンプリング
    返り値: なし, 信号のプロットのみ
    """
    data, sample_frequency = read_edf_file(filename)

    """ 標準化 """
    data = standardize(data)

    """ 正規化 """
    data = normalize(data)

    """ ロバスト統計 """
    data = replace_outliers_with_robust_statistics(data)

    """ バンドパスフィルタを適用する(0.5~70.0Hz) """
    data = apply_bandpass_filter(data, sample_frequency, low_cut=cutoff_freq[0], high_cut=cutoff_freq[1])

    return data


def plot_data(filename, label, cutoff_freq=[0.5, 70.0], method=None, start=0, end=10, downsampling=False):
    """
    引数: ファイル名, KSSラベル, カットオフ周波数, 外れ値除去の手法(iqr, robust, None), 
            プロットするスタート時間, プロットする終わり時間, ダウンサンプリング
    返り値: なし, 信号のプロットのみ
    """
    num_subject, num_experiment = get_subject_experiment_numbers(filename)
    data, sample_frequency = read_edf_file(filename)
    print(f"sampling freq: {sample_frequency}")

    """ 標準化 """
    data = standardize(data)

    """ 正規化 """
    data = normalize(data)

    """ ロバスト統計 """
    data = replace_outliers_with_robust_statistics(data)

    """ バンドパスフィルタを適用する(0.5~70.0Hz) """
    data = apply_bandpass_filter(data, sample_frequency, low_cut=cutoff_freq[0], high_cut=cutoff_freq[1])

    start_second = start
    # end_second = len(preprocessed_data_robust) / sample_frequency
    end_second = end
    duration = end_second - start_second

    data = data[start_second*sample_frequency:end_second*sample_frequency]
    time_axis = np.arange(start_second, end_second, 1 / sample_frequency)

    if downsampling == True:
        new_fs = 64  # 新しいサンプリングレート
        decimation_factor = sample_frequency // new_fs  # デシメーションファクターを計算

        # 信号をデシメーション（ダウンサンプリング）する
        data = signal.decimate(data, decimation_factor, zero_phase=True)
        time_axis = np.arange(start_second, end_second, 1/new_fs)

    plt.figure(figsize=(20, 3))
    # plt.figure(figsize=(10, 3))
    plt.plot(time_axis, data, label=f"{method}", alpha=1.0)
    # plt.title(f"EEG signal", fontsize=18)
    # plt.xlabel("Time (s)", fontsize=16)
    # plt.ylabel("Amplitude", fontsize=16)
    # plt.legend(loc="upper right")
    plt.grid(True)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=16)
    plt.show()

def get_subject_experiment_numbers(filename):
    """
    引数: 読み込むファイル名
    返り値: 被験者番号(0~13), 実験No.(0~2)
    """
    if sys.platform == 'win32':
        nums = filename.split("\\")[-1].split(".")[0].split("-")
    elif sys.platform == 'darwin':
        nums = filename.split("/")[-1].split(".")[0].split("-")

    num_subject = int(nums[0]) - 1
    num_experiment = int(nums[-1]) - 1

    return num_subject, num_experiment

def read_edf_file(filename):
    """
    引数: 読み込むファイル名
    返り値: 信号データ, サンプリング周波数
    """
    try:
        edf = pyedflib.EdfReader(filename)
    except OSError as e:
        if "file has already been opened" in str(e):
            print(f"ファイル '{filename}' はすでに開いています。閉じてから再度開いてください。")
            return None, None

    channel_index = edf.getSignalLabels().index("EOG-H")  # EOG-H, EOG-V, ECG, EMG, Fz, Pz, Cz, C3, C4
    sample_frequency = int(edf.getSampleFrequency(channel_index))
    data = edf.readSignal(channel_index)
    edf.close()

    return data, sample_frequency



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




import json

""" データセットの作成 """
def create_all_dataset(channel=None,  # 生体信号チャンネル
                       subject_list=None,  # 訓練に使う被験者リスト
                       bandpass=None, fs=512, lowcut=None, highcut=None,  # バンドパスフィルタ
                       standardization=None,  # 標準化
                       normalizaion=None,  # 正規化
                       remove_outliers=None,  # 外れ値除去
                       downsampling=False, cutoff=None, fr=None,
                       window_sec=None, over_sec=None):
    # filenames = glob.glob("../DROZY/psg/*.edf")  # 読み込むファイル名フォーマット
    # print(filenames)

    """ クラスに分けた被験者ファイ名の読み込み"""
    # 読み込むJSONファイルのパス
    file_path = "./2class.json"
    # file_path = "/Users/tk/Documents/法政大学/lab/drowsy/drowsy_eeg_eog/dataset_independent/2class.json"
    # JSONファイルを開いて読み込む
    with open(file_path, 'r') as f:
        files_two_class = json.load(f)
    class_filenames = [files_two_class[0], files_two_class[1]]


    """ ラベルファイルの読み込み """
    file_path = "../dataset/KSS.txt"
    # file_path = "/Users/tk/Documents/法政大学/lab/drowsy/DROZY/KSS.txt"
    with open(file_path, "r") as f: # ラベルファイルの読み込み
        original_labels = f.read().split("\n")


    data_list = []  # クラスごとのデータリスト
    for filenames in tqdm(class_filenames):
        class_data_list = []  # クラス内のデータリスト
        for filename in filenames:
            
            """ 被験者No.と、ラベルNo.の取得 """
            if sys.platform == 'win32':  # 'win32'はWindowsの場合
                # print("Windows用の処理")
                nums = filename.split("\\")[-1].split(".")[0].split("-")  # windows
            elif sys.platform == 'darwin':  # 'darwin'はMacの場合
                # print("Mac用の処理")
                nums = filename.split("/")[-1].split(".")[0].split("-")  # mac [被験者番号（1~14）, 実験回数（1~3）]
            num_subject, num_experiment = int(nums[0])-1, int(nums[-1])-1  # KSSスコアファイル参照のため被験者No.と実験No.からマイナス1

            """ 指定した被験者No.のデータを読み込み"""
            if num_subject in subject_list:
                label = original_labels[num_subject].split(" ")[num_experiment]
                # print(label)  # ラベルの確認
                # print(num_subject, label)

                """ edfファイルの読み込み """
                print(filename)  # ../DROZY/psg/2-1.edf
                # edf = pyedflib.EdfReader(os.path.join("..", filename))  # edfファイルの読み込み(filename)
                # file_dir = "/Users/tk/Documents/法政大学/lab/drowsy/DROZY/psg"
                file_dir = "../../dataset/psg"
                edf = pyedflib.EdfReader(os.path.join(file_dir, filename.split("/")[-1]))  # edfファイルの読み込み

                """ データの読み込み """
                # ECG信号が含まれるチャンネルの名前を指定します
                # チャンネルのインデックスを取得します
                channel_index = edf.getSignalLabels().index(channel)
                data = edf.readSignal(channel_index)  # 0:Fz, 11:ECG
                edf.close()

                """ 処理ver1 """
                # バンドパス
                if bandpass:
                    data = apply_bandpass_filter(data, fs, lowcut, highcut)
                    # print("バンドパスフィルタを適用しました")

                # 標準化
                """ 標準化 """
                if standardization:
                    data = standardize(data)

                # 正規化
                """ 正規化 """
                if normalizaion:
                    data = normalize(data)
                    

                # 外れ値の処理
                """ ロバスト統計 """
                if remove_outliers:
                    data = replace_outliers_with_robust_statistics(data)
                    # print("外れ値を除去しました")

                # """ 処理ver2 """
                # # 正規化
                # if normalizaion == True:
                #     data = normalize(data)
                #     print("正規化しました")

                # # 標準化
                # if standardization == True:
                #     scaler = StandardScaler()
                #     data = scaler.fit_transform(data.reshape(-1, 1))
                #     print("標準化しました")

                # # 外れ値の処理
                # if remove_outliers == True:
                #     data = replace_outliers_with_iqr(data)
                #     print("外れ値を除去しました")

                class_data_list.append(data)  # クラス内のデータリスト
            # edf.close()
        data_list.append(class_data_list)  # クラスごとのデータリストをまとめる

    return data_list



# ファイル名を指定して該当するサンプルを取り出す関数
def get_samples_by_filename(filename, X):
    # file_path = "/Users/tk/Documents/法政大学/lab/drowsy/drowsy_eeg_eog/dataset_independent/2class.json"
    file_path = "./2class.json"
    # JSONファイルを開いて読み込む
    with open(file_path, 'r') as f:
        files_two_class = json.load(f)
    class_filenames = [files_two_class[0], files_two_class[1]]
    # ファイル名だけを取り出してを1次元リストに変換
    class_filenames = [item.replace('../dataset/psg/', '') for sublist in class_filenames for item in sublist]
    if filename not in class_filenames:
        raise ValueError(f"File name {filename} not found in the list.")
    
    samples_per_file = 588
    index = class_filenames.index(filename)  # 指定したファイル名のインデックスを取得
    start_idx = index * samples_per_file  # サンプルの開始インデックス
    end_idx = start_idx + samples_per_file  # サンプルの終了インデックス
    
    return X[start_idx:end_idx]  # 指定した範囲のサンプルを取得


# ファイル名を指定して該当するサンプルを取り出す関数
def get_lyap_results_by_filename(filename, lyap_results):
    # file_path = "/Users/tk/Documents/法政大学/lab/drowsy/drowsy_eeg_eog/dataset_independent/2class.json"
    file_path = "./2class.json"
    # JSONファイルを開いて読み込む
    with open(file_path, 'r') as f:
        files_two_class = json.load(f)
    class_filenames = [files_two_class[0], files_two_class[1]]
    # ファイル名だけを取り出してを1次元リストに変換
    class_filenames = [item.replace('../dataset/psg/', '') for sublist in class_filenames for item in sublist]
    # 取り除きたい値のリスト (テスト被験者のファイル名)
    values_to_remove = ["2-1.edf", "2-2.edf"]
    # 指定の値を取り除く (学習被験者のファイル名のみにする)
    class_filenames = [item for item in class_filenames if item not in values_to_remove]

    # 学習被験者の場合
    if filename in class_filenames:
        samples_per_file = 588
        index = class_filenames.index(filename)  # 指定したファイル名のインデックスを取得
        start_idx = index * samples_per_file  # サンプルの開始インデックス
        end_idx = start_idx + samples_per_file  # サンプルの終了インデックス

    # テスト被験者の場合
    if filename == "2-1.edf":
        start_idx = 0
        end_idx = 588
    elif filename == "2-2.edf":
        start_idx = 588
        end_idx = 588 + 588
    
    return lyap_results[start_idx:end_idx]  # 指定した範囲のサンプルを取得


# ファイル名を指定して該当するサンプルを取り出す関数
def get_subject_data_by_filename(filename, dataset, win_sec):
    file_path = "./2class.json"
    # JSONファイルを開いて読み込む
    with open(file_path, 'r') as f:
        files_two_class = json.load(f)
    class_filenames = [files_two_class[0], files_two_class[1]]
    # ファイル名だけを取り出してを1次元リストに変換
    class_filenames = [item.replace('../dataset/psg/', '') for sublist in class_filenames for item in sublist]
    # 取り除きたい値のリスト (テスト被験者のファイル名)
    values_to_remove = ["2-1.edf", "2-2.edf"]
    # 指定の値を取り除く (学習被験者のファイル名のみにする)
    class_filenames = [item for item in class_filenames if item not in values_to_remove]

    segment_num = 600 - win_sec + 1  # セグメント数

    # 学習被験者の場合
    if filename in class_filenames:
        samples_per_file = segment_num
        index = class_filenames.index(filename)  # 指定したファイル名のインデックスを取得
        start_idx = index * samples_per_file  # サンプルの開始インデックス
        end_idx = start_idx + samples_per_file  # サンプルの終了インデックス

    # テスト被験者の場合
    if filename == "2-1.edf":
        start_idx = 0
        end_idx = segment_num
    elif filename == "2-2.edf":
        start_idx = segment_num
        end_idx = segment_num + segment_num
    
    return dataset[start_idx:end_idx]  # 指定した範囲のサンプルを取得




""" セグメントされた辞書内segmentsの取り出し """
def get_segments_from_split_data(split_data_dic, subj_num):
    """
    入力:
        split_data_dic <-- preprocessing.split_data_by_subject(data_dic, WINDOW_SEC, SLIDE_SEC)
        subj_num <-- "1-1"
    出力:
        np.array(セグメント数, データポイント数) <-- 被験者の実験ラウンドにおけるセグメントされた行列データ
    """
    label = list(split_data_dic[subj_num])  # ラベルの取得
    segments = np.array(list(split_data_dic[subj_num].values())[0])  # セグメントされたデータの取得
    print("----------------------------------------")
    print("セグメントされたデータの取り出しが完了しました。")

    return segments  # セグメントされたデータを返す --> (セグメント数, サンプル数)

# セグメントされたデータの取り出し（例）
# segments = get_segments_from_split_data(split_data_dic, "1-1")
# print("セグメントされたデータの形状: ", segments.shape)  # セグメントされたデータの形状を表示 --> (セグメント数, サンプル数)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(20, 3))
# plt.plot(segments[0])
# plt.show()
""" セグメントされた辞書内segmentsの取り出し """




""" 訓練データとテストデータの作成 """
def make_train_test_dataset(split_data_dic, test_subj_list):
    """
    入力:
        split_data_dic <-- preprocessing.split_data_by_subject(data_dic, WINDOW_SEC, SLIDE_SEC)
        test_subj_list <-- ["1-1", "1-3"]   # テスト被験者
    出力:
        X_train_segments, y_train_segments, X_test_segments, y_test_segments
            # セグメントの形状: (実験数, セグメント数, サンプル数)
            # 訓練データセグメント:  (25, 588, 6656)
            # 訓練ラベルセグメント:  (25, 588)
            # テストデータセグメント:  (1, 588, 6656)
            # テストラベルセグメント:  (1, 588)
        """
    X_train_segments = []  # 訓練データ
    y_train_segments = []  # 訓練ラベル
    X_test_segments = []  # テストデータ
    y_test_segments = []  # テストラベル
    for subj_num in split_data_dic:  # 被験者ごとに処理  (例: "1-1")
        label = list(split_data_dic[subj_num].keys())[0]  # ラベル  (例: 0)
        segments = np.array(split_data_dic[subj_num][label])  # セグメントされたデータ  (例: (セグメント数, サンプル数))
        if subj_num in test_subj_list:  # テストデータの場合  (例: ["1-1", "1-3"])
            X_test_segments.append(segments)  # テストデータに追加  
            y_test_segments.append(np.array([label] * len(segments)))  # テストラベルに追加   
        else:  # 訓練データの場合
            X_train_segments.append(segments)  # 訓練データに追加
            y_train_segments.append(np.array([label]*len(segments)))  # 訓練ラベルに追加
    X_train_segments = np.array(X_train_segments)  # 訓練データ
    y_train_segments = np.array(y_train_segments)  # 訓練ラベル
    X_test_segments = np.array(X_test_segments)  # テストデータ
    y_test_segments = np.array(y_test_segments)  # テストラベル
    print("セグメントの形状: (実験数, セグメント数, サンプル数)")
    print("訓練データセグメント: ", X_train_segments.shape)
    print("訓練ラベルセグメント: ", y_train_segments.shape)
    print("テストデータセグメント: ", X_test_segments.shape)
    print("テストラベルセグメント: ", y_test_segments.shape)

    return X_train_segments, y_train_segments, X_test_segments, y_test_segments