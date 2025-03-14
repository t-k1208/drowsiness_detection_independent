import copy
import numpy as np

import sys
sys.path.append('./') 
from need_utils import *



""" データセット作成の実行 """
##################################

VAL_SUBJECT_LIST = [
    # 0,  #1,1
    # 1,  # 1,1
    # 2,  # 2,0
    # 3,  # 0,2
    # 4,  # 1,2
    # 5,  # 2,1  (覚醒ラベルの実験の数, 眠気ラベルの実験の数)
    # 6,  # 0,1
    # 7,  # 1,1
    # 8,  # 0,1
    # 9,  # 1,1
    # 10,  # 0,2
    # 11,  # 1,0
    # 12,  # 1,0
    # 13,  # 0,2
]

SUBJECT_LIST = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ]

X_train_all_ch = []
X_test_all_ch = []
y_train_all_ch = []
y_test_all_ch = []

def load_data(  # pz_data_list[クラスのラベル:0か1][被験者]  0:11個、1:15個のデータ
    CHANNEL, test_subject_no, BANDPASS, FS, LOWCUT, HIGHCUT,
    STANDARDIZATION, NORMALIZATION, REMOVE_OUTLIERS, DOWNSAMPLING, CUTOFF, FR,
    WINDOW_SEC, OVER_SEC
):
    print("チャンネル: ", CHANNEL)

    train_subject_list = copy.deepcopy(SUBJECT_LIST)  # 深いコピーを作成
    test_subject = [train_subject_list.pop(test_subject_no)]  # コピーから要素を削除し、値を取得
    print(test_subject)  # 出力: 3
    print(train_subject_list)  # 出力: [1, 2, 3, 4, 5] (元のリストは変更されていない)


    train_data_list = create_all_dataset(  # pz_data_list[クラスのラベル:0か1][被験者]  0:11個、1:15個のデータ
        channel=CHANNEL,
        subject_list=train_subject_list,
        bandpass=BANDPASS, fs=FS, lowcut=LOWCUT, highcut=HIGHCUT,
        standardization=STANDARDIZATION,
        normalizaion=NORMALIZATION,
        remove_outliers=REMOVE_OUTLIERS,
        downsampling=DOWNSAMPLING, cutoff=CUTOFF, fr=FR,
        window_sec=WINDOW_SEC, over_sec=OVER_SEC
    )

    val_data_list = create_all_dataset(  # pz_data_list[クラスのラベル:0か1][被験者]  0:11個、1:15個のデータ
        channel=CHANNEL,
        subject_list=test_subject,
        bandpass=BANDPASS, fs=FS, lowcut=LOWCUT, highcut=HIGHCUT,
        standardization=STANDARDIZATION,
        normalizaion=NORMALIZATION,
        remove_outliers=REMOVE_OUTLIERS,
        downsampling=DOWNSAMPLING, cutoff=CUTOFF, fr=FR,
        window_sec=WINDOW_SEC, over_sec=OVER_SEC
    )

    print("ラベルが0の訓練データ数: ", len(train_data_list[0]))
    print("ラベルが1の訓練データ数: ", len(train_data_list[1]))
    print("ラベルが0の検証データ数: ", len(val_data_list[0]))
    print("ラベルが1の検証データ数: ", len(val_data_list[1]))



    train_awake = train_data_list[0]  # 訓練データ awake
    train_drowsy = train_data_list[1]  # 訓練データ drowsy

    val_awake = val_data_list[0]  # テストデータ awake
    val_drowsy = val_data_list[1]  # テストデータ drowsy


    fs = FS
    sec = WINDOW_SEC
    slide = WINDOW_SEC - OVER_SEC

    # 訓練-覚醒のセグメンテーション
    signals = train_awake
    segments_list = []  # セグメントしたものを保存
    for i in range(len(signals)):  # 被験者の実験ごとにセグメンテーション
        segments = segmentation(signals[i], fs, sec, slide)
        segments_list.append(segments)
    # awakeのデータセットを作成
    # (セグメント数, サンプリング数)に変換
    print("セグメント数: ", np.array(segments_list).shape)
    segmented_train_awake = np.array(segments_list).reshape(-1, fs * sec)  
    label_train_awake = np.array([0] * len(segmented_train_awake))  # セグメントの数だけラベルを作成
    print("訓練データ(覚醒): ", segmented_train_awake.shape)  # (3990, 1536)
    print("訓練データ(覚醒ラベル): ", label_train_awake.shape)  # (3990, )


    # 訓練-眠気のセグメンテーション
    signals = train_drowsy
    segments_list = []  # セグメントしたものを保存
    for i in range(len(signals)):  # 被験者の実験ごとにセグメンテーション
        segments = segmentation(signals[i], fs, sec, slide)
        segments_list.append(segments)
    # awakeのデータセットを作成
    # (セグメント数, サンプリング数)に変換
    segmented_train_drowsy = np.array(segments_list).reshape(-1, fs * sec)  
    label_train_drowsy = np.array([1] * len(segmented_train_drowsy))  # セグメントの数だけラベルを作成
    print("訓練データ(眠気): ", segmented_train_drowsy.shape)  # (3990, 1536)
    print("訓練データ(眠気ラベル): ", label_train_drowsy.shape)  # (3990, )
    print()


    # 検証-覚醒のセグメンテーション
    signals = val_awake
    segments_list = []  # セグメントしたものを保存
    for i in range(len(signals)):  # 被験者の実験ごとにセグメンテーション
        segments = segmentation(signals[i], fs, sec, slide)
        segments_list.append(segments)
    # awakeのデータセットを作成
    # (セグメント数, サンプリング数)に変換
    segmented_val_awake = np.array(segments_list).reshape(-1, fs * sec)  
    label_val_awake = np.array([0] * len(segmented_val_awake))  # セグメントの数だけラベルを作成
    print("検証データ(覚醒): ", segmented_val_awake.shape)  # (3990, 1536)
    print("検証データ(覚醒ラベル): ", label_val_awake.shape)  # (3990, )


    # 検証-眠気のセグメンテーション
    signals = val_drowsy
    segments_list = []  # セグメントしたものを保存
    for i in range(len(signals)):  # 被験者の実験ごとにセグメンテーション
        segments = segmentation(signals[i], fs, sec, slide)
        segments_list.append(segments)
    # awakeのデータセットを作成
    # (セグメント数, サンプリング数)に変換
    segmented_val_drowsy = np.array(segments_list).reshape(-1, fs * sec)  
    label_val_drowsy = np.array([1] * len(segmented_val_drowsy))  # セグメントの数だけラベルを作成
    print("検証データ(眠気): ", segmented_val_drowsy.shape)  # (3990, 1536)
    print("検証データ(眠気ラベル): ", label_val_drowsy.shape)  # (3990, )
    print()


    # モデルに入力する学習データセットの作成
    X_train = np.concatenate([segmented_train_awake, segmented_train_drowsy])
    y_train = np.concatenate([label_train_awake, label_train_drowsy])
    X_test = np.concatenate([segmented_val_awake, segmented_val_drowsy])
    y_test = np.concatenate([label_val_awake, label_val_drowsy])

    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    print("X_test: ", X_test.shape)
    print("y_test: ", y_test.shape)

    return X_train, y_train, X_test, y_test


# if __name__ == '__main__':

#     CHANNEL = "Pz"
#     test_subject_no = 1

#     BANDPASS = False  # True: バンドパスフィルタをかける / False: バンドパスフィルタをかけない
#     FS = 512
#     LOWCUT = 0.1
#     HIGHCUT = 40.0

#     STANDARDIZATION = True
#     NORMALIZATION = True
#     REMOVE_OUTLIERS = True

#     DOWNSAMPLING = False  # True: ダウンサンプリングを行う / False: ダウンサンプリングを行わない
#     FR = 64
#     CUTOFF = FR/2

#     WINDOW_SEC = 13  # ウィンドウサイズ(秒)
#     OVER_SEC = 12   # オーバーラップサイズ(秒)
#     ##################################

#     X_train, y_train, X_test, y_test = load_data(
#         CHANNEL, test_subject_no, BANDPASS, FS, LOWCUT, HIGHCUT,
#         STANDARDIZATION, NORMALIZATION, REMOVE_OUTLIERS, DOWNSAMPLING, CUTOFF, FR,
#         WINDOW_SEC, OVER_SEC
#     )



# 被験者非依存用のデータセットを作成する関数
""" データセットの作成 """
def create_dataset_for_independent(channel=None,  # 生体信号チャンネル
                    #    subject_list=None,  # 訓練に使う被験者リスト
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
    file_path = "../2class.json"
    # file_path = "/Users/tk/Documents/法政大学/lab/drowsy/drowsy_eeg_eog/dataset_independent/2class.json"
    # JSONファイルを開いて読み込む
    with open(file_path, 'r') as f:
        files_two_class = json.load(f)
    class_filenames = [files_two_class[0], files_two_class[1]]


    """ ラベルファイルの読み込み """
    file_path = "../../dataset/KSS.txt"
    # file_path = "/Users/tk/Documents/法政大学/lab/drowsy/DROZY/KSS.txt"
    with open(file_path, "r") as f: # ラベルファイルの読み込み
        original_labels = f.read().split("\n")


    data_list = []  # クラスごとのデータリスト
    class_dic = {}  # クラス内のデータをまとめる辞書
    for state_index, filenames in tqdm(enumerate(class_filenames)):
        # class_data_list = []  # クラス内のデータリスト
        for filename in filenames:
            
            """ 被験者No.と、ラベルNo.の取得 """
            if sys.platform == 'win32':  # 'win32'はWindowsの場合
                # print("Windows用の処理")
                nums = filename.split("\\")[-1].split(".")[0].split("-")  # windows
            elif sys.platform == 'darwin':  # 'darwin'はMacの場合
                # print("Mac用の処理")
                nums = filename.split("/")[-1].split(".")[0].split("-")  # mac [被験者番号（1~14）, 実験回数（1~3）]
            num_subject, num_experiment = int(nums[0])-1, int(nums[-1])-1  # KSSスコアファイル参照のため被験者No.と実験No.からマイナス1

            label = original_labels[num_subject].split(" ")[num_experiment]
            # print(label)  # ラベルの確認
            # print(num_subject, label)

            """ edfファイルの読み込み """
            # print("ファイル名", filename)  # ../DROZY/psg/2-1.edf
            # edf = pyedflib.EdfReader(os.path.join("..", filename))  # edfファイルの読み込み(filename)
            file_dir = "../../psg"
            # file_dir = "/Users/tk/Documents/法政大学/lab/drowsy/DROZY/psg"
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

            # class_data_list.append(data)  # クラス内のデータリスト
            sub_trial_num = filename.split("/")[-1].split(".")[0] 
            if sub_trial_num not in class_dic:  # キーが存在しない場合
                class_dic[sub_trial_num] = {} # 空の辞書で初期化 <= 追加
                print(f"{sub_trial_num} をキーに追加しました")
            class_dic[f"{sub_trial_num}"][state_index] = data
            edf.close()
        # data_list.append(class_data_list)  # クラスごとのデータリストをまとめる

    # return data_list
    return class_dic