# drowsiness_detection_independent

# Structure of Directory

drowsy/
├━━ DROZY/
│  　├━━ psg/1-1.edf ...  # 実験ファイル
│  　├━━ KSS.txt  # 実験のKSSラベル
│━━ drowsy_eeg_eog/  # 被験者非依存のファイルはここのディレクトリ
│   　├━━ dataset_independent/
│  　 │      ├━━ 2class.json  # 被験者非依存用に実験ファイルを分割したファイル名を格納
│　　　├━━ independent_project/
│　　　│     ├━━ eogv.ipynb
│     │     ├━━ load_data.py
│     │     ├━━ main_plot.py
│     │     ├━━ need_utils.py
│     │     ├━━ preprocessing.py
