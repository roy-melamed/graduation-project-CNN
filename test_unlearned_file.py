import librosa
import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow import keras
import utils

if __name__ == '__main__':
    nfft_frame = 500
    # hop = 290
    hop = 250
    audio_file_path = "data/S4A13411_20210613_150000_short.wav"
    label_file_path = "data/S4A13411_20210613_150000_short.txt"
    model_checkpoint_filepath = 'model_checkpoint'
    dur = 1
    y_sdb = []
    blocks = []
    labels = []
    print(audio_file_path, label_file_path)
    sr = librosa.get_samplerate(audio_file_path)
    frame_length = int(sr * dur)  # add control over hop size and frame size
    hop_length = frame_length // 2  # add control over hop size and frame size
    stream = librosa.stream(audio_file_path,  # Stream the data, working on one frame at a time
                            block_length=1,
                            frame_length=frame_length,
                            hop_length=hop_length,
                            mono=True)
    for y in stream:
        S = librosa.feature.melspectrogram(y, n_fft=nfft_frame, hop_length=hop, center=False, n_mels=90)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = S_db[6:-14, :]
        y_sdb.append(pd.DataFrame(S_db))
        # librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    del y_sdb[-1]

    block_size = len(y_sdb)

    blocks.extend(y_sdb)

    # BUILD LABELS
    labels_time_stamps = []
    with open(label_file_path) as f:
        lines = f.readlines()
    for line in lines:
        a, b = line.split('\t')[0:2]
        labels_time_stamps.append((float(a), float(b)))

    labels = []
    start, end = np.inf, np.inf

    # defining minimum thresh for each file
    min_thresh = np.inf
    for time in labels_time_stamps:
        min_thresh = min(time[1] - time[0], min_thresh)
    threshold = min_thresh

    for i in range(block_size):
        for time in labels_time_stamps:
            start = time[0] if time[0] <= (i / 2 + 1) else np.inf
            end = time[1] if time[1] >= (i / 2) else np.inf
            if start == np.inf or end == np.inf:
                continue
            break
        if (i / 2) <= start <= (i / 2 + 1) and (i / 2) <= end <= (i / 2 + 1):
            d = end - start
            if d > threshold:
                labels.append(1)
            else:
                labels.append(0)
            continue
        if start <= (i / 2) and end >= (i / 2 + 1):
            labels.append(1)
            continue
        if (i / 2) <= start <= (i / 2 + 1):
            d = (i / 2 + 1) - start
            if d > threshold:
                labels.append(1)
            else:
                labels.append(0)
            continue
        if (i / 2) <= end <= (i / 2 + 1):
            d = end - (i / 2)
            if d > threshold:
                labels.append(1)
            else:
                labels.append(0)
            continue
        labels.append(0)

    tmp = np.array([np.array(df) for df in blocks])
    reshaped_blocks = np.expand_dims(tmp, -1)
    # labels = keras.utils.to_categorical(labels, 2)
    model = keras.models.load_model(model_checkpoint_filepath)
    pred = np.argmax(model.predict(reshaped_blocks), axis=-1)
    pred = pred.reshape(-1, 1)
    cm = metrics.confusion_matrix(labels, pred[:, 0])

    utils.plot_confusion_matrix(cm, ['no bird', 'bird'])
