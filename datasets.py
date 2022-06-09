import librosa
import numpy as np
import librosa.display
import pandas as pd
import os


class DataSets:
    def __init__(self, audio_files_path, label_files_path, dur=1, thresh=0.05):
        self.audio_files_path = audio_files_path
        self.label_files_path = label_files_path
        # self.audio_file_path = r"audacity-Yuri\S4A13335_20210408_060000.wav"
        # self.labels_file_path = r"audacity-Yuri\S4A13335_20210408_060000.txt"
        self.threshold = thresh  # add control over thresh
        self.dur = dur
        self.blocks = []
        self.labels = []
        self.image_size = ()

    def build(self):  # build blocks and labels
        audio_file_paths = []
        label_file_paths = []
        for filename in os.listdir(self.audio_files_path):
            audio_file_paths.append(os.path.join(self.audio_files_path, filename))
        for filename in os.listdir(self.label_files_path):
            label_file_paths.append(os.path.join(self.label_files_path, filename))
        augment_time_shifts = [0, 0.2, 0.4]
        # augment_shifts = [0]
        # nfft_frame = 580
        nfft_frame = 500
        # hop = 290
        hop = 250

        # BUILD BLOCKS
        # Time Shift augmentation
        print('Starting Time Shift augmentation...')
        for audio_file_path, label_file_path in zip(audio_file_paths, label_file_paths):
            y_sdb = []
            print(audio_file_path, label_file_path)
            sr, frame_length, hop_length = self.get_file_details(audio_file_path)
            for augment_time_shift in augment_time_shifts:
                stream = librosa.stream(audio_file_path,  # Stream the data, working on one frame at a time
                                        block_length=1,
                                        frame_length=frame_length,
                                        hop_length=hop_length,
                                        mono=True,
                                        offset=augment_time_shift)
                for y in stream:
                    S = librosa.feature.melspectrogram(y, n_fft=nfft_frame, hop_length=hop, center=False, n_mels=90)
                    S_db = librosa.power_to_db(S, ref=np.max)
                    S_db = S_db[6: -14, :]
                    y_sdb.append(pd.DataFrame(S_db))
                    # librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
                del y_sdb[-1]

            block_size = len(y_sdb) // len(augment_time_shifts)

            self.blocks.extend(y_sdb)
            self.image_size = self.blocks[0].shape + (1,)

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
            self.threshold = min_thresh

            # create and shift labels' time for data augmentation
            for augment_time_shift in augment_time_shifts:
                for i in range(block_size):  # i/2 represents the time in seconds
                    for time in labels_time_stamps:
                        start = time[0] if time[0] <= (i / 2 + 1) else np.inf
                        start = 0 if start - augment_time_shift < 0 else start - augment_time_shift
                        end = time[1] if time[1] >= (i / 2) else np.inf
                        end = end - augment_time_shift if end > augment_time_shift else 0.01
                        if start == np.inf or end == np.inf or end <= 0:
                            continue
                        break
                    if (i / 2) <= start <= (i / 2 + 1) and (i / 2) <= end <= (i / 2 + 1):
                        d = end - start
                        if d > self.threshold:
                            labels.append(1)
                        else:
                            labels.append(0)
                        continue
                    if start <= (i / 2) and end >= (i / 2 + 1):
                        labels.append(1)
                        continue
                    if (i / 2) <= start <= (i / 2 + 1):
                        d = (i / 2 + 1) - start
                        if d > self.threshold:
                            labels.append(1)
                        else:
                            labels.append(0)
                        continue
                    if (i / 2) <= end <= (i / 2 + 1):
                        d = end - (i / 2)
                        if d > self.threshold:
                            labels.append(1)
                        else:
                            labels.append(0)
                        continue
                    labels.append(0)

            self.labels.extend(labels)

        # BUILD BLOCKS
        # Stretch augmentation
        print('Starting Stretch augmentation...')
        stretch_rate = 0.9
        for audio_file_path, label_file_path in zip(audio_file_paths, label_file_paths):
            y_sdb = []
            print(audio_file_path, label_file_path)
            sr, frame_length, hop_length = self.get_file_details(audio_file_path)
            stream = librosa.stream(audio_file_path,  # Stream the data, working on one frame at a time
                                    block_length=1,
                                    frame_length=frame_length,
                                    hop_length=hop_length,
                                    mono=True)
            image_slicing_size = int(sr*(1-stretch_rate) / 2)
            for y in stream:
                y_slow = librosa.effects.time_stretch(y[image_slicing_size:-(image_slicing_size + 1)], stretch_rate)
                S = librosa.feature.melspectrogram(y_slow, n_fft=nfft_frame, hop_length=hop, center=False, n_mels=90)
                S_db = librosa.power_to_db(S, ref=np.max)
                S_db = S_db[6:-14, :]
                y_sdb.append(pd.DataFrame(S_db))
                # librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
            del y_sdb[-1]

            block_size = len(y_sdb)

            self.blocks.extend(y_sdb)

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
            self.threshold = min_thresh

            for i in range(block_size):
                for time in labels_time_stamps:
                    start = time[0] if time[0] <= (i / 2 + 1) else np.inf
                    end = time[1] if time[1] >= (i / 2) else np.inf
                    if start == np.inf or end == np.inf:
                        continue
                    break
                if (i / 2) <= start <= (i / 2 + 1) and (i / 2) <= end <= (i / 2 + 1):
                    d = end - start
                    if d > self.threshold:
                        labels.append(1)
                    else:
                        labels.append(0)
                    continue
                if start <= (i / 2) and end >= (i / 2 + 1):
                    labels.append(1)
                    continue
                if (i / 2) <= start <= (i / 2 + 1):
                    d = (i / 2 + 1) - start
                    if d > self.threshold:
                        labels.append(1)
                    else:
                        labels.append(0)
                    continue
                if (i / 2) <= end <= (i / 2 + 1):
                    d = end - (i / 2)
                    if d > self.threshold:
                        labels.append(1)
                    else:
                        labels.append(0)
                    continue
                labels.append(0)
            self.labels.extend(labels)

        # BUILD BLOCKS
        # Pitch Shift augmentation
        print('Starting Pitch Shift augmentation...')
        for audio_file_path, label_file_path in zip(audio_file_paths, label_file_paths):
            y_sdb = []
            print(audio_file_path, label_file_path)
            sr, frame_length, hop_length = self.get_file_details(audio_file_path)
            stream = librosa.stream(audio_file_path,  # Stream the data, working on one frame at a time
                                    block_length=1,
                                    frame_length=frame_length,
                                    hop_length=hop_length,
                                    mono=True)
            for y in stream:
                S = librosa.feature.melspectrogram(y, n_fft=nfft_frame, hop_length=hop, center=False, n_mels=90)
                S_db = librosa.power_to_db(S, ref=np.max)
                S_db = S_db[4:-16, :]
                y_sdb.append(pd.DataFrame(S_db))
                # librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
            del y_sdb[-1]

            block_size = len(y_sdb)

            self.blocks.extend(y_sdb)

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
            self.threshold = min_thresh

            for i in range(block_size):
                for time in labels_time_stamps:
                    start = time[0] if time[0] <= (i / 2 + 1) else np.inf
                    end = time[1] if time[1] >= (i / 2) else np.inf
                    if start == np.inf or end == np.inf:
                        continue
                    break
                if (i / 2) <= start <= (i / 2 + 1) and (i / 2) <= end <= (i / 2 + 1):
                    d = end - start
                    if d > self.threshold:
                        labels.append(1)
                    else:
                        labels.append(0)
                    continue
                if start <= (i / 2) and end >= (i / 2 + 1):
                    labels.append(1)
                    continue
                if (i / 2) <= start <= (i / 2 + 1):
                    d = (i / 2 + 1) - start
                    if d > self.threshold:
                        labels.append(1)
                    else:
                        labels.append(0)
                    continue
                if (i / 2) <= end <= (i / 2 + 1):
                    d = end - (i / 2)
                    if d > self.threshold:
                        labels.append(1)
                    else:
                        labels.append(0)
                    continue
                labels.append(0)
            self.labels.extend(labels)

    def get_file_details(self, audio_file_path):
        sr = librosa.get_samplerate(audio_file_path)
        frame_length = int(sr * self.dur)  # add control over hop size and frame size
        hop_length = frame_length // 2  # add control over hop size and frame size
        return sr, frame_length, hop_length
