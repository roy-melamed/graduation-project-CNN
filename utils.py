from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
import keras.utils
import numpy as np
import matplotlib.pyplot as plt
import itertools


def get_3_datasets(blocks, labels, train_size=0.75):
    num_classes = 2
    # reshaped_blocks = np.array(blocks)
    tmp = np.array([np.array(df) for df in blocks])
    reshaped_blocks = np.expand_dims(tmp, -1)
    x_train, x_test, y_train, y_test = train_test_split(reshaped_blocks, labels, train_size=train_size,
                                                        random_state=4, stratify=labels)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.3, random_state=4, stratify=y_test)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    # y_train = np.array(y_train)
    # y_val = np.array(y_val)
    # y_test = np.array(y_test)
    return x_train, x_val, x_test, y_train, y_val, y_test


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap='Blues'):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f'{title}, accuracy = {round(((cm[0, 0] + cm[1, 1]) / sum(sum(cm))) * 100, 2)}%')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=-1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if (i == 0 and j == 1) or (i == 1 and j == 0):
            plt.text(j, i, f'{cm[i, j]}\n{round( (cm[i , j] / sum(cm[:, j])*100), 2)}%', horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
            continue
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_history(hist):
    plt.figure(1)
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.figure(2)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def my_k_fold(train_dataset, train_labels, test_dataset, test_labels, cnn, num_folds=5, epochs=20):
    # Define per-fold score containers
    acc_per_fold = []
    fp_per_fold = []  # false positive

    # Merge inputs and labels
    inputs = np.concatenate((train_dataset, test_dataset), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(inputs, labels):
        cnn.model.fit(x=inputs[train], y=labels[train], batch_size=32, epochs=epochs)

        # Generate generalization metrics
        pred = np.argmax(cnn.model.predict(inputs[test]), axis=-1)
        pred = pred.reshape(-1, 1)
        cm = metrics.confusion_matrix([np.argmax(y, axis=None, out=None) for y in labels[test]], pred[:, 0])
        score = round(((cm[0, 0] + cm[1, 1]) / sum(sum(cm))) * 100, 2)
        fp = round((cm[0, 1] / sum(cm[:, 1])*100), 2)

        print(f'for fold {fold_no}: False positive of {fp}%, Accuracy {score}%')

        acc_per_fold.append(score)
        fp_per_fold.append(fp)

        # Increase fold number
        fold_no = fold_no + 1
        cnn.build()  # reset model

    # == Provide average scores ==
    with open('KfoldCNN.txt', 'w') as file:
        file.writelines('------------------------------------------------------------------------\n')
        file.writelines('Score per fold\n')
        for i in range(0, len(acc_per_fold)):
            file.writelines('------------------------------------------------------------------------\n')
            file.writelines(f'> Fold {i + 1} - False Positive: {fp_per_fold[i]}% - Accuracy: {acc_per_fold[i]}%\n')
        file.writelines('------------------------------------------------------------------------\n')
        file.writelines('Average scores for all folds:\n')
        file.writelines(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})\n')
        file.writelines(f'> False Positive: {np.mean(fp_per_fold)} (+- {np.std(fp_per_fold)})\n')
        file.writelines('------------------------------------------------------------------------\n')
