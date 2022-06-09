from CNN import CNN
from NN import NN
from datasets import DataSets
from sklearn import metrics
import numpy as np
import pickle
import utils
import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

if __name__ == '__main__':
    create_pickle_flag = False
    save_model_flag = True
    train_flag = True
    load_model_flag = False
    k_fold_cv_flag = False
    model_checkpoint_filepath = 'model_checkpoint3'
    # import datasets
    if create_pickle_flag:
        data_set = DataSets("data/audio", "data/labels")
        data_set.build()
        with open('pickles/blocks_mel_70_175_augments_17.pkl', 'wb') as f:
            pickle.dump(data_set.blocks, f)
            data_set_blocks = data_set.blocks
        with open('pickles/labels_mel_70_175_augments_17.pkl', 'wb') as f:
            pickle.dump(data_set.labels, f)
            data_set_labels = data_set.labels
        del data_set
    else:
        with open('pickles/blocks_mel_70_175_augments_17.pkl', 'rb') as f:
            data_set_blocks = pickle.load(f)
        with open('pickles/labels_mel_70_175_augments_17.pkl', 'rb') as f:
            data_set_labels = pickle.load(f)
    if train_flag:
        # data_set_blocks = [data_set_blocks[i].values.flatten() for i in range(len(data_set_blocks))]  # for NN
        train_batch_step = 64
        valid_batch_step = 32
        train_dataset, valid_dataset, \
            test_dataset, train_labels, \
            valid_labels, test_labels = utils.get_3_datasets(train_size=0.7, blocks=data_set_blocks, labels=data_set_labels)
        cnn = CNN(train_dataset[0].shape)
        # nn = NN(train_dataset[0].shape)
        if load_model_flag:
            cnn.model = keras.models.load_model(model_checkpoint_filepath)
            # nn.model = keras.models.load_model(model_checkpoint_filepath)
        else:
            cnn.build()
            # nn.build()

        es = EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=1)
        mcp = ModelCheckpoint(model_checkpoint_filepath, save_best_only=True, monitor='val_loss', mode='min')
        if not k_fold_cv_flag:
            history = cnn.model.fit(x=train_dataset, y=train_labels, validation_data=(valid_dataset, valid_labels),
                                    batch_size=64, epochs=200, callbacks=[es, mcp] if save_model_flag else [es])
            # history = nn.model.fit(x=train_dataset, y=train_labels, validation_data=(valid_dataset, valid_labels),
            #                        batch_size=32, epochs=200, callbacks=[es, mcp] if save_model_flag else [es])

            utils.plot_history(history)

            print("testing...")
            model = mcp.model if save_model_flag else cnn.model
            # model = mcp.model if save_model_flag else nn.model
            model.summary()
            pred = np.argmax(model.predict(test_dataset), axis=-1)
            pred = pred.reshape(-1, 1)
            test_labels = [np.argmax(y, axis=None, out=None) for y in test_labels]
            cm = metrics.confusion_matrix(test_labels, pred[:, 0])

            utils.plot_confusion_matrix(cm, ['no bird', 'bird'])
        else:
            utils.my_k_fold(train_dataset, train_labels, test_dataset, test_labels, cnn, num_folds=10, epochs=20)
            # utils.my_k_fold(train_dataset, train_labels, test_dataset, test_labels, nn, num_folds=10, epochs=20)
