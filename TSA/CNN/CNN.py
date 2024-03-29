from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from TSA.CNN.data_handler import load_data
from sklearn import metrics
from TSA.Preproc import Preproc
import numpy as np


def cnn_make_predict_lable(pred):
    labels = [0, 1]
    arglist = []
    # Swap from [0.423, 0.677] -> lable = 1
    for p in pred:
        arglist.append(labels[np.argmax(p)])

    return arglist


def normalize_lables(y_test):
    y_normalized = []
    for y in y_test:
        if y[0] == 1:
            y_normalized.append(0)
        else:
            y_normalized.append(1)
    return y_normalized


def train_CNN():
    # Load data
    pp = Preproc.Preproc()
    pp.loadCsv("TSA/datasets/SemEval/4A-English/", "SemEval.csv")
    # pp.loadCsv("TSA/datasets/STS/", "preproc_STS.csv")
    # pp.clean_data()
    df = pp.get_twitter_df()

    x, y, vocabulary, vocabulary_inv = load_data(df)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    print(y_test.shape)

    sequence_length = x.shape[1]
    vocabulary_size = len(vocabulary_inv)
    embedding_dim = 128
    filter_sizes = [3, 4, 5]
    num_filters = 128
    drop = 0.5

    epochs = 100
    batch_size = 64

    print("Building model")
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=2, activation='softmax')(dropout)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)

    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='auto')

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1, mode='auto')

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    print("Training")
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint, earlystop],
              validation_data=(X_test, y_test))  # starts training

    # Make prediction so we can take out desired metrics
    prediction = model.predict(X_test)

    target_names = ['Negative', 'Positive']

    print(metrics.classification_report(normalize_lables(y_test), cnn_make_predict_lable(prediction),
                                        target_names=target_names, digits=3))

    # To save the trained model
    # model_json = model.to_json()
    # with open('../../../TrainedModels/CNN_base_SemEval.json', 'w') as json_file:
    #     json_file.write(model_json)
    #
    # model.save_weights('../../../TrainedModels/CNN_base_SemEval_w.h5')

    # with open('../../TrainedModels/CNN_base_STS.json', 'w') as json_file:
    #    json_file.write(model_json)

    # model.save_weights('../../TrainedModels/CNN_base_STS_w.h5')
