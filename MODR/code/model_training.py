# import the necessary packages
import os, shutil, pickle, cv2, errno, time
from tensorflow.keras import applications
import tensorflow.keras.applications as tf_applications
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, GlobalMaxPooling2D, AveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from math import ceil, floor
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from helper_functions.custom_loss_functions import get_weighted_loss, weighted_categorical_crossentropy
from helper_functions.custom_accuracy_functions import count_ones, fbeta, precision, recall, specificity


def plot_history(x1, x2, t, xlabel, ylabel, legend, pth):
    plt.figure(dpi=400)
    plt.plot(x1, 'b')
    plt.plot(x2, 'c')
    plt.title(t)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend, loc='upper left')
    plt.savefig(pth)


def get_class_weights(data, columns, cw_setting):
    if cw_setting is None:
        print('Class Weights are set to None.')
        return None
    elif cw_setting == 'auto':
        print('Class Weights are set to "auto".')
        return 'auto'
    elif cw_setting == 'mc':
        # TODO TEMPORARY - INPLACE to see if weights can be better balanced
        # if 'BOV' in columns:
        #     data[['BOV', 'BCP']] = data[['BOV', 'BCP']]*2
        ################################################################################################
        summed_arr = np.sum(data[columns].values, axis=0)
        class_weights = np.max(summed_arr) / summed_arr
        # return {idx: w for idx, w in zip(range(len(class_weights)), class_weights)}
        return class_weights
    elif cw_setting == 'ml':
        number_dim = np.shape(data[columns].values)[1]
        weights = np.empty([number_dim, 2])
        for i in range(number_dim):
            weights[i] = compute_class_weight('balanced', [0., 1.], data[columns].values[:, i])
        return weights
    else:
        raise ValueError("Specified class weight setting: {} not found.".format(cw_setting))


def get_loss_function(lf, weights):
    if lf == "binary_crossentropy" or lf == 'categorical_crossentropy':
        return lf
        return hexgraph_weighted_loss(weights, k, v, idx)
    elif lf == "weighted_loss":
        return get_weighted_loss(weights)
    elif lf == 'weighted_categorical_crossentropy':
        return weighted_categorical_crossentropy(weights)
    else:
        raise ValueError("Specified loss function setting: {} not found.".format(lf))


def get_metrics(m_setting):
    metrics = []
    for m in m_setting:
        if m == 'fbeta' or m == 'recall' or m == 'precision' or m == 'count_ones' or m == 'specificity':
            metrics.append(globals()[m])
        else:
            metrics.append(m)
    return metrics

def get_multioutput_pred(prediction, labels):
    # 1. Find argmax of RIMG - if BCP or BOC, then proceed to also find argmax for
    i = np.argmax(prediction[0][:9])
    prediction[0][:9] = 0
    prediction[0][i] = 1
    #   if BCP: CPDMG, CPTYPE, and LOC
    if labels[i] == 'BCP':
        # CPDMG
        i = np.argmax(prediction[0][9:13]) + 9
        prediction[0][9:13] = 0
        prediction[0][i] = 1
        # CPTYPE
        i = np.argmax(prediction[0][13:15]) + 13
        prediction[0][13:15] = 0
        prediction[0][i] = 1
        # LOC
        i = np.argmax(prediction[0][15:17]) + 15
        prediction[0][15:17] = 0
        prediction[0][i] = 1
        # wipe BOV predictions
        prediction[0][17:21] = 0
    # if BOV: OVDMG, OVANG, and LOC
    elif labels[i] == 'BOV':
        # OVDMG
        i = np.argmax(prediction[0][17:19]) + 17
        prediction[0][17:19] = 0
        prediction[0][i] = 1
        # OVANG
        i = np.argmax(prediction[0][19:21]) + 19
        prediction[0][19:21] = 0
        prediction[0][i] = 1
        # LOC
        i = np.argmax(prediction[0][15:17]) + 15
        prediction[0][15:17] = 0
        prediction[0][i] = 1
        # wipe BCP predictions
        prediction[0][9:15] = 0
    else:  # if not BCP or BOV, wipe all predictions
        prediction[0][9:] = 0

    return prediction

def get_model(model_setting, model_img_width, model_img_height, model_img_channels, weights, retrain):
    if model_setting == 'xception':
        if retrain:
            return applications.xception.preprocess_input, None
        else:
            return applications.xception.preprocess_input, applications.Xception(
                input_shape=(model_img_width, model_img_height, model_img_channels),
                weights=weights, include_top=False)
    elif model_setting == 'efficientnet':
        if retrain:
            return tf_applications.efficientnet.preprocess_input, None
        else:
            return tf_applications.efficientnet.preprocess_input, tf_applications.EfficientNetB4(
                input_shape=(model_img_width, model_img_height, model_img_channels),
                weights=weights, include_top=False)
    elif model_setting == 'vgg16':
        if retrain:
            return applications.vgg16.preprocess_input, None
        else:
            return applications.vgg16.preprocess_input, applications.VGG16(
                input_shape=(model_img_width, model_img_height, model_img_channels),
                weights=weights, include_top=False)
    elif model_setting == 'resnetV2':
        if retrain:
            return applications.inception_resnet_v2.preprocess_input, None
        else:
            return applications.inception_resnet_v2.preprocess_input, applications.InceptionResNetV2(
                input_shape=(model_img_width, model_img_height, model_img_channels),
                weights=weights, include_top=False)
    elif model_setting == 'mnetV2':
        if retrain:
            return applications.mobilenet_v2.preprocess_input, None
        else:
            return applications.mobilenet_v2.preprocess_input, applications.MobileNetV2(
                input_shape=(model_img_width, model_img_height, model_img_channels),
                weights=weights, include_top=False)
    elif model_setting == 'densenet201':
        if retrain:
            return applications.densenet.preprocess_input, None
        else:
            return applications.densenet.preprocess_input, applications.DenseNet201(
                input_shape=(model_img_width, model_img_height, model_img_channels),
                weights=weights, include_top=False)

def train_model(take_out_cols=None,
                validation_split=0.2,
                learning_rate=0.0001,
                epochs=20,
                batch_size=16,
                pth_to_data="../datasets/V2/unique_dataset",
                pth_to_labels="../datasets/V2/multilabels.csv",
                image_augmentations={
                    "brightness_range": [0.9,1.1],
                    "width_shift_range": 0.05,
                    "height_shift_range": 0.05,
                    "zoom_range": 0.03,
                    "rotation_range": 15,
                    "horizontal_flip": True
                },
                model_img_width=299,
                model_img_height=299,
                model_img_channels=3,
                df_file_name='file',
                model_name='model.h5',
                output_pth='../output/1_model',
                cw_setting=None,
                lf_setting='binary_crossentropy',
                af_setting='sigmoid',
                ws_settings='imagenet',
                m_setting=['acc'],
                model_setting='xception',
                top_layer_option="basic",
                random_state=42,
                equal_loss_weights=False,
                ml_to_mc=False,
                retrain=None,
                custom_step_size=None,
                split=None
                ):

    """This is for multilabel training - takes in a dataframe with file paths and multi-hot-encoded labels on the right side."""
    # Read csv and create list of columns
    data = pd.read_csv(pth_to_labels)
    columns = list(data.columns[1:])
    columns_w_f_pth = np.append('file', columns)

    ############################################################################################################
    # Train Model
    ############################################################################################################
    model_pth = os.path.join(output_pth, model_name)

    # Split dataset into training and validation
    if split is None:
        train, valid = train_test_split(data, test_size=validation_split, random_state=random_state)
    else:
        train = data[:split]
        valid = data[split:]
    train.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)

    """
    1. adjust get_class_weights to work for multioutput
    2. adjust get_loss_function to work for multioutput
    """
    ########################################################################################################################
    # Get Class Weights
    class_weights = get_class_weights(data, columns, cw_setting)
    if cw_setting == 'ml' or top_layer_option == 'multioutput' or top_layer_option == 'rimg-softmax' or lf_setting=='weighted_categorical_crossentropy':
        ml_weights = class_weights.copy()  # Weights for ml (sigmoid, each class has probability between 0 and 1)
        class_weights = None               # Weights for mc (softmax, classes sum upto probability of 1)
    else:
        ml_weights = None

    # Get Loss Weights
    loss_function = get_loss_function(lf_setting, ml_weights)

    # Get Metrics
    metrics = get_metrics(m_setting)

    # Import preprocessing input and base model from Bag of Models
    preprocess_input, base_model = get_model(model_setting, model_img_width, model_img_height, model_img_channels, ws_settings, retrain)
    # Required image size for mobilenet
    if model_setting == 'mnetV2':
        model_img_width, model_img_height = 224, 224

    if retrain is None:  # Train new model
        sgd = SGD(lr=learning_rate, decay=1e-3, momentum=0.9, nesterov=True)
        # Option 1
        # POOL = AveragePooling2D(name='pool')(base_model.output)
        # POOL = Flatten()(POOL)
        # Option 2
        POOL = GlobalAveragePooling2D(name='pool')(base_model.output)
        POOL = Flatten()(POOL)
        # # Option 3
        # POOL = GlobalMaxPooling2D(name='pool')(base_model.output)
        # # Option a
        # POOL = Dense(64)(POOL)
        # POOL = BatchNormalization()(POOL)
        # POOL = Flatten()(base_model.output)
        # POOL = Dropout(0.5, name='general_dropout_1')(POOL)
        if top_layer_option == 'multioutput' or top_layer_option == 'rimg-softmax':
            RIMG = Dropout(0.5, name='RIMG_dropout')(POOL)
            # RIMG
            # RIMG = Flatten()(base_model.output)
            # RIMG = Dense(16, activation='relu')(RIMG)
            # RIMG = BatchNormalization()(RIMG)
            # RIMG = Dropout(0.5)(RIMG)
            if top_layer_option == "rimg-softmax":
                RIMG = Dense(9, activation='softmax', name='RIMG')(RIMG)
            else:
                RIMG = Dense(9, activation=af_setting, name='RIMG')(RIMG)
            # CPDMG
            # CPDMG = Flatten()(base_model.output)
            # CPDMG = Dense(16, activation='relu')(CPDMG)
            # CPDMG = BatchNormalization()(CPDMG)
            # CPDMG = Dropout(0.5)(CPDMG)
            CPDMG = Dropout(0.5, name='CPDMG_dropout')(POOL)
            CPDMG = Dense(4, activation=af_setting, name='CPDMG')(CPDMG)
            # CPTYPE
            # CPTYPE = Flatten()(base_model.output)
            # CPTYPE = Dense(16, activation='relu')(CPTYPE)
            # CPTYPE = BatchNormalization()(CPTYPE)
            # CPTYPE = Dropout(0.5)(CPTYPE)
            CPTYPE = Dropout(0.5, name='CPTYPE_dropout')(POOL)
            CPTYPE = Dense(2, activation=af_setting, name='CPTYPE')(CPTYPE)
            # LOC
            # LOC = Flatten()(base_model.output)
            # LOC = Dense(16, activation='relu')(LOC)
            # LOC = BatchNormalization()(LOC)
            # LOC = Dropout(0.5)(LOC)
            LOC = Dropout(0.5, name='LOC_dropout')(POOL)
            LOC = Dense(2, activation=af_setting, name='LOC')(LOC)
            # OVDMG
            # OVDMG = Flatten()(base_model.output)
            # OVDMG = Dense(16, activation='relu')(OVDMG)
            # OVDMG = BatchNormalization()(OVDMG)
            # OVDMG = Dropout(0.5)(OVDMG)
            OVDMG = Dropout(0.5, name='OVDMG_dropout')(POOL)
            OVDMG = Dense(2, activation=af_setting, name='OVDMG')(OVDMG)
            # OVANG
            # OVANG = Flatten()(base_model.output)
            # OVANG = Dense(16, activation='relu')(OVANG)
            # OVANG = BatchNormalization()(OVANG)
            # OVANG = Dropout(0.5)(OVANG)
            OVANG = Dropout(0.5, name='OVANG_dropout')(POOL)
            OVANG = Dense(2, activation=af_setting, name='OVANG')(OVANG)
            # Create model
            model = Model(base_model.input, outputs=[RIMG, CPDMG, CPTYPE, LOC, OVDMG, OVANG])
            model.summary()
            # Get Loss Weights
            if cw_setting is None:
                loss_function = {
                    "RIMG": get_loss_function(lf_setting, None),
                    "CPDMG": get_loss_function(lf_setting, None),
                    "CPTYPE": get_loss_function(lf_setting, None),
                    "LOC": get_loss_function(lf_setting, None),
                    "OVDMG": get_loss_function(lf_setting, None),
                    "OVANG": get_loss_function(lf_setting, None)
                }
            else:
                if top_layer_option == 'multioutput':
                    if cw_setting == 'ml':
                        loss_function = {
                            "RIMG": get_loss_function(lf_setting, ml_weights[:9, :]),
                            "CPDMG": get_loss_function(lf_setting, ml_weights[9:13, :]),
                            "CPTYPE": get_loss_function(lf_setting, ml_weights[13:15, :]),
                            "LOC": get_loss_function(lf_setting, ml_weights[15:17, :]),
                            "OVDMG": get_loss_function(lf_setting, ml_weights[17:19, :]),
                            "OVANG": get_loss_function(lf_setting, ml_weights[19:21, :])
                        }
                    elif cw_setting == 'mc':
                        loss_function = {
                            "RIMG": get_loss_function(lf_setting, ml_weights[:9]),
                            "CPDMG": get_loss_function(lf_setting, ml_weights[9:13]),
                            "CPTYPE": get_loss_function(lf_setting, ml_weights[13:15]),
                            "LOC": get_loss_function(lf_setting, ml_weights[15:17]),
                            "OVDMG": get_loss_function(lf_setting, ml_weights[17:19]),
                            "OVANG": get_loss_function(lf_setting, ml_weights[19:21])
                        }
                elif top_layer_option == 'rimg-softmax':
                    mc_weights = get_class_weights(data, columns[:9], 'mc')
                    loss_function = {
                        "RIMG": get_loss_function('weighted_categorical_crossentropy', mc_weights),
                        "CPDMG": get_loss_function(lf_setting, ml_weights[9:13, :]),
                        "CPTYPE": get_loss_function(lf_setting, ml_weights[13:15, :]),
                        "LOC": get_loss_function(lf_setting, ml_weights[15:17, :]),
                        "OVDMG": get_loss_function(lf_setting, ml_weights[17:19, :]),
                        "OVANG": get_loss_function(lf_setting, ml_weights[19:21, :])
                    }

        else:  # Basic option
            x = Dropout(0.5, name='general_dropout_2')(POOL)
            top_layer = Dense(len(columns), activation=af_setting, name='predictions')(x)
            # Create model
            model = Model(base_model.input, top_layer)
            model.summary()
        if (top_layer_option == 'rimg-softmax' or (top_layer_option == 'multioutput' and cw_setting=='mc') and not equal_loss_weights):
            model.compile(loss=loss_function, optimizer=sgd, metrics=metrics, loss_weights=[0.7, 1, 1, 1, 1, 1])
            # model.compile(loss=loss_function, optimizer=sgd, metrics=metrics, loss_weights=[0.1593, 0.72, 0.72, 0.41855, 1, 1])
            # model.compile(loss=loss_function, optimizer=sgd, metrics=metrics, loss_weights=[0.3, 1, 1, 0.8, 1, 1])
            # model.compile(loss=loss_function, optimizer=sgd, metrics=metrics)
        else:
            model.compile(loss=loss_function, optimizer=sgd, metrics=metrics)

    else:  # Re-train model
        model = load_model(retrain, custom_objects={
            'fbeta': fbeta,
            'recall': recall,
            'weighted_loss': loss_function,
            'precision': precision,
            'specificity': specificity})

    # Create Data Generators
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input, **image_augmentations)
    valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, **image_augmentations)
    if top_layer_option == 'multioutput' or top_layer_option == 'rimg-softmax':
        train_mlmo = pd.DataFrame(train['file'])
        train_mlmo["RIMG"] = train[columns[:9]].values.tolist()
        train_mlmo["CPDMG"] = train[columns[9:13]].values.tolist()
        train_mlmo["CPTYPE"] = train[columns[13:15]].values.tolist()
        train_mlmo["LOC"] = train[columns[15:17]].values.tolist()
        train_mlmo["OVDMG"] = train[columns[17:19]].values.tolist()
        train_mlmo["OVANG"] = train[columns[19:21]].values.tolist()
        train_generator = datagen.flow_from_dataframe(dataframe=train_mlmo, directory=pth_to_data, x_col=df_file_name,
                                                      y_col=["RIMG", "CPDMG", "CPTYPE", "LOC", "OVDMG", "OVANG"],
                                                      batch_size=batch_size, seed=random_state,
                                                      shuffle=True, class_mode="multi_output",
                                                      target_size=(model_img_width, model_img_height))
        valid_mlmo = pd.DataFrame(valid['file'])
        valid_mlmo["RIMG"] = valid[columns[:9]].values.tolist()
        valid_mlmo["CPDMG"] = valid[columns[9:13]].values.tolist()
        valid_mlmo["CPTYPE"] = valid[columns[13:15]].values.tolist()
        valid_mlmo["LOC"] = valid[columns[15:17]].values.tolist()
        valid_mlmo["OVDMG"] = valid[columns[17:19]].values.tolist()
        valid_mlmo["OVANG"] = valid[columns[19:21]].values.tolist()
        valid_generator = valid_datagen.flow_from_dataframe(dataframe=valid_mlmo, directory=pth_to_data, x_col=df_file_name,
                                                           y_col=["RIMG", "CPDMG", "CPTYPE", "LOC", "OVDMG", "OVANG"],
                                                           batch_size=batch_size, seed=random_state,
                                                           shuffle=True, class_mode="multi_output",
                                                           target_size=(model_img_width, model_img_height))
    else:
        train_generator = datagen.flow_from_dataframe(dataframe=train, directory=pth_to_data, x_col=df_file_name,
                                                      y_col=columns, batch_size=batch_size, seed=random_state,
                                                      shuffle=True, class_mode="raw",
                                                      target_size=(model_img_width, model_img_height))
        valid_generator = valid_datagen.flow_from_dataframe(dataframe=valid, directory=pth_to_data, x_col=df_file_name,
                                                           y_col=columns, batch_size=batch_size, seed=random_state,
                                                           shuffle=True, class_mode="raw",
                                                           target_size=(model_img_width, model_img_height))

    if custom_step_size is None:
        train_step_size = train_generator.n//batch_size
        valid_step_size = valid_generator.n//batch_size
    else:
        train_step_size = custom_step_size
        valid_step_size = custom_step_size

    # Train Model
    callbacks_list = [ModelCheckpoint(model_pth, monitor='val_acc', verbose=1, )]

    # remove all previous files from destination directory
    if os.path.exists(output_pth):
        shutil.rmtree(output_pth)
    # Make output destination
    os.makedirs(output_pth)
    t0 = time.time()
    H = model.fit_generator(generator=train_generator,
                            callbacks=callbacks_list,
                            class_weight=class_weights,
                            steps_per_epoch=train_step_size,
                            validation_data=valid_generator,
                            validation_steps=valid_step_size,
                            epochs=epochs,
                            verbose=1
                            )
    t1 = time.time()
    training_time = t1 - t0
    ############################################################################################################
    # Generate Statistics using VALIDATION DATA
    ############################################################################################################
    # Pre-process images and generate predictions
    result = []
    raw_result = []
    print("There are {} valid rows.".format(len(valid[columns_w_f_pth].values)))
    for idx, row in tqdm(valid[columns_w_f_pth].iterrows()):
        # Pre-process image
        tmp = cv2.resize(cv2.imread(os.path.join(pth_to_data, row['file'])),
                         dsize=(model_img_height, model_img_width),
                         interpolation=cv2.INTER_LINEAR).astype(np.float64)
        tmp = np.expand_dims(tmp, axis=0)
        tmp = preprocess_input(tmp)
        # Predict and store results
        prediction = model.predict(tmp)
        if top_layer_option == 'multioutput' or top_layer_option == 'rimg-softmax':  # Fix dimension for moml
            prediction = np.concatenate(prediction, axis=1)
        raw_result.append(np.append(row['file'], prediction))
        #########################################################################################################
        prediction = get_multioutput_pred(prediction, columns)
        # if af_setting == 'softmax':
        #     idx = np.argmax(prediction)
        #     prediction.fill(0)
        #     prediction[0, idx] = 1
        # else:
        #     prediction[prediction >= 0.5] = 1
        #     prediction[prediction < 0.5] = 0
        #########################################################################################################
        prediction = np.append(row['file'], prediction)
        result.append(prediction)

    # Save predictions in a DataFrame
    predictions = pd.DataFrame(result, columns=columns_w_f_pth)
    raw_predictions = pd.DataFrame(raw_result, columns=columns_w_f_pth)

    # Generate a classification report
    c_rprt = classification_report(valid[columns], predictions[columns].values.astype('float'),
                                   target_names=list(columns))
    ################################################################################################
    # Generate a confusion matrix
    # if af_setting == 'softmax':
    #     # Compute confusion matrix
    #     c_matrix = confusion_matrix(np.array([np.argmax(l) for l in valid[columns].values]),
    #                           np.array([np.argmax(l) for l in predictions[columns].values]))
    #     fig, ax = plt.subplots()
    #     im = ax.imshow(c_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    #     ax.figure.colorbar(im, ax=ax)
    #     # We want to show all ticks...
    #     ax.set(xticks=np.arange(c_matrix.shape[1]),
    #            yticks=np.arange(c_matrix.shape[0]),
    #            # ... and label them with the respective list entries
    #            xticklabels=columns, yticklabels=columns,
    #            ylabel='True label',
    #            xlabel='Predicted label')
    #
    #     # Rotate the tick labels and set their alignment.
    #     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #              rotation_mode="anchor")
    #
    #     # Loop over data dimensions and create text annotations.
    #     thresh = c_matrix.max() / 2.
    #     for i in range(c_matrix.shape[0]):
    #         for j in range(c_matrix.shape[1]):
    #             ax.text(j, i, format(c_matrix[i, j], 'd'),
    #                     ha="center", va="center",
    #                     color="white" if c_matrix[i, j] > thresh else "black")
    # else:
    c_matrix = multilabel_confusion_matrix(valid[columns].values, predictions[columns].values.astype('float'))
    fig, axes = plt.subplots(nrows=floor(len(columns) ** 0.5)+1, ncols=ceil(len(columns) ** 0.5), figsize=(15, 15))
    for ax, cnt, c in zip(axes.flat, range(len(columns)), columns):
        im = ax.imshow(c_matrix[cnt, :, :], interpolation='nearest', cmap=plt.cm.Blues)
        # We want to show all ticks...
        ax.set(xticks=np.arange(c_matrix.shape[2]),
               yticks=np.arange(c_matrix.shape[1]),
               # ... and label them with the respective list entries
               xticklabels=['False', 'True'], yticklabels=['False', 'True'],
               title=c,
               ylabel='True label',
               xlabel='Predicted label')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = c_matrix[cnt, :, :].max() / 2.
        for i in range(c_matrix.shape[1]):
            for j in range(c_matrix.shape[2]):
                ax.text(j, i, format(c_matrix[cnt, i, j], fmt),
                        ha="center", va="center",
                        color="white" if c_matrix[cnt, i, j] > thresh else "black")
    ########################################################################################################
    fig.tight_layout()
    plt.savefig(os.path.join(output_pth, 'confusion_matrix'))
    plt.show()

    # Generate % distribution of labels in training vs valid set
    train_bar = train[columns].sum() / train[columns].sum().sum()
    train_df = pd.DataFrame(np.array([train_bar.values, ['train'] * len(train_bar), train_bar.index]),
                            index=['value', 'type', 'category'])
    valid_bar = valid[columns].sum() / valid[columns].sum().sum()
    valid_df = pd.DataFrame(np.array([valid_bar.values, ['valid'] * len(valid_bar), valid_bar.index]),
                           index=['value', 'type', 'category'])
    data_dist = pd.concat([train_df, valid_df], axis=1).T

    ############################################################################################################
    # Save all results
    ############################################################################################################
    # Save training, validation and prediction data
    train.to_csv(os.path.join(output_pth, 'train.csv'), index=False)
    valid.to_csv(os.path.join(output_pth, 'valid.csv'), index=False)
    predictions.to_csv(os.path.join(output_pth, "pred.csv"), index=False)
    raw_predictions.to_csv(os.path.join(output_pth, 'pred_raw.csv'), index=False)

    # Save confusion matrix
    with open(os.path.join(output_pth, "c_matrix.pkl"), 'wb') as f:
        pickle.dump(c_matrix, f)

    ####
    # Convert to readable files
    ####
    # Save Training History
    with open(os.path.join(output_pth, 'train_hist.pkl'), 'wb') as f:
        pickle.dump(H.history, f)
    # Classification Report
    with open(os.path.join(output_pth, 'classification_report.txt'), 'w') as f:
        f.write(c_rprt)

    # summarize history
    for m in list(H.history.keys()):
        if m[:4] != 'val_':
            plot_history(H.history[m], H.history['val_{}'.format(m)], 'model {}'.format(m), 'epoch', m, ['train', 'valid'],
                         os.path.join(output_pth, m))

    # Save history
    pd.DataFrame(H.history).to_csv(os.path.join(output_pth, "training_history.csv"))

    # Save plot of % of labels in training vs valid set
    data_dist.to_csv(os.path.join(output_pth, 'data_distribution.csv'), index=False)
    sns.barplot(x="category", y="value", hue='type', data=data_dist)
    plt.yscale('log')
    plt.xticks(rotation=90)
    plt.title('Data Distribution between Training and Testing Dataset')
    plt.savefig(os.path.join(output_pth, 'data_distribution'))

    # Output a training configuration json
    config = {
        "validation_split": validation_split,
        "epochs": epochs,
        "batch_size": batch_size,
        "pth_to_data": pth_to_data,
        "pth_to_labels": pth_to_labels,
        "image_augmentations": image_augmentations,
        "model_img_width": model_img_width,
        "model_img_height": model_img_height,
        "df_file_name": df_file_name,
        "model_name": model_name,
        "output_pth": output_pth,
        "cw_setting": cw_setting,
        "random_state": random_state,
        "columns": columns,
        "model_pth": model_pth,
        "lf_setting": lf_setting,
        "columns_w_f_pth": columns_w_f_pth,
        "af_setting": af_setting,
        "ws_settings": ws_settings,
        "m_settings":m_setting,
        "learning_rate": learning_rate,
        "class_weights": class_weights,
        "model_img_channels": model_img_channels,
        'custom_step_size': custom_step_size,
        'training_time': training_time
    }
    with open(os.path.join(output_pth, 'training_config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    # Output predictions to csv and images
    pth_output = os.path.join(output_pth, "Prediction")
    valid_pth_output = os.path.join(output_pth, "Prediction_valid")
    train_pth_output = os.path.join(output_pth, "Prediction_train")
    # Remove old predicted folder
    if os.path.exists(os.path.dirname(pth_output + '/')):
        shutil.rmtree(pth_output)
    labels = list(pd.read_csv(os.path.join(output_pth, "valid.csv"), nrows=1).columns)[1:]
    all_imgs = data[('file,' + ','.join(labels)).split(',')]
    valid_df = pd.read_csv(os.path.join(output_pth, "valid.csv"))
    train_df = pd.read_csv(os.path.join(output_pth, "train.csv"))
    ##################################################################################################################
    # TEMPORARY COMMENTOUT outputting images takes waaaaay to long, so I'll do it only for the final version when we want to look at the images
    ##################################################################################################################
    # cnt = len(labels)
    # for p, df in zip([valid_pth_output, train_pth_output], [valid_df, train_df]):
    #     # Using pre-process for input into trained model and generate predictions
    #     actual_df = all_imgs[all_imgs['file'].isin(df['file'].values)]
    #     actual_df.sort_values('file', inplace=True)
    #     df.sort_values('file', inplace=True)
    #     # Start loop
    #     print(f"Total number of rows: {len(df)}")
    #     for (idx, row), (idx_a, row_a) in tqdm(zip(df.iterrows(), actual_df.iterrows())):
    #         if row['file'] != row_a['file']: raise ValueError('Wrong!')
    #         f_name = row['file']
    #         # Pre-process image
    #         img = cv2.imread(os.path.join(pth_to_data, f_name))
    #         tmp = cv2.resize(img, dsize=(model_img_height, model_img_width), interpolation=cv2.INTER_LINEAR).astype(
    #             np.float64)
    #         tmp = np.expand_dims(tmp, axis=0)
    #         tmp = preprocess_input(tmp)
    #         # Predict and get actual values from df_actual
    #         prediction = model.predict(tmp)
    #         if top_layer_option == 'multioutput':  # Fix dimension for moml
    #             prediction = np.concatenate(prediction, axis=1)
    #         if af_setting == 'softmax':
    #             i = np.argmax(prediction)
    #             prediction[0][:] = 0
    #             prediction[0][i] = 1
    #             prediction = prediction[0]
    #         else:
    #             prediction[prediction >= 0.5] = 1
    #             prediction[prediction < 0.5] = 0
    #             prediction = prediction[0]
    #
    #         # Write labels to image
    #         WHITE = (255, 255, 255)
    #         # W = 400
    #         # H = 100
    #         W = 600
    #         H = 200
    #         fs = 0.001
    #         spacing = 2
    #         img = cv2.copyMakeBorder(img, H, H, W, 0, cv2.BORDER_CONSTANT)
    #         h, w, c = img.shape
    #         img = cv2.putText(img, "Actual", (0, int(h / (cnt + 1))), cv2.QT_FONT_NORMAL, fs * h, WHITE, 1,
    #                           cv2.LINE_AA)
    #         img = cv2.putText(img, 'Prediction', (int(W / 3), int(h / (cnt + 1))), cv2.QT_FONT_NORMAL, fs * h,
    #                           WHITE, 1, cv2.LINE_AA)
    #         img = cv2.putText(img, 'Condition', (int(2 * W / 3), int(h / (cnt + 1))), cv2.QT_FONT_NORMAL, fs * h,
    #                           WHITE, 1, cv2.LINE_AA)
    #         save_in = []
    #         for cls, val, pred, idx in zip(list(row[1:].index), row[1:].values, prediction, range(len(prediction))):
    #             # Classify detections
    #             if val:
    #                 img = cv2.putText(img, labels[idx], (0, int((idx + spacing) * h / (cnt + 1))), cv2.QT_FONT_NORMAL,
    #                                   fs * h, WHITE, 1, cv2.LINE_AA)
    #             if pred:
    #                 img = cv2.putText(img, labels[idx], (int(W / 3), int((idx + spacing) * h / (cnt + 1))),
    #                                   cv2.QT_FONT_NORMAL, fs * h, WHITE, 1, cv2.LINE_AA)
    #             if val and pred:
    #                 img = cv2.putText(img, "TP", (int(2 * W / 3), int((idx + spacing) * h / (cnt + 1))),
    #                                   cv2.QT_FONT_NORMAL, fs * h, WHITE, 1, cv2.LINE_AA)
    #                 save_in.append((cls, "TP"))
    #             if val and not pred:
    #                 img = cv2.putText(img, "FN", (int(2 * W / 3), int((idx + spacing) * h / (cnt + 1))),
    #                                   cv2.QT_FONT_NORMAL, fs * h, WHITE, 1, cv2.LINE_AA)
    #                 save_in.append((cls, "FN"))
    #             if not val and pred:
    #                 img = cv2.putText(img, "FP", (int(2 * W / 3), int((idx + spacing) * h / (cnt + 1))),
    #                                   cv2.QT_FONT_NORMAL, fs * h, WHITE, 1, cv2.LINE_AA)
    #                 save_in.append((cls, "FP"))
    #
    #         for sv in save_in:
    #             # Save all non-FN detections to folder
    #             new_filename = os.path.join(p, sv[0], sv[1], f_name)
    #             if not os.path.exists(os.path.dirname(new_filename)):
    #                 try:
    #                     os.makedirs(os.path.dirname(new_filename))
    #                 except OSError as exc:  # Guard against race condition
    #                     if exc.errno != errno.EEXIST:
    #                         raise
    #             cv2.imwrite(new_filename, img)
    ##################################################################################################################


if __name__ == '__main__':
    configs=[
        {  # sigmoid - single dense layer
            "epochs": 30,
            "output_pth": '../output/V6_efficientnet_single_dense_layer_sigmoid',
            "cw_setting": 'ml',
            'model_setting': 'efficientnet',
            "learning_rate": 0.01,
            "pth_to_data": "../datasets/V5/unique_dataset",
            "pth_to_labels": "../datasets/V5/multilabels_V2_moml.csv",
            "lf_setting": 'weighted_loss',
            "model_img_width": 299,
            "model_img_height": 299,
            "top_layer_option": "basic",
            'custom_step_size': 5,
            'm_setting': ['acc', 'fbeta', 'recall', 'precision', 'specificity'],
        },
        {  # all softmax
            "epochs": 30,
            "output_pth": '../output/V6_efficientnet_all_softmax',
            "cw_setting": 'mc',
            'model_setting': 'efficientnet',
            "learning_rate": 0.01,
            "pth_to_data": "../datasets/V5/unique_dataset",
            "pth_to_labels": "../datasets/V5/multilabels_V2_moml.csv",
            "lf_setting": 'weighted_categorical_crossentropy',
            "model_img_width": 299,
            "model_img_height": 299,
            "top_layer_option": "multioutput",
            "af_setting": "softmax",
            'm_setting': ['acc', 'fbeta', 'recall', 'precision', 'specificity'],
        },
        {  # softmax - single dense layer
            "epochs": 30,
            "output_pth": '../output/V6_efficientnet_single_dense_layer_softmax',
            "cw_setting": 'mc',
            'model_setting': 'efficientnet',
            'af_setting': 'softmax',
            "learning_rate": 0.001,
            "pth_to_data": "../datasets/V5/unique_dataset",
            "pth_to_labels": "../datasets/V5/multilabels_V2_moml.csv",
            "lf_setting": 'weighted_categorical_crossentropy',
            "model_img_width": 299,
            "model_img_height": 299,
            "top_layer_option": "basic",
            'm_setting': ['acc', 'fbeta', 'recall', 'precision', 'specificity'],
        }
    ]

    for config in configs:
        train_model(**config)
