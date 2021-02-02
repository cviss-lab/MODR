
import tensorflow as tf
import keras.backend as K
from sklearn.utils.class_weight import compute_class_weight
from keras.models import load_model
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2, os, errno
from model_training_v5_dataset import get_model, get_loss_function
from helper_functions.custom_accuracy_functions import count_ones, fbeta, precision, recall, specificity
from helper_functions.custom_loss_functions import focal_loss, get_weighted_loss, hexgraph_weighted_loss
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt
from math import ceil, floor

def dummy_weighted_categorical_loss(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred)
    loss = -K.sum(loss, -1)
    # If true label is all zero, zero the loss
    condition = K.greater(K.sum(y_true), 0)
    return K.switch(condition, loss, K.zeros_like(loss))

def dummy_weighted_loss(y_true, y_pred):
    """Originally get_weighted_loss, but removed weights portion so that the model can be loaded for inference"""
    return K.mean((K.binary_crossentropy(y_true, y_pred)))

def get_weights(data, columns):
    number_dim = np.shape(data[columns].values)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0., 1.], data[columns].values[:, i])
    return weights

def output_results(
        pth_to_data,
        out_path,
        true_df,
        # Option 1: specify a prediction csv
        pred_df=None,
        # Option 2: specify model
        model_img_height=299,
        model_img_width=299,
        preprocess_input=None,
        top_layer_option="basic",
        af_setting='softmax',
        model=None,
        toggle_high_prob=False,
        prob_threshold=0.5,
        flatten=True
):
    # Get class category labels
    labels = list(true_df.columns)[1:]
    labels_w_f = np.append('file', labels)
    labels_len = len(labels)
    # Using pre-process for input into trained model and generate predictions
    true_df.sort_values('file', inplace=True)
    true_df.reset_index(drop=True, inplace=True)
    if pred_df is not None:
        pred_df.sort_values('file', inplace=True)
        pred_df.reset_index(drop=True, inplace=True)
    # Start loop
    print(f"Total number of rows: {len(true_df)}")
    result = []
    raw_result = []
    for idx in tqdm(range(len(true_df))):
        # Load rows
        row = true_df.loc[idx]
        f_name = row['file']
        # Read image
        img = cv2.imread(os.path.join(pth_to_data, f_name))
        # If we have a prediction dataframe
        if pred_df is not None:
            row_pred = pred_df.loc[idx]
            # Sanity check
            if row['file'] != row_pred['file']: raise ValueError('Wrong!')
        # Else, run prediction
        else:
            # Pre-process image
            tmp = cv2.resize(img, dsize=(model_img_height, model_img_width), interpolation=cv2.INTER_LINEAR).astype(
                np.float64)
            tmp = np.expand_dims(tmp, axis=0)
            tmp = preprocess_input(tmp)
            # Predict and get actual values from df_actual
            raw_prediction = model.predict(tmp)
            if top_layer_option == 'multioutput_multilabel' and flatten==True:  # Fix dimension for moml
                raw_prediction = np.concatenate(raw_prediction, axis=1)
            # Create prediction object
            prediction = raw_prediction.copy()
            if af_setting == 'softmax':  # Typical multiclass
                i = np.argmax(prediction)
                prediction[0][:] = 0
                prediction[0][i] = 1
                row_pred = pd.Series(f_name, index='file').append(pd.Series(prediction[0], index=labels))
            elif top_layer_option == 'multioutput_multilabel':  # My own weird algorithm
                # 1. Find argmax of RIMG - if BCP or BOC, then proceed to also find argmax for
                i = np.argmax(prediction[0][:9])
                prediction[0][:9] = 0
                prediction[0][i] = 1
                #   if BCP: CPDMG, CPTYPE, and LOC
                if labels[i] == 'BCP':
                    # CPDMG
                    i = np.argmax(prediction[0][9:13])+9
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
                # Override hierarchical restrictions and allow high confidence predictions to "pop out"
                if toggle_high_prob:
                    prediction[0][raw_prediction[0] > prob_threshold] = 1
            else:  # Typical multilabel
                prediction[prediction >= 0.5] = 1
                prediction[prediction < 0.5] = 0
            # Create prediction series
            row_pred = pd.Series(f_name, index=['file']).append(pd.Series(prediction[0], index=labels))
            # Save prediction results
            raw_result.append(np.append(row['file'], np.round(raw_prediction, 4)))
            result.append(np.append(row['file'], prediction))

        # Write labels to image
        WHITE = (255, 255, 255)
        # W = 400
        # H = 100
        W = 600
        H = 200
        fs = 0.001
        spacing = 2
        img = cv2.copyMakeBorder(img, H, H, W, 0, cv2.BORDER_CONSTANT)
        h, w, c = img.shape
        img = cv2.putText(img, "Actual", (0, int(h / (labels_len + 1))), cv2.QT_FONT_NORMAL, fs * h, WHITE, 1,
                          cv2.LINE_AA)
        img = cv2.putText(img, 'Prediction', (int(W / 3), int(h / (labels_len + 1))), cv2.QT_FONT_NORMAL, fs * h,
                          WHITE, 1, cv2.LINE_AA)
        img = cv2.putText(img, 'Condition', (int(2 * W / 3), int(h / (labels_len + 1))), cv2.QT_FONT_NORMAL, fs * h,
                          WHITE, 1, cv2.LINE_AA)
        save_in = []
        for cls, val, pred, idx in zip(list(row[1:].index), row[1:].values, row_pred[1:].values, range(len(row_pred[1:].values))):
            # Classify detections
            if val:
                img = cv2.putText(img, labels[idx], (0, int((idx + spacing) * h / (labels_len + 1))), cv2.QT_FONT_NORMAL,
                                  fs * h, WHITE, 1, cv2.LINE_AA)
            if pred:
                img = cv2.putText(img, labels[idx], (int(W / 3), int((idx + spacing) * h / (labels_len + 1))),
                                  cv2.QT_FONT_NORMAL, fs * h, WHITE, 1, cv2.LINE_AA)
            if val and pred:
                img = cv2.putText(img, "TP", (int(2 * W / 3), int((idx + spacing) * h / (labels_len + 1))),
                                  cv2.QT_FONT_NORMAL, fs * h, WHITE, 1, cv2.LINE_AA)
                save_in.append((cls, "TP"))
            if val and not pred:
                img = cv2.putText(img, "FN", (int(2 * W / 3), int((idx + spacing) * h / (labels_len + 1))),
                                  cv2.QT_FONT_NORMAL, fs * h, WHITE, 1, cv2.LINE_AA)
                save_in.append((cls, "FN"))
            if not val and pred:
                img = cv2.putText(img, "FP", (int(2 * W / 3), int((idx + spacing) * h / (labels_len + 1))),
                                  cv2.QT_FONT_NORMAL, fs * h, WHITE, 1, cv2.LINE_AA)
                save_in.append((cls, "FP"))

        for sv in save_in:
            # Save all non-FN detections to folder
            new_filename = os.path.join(out_path, sv[0], sv[1], f_name)
            if not os.path.exists(os.path.dirname(new_filename)):
                try:
                    os.makedirs(os.path.dirname(new_filename))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            cv2.imwrite(new_filename, img)
    # Save predictions in a DataFrame
    true_df.to_csv(os.path.join(out_path, "true.csv"))
    result = pd.DataFrame(result, columns=labels_w_f)
    result.to_csv(os.path.join(out_path, "pred.csv"))
    raw_result = pd.DataFrame(raw_result, columns=labels_w_f)
    raw_result.to_csv(os.path.join(out_path, "raw_pred.csv"))

    if len(true_df) > 0:
        # Generate a classification report
        c_rprt = classification_report(true_df[labels], result[labels].values.astype('float'),
                                       target_names=list(labels))
        # Classification Report
        with open(os.path.join(out_path, 'classification_report.txt'), 'w') as f:
            f.write(c_rprt)
        # Generate a confusion matrix
        if af_setting == 'softmax':
            # Compute confusion matrix
            c_matrix = confusion_matrix(np.array([np.argmax(l) for l in true_df[labels].values]),
                                        np.array([np.argmax(l) for l in result[labels].values]))
            fig, ax = plt.subplots()
            im = ax.imshow(c_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            # We want to show all ticks...
            ax.set(xticks=np.arange(c_matrix.shape[1]),
                   yticks=np.arange(c_matrix.shape[0]),
                   # ... and label them with the respective list entries
                   xticklabels=labels, yticklabels=labels,
                   ylabel='True label',
                   xlabel='Predicted label')

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            thresh = c_matrix.max() / 2.
            for i in range(c_matrix.shape[0]):
                for j in range(c_matrix.shape[1]):
                    ax.text(j, i, format(c_matrix[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if c_matrix[i, j] > thresh else "black")
        else:
            c_matrix = multilabel_confusion_matrix(true_df[labels].values, result[labels].values.astype('float'))
            fig, axes = plt.subplots(nrows=floor(len(labels) ** 0.5)+1, ncols=ceil(len(labels) ** 0.5), figsize=(15, 15))
            for ax, cnt, c in zip(axes.flat, range(len(labels)), labels):
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
        fig.tight_layout()
        plt.savefig(os.path.join(out_path, 'confusion_matrix'))
        plt.show()

def run_inference(
        true_df,
        pred_csv_path,
        pth_to_data,
        output_pth,
        output_folder,
        model_setting,
        model_img_width,
        model_img_height,
        model_img_channels,
        top_layer_option,
        af_setting,
        model,
        toggle_high_prob=False,
        flatten=True
):
    if pred_csv_path is not None:
        pred_df = pd.read_csv(pred_csv_path)
        output_results(
            pth_to_data,
            os.path.join(output_pth, output_folder),
            true_df,
            pred_df,
            toggle_high_prob=toggle_high_prob,
            flatten=flatten
        )
    else:
        # Get preprocess input for model
        preprocess_input, _ = get_model(model_setting, model_img_width, model_img_height, model_img_channels, None, True)
        output_results(
            pth_to_data,
            os.path.join(output_pth, output_folder),
            true_df,
            model_img_height=model_img_height,
            model_img_width=model_img_width,
            preprocess_input=preprocess_input,
            top_layer_option=top_layer_option,
            af_setting=af_setting,
            model=model,
            toggle_high_prob=toggle_high_prob,
            flatten=flatten
        )

def run_inference_and_save_results(
    # Select which one to do inference on
    run_inference_on_training_data,
    run_inference_on_validation_data,
    run_inference_on_testing_data,
    # output_pth
    output_pth,  # ie. '../output/1_model',
    # Parameters for validating training and or testing test
    truth_train_valid_csv_path,         # ie. '../datasets/V4_CVISS_Version_filtered/multilabels.csv',
    truth_train_csv_path,            # ie. '../output/multioutput_multilabel_2/train.csv',
    # Parameters for image retrieval and prediction
    pth_to_data,                        # ie. "../datasets/V3/unique_dataset"
    # Specify csv files housing model prediction here
    pred_train_csv_path=None,      # ie. '../output/multioutput_multilabel_2/pred_train.csv',  # model training data
    pred_valid_csv_path=None,      # ie. '../output/multioutput_multilabel_2/pred_valid.csv',  # Benchmark "unseen" data
    pred_test_csv_path=None,    # ie. '../output/multioutput_multilabel_2/pred_test.csv',
    # Parameters for validating testing set
    f=0.1,                                                          # Sample a fraction of the testing set
    truth_test_csv_path='../datasets/V3/multilabels_format_V4_CVISS_Version_filtered.csv',        # Path to ALL labels
                                                                    # (un-filtered, erroneous version)
                                                                    # Code uses ALL labels and filters out labels used
                                                                    # for training and validation to create the test set
    # Model parameters
    model_pth="../output/multioutput_multilabel_2/model.h5",
    model_img_height=299,
    model_img_width=299,
    model_img_channels=3,
    model_setting=None,
    top_layer_option="basic",
    af_setting='softmax',
    toggle_high_prob=False,
    # Output of training/validation/testing results will automatically be outputted to
    # these folder (created automatically) names in the folder specified in output_pth.
    train_folder='results_train',
    valid_folder='results_valid',
    test_folder ='results_test',
    flatten=True
):
    """
    Runs inference on either training, validation, or testing data and outputs them to a specified folder.

    - If csv paths of model predictions are given, images are generated a lot faster.
    """
    col_org = ["file", "DWG", "GPS", "IRR", "MEAS", "NON", "SGN", "WAT", "BCP", "BOV", "CD0", "CD1", "CDM", "CDR",
            "CPCOL", "CPWAL", "LOCEX", "LOCIN", "ODM", "ODS", "OVCAN", "OVFRT"]
    # Load true value training csv and derive true training and validation sets
    truth_train_valid_df = pd.read_csv(truth_train_valid_csv_path)[col_org]
    truth_train_df = pd.read_csv(truth_train_csv_path)[col_org]
    truth_valid_df = truth_train_valid_df[~truth_train_valid_df.file.isin(truth_train_df['file'])]
    truth_train_df = truth_train_valid_df[truth_train_valid_df.file.isin(truth_train_df['file'])]
    ########################################################################################################################
    # OPTIONAL TEST DATA TESTING
    truth_test_df = pd.read_csv(truth_test_csv_path)[col_org]
    truth_test_df = truth_test_df[~truth_test_df.file.isin(truth_train_valid_df['file'])]
    truth_test_df = truth_test_df.sample(frac=f, random_state=42)  # samples the testing dataframe for inference
    ########################################################################################################################
    # Create fake weights and load model
    n = len(truth_train_df.columns)-1
    cols = [str(l) for l in list(range(n))]
    dummy_data = pd.DataFrame(np.random.random_integers(0, 1, (n, n)), columns=cols)
    # Load model
    ml_weights = get_weights(dummy_data, cols)
    # lf_setting = "weighted_loss"
    # RIMG, CPDMG, CPTYPE, LOC, OVDMG, OVANG = \
    #     get_loss_function(lf_setting, ml_weights[:9, :]),\
    #     get_loss_function(lf_setting, ml_weights[9:13, :]),\
    #     get_loss_function(lf_setting, ml_weights[13:15, :]),\
    #     get_loss_function(lf_setting, ml_weights[15:17, :]),\
    #     get_loss_function(lf_setting, ml_weights[17:19, :]), \
    #     get_loss_function(lf_setting, ml_weights[19:21, :])
    # lf_setting = "weighted_categorical_crossentropy"
    # RIMG, CPDMG, CPTYPE, LOC, OVDMG, OVANG = \
    #     get_loss_function(lf_setting, ml_weights[:9]), \
    #     get_loss_function(lf_setting, ml_weights[9:13]), \
    #     get_loss_function(lf_setting, ml_weights[13:15]), \
    #     get_loss_function(lf_setting, ml_weights[15:17]), \
    #     get_loss_function(lf_setting, ml_weights[17:19]), \
    #     get_loss_function(lf_setting, ml_weights[19:21])
    lf_setting = "weighted_loss"
    lf_setting2 = 'weighted_categorical_crossentropy'
    RIMG, CPDMG, CPTYPE, LOC, OVDMG, OVANG = \
        get_loss_function(lf_setting2, ml_weights[:9, :]), \
        get_loss_function(lf_setting, ml_weights[9:13, :]), \
        get_loss_function(lf_setting, ml_weights[13:15, :]), \
        get_loss_function(lf_setting, ml_weights[15:17, :]), \
        get_loss_function(lf_setting, ml_weights[17:19, :]), \
        get_loss_function(lf_setting, ml_weights[19:21, :])
    model = load_model(model_pth,
                       custom_objects={
                           'fbeta': fbeta,
                           'recall': recall,
                           'precision': precision,
                           'specificity': specificity,
                           "weighted_loss": dummy_weighted_loss,
                           "loss": dummy_weighted_categorical_loss
                       })
    if run_inference_on_testing_data:
        run_inference(truth_test_df,
                      pred_test_csv_path,
                      pth_to_data,
                      output_pth,
                      test_folder,
                      model_setting,
                      model_img_width,
                      model_img_height,
                      model_img_channels,
                      top_layer_option,
                      af_setting,
                      model,
                      toggle_high_prob,
                      flatten=flatten)
    if run_inference_on_training_data:
        run_inference(truth_train_df,
                      pred_train_csv_path,
                      pth_to_data,
                      output_pth,
                      train_folder,
                      model_setting,
                      model_img_width,
                      model_img_height,
                      model_img_channels,
                      top_layer_option,
                      af_setting,
                      model,
                      toggle_high_prob,
                      flatten=flatten)
    if run_inference_on_validation_data:
        run_inference(truth_valid_df,
                      pred_valid_csv_path,
                      pth_to_data,
                      output_pth,
                      valid_folder,
                      model_setting,
                      model_img_width,
                      model_img_height,
                      model_img_channels,
                      top_layer_option,
                      af_setting,
                      model,
                      toggle_high_prob,
                      flatten=flatten)


if __name__ == '__main__':
    flds = [
        ('V6_efficientnet_single_dense_layer_sigmoid', False),
        ('V6_efficientnet_single_dense_layer_softmax', False),
        ('V6_efficientnet_RIMG_softmax_rest_sigmoid', True),
        ('V6_efficientnet_all_softmax', True),
        ('V6_efficientnet_all_sigmoid', True)
    ]
    for fld, flatten in flds:
        run_inference_on_training_data = False
        run_inference_on_validation_data = True
        run_inference_on_testing_data = False
        output_pth = f'../output/{fld}'
        truth_train_csv_path = f"../output/{fld}/train.csv"
        model_pth = f"../output/{fld}/model.h5"
        pred_valid_csv_path = None # '../output/V6_efficientnet_single_dense_layer_softmax/pred_raw.csv'  # Benchmark "unseen" data
        pred_train_csv_path = None  # ie. '../output/multioutput_multilabel_2/pred_train.csv',  # model training data
        pred_test_csv_path = None  # ie. '../output/multioutput_multilabel_2/pred_test.csv',

        truth_train_valid_csv_path = '../datasets/V5/multilabels_V2_moml.csv'
        pth_to_data = "../datasets/V5/unique_dataset"
        f = 0.2  # Sample a fraction of the testing set
        truth_test_csv_path = '../datasets/V5/multilabels_V2_moml.csv'  # Path to ALL labels
        model_img_height = 299
        model_img_width = 299
        model_img_channels = 3
        model_setting = "efficientnet"
        top_layer_option = "multioutput_multilabel"
        af_setting = 'sigmoid'
        toggle_high_prob = True

        run_inference_and_save_results(
            run_inference_on_training_data,
            run_inference_on_validation_data,
            run_inference_on_testing_data,
            output_pth,
            truth_train_valid_csv_path,
            truth_train_csv_path,
            pth_to_data,
            pred_train_csv_path,
            pred_valid_csv_path,
            pred_test_csv_path,
            f,
            truth_test_csv_path,
            model_pth,
            model_img_height,
            model_img_width,
            model_img_channels,
            model_setting,
            top_layer_option,
            af_setting,
            False,
            valid_folder='results_valid_high_prob_off',
            test_folder='results_test_high_prob_off',
            flatten=flatten
        )
        run_inference_and_save_results(
            run_inference_on_training_data,
            run_inference_on_validation_data,
            run_inference_on_testing_data,
            output_pth,
            truth_train_valid_csv_path,
            truth_train_csv_path,
            pth_to_data,
            pred_train_csv_path,
            pred_valid_csv_path,
            pred_test_csv_path,
            f,
            truth_test_csv_path,
            model_pth,
            model_img_height,
            model_img_width,
            model_img_channels,
            model_setting,
            top_layer_option,
            af_setting,
            True,
            valid_folder='results_valid_high_prob_on',
            test_folder='results_test_high_prob_on',
            flatten=flatten
        )
        #
        # run_inference_and_save_results(
        #     True,
        #     False,
        #     False,
        #     output_pth,
        #     truth_train_valid_csv_path,
        #     truth_train_csv_path,
        #     pth_to_data,
        #     pred_train_csv_path,
        #     pred_valid_csv_path,
        #     pred_test_csv_path,
        #     f,
        #     truth_test_csv_path,
        #     model_pth,
        #     model_img_height,
        #     model_img_width,
        #     model_img_channels,
        #     model_setting,
        #     top_layer_option,
        #     af_setting,
        #     flatten=flatten
        # )