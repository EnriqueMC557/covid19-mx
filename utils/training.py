import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

from utils import paths
from utils.metrics import sensitivity, specificity, precision, f1_score
from utils.tensorflow_keras import reset_session, reset_weights


def compile_fit_evaluate_model(model_, train_set, validation_set, test_set, test, **kwargs):
    verbose = kwargs.get('verbose', 1)
    print_cm = kwargs.get('print_cm', False)
    return_model = kwargs.get('return_model', False)
    save_model = kwargs.get('save_model', False)
    if save_model:
        try:
            model_name = kwargs['model_name']
        except KeyError:
            print('Error: model_name parameter is required when save_model is True')
            return

    reset_session()

    loss_ = test['loss']
    optimizer_ = test['optimizer']
    epochs_ = test['epochs']
    batch_size_ = test['batch_size']

    model_.compile(
        loss=loss_, optimizer=optimizer_,
        metrics=[
            'accuracy',
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.TruePositives()
        ]
    )

    model_.fit(
        train_set[0], train_set[1], validation_data=validation_set,
        epochs=epochs_, batch_size=batch_size_, verbose=verbose
    )

    results = model_.evaluate(test_set[0], test_set[1], batch_size=batch_size_, verbose=verbose)
    FN = results[2]
    FP = results[3]
    TN = results[4]
    TP = results[5]

    if print_cm:
        print('FN\tFP\tTN\tTP')
        print(f'{FN}\t{FP}\t{TN}\t{TP}')

    evaluation_results = dict()
    evaluation_results['accuracy'] = results[1]
    evaluation_results['sensitivity'] = sensitivity(TP, FN)
    evaluation_results['specificity'] = specificity(TP, FP)
    evaluation_results['precision'] = precision(TP, FP)
    evaluation_results['f1_score'] = f1_score(TP, FP, FN)

    if save_model:
        model_.save(f'{paths.MODELS_PATH}/models/{model_name}.h5')

    if return_model:
        return evaluation_results, model_

    return evaluation_results


def apply_kfold(model_, test_config, train_data, test_data, splits, return_models=False, verbose=0, **kwargs):
    kf_results = []
    models = []
    if "rebuild_model" in kwargs:
        model_function = model_

    if "load_model" in kwargs:
        model_ = tf.keras.models.load_model(f"{paths.MODELS_PATH}/{model_}")


    for train_idx, val_idx in tqdm(StratifiedKFold(splits).split(train_data[0],  train_data[1]), total=splits):
        x_train_s, x_val_s = train_data[0][train_idx], train_data[0][val_idx]
        y_train_s, y_val_s = train_data[1][train_idx], train_data[1][val_idx]
        x_test, y_test = test_data[0], test_data[1]

        reset_session()
        reset_weights(model_)

        if "categorical_target" in kwargs:
            y_train_s, y_val_s = to_categorical(y_train_s), to_categorical(y_val_s)
            y_test = to_categorical(y_test)

        if "rebuild_model" in kwargs:
            model_ = model_function(*kwargs["args"])

        fold_results = compile_fit_evaluate_model(
            model_, (x_train_s, y_train_s), (x_val_s, y_val_s),
            (x_test, y_test), test_config, verbose=verbose
        )

        if return_models:
            models.append(model_)

        kf_results.append(fold_results)
        reset_session()

    if return_models:
        return pd.DataFrame(kf_results), models

    return pd.DataFrame(kf_results)
