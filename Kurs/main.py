
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from contextlib import contextmanager
import sys, os

def load():
    data = pd.read_csv("heart.csv")
    return data


TEST_SIZE = 0.3
RANDOM_STATE = 0

#Готовим данные к ML
def preprocess_data(data):
    scale_cols = ['trestbps', 'chol', 'thalach', 'oldpeak', 'age'];
    sc1 = MinMaxScaler()
    data[scale_cols] = sc1.fit_transform(data[scale_cols])
    data.drop(columns=['slope'], inplace=True)
    TEST_SIZE = 0.3
    RANDOM_STATE = 1
    data_X = data.drop(columns=['target'])
    data_Y = data['target']
    data_X_train, data_X_test, data_Y_train, data_Y_test = train_test_split \
        (data_X, data_Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return data_X_train, data_X_test, data_Y_train, data_Y_test


class MetricLogger:

    def __init__(self):
        self.df = pd.DataFrame(
            {'metric': pd.Series([], dtype='str'),
             'alg': pd.Series([], dtype='str'),
             'value': pd.Series([], dtype='float')})

    def add(self, metric, alg, value):
        """
        Добавление значения
        """
        # Удаление значения если оно уже было ранее добавлено
        self.df.drop(self.df[(self.df['metric'] == metric) & (self.df['alg'] == alg)].index, inplace=True)
        # Добавление нового значения
        temp = [{'metric': metric, 'alg': alg, 'value': value}]
        self.df = self.df.append(temp, ignore_index=True)

    def get_data_for_metric(self, metric, ascending=True):
        """
        Формирование данных с фильтром по метрике
        """
        temp_data = self.df[self.df['metric'] == metric]
        temp_data_2 = temp_data.sort_values(by='value', ascending=ascending)
        return temp_data_2['alg'].values, temp_data_2['value'].values

    def plot(self, str_header, metric, ascending=True, figsize=(5, 5)):
        """
        Вывод графика
        """
        array_labels, array_metric = self.get_data_for_metric(metric, ascending)
        fig, ax1 = plt.subplots(figsize=figsize)
        pos = np.arange(len(array_metric))
        rects = ax1.barh(pos, array_metric,
                         align='center',
                         height=0.5,
                         tick_label=array_labels)
        ax1.set_title(str_header)
        for a, b in zip(pos, array_metric):
            plt.text(0.5, a - 0.05, str(round(b, 3)), color='white')
        plt.show()

clas_models = {'LogR': LogisticRegression(),
               'KNN_5':KNeighborsClassifier(n_neighbors=5),
               'SVC':SVC(probability=True),
               'Tree':DecisionTreeClassifier(),
               'RF':RandomForestClassifier(),
               'GB':GradientBoostingClassifier()}

def draw_roc_curve(y_true, y_score, ax, pos_label=1, average='micro'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score,
                                     pos_label=pos_label)
    roc_auc_value = roc_auc_score(y_true, y_score, average=average)
    #plt.figure()
    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_value)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_xlim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")


def plot_learning_curve(data_X, data_y, clf, name='accuracy', scoring='accuracy'):
    train_sizes, train_scores, test_scores = learning_curve(estimator=clf, scoring=scoring, X=data_X, y=data_y,
                                                            train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    fig = plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label=f'тренировочная {name}-мера')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
             label=f'проверочная {name}-мера')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.legend(loc='lower right')
    plt.xlabel('Число тренировочных образцов')
    plt.ylabel(f'{name}-мера')
    st.pyplot(fig)



if __name__ == '__main__':
    st.title('Курсовая Работа Кожуро Б.Е.')
    data = load()
    data_X_train, data_X_test, data_Y_train, data_Y_test = preprocess_data(data)


    def clas_train_model(model_name, model, clasMetricLogger):
        model.fit(data_X_train, data_Y_train)
        # Предсказание значений
        Y_pred = model.predict(data_X_test)
        # Предсказание вероятности класса "1" для roc auc
        Y_pred_proba_temp = model.predict_proba(data_X_test)
        Y_pred_proba = Y_pred_proba_temp[:, 1]

        precision = precision_score(data_Y_test.values, Y_pred)
        recall = recall_score(data_Y_test.values, Y_pred)
        f1 = f1_score(data_Y_test.values, Y_pred)
        roc_auc = roc_auc_score(data_Y_test.values, Y_pred_proba)

        clasMetricLogger.add('precision', model_name, precision)
        clasMetricLogger.add('recall', model_name, recall)
        clasMetricLogger.add('f1', model_name, f1)
        clasMetricLogger.add('roc_auc', model_name, roc_auc)

        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
        draw_roc_curve(data_Y_test.values, Y_pred_proba, ax[0])
        plot_confusion_matrix(model, data_X_test, data_Y_test.values, ax=ax[1],
                              display_labels=['0', '1'],
                              cmap=plt.cm.Blues, normalize='true')
        fig.suptitle(model_name)
    clasMetricLogger = MetricLogger()
    clasGradLogger = MetricLogger()

    # выбор гиперпараметров в сайдбаре

    st.sidebar.subheader('Гиперпараметры :')
    estimators = st.sidebar.slider('Количество деревьев:', min_value=10, max_value=300, value=100, step=10)
    max_depth = st.sidebar.slider('Максимальная глубина', min_value=1, max_value=10, value=3, step=1)
    min_samples_split = st.sidebar.slider('Минимальное количество образцов для разделения ноды', min_value=0.1, max_value=0.5, value=0.2, step= 0.04)
    min_samples_leaf = st.sidebar.slider('Минимальное количество образцов в ноде', min_value=0.1, max_value=0.5, value=0.1, step=0.04)
    max_features = st.sidebar.radio('Максимальное кольчество параметров', ["log2", "sqrt"], index=0)
    subsample = st.sidebar.radio('Размер подвыборки', [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0], index=0)
    learning_rate = st.sidebar.radio('Обучаемость', [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2], index=0)
    score = st.sidebar.radio('Метрика кривой обучения', ['roc_auc', 'f1', 'precision', 'recall'], index=0)


    # Вывод результатов
    gd = GradientBoostingClassifier(n_estimators=estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                    min_samples_split=min_samples_split, max_features=max_features,
                                    subsample=subsample, learning_rate=learning_rate)
    data_X = pd.concat([data_X_train, data_X_test])
    data_Y = pd.concat([data_Y_train, data_Y_test])
    clas_train_model("GradBoost", gd, clasGradLogger)
    clas_metrics = clasGradLogger.df['metric'].unique()
    for metric in clas_metrics:
        st.text(str(clasGradLogger.get_data_for_metric(metric)[1][0]) + " - " + str(metric))
    plot_learning_curve(data_X, data_Y, gd, name=score, scoring=score)


    #  показать данные
    if st.checkbox('Показать данные'):
        st.write(data.head(10))

    #Показать матрицу
    if st.checkbox('Показать корреляционную матрицу'):
        fig_corr, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(data.corr(), annot=True, fmt='.2f')
        st.pyplot(fig_corr)

    #показать КР
    if st.checkbox('Показать Исследование'):
        for model_name, model in clas_models.items():
            clas_train_model(model_name, model, clasMetricLogger)

        metrics = ['precision', 'recall', 'f1', 'roc_auc']
        data_X = pd.concat([data_X_train, data_X_test])
        data_Y = pd.concat([data_Y_train, data_Y_test])
        for metric in metrics:
            st.markdown("# " + metric)
            for model_name, model in clas_models.items():
                st.markdown("## " + model_name)
                model.fit(data_X_train, data_Y_train)
                plot_learning_curve(data_X, data_Y, model, name=metric, scoring=metric)
        #KNN
        n_range = np.array(range(1, 170, 20))
        tuned_parameters = [{'n_neighbors': n_range}]
        clf_gs = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring='roc_auc')
        clf_gs.fit(data_X_train, data_Y_train)
        clas_models_grid = {'KNN_5': KNeighborsClassifier(n_neighbors=5),
                            str('KNN_' + str(clf_gs.best_params_['n_neighbors'])): clf_gs.best_estimator_}
        for model_name, model in clas_models_grid.items():
            clas_train_model(model_name, model, clasMetricLogger)
        #Tree
        n_range = np.array(range(1, 10, 1))
        tuned_parameters = [{'max_depth': n_range}]
        clf_gs = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='roc_auc')
        clf_gs.fit(data_X_train, data_Y_train)
        clas_models_grid = {'Tree': DecisionTreeClassifier(),
                            str('Tree_' + str(clf_gs.best_params_['max_depth'])): clf_gs.best_estimator_}
        for model_name, model in clas_models_grid.items():
            clas_train_model(model_name, model, clasMetricLogger)

        #SVC
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000], 'probability': [True]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000], 'probability': [True]}]
        clf_gs = GridSearchCV(SVC(), tuned_parameters, cv=7, scoring='roc_auc')
        clf_gs.fit(data_X_train, data_Y_train)
        clas_models_grid = {'SVC': SVC(probability=True),
                            str('SVC_' + str(clf_gs.best_params_['kernel']) + "_" + str(
                                clf_gs.best_params_['C'])): clf_gs.best_estimator_}
        for model_name, model in clas_models_grid.items():
            clas_train_model(model_name, model, clasMetricLogger)

        #RandForest
        _range = np.array(range(1, 10, 1))
        n_est = np.array(range(1, 251, 50))
        tuned_parameters = [{'n_estimators': n_est, 'max_depth': n_range}]
        clf_gs = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='roc_auc')
        clf_gs.fit(data_X_train, data_Y_train)
        clas_models_grid = {'RF': RandomForestClassifier(),
                            str('RF_' + str(clf_gs.best_params_['n_estimators']) + "_" + str(
                                clf_gs.best_params_['max_depth'])): clf_gs.best_estimator_}
        for model_name, model in clas_models_grid.items():
            clas_train_model(model_name, model, clasMetricLogger)

        #GB
        tuned_parameters = [{
            "loss": ["deviance"],
            "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
            "max_depth": [3, 5, 8],
            "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
            "n_estimators": [100]
        }]
        clf_gs = GridSearchCV(GradientBoostingClassifier(), tuned_parameters, cv=5, scoring='roc_auc')
        clf_gs.fit(data_X_train, data_Y_train)
        clas_models_grid = {'GB': GradientBoostingClassifier(),
                            str('GB_' + str(clf_gs.best_params_['learning_rate']) + "_" + str(
                                clf_gs.best_params_['max_depth']) + '_'
                                + str(clf_gs.best_params_['subsample']) + "_" + str(
                                clf_gs.best_params_['n_estimators'])): clf_gs.best_estimator_}
        for model_name, model in clas_models_grid.items():
            clas_train_model(model_name, model, clasMetricLogger)
        clas_metrics = clasMetricLogger.df['metric'].unique()
        for metric in clas_metrics:
            clasMetricLogger.plot('Метрика: ' + metric, metric, figsize=(9, 12))
