from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import metrics

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Подборка вин', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(ROOT / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(ROOT / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


model = load_model()
test = load_test()
features = list(model.feature_names_in_)
X = test[features]
y = test['target']
prob = test_proba(model, X)
config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
st.title('Подборка хороших вин')
st.caption('Хорошее вино — экспертная оценка не ниже 7. Модель использует результаты лабораторного анализа.')
threshold = st.sidebar.slider('Порог отбора', 0.0, 1.0, float(config['threshold']), 0.001,
                              format='%.3f', key='threshold')
st.sidebar.caption('Один порог применяется ко всем вкладкам. Порог по умолчанию выбран на валидации.')
one, quality, batch, weights = st.tabs(['Одно вино', 'Модельные метрики', 'Загрузить CSV', 'Что выучила модель'])

with one:
    values = {}
    columns = st.columns(3)
    for i, name in enumerate(features):
        with columns[i % 3]:
            if name == 'is_red':
                values[name] = int(st.toggle('Красное вино', value=bool(X[name].median())))
            else:
                values[name] = st.number_input(name, min_value=float(X[name].min()),
                    max_value=float(X[name].max()), value=float(X[name].median()), format='%.5f')
    single = pd.DataFrame([values], columns=features)
    p = float(model.predict_proba(single)[0, 1])
    st.metric('Вероятность хорошего вина', f'{p:.1%}')
    if p >= threshold:
        st.success('Добавить в подборку')
    else:
        st.info('Не добавлять в подборку')
    st.caption('Границы ввода и начальные значения взяты из минимума, максимума и медианы test.csv по условию задания.')

with quality:
    pred = prob >= threshold
    cols = st.columns(3)
    cols[0].metric('Precision', f'{metrics.precision_score(y, pred, zero_division=0):.3f}')
    cols[1].metric('Recall', f'{metrics.recall_score(y, pred, zero_division=0):.3f}')
    cols[2].metric('F1', f'{metrics.f1_score(y, pred, zero_division=0):.3f}')
    cm = metrics.confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fig, ax = plt.subplots(figsize=(5, 4))
    metrics.ConfusionMatrixDisplay(cm, display_labels=['Остальные', 'Хорошие']).plot(ax=ax, colorbar=False)
    st.pyplot(fig); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fpr, tpr, _ = metrics.roc_curve(y, prob)
    pr, re, _ = metrics.precision_recall_curve(y, prob)
    current_p = metrics.precision_score(y, pred, zero_division=0)
    current_r = metrics.recall_score(y, pred, zero_division=0)
    axes[0].plot(fpr, tpr, label=f'ROC-AUC={metrics.roc_auc_score(y, prob):.3f}')
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].scatter(fp / (fp + tn), current_r, color='red', label='Текущий порог', zorder=3)
    axes[0].set(xlabel='FPR', ylabel='TPR')
    axes[1].plot(re, pr, label=f'AP={metrics.average_precision_score(y, prob):.3f}')
    axes[1].axhline(y.mean(), ls='--', color='gray')
    # Для пустой подборки precision математически не определён.
    if pred.any():
        axes[1].scatter(current_r, current_p, color='red', label='Текущий порог', zorder=3)
    else:
        axes[1].scatter(0, 1, color='red', label='Пустая подборка (условная точка)', zorder=3)
        st.info('При этом пороге подборка пуста. Precision показан как 0 по соглашению; на PR-кривой используется крайняя точка (0, 1).')
    axes[1].set(xlabel='Recall', ylabel='Precision')
    for ax in axes:
        ax.legend()
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.write('Если увеличить порог, подборка станет меньше, а хороших вин мы пропустим больше. '
             'Обычно доля хороших вин в подборке при этом растёт, но на конечной выборке precision может колебаться. '
             'Если уменьшить порог, найдём больше хороших вин, но добавим и больше посредственных.')
    st.caption('Это демонстрация на отложенном тесте. Подбирать окончательный порог по этой вкладке нельзя.')


def prepare_batch(frame, expected):
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
    if frame.empty:
        raise ValueError('В CSV нет строк с винами.')
    numeric = frame[expected].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError('Есть пропуски или бесконечные значения. Заполните их перед загрузкой.')
    if not numeric['is_red'].isin([0, 1]).all():
        raise ValueError('is_red должен содержать только 0 (белое) или 1 (красное).')
    return numeric


with batch:
    st.write('Загрузите CSV с 12 признаками. Названия столбцов должны совпадать с примером.')
    st.download_button('Скачать пример входного CSV', X.head(5).to_csv(index=False).encode('utf-8'),
                       file_name='example.csv', mime='text/csv')
    uploaded = st.file_uploader('CSV с винами', type=['csv'])
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
            data = prepare_batch(frame, features)
            scores = model.predict_proba(data)[:, 1]
            answer = frame.copy()
            answer['probability_good'] = scores
            answer['recommend'] = (scores >= threshold).astype(int)
            st.dataframe(answer)
            st.download_button('Скачать предсказания', answer.to_csv(index=False).encode('utf-8-sig'),
                               file_name='predictions.csv', mime='text/csv')
        except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as exc:
            st.error(f'Не удалось обработать CSV: {exc}')

with weights:
    coef = pd.Series(model[-1].coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    coef.plot.barh(ax=ax, color=['#c65d57' if v < 0 else '#348c70' for v in coef])
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Коэффициент при стандартизованном признаке')
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.write('Сильнее всего за хорошее вино: ' + ', '.join(coef[coef > 0].nlargest(3).index) + '.')
    st.write('Сильнее всего против: ' + ', '.join(coef[coef < 0].nsmallest(3).index) + '.')
    st.caption('Коэффициенты относятся к признакам после StandardScaler. Знак показывает направление изменения '
               'логарифма шансов при прочих равных. Это связи в данных, а не доказательство причинности.')
