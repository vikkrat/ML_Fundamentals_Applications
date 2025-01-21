# =============================================================================
# 0. Команда для встановлення необхідних пакетів
# =============================================================================
# pip install pandas numpy scikit-learn imbalanced-learn optuna category_encoders


# =============================================================================
# 1. Імпорт необхідних бібліотек та monkey patch для pandas
# =============================================================================
import pandas as pd
import numpy as np

# Monkey patch: якщо pd.api.types не має is_categorical, задаємо його за допомогою рекомендованої перевірки
if not hasattr(pd.api.types, 'is_categorical'):
    pd.api.types.is_categorical = lambda x: isinstance(x, pd.CategoricalDtype)

# Для виявлення аномалій
from sklearn.ensemble import IsolationForest

# Для масштабування числових ознак
from sklearn.preprocessing import StandardScaler

# Для імпутації (SimpleImputer)
from sklearn.impute import SimpleImputer

# Побудова конвеєрів та балансування даних
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Класифікатор
from sklearn.ensemble import RandomForestClassifier

# Пошук кращих параметрів (Optuna)
import optuna

# Для крос-валідації та train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# Для оцінки якості моделі
from sklearn.metrics import classification_report, balanced_accuracy_score

# Для побудови препроцесора
from sklearn.compose import ColumnTransformer

# Для частотного кодування – використовуємо CountEncoder для отримання кількості появ категорій, 
# після чого ці значення діляться на загальну кількість рядків для нормалізації.
from category_encoders import CountEncoder


# =============================================================================
# 2. Допоміжні функції
# =============================================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Крок 2.1: Видалення дублікатів.
    Перевіряє та прибирає дублікати у DataFrame.
    """
    dup_mask = df.duplicated(keep='first')
    num_dup = dup_mask.sum()
    if num_dup > 0:
        df = df.loc[~dup_mask].copy()
        print(f"[INFO] Видалено {num_dup} дублікатів.")
    else:
        print("[INFO] Дублікатів не знайдено.")
    return df

def generate_missing_count_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Крок 2.2: Створення колонок з кількістю пропусків:
      - 'cat_missing_count': кількість пропусків у категоріальних стовпцях,
      - 'float_missing_count': кількість пропусків у числових (float) стовпцях,
      - 'int_missing_count': кількість пропусків у числових (int) стовпцях.
    """
    cat_cols = df.select_dtypes(include=['object']).columns
    float_cols = df.select_dtypes(include=[np.float64, np.float32]).columns
    int_cols = df.select_dtypes(include=[np.int64, np.int32, np.int16, np.int8]).columns

    df['cat_missing_count'] = df[cat_cols].isnull().sum(axis=1) if len(cat_cols) > 0 else 0
    df['float_missing_count'] = df[float_cols].isnull().sum(axis=1) if len(float_cols) > 0 else 0
    df['int_missing_count'] = df[int_cols].isnull().sum(axis=1) if len(int_cols) > 0 else 0

    return df

def remove_high_cardinality(df_train: pd.DataFrame, df_test: pd.DataFrame, max_unique: int = 50):
    """
    Крок 2.3: Видалення категоріальних стовпців з високою кардинальністю.
    Якщо кількість унікальних значень у стовпці перевищує max_unique у спільних даних train і test,
    такий стовпець видаляється.
    """
    cat_train = df_train.select_dtypes(include=['object']).columns
    cat_test = df_test.select_dtypes(include=['object']).columns

    high_train = [c for c in cat_train if df_train[c].nunique() > max_unique]
    high_test  = [c for c in cat_test if df_test[c].nunique() > max_unique]

    common_high = set(high_train).intersection(high_test)
    if common_high:
        print(f"[INFO] Видалення стовпців з високою кардинальністю: {common_high}")
        df_train.drop(columns=common_high, inplace=True, errors='ignore')
        df_test.drop(columns=common_high, inplace=True, errors='ignore')
    return df_train, df_test

def generate_string_length_features(df: pd.DataFrame, prefix: str = 'catlen_') -> pd.DataFrame:
    """
    Крок 2.4: Додавання колонок з довжинами рядків для кожного категоріального стовпця
    та колонка 'total_cat_length' як сума довжин.
    """
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) == 0:
        df['total_cat_length'] = 0
        return df

    total_len_array = np.zeros(len(df), dtype=int)
    for col in cat_cols:
        col_str = df[col].fillna('')
        lengths = col_str.apply(len).values
        df[f"{prefix}{col}"] = lengths
        total_len_array += lengths

    df['total_cat_length'] = total_len_array
    return df

def frequency_encoding_ce(train_df: pd.DataFrame, test_df: pd.DataFrame, col: str):
    """
    Крок 2.5: Частотне кодування за допомогою CountEncoder з нормалізацією.
    Обчислює нормалізовану частоту (кількість появ / загальна кількість рядків)
    та зберігає результат у колонці <col>_Freq.
    """
    encoder = CountEncoder(cols=[col])
    train_counts = encoder.fit_transform(train_df[[col]])
    test_counts = encoder.transform(test_df[[col]])
    
    total_train = len(train_df)
    total_test = len(test_df)
    
    train_df[col + '_Freq'] = train_counts[col] / total_train
    test_df[col + '_Freq'] = test_counts[col] / total_test
    return train_df, test_df

def custom_imputation(df_train: pd.DataFrame, df_test: pd.DataFrame, target_col: str = 'y') -> tuple:
    """
    Крок 2.6: Імпутація значень:
      - Для числових: медіана, якщо пропусків <30%, інакше – mean.
      - Для категоріальних: most_frequent, якщо пропусків <30%, інакше – "Unknown".
    Використовуємо SimpleImputer для кожного стовпця.
    """
    num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_train.select_dtypes(include=['object']).columns.tolist()

    if target_col in num_cols:
        num_cols.remove(target_col)

    for col in num_cols:
        missing_percent = df_train[col].isna().mean() * 100
        strategy = 'median' if missing_percent < 30 else 'mean'
        imputer = SimpleImputer(strategy=strategy)
        df_train.loc[:, col] = imputer.fit_transform(df_train[[col]]).ravel()
        df_test.loc[:, col] = imputer.transform(df_test[[col]]).ravel()

    for col in cat_cols:
        missing_percent = df_train[col].isna().mean() * 100
        if missing_percent < 30:
            strategy = 'most_frequent'
            fill_value = None
        else:
            strategy = 'constant'
            fill_value = "Unknown"
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        df_train.loc[:, col] = imputer.fit_transform(df_train[[col]]).ravel()
        df_test.loc[:, col] = imputer.transform(df_test[[col]]).ravel()

    return df_train, df_test

def check_no_missing(df: pd.DataFrame, df_name: str = 'DataFrame'):
    """
    Крок 2.7: Перевірка, що у DataFrame немає пропусків.
    Якщо є – видається помилка.
    """
    if df.isnull().sum().sum() > 0:
        raise ValueError(f"[ERROR] {df_name} містить пропущені значення!")
    else:
        print(f"[INFO] {df_name} не містить пропущених значень.")

def add_numerical_aggregations(df: pd.DataFrame, target_col: str = 'y') -> pd.DataFrame:
    """
    Крок 2.8: Додавання агрегованих ознак:
      - sum_row, sum_sq, mean_row, max_row, min_row, std_row.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    df['sum_row'] = df[num_cols].sum(axis=1)
    df['sum_sq'] = (df[num_cols] ** 2).sum(axis=1)
    df['mean_row'] = df[num_cols].mean(axis=1)
    df['max_row'] = df[num_cols].max(axis=1)
    df['min_row'] = df[num_cols].min(axis=1)
    df['std_row'] = df[num_cols].std(axis=1)
    return df


# =============================================================================
# 3. Завантаження даних
# =============================================================================
print("\n[INFO] Завантаження даних...")
train_data = pd.read_csv('final_proj_data.csv')
test_data = pd.read_csv('final_proj_test.csv')
print(f"[INFO] Початкові розміри - Train: {train_data.shape}, Test: {test_data.shape}")


# =============================================================================
# 4. Попередня обробка даних
# =============================================================================

# 4.1. Видалення дублікатів
train_data = remove_duplicates(train_data)
print(f"[INFO] Розміри після видалення дублікатів - Train: {train_data.shape}")

# 4.2. Додавання ознак з кількістю пропусків
train_data = generate_missing_count_features(train_data)
test_data = generate_missing_count_features(test_data)

# 4.3. Видалення стовпців з >=50% пропусків
train_missing_perc = train_data.isnull().mean() * 100
test_missing_perc = test_data.isnull().mean() * 100
train_high_missing = set(train_missing_perc[train_missing_perc >= 50].index)
test_high_missing = set(test_missing_perc[test_missing_perc >= 50].index)
common_high_missing = train_high_missing.intersection(test_high_missing)
if common_high_missing:
    print(f"[INFO] Видалення стовпців з >=50% пропусків: {common_high_missing}")
    train_data.drop(columns=common_high_missing, inplace=True, errors='ignore')
    test_data.drop(columns=common_high_missing, inplace=True, errors='ignore')

# 4.4. Видалення категоріальних стовпців з високою кардинальністю (>50 унікальних значень)
train_data, test_data = remove_high_cardinality(train_data, test_data, max_unique=50)

# 4.5. Додавання ознак довжин для категоріальних стовпців
train_data = generate_string_length_features(train_data)
test_data = generate_string_length_features(test_data)

# 4.6. Частотне кодування категоріальних ознак за допомогою CountEncoder
cat_columns_train = set(train_data.select_dtypes(include=['object']).columns)
cat_columns_test = set(test_data.select_dtypes(include=['object']).columns)
common_cat = cat_columns_train.intersection(cat_columns_test)
for col in common_cat:
    train_data, test_data = frequency_encoding_ce(train_data, test_data, col=col)
# Видаляємо оригінальні категоріальні стовпці, щоб у навчальну вибірку потрапили лише числові ознаки
train_data.drop(columns=list(common_cat), inplace=True, errors='ignore')
test_data.drop(columns=list(common_cat), inplace=True, errors='ignore')

# 4.7. Імпутація відсутніх значень за допомогою SimpleImputer
train_data, test_data = custom_imputation(train_data, test_data, target_col='y')

# 4.8. Перевірка відсутності пропусків
check_no_missing(train_data, 'train_data')
check_no_missing(test_data, 'test_data')

# 4.9. Видалення викидів за допомогою IsolationForest (тільки за числовими ознаками, окрім 'y')
num_cols_for_if = train_data.select_dtypes(include=[np.number]).columns.tolist()
if 'y' in num_cols_for_if:
    num_cols_for_if.remove('y')
iso = IsolationForest(contamination=0.05, random_state=42)
out_preds = iso.fit_predict(train_data[num_cols_for_if])
mask = (out_preds == 1)
orig_len = len(train_data)
train_data = train_data[mask].copy()
print(f"[ISOLATION FOREST] Видалено {orig_len - len(train_data)} викидів.")

# 4.10. Фіче-інженіринг числових ознак
train_data = add_numerical_aggregations(train_data, target_col='y')
test_data = add_numerical_aggregations(test_data, target_col='y')


# =============================================================================
# 5. Підготовка даних для моделювання
# =============================================================================
X_train = train_data.drop(columns=['y'], errors='ignore')
y_train = train_data['y'].values
X_test = test_data.copy()  # Цільова змінна відсутня в тесті

# Визначення списку числових колонок для масштабування
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()


# =============================================================================
# 6. Побудова препроцесора та конвеєра
# =============================================================================
# 6.1. Масштабування числових ознак
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# 6.2. Побудова препроцесора: масштабування числових ознак; інші залишаються без змін
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols)
], remainder='passthrough')

# 6.3. Побудова конвеєра з балансуванням (SMOTE) та RandomForest
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])

# 6.4. Оцінка базової моделі за допомогою крос-валідації (StratifiedKFold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_scores = cross_val_score(pipeline, X_train, y_train,
                                  scoring='balanced_accuracy', cv=cv, n_jobs=-1)
print(f"[BASE RF] CV Balanced Accuracy: {np.mean(baseline_scores):.4f} (+/- {np.std(baseline_scores):.4f})")


# =============================================================================
# 7. Оптимізація гіперпараметрів за допомогою Optuna (Bayesian Search)
# =============================================================================
def objective_optuna(trial):
    """
    Функція цілі для Optuna. Підбирає гіперпараметри для RandomForest та оцінює модель за Balanced Accuracy.
    """
    n_estimators = trial.suggest_int('n_estimators', 50, 300, step=50)
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(
            random_state=42,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        ))
    ])
    cv_scores = cross_val_score(model, X_train, y_train,
                                scoring='balanced_accuracy', cv=cv, n_jobs=-1)
    return np.mean(cv_scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective_optuna, n_trials=20, show_progress_bar=False)

print("[OPTUNA] Кращі параметри:", study.best_params)
print("[OPTUNA] Кращий Balanced Accuracy:", study.best_value)

# n_estimators: 250
# max_depth: 9
# min_samples_split: 10
# min_samples_leaf: 3

with open('best_params.txt', 'w') as f:
    for key, value in study.best_params.items():
        f.write(f"{key}: {value}\n")
print("[INFO] Кращі параметри збережені у файл best_params.txt.")

best_params = study.best_params

# =============================================================================
# 8. Побудова фінальної моделі з кращими параметрами
# =============================================================================
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(
        random_state=42,
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf']
    ))
])

final_pipeline.fit(X_train, y_train)

cv_scores_final = cross_val_score(final_pipeline, X_train, y_train,
                                  scoring='balanced_accuracy', cv=cv, n_jobs=-1)
print(f"[FINAL RF] CV Balanced Accuracy: {np.mean(cv_scores_final):.4f}")

y_pred_train = final_pipeline.predict(X_train)
print("\nClassification Report (TRAIN):")
print(classification_report(y_train, y_pred_train))

# =============================================================================
# 9. Прогнозування та збереження результатів
# =============================================================================
test_preds = final_pipeline.predict(X_test)
submission = pd.DataFrame({
    'index': X_test.index,
    'y': test_preds
})
submission.to_csv('submission_rf.csv', index=False)
print("[INFO] Результати збережені у файл submission_rf.csv.")


# =============================================================================
# 10. Публічна оцінка на hold-out set
# =============================================================================
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                              stratify=y_train, random_state=42)

final_pipeline.fit(X_tr, y_tr)
y_val_pred = final_pipeline.predict(X_val)
public_balanced_acc = balanced_accuracy_score(y_val, y_val_pred)
print(f"[PUBLIC EVALUATION] Balanced Accuracy на hold-out set: {public_balanced_acc:.4f}")

# =============================================================================
# 11. (Опційно) Візуалізація схеми пайплайна (Jupyter Notebook)
# =============================================================================
# from sklearn import set_config
# set_config(display='diagram')
# final_pipeline
