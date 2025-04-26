import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import pandas as pd

# Importar utilidades de Modelo2.py
from Modelo2 import load_data, convert_numeric

# Crear carpeta de resultados de pruebas
RESULTS_DIR = Path.cwd() / 'resultados pruebas'
RESULTS_DIR.mkdir(exist_ok=True)

# Definición de clasificadores base
classifiers = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

# Intentar añadir LightGBM, omitir si falla (macOS + libomp)
try:
    from lightgbm import LGBMClassifier
    classifiers['LightGBM'] = LGBMClassifier(random_state=42)
except Exception as e:
    print(f"[Advertencia] LightGBM no disponible y se omitirá: {e}")


def run_classification_tests(df):
    """
    1) Binariza 'Active users' (> mediana → 1, else → 0).
    2) Comprueba que hay ambas clases; si no, aborta.
    3) Entrena clasificadores, grafica ROC y feature importances.
    4) Muestra métricas con round(3).
    """
    cols = [
        'New users', 'Engaged sessions', 'Engagement rate',
        'Engaged sessions per active user',
        'Average engagement time per active user', 'Event count'
    ]
    df = df.copy()
    df = convert_numeric(df, cols + ['Active users']).dropna(subset=cols + ['Active users'])
    
    # Binarizar según mediana
    median_val = df['Active users'].median()
    df['y'] = (df['Active users'] > median_val).astype(int)
    
    # Verificar que hay ambas clases
    if df['y'].nunique() < 2:
        print("[Error] Sólo hay una clase tras binarizar; no se pueden entrenar modelos.")
        return None

    X = df[cols].values
    y = df['y'].values

    # División entrenamiento/prueba (estratificado cuando sea posible)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        print("[Advertencia] Stratify omitido por pocas muestras.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )

    results = {}

    # Curva ROC comparada
    plt.figure(figsize=(6, 4))
    any_model_plotted = False
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        # Predict & proba
        try:
            y_proba = clf.predict_proba(X_test)[:, 1]
        except Exception:
            print(f"[Aviso] {name} no soporta predict_proba; omitido de ROC.")
            continue

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        # Intentar ROC-AUC
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            print(f"[Aviso] ROC-AUC no definido para {name}.")
            continue

        results[name] = {'accuracy': acc, 'roc_auc': auc, 'clf': clf}

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.2f})')
        any_model_plotted = True

    if not any_model_plotted:
        plt.text(0.5, 0.5, 'Curvas ROC no disponibles\npor falta de clases suficientes.',
                 ha='center', va='center', fontsize=10)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.title('Curvas ROC Comparadas')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'roc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Importancia de características
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (name, res) in enumerate(results.items()):
        clf = res['clf']
        imp = None

        if hasattr(clf, 'coef_'):
            imp = np.abs(clf.coef_[0])
        elif hasattr(clf, 'feature_importances_'):
            imp = clf.feature_importances_

        ax = axes[idx]
        if imp is not None:
            order = np.argsort(imp)[::-1]
            ax.barh(np.array(cols)[order], imp[order])
            ax.set_title(f'{name} - Importancias')
            ax.tick_params(axis='y', labelsize=8)
        else:
            if name == 'SVM':
                ax.text(0.5, 0.5, 'Importancia no aplicable\npara SVM (RBF Kernel)',
                        ha='center', va='center', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No disponible', ha='center', va='center', fontsize=8)
            ax.set_title(f'{name}')

    for j in range(len(results), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'feature_importances.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Resumen de métricas
    summary = pd.DataFrame(results).T[['accuracy', 'roc_auc']].round(3)
    print("\nResumen de métricas:\n", summary)

    return {
        'roc_image': 'roc_comparison.png',
        'featimp_image': 'feature_importances.png',
        'metrics': summary
    }

if __name__ == '__main__':
    dfs = load_data()
    dem_df = dfs.get('demographics', pd.DataFrame())
    results = run_classification_tests(dem_df)
    if results:
        print("\nImágenes generadas en 'resultados pruebas':",
              results['roc_image'], results['featimp_image'])