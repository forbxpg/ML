"""
Полное решение лабораторной работы по методу итеративно взвешенных наименьших квадратов (IRLS)
по техническому заданию из файла tasks.md

Задания:
1. Построить простейшую линейную регрессионную модель, используя метод OLS
2. Рассчитать стандартизованные остатки и робастные стандартизованные остатки
3. Рассчитать биквадратные веса наблюдений и построить WLS модель
4. Повторить пп. 2-3 до сходимости модели (итеративный процесс IRLS)
5. Анализ остатков IRLS-модели и идентификация выбросов
6. Сравнение методов OLS и IRLS
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import median_abs_deviation, probplot
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Config:
    """Конфигурация для лабораторной работы."""

    DATA_PATH: str = "data_v1-08.csv"
    SPLIT_TEST_SIZE: float = 0.3  # 70/30
    SPLIT_RAND_STATE: int = 42
    NORMAL_DISTRIBUTION_CONST: float = 0.67448975
    IRLS_TOLERANCE: float = 1e-6  # Критерий сходимости для IRLS
    IRLS_MAX_ITER: int = 100  # Максимальное количество итераций


config = Config()


def load_data():
    """Загрузка и подготовка данных."""
    data = pd.read_csv(config.DATA_PATH)
    df = pd.DataFrame(data)

    X = df[["x"]]
    y = df["y"]

    return X, y


def split_data(X, y):
    """Разделение данных на обучающую и тестовую выборки."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.SPLIT_TEST_SIZE, random_state=config.SPLIT_RAND_STATE
    )

    return X_train, X_test, y_train, y_test


def task_1_ols_model(X_train, X_test, y_train, y_test):
    """
    Задание 1: Построить простейшую линейную регрессионную модель, используя метод OLS.

    Возвращает:
        ols_model: обученная модель OLS
        r2_train: коэффициент детерминации на обучающей выборке
        r2_test: коэффициент детерминации на тестовой выборке
        ols_pred_test: предсказания на тестовой выборке
    """
    print("=" * 60)
    print("ЗАДАНИЕ 1: OLS модель")
    print("=" * 60)

    # Добавляем константу для модели
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    # Обучаем модель OLS
    ols_model = sm.OLS(y_train, X_train_const).fit()

    # Выводим результаты модели
    print(ols_model.summary())

    # Коэффициенты детерминации
    r2_train = ols_model.rsquared
    ols_pred_test = ols_model.predict(X_test_const)
    r2_test = r2_score(y_test, ols_pred_test)

    print(f"\nR² на обучающей выборке: {r2_train:.4f}")
    print(f"R² на тестовой выборке: {r2_test:.4f}")

    # Визуализация с линией регрессии
    plt.figure(figsize=(12, 8))

    # Точки данных
    plt.scatter(X_train["x"], y_train, label="Обучающая выборка", alpha=0.6, s=50)
    plt.scatter(X_test["x"], y_test, label="Тестовая выборка", alpha=0.6, s=50)

    # Линия регрессии
    x_vals = np.linspace(X_train["x"].min(), X_train["x"].max(), 100)
    x_vals_const = sm.add_constant(x_vals)
    y_vals = ols_model.predict(x_vals_const)
    plt.plot(x_vals, y_vals, color="red", linewidth=3, label="OLS регрессия")

    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.title("Диаграмма рассеяния с функцией регрессии (OLS)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return ols_model, r2_train, r2_test, ols_pred_test


def task_2_residuals_analysis(
    ols_model, X_train, X_test, y_train, y_test, ols_pred_test
):
    """
    Задание 2: Анализ остатков OLS модели.

    Визуализирует стандартизованные и робастные стандартизованные остатки,
    проверяет распределения на нормальность.
    """
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 2: Анализ остатков OLS модели")
    print("=" * 60)

    # Расчет остатков
    residuals_train = ols_model.resid
    residuals_test = y_test - ols_pred_test

    # Стандартизованные остатки
    std_residuals_train = residuals_train / residuals_train.std()
    std_residuals_test = residuals_test / residuals_test.std()

    # Робастные стандартизованные остатки (MAD-based)
    mad_train = median_abs_deviation(residuals_train)
    scale_train = mad_train / config.NORMAL_DISTRIBUTION_CONST
    robust_std_residuals_train = residuals_train / scale_train
    robust_std_residuals_test = residuals_test / scale_train

    # Графики остатков vs x
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Стандартизованные остатки
    axes[0, 0].scatter(X_train["x"], std_residuals_train, alpha=0.7, s=30)
    axes[0, 0].axhline(y=0, color="red", linestyle="--", alpha=0.7)
    axes[0, 0].set_title("Стандартизованные остатки (обучающая выборка)", fontsize=12)
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("Стандартизованные остатки")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(X_test["x"], std_residuals_test, alpha=0.7, s=30)
    axes[0, 1].axhline(y=0, color="red", linestyle="--", alpha=0.7)
    axes[0, 1].set_title("Стандартизованные остатки (тестовая выборка)", fontsize=12)
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("Стандартизованные остатки")
    axes[0, 1].grid(True, alpha=0.3)

    # Робастные стандартизованные остатки
    axes[1, 0].scatter(X_train["x"], robust_std_residuals_train, alpha=0.7, s=30)
    axes[1, 0].axhline(y=0, color="red", linestyle="--", alpha=0.7)
    axes[1, 0].set_title(
        "Робастные стандартизованные остатки (обучающая выборка)", fontsize=12
    )
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("Робастные стандартизованные остатки")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(X_test["x"], robust_std_residuals_test, alpha=0.7, s=30)
    axes[1, 1].axhline(y=0, color="red", linestyle="--", alpha=0.7)
    axes[1, 1].set_title(
        "Робастные стандартизованные остатки (тестовая выборка)", fontsize=12
    )
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("Робастные стандартизованные остатки")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Гистограммы и Q-Q plots для проверки нормальности
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Обучающая выборка
    sns.histplot(std_residuals_train, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title(
        "Гистограмма стандартизованных остатков (обучающая)", fontsize=11
    )
    axes[0, 0].set_xlabel("Стандартизованные остатки")
    axes[0, 0].set_ylabel("Частота")

    probplot(std_residuals_train, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Q-Q plot стандартизованных остатков (обучающая)", fontsize=11)

    sns.histplot(robust_std_residuals_train, kde=True, ax=axes[0, 2])
    axes[0, 2].set_title("Гистограмма робастных остатков (обучающая)", fontsize=11)
    axes[0, 2].set_xlabel("Робастные стандартизованные остатки")
    axes[0, 2].set_ylabel("Частота")

    probplot(robust_std_residuals_train, dist="norm", plot=axes[0, 3])
    axes[0, 3].set_title("Q-Q plot робастных остатков (обучающая)", fontsize=11)

    # Тестовая выборка
    sns.histplot(std_residuals_test, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title(
        "Гистограмма стандартизованных остатков (тестовая)", fontsize=11
    )
    axes[1, 0].set_xlabel("Стандартизованные остатки")
    axes[1, 0].set_ylabel("Частота")

    probplot(std_residuals_test, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q plot стандартизованных остатков (тестовая)", fontsize=11)

    sns.histplot(robust_std_residuals_test, kde=True, ax=axes[1, 2])
    axes[1, 2].set_title("Гистограмма робастных остатков (тестовая)", fontsize=11)
    axes[1, 2].set_xlabel("Робастные стандартизованные остатки")
    axes[1, 2].set_ylabel("Частота")

    probplot(robust_std_residuals_test, dist="norm", plot=axes[1, 3])
    axes[1, 3].set_title("Q-Q plot робастных остатков (тестовая)", fontsize=11)

    plt.tight_layout()
    plt.show()

    return {
        "std_residuals_train": std_residuals_train,
        "std_residuals_test": std_residuals_test,
        "robust_std_residuals_train": robust_std_residuals_train,
        "robust_std_residuals_test": robust_std_residuals_test,
        "scale_train": scale_train,
    }


def calculate_bisquare_weights(residuals, scale):
    """
    Расчет биквадратных весов для наблюдений.

    Формула: w_i = (1 - (r_i / (6 * scale))^2)^2 для |r_i / (6 * scale)| < 1
             w_i = 0 для |r_i / (6 * scale)| >= 1

    где r_i - остатки, scale - MAD-based оценка масштаба
    """
    standardized_residuals = residuals / (6 * scale)
    weights = np.where(
        np.abs(standardized_residuals) < 1, (1 - standardized_residuals**2) ** 2, 0
    )
    return weights


def task_3_wls_model(X_train, X_test, y_train, y_test, residuals_train, scale_train):
    """
    Задание 3: Рассчитать биквадратные веса и построить WLS модель.

    Возвращает:
        wls_model: обученная WLS модель
        weights: рассчитанные биквадратные веса
    """
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 3: WLS модель с биквадратными весами")
    print("=" * 60)

    # Расчет биквадратных весов
    weights = calculate_bisquare_weights(residuals_train, scale_train)

    print(f"Количество нулевых весов (выбросы): {np.sum(weights == 0)}")
    print(f"Минимальный вес (ненулевые): {np.min(weights[weights > 0]):.6f}")
    print(f"Максимальный вес: {np.max(weights):.6f}")
    print(f"Средний вес: {np.mean(weights):.6f}")

    # Добавляем константу для модели
    X_train_const = sm.add_constant(X_train)

    # Обучаем WLS модель
    wls_model = sm.WLS(y_train, X_train_const, weights=weights).fit()

    print("\nРезультаты WLS модели:")
    print(wls_model.summary())

    # Коэффициенты детерминации для WLS
    wls_pred_train = wls_model.predict(X_train_const)
    wls_pred_test = wls_model.predict(sm.add_constant(X_test))

    r2_train_wls = r2_score(y_train, wls_pred_train)
    r2_test_wls = r2_score(y_test, wls_pred_test)

    print(f"\nR² WLS на обучающей выборке: {r2_train_wls:.4f}")
    print(f"R² WLS на тестовой выборке: {r2_test_wls:.4f}")

    # Визуализация с выделением выбросов (нулевые веса)
    plt.figure(figsize=(12, 8))

    # Все точки
    plt.scatter(X_train["x"], y_train, label="Обучающая выборка", alpha=0.6, s=50)

    # Точки с нулевыми весами (выбросы) - выделяем красным
    outlier_indices = weights == 0
    if np.any(outlier_indices):
        plt.scatter(
            X_train["x"][outlier_indices],
            y_train[outlier_indices],
            color="red",
            s=80,
            label="Выбросы (нулевые веса)",
            edgecolors="black",
            linewidth=1,
        )

    # Линия регрессии WLS
    x_vals = np.linspace(X_train["x"].min(), X_train["x"].max(), 100)
    x_vals_const = sm.add_constant(x_vals)
    y_vals = wls_model.predict(x_vals_const)
    plt.plot(x_vals, y_vals, color="green", linewidth=3, label="WLS регрессия")

    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.title(
        "Диаграмма рассеяния с WLS регрессией и выделенными выбросами", fontsize=14
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return wls_model, weights, r2_train_wls, r2_test_wls


def task_4_irls_process(X_train, X_test, y_train, y_test, max_iter=100, tolerance=1e-6):
    """
    Задание 4: Итеративный процесс IRLS до сходимости модели.

    Возвращает:
        final_model: финальная модель после сходимости
        models_history: история моделей по итерациям
        convergence_info: информация о сходимости
    """
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 4: Итеративный процесс IRLS")
    print("=" * 60)

    # Начальная модель (OLS)
    X_train_const = sm.add_constant(X_train)
    current_model = sm.OLS(y_train, X_train_const).fit()
    models_history = [current_model]

    print("Итерация 0 (начальная OLS модель):")
    print(f"  Коэффициенты: {current_model.params.values}")
    print(f"  R²: {current_model.rsquared:.6f}")

    for iteration in range(1, max_iter + 1):
        # Расчет остатков текущей модели
        residuals = current_model.resid

        # Оценка масштаба (MAD)
        mad = median_abs_deviation(residuals)
        scale = mad / config.NORMAL_DISTRIBUTION_CONST

        # Расчет биквадратных весов
        weights = calculate_bisquare_weights(residuals, scale)

        # Обучение новой модели с весами
        try:
            new_model = sm.WLS(y_train, X_train_const, weights=weights).fit()
        except:
            print(
                f"  Итерация {iteration}: Не удалось обучить модель с текущими весами"
            )
            break

        models_history.append(new_model)

        print(f"Итерация {iteration}:")
        print(f"  Коэффициенты: {new_model.params.values}")
        print(f"  R²: {new_model.rsquared:.6f}")
        print(f"  Нулевых весов: {np.sum(weights == 0)}")

        # Проверка сходимости (по изменению коэффициентов)
        if len(models_history) >= 2:
            prev_params = models_history[-2].params.values
            current_params = new_model.params.values
            max_change = np.max(np.abs(current_params - prev_params))

            print(f"  Максимальное изменение коэффициентов: {max_change:.8f}")

            if max_change < tolerance:
                print(f"  Сходимость достигнута на итерации {iteration}")
                break

    final_model = models_history[-1]
    convergence_info = {
        "iterations": len(models_history) - 1,
        "converged": len(models_history) < max_iter,
        "final_weights_zeros": np.sum(weights == 0) if "weights" in locals() else 0,
    }

    print("\nФинальная модель:")
    print(f"  Количество итераций: {convergence_info['iterations']}")
    print(
        f"  Сходимость: {'достигнута' if convergence_info['converged'] else 'не достигнута'}"
    )
    print(f"  Финальные коэффициенты: {final_model.params.values}")

    # Коэффициенты детерминации финальной модели
    final_pred_train = final_model.predict(X_train_const)
    final_pred_test = final_model.predict(sm.add_constant(X_test))

    r2_train_final = r2_score(y_train, final_pred_train)
    r2_test_final = r2_score(y_test, final_pred_test)

    print(f"  R² на обучающей выборке: {r2_train_final:.4f}")
    print(f"  R² на тестовой выборке: {r2_test_final:.4f}")

    # Визуализация процесса сходимости
    plt.figure(figsize=(12, 8))

    # Все точки
    plt.scatter(X_train["x"], y_train, label="Обучающая выборка", alpha=0.6, s=50)

    # Линии регрессии для разных итераций
    x_vals = np.linspace(X_train["x"].min(), X_train["x"].max(), 100)
    x_vals_const = sm.add_constant(x_vals)

    # Начальная модель (OLS)
    y_vals_0 = models_history[0].predict(x_vals_const)
    plt.plot(x_vals, y_vals_0, "blue", linewidth=2, alpha=0.7, label="OLS (итерация 0)")

    # Промежуточные модели
    colors = ["orange", "green", "purple"]
    for i, model in enumerate(models_history[1:-1]):
        if i >= len(colors):
            break
        y_vals = model.predict(x_vals_const)
        plt.plot(
            x_vals,
            y_vals,
            color=colors[i],
            linewidth=2,
            alpha=0.7,
            label=f"IRLS (итерация {i + 1})",
        )

    # Финальная модель
    y_vals_final = final_model.predict(x_vals_const)
    plt.plot(x_vals, y_vals_final, "red", linewidth=3, label="Финальная IRLS модель")

    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.title("Процесс сходимости IRLS модели", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return final_model, models_history, convergence_info, r2_train_final, r2_test_final


def task_5_outlier_analysis(final_model, X_train, y_train):
    """
    Задание 5: Анализ остатков финальной IRLS-модели и идентификация выбросов.

    Критерий: наблюдение является выбросным, если робастный стандартизованный
    остаток превышает робастное стандартное отклонение остатков.
    """
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 5: Анализ остатков IRLS-модели и идентификация выбросов")
    print("=" * 60)

    # Расчет остатков финальной модели
    X_train_const = sm.add_constant(X_train)
    residuals = final_model.resid

    # Стандартизованные остатки
    std_residuals = residuals / residuals.std()

    # Робастные стандартизованные остатки (MAD-based)
    mad = median_abs_deviation(residuals)
    scale = mad / config.NORMAL_DISTRIBUTION_CONST
    robust_std_residuals = residuals / scale

    # Робастное стандартное отклонение остатков
    robust_std = np.std(robust_std_residuals)

    print(f"Робастное стандартное отклонение остатков: {float(robust_std):.4f}")

    # Критерий для выбросов: |robust_std_residual| > robust_std
    outlier_mask = np.abs(robust_std_residuals) > robust_std
    outliers = X_train["x"][outlier_mask]

    print(f"Количество выбросов по критерию: {np.sum(outlier_mask)}")
    if len(outliers) > 0:
        print(f"Значения x для выбросов: {outliers.values}")

    # График зависимости остатков от x
    plt.figure(figsize=(15, 10))

    # Стандартизованные остатки
    plt.subplot(2, 2, 1)
    plt.scatter(X_train["x"], std_residuals, alpha=0.7, s=50)
    plt.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    if np.sum(outlier_mask) > 0:
        plt.scatter(
            X_train["x"][outlier_mask],
            std_residuals[outlier_mask],
            color="red",
            s=80,
            label="Выбросы",
            edgecolors="black",
            linewidth=1,
        )
    plt.xlabel("x")
    plt.ylabel("Стандартизованные остатки")
    plt.title("Стандартизованные остатки vs x")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Робастные стандартизованные остатки
    plt.subplot(2, 2, 2)
    plt.scatter(X_train["x"], robust_std_residuals, alpha=0.7, s=50)
    plt.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    plt.axhline(
        y=robust_std,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"±{robust_std:.2f}",
    )
    plt.axhline(y=-robust_std, color="orange", linestyle="--", alpha=0.7)
    if np.sum(outlier_mask) > 0:
        plt.scatter(
            X_train["x"][outlier_mask],
            robust_std_residuals[outlier_mask],
            color="red",
            s=80,
            label="Выбросы",
            edgecolors="black",
            linewidth=1,
        )
    plt.xlabel("x")
    plt.ylabel("Робастные стандартизованные остатки")
    plt.title("Робастные стандартизованные остатки vs x")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Гистограмма робастных остатков
    plt.subplot(2, 2, 3)
    plt.hist(robust_std_residuals, bins=20, alpha=0.7, edgecolor="black")
    plt.axvline(
        x=robust_std,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"+{robust_std:.2f}",
    )
    plt.axvline(
        x=-robust_std,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"-{robust_std:.2f}",
    )
    if np.sum(outlier_mask) > 0:
        for outlier in robust_std_residuals[outlier_mask]:
            plt.axvline(x=outlier, color="red", alpha=0.8, linewidth=2)
    plt.xlabel("Робастные стандартизованные остатки")
    plt.ylabel("Частота")
    plt.title("Гистограмма робастных стандартизованных остатков")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Q-Q plot для проверки нормальности
    plt.subplot(2, 2, 4)
    probplot(robust_std_residuals, dist="norm", plot=plt)
    plt.title("Q-Q plot робастных стандартизованных остатков")

    plt.tight_layout()
    plt.show()

    return {
        "outliers": outliers,
        "outlier_mask": outlier_mask,
        "robust_std": robust_std,
        "std_residuals": std_residuals,
        "robust_std_residuals": robust_std_residuals,
    }


def task_6_comparison(
    ols_model, final_model, r2_train_ols, r2_test_ols, r2_train_irls, r2_test_irls
):
    """
    Задание 6: Сравнение методов OLS и IRLS.
    """
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 6: Сравнение методов OLS и IRLS")
    print("=" * 60)

    print("\nСравнение коэффициентов:")
    print(f"{'Метод':<10} {'const':<12} {'x':<12} {'R² (train)':<12} {'R² (test)':<12}")
    print("-" * 60)
    print(
        f"{'OLS':<10} {ols_model.params['const']:<12.6f} {ols_model.params['x']:<12.6f} {r2_train_ols:<12.4f} {r2_test_ols:<12.4f}"
    )
    print(
        f"{'IRLS':<10} {final_model.params['const']:<12.6f} {final_model.params['x']:<12.6f} {r2_train_irls:<12.4f} {r2_test_irls:<12.4f}"
    )

    # Определение лучшей модели по R² на тестовой выборке
    if r2_test_irls > r2_test_ols:
        best_method = "IRLS"
        improvement = r2_test_irls - r2_test_ols
        print("\n✅ IRLS показывает лучшие результаты на тестовой выборке!")
        print(f"   Улучшение R²: {improvement:.4f}")
    elif r2_test_irls < r2_test_ols:
        best_method = "OLS"
        degradation = r2_test_ols - r2_test_irls
        print("\n✅ OLS показывает лучшие результаты на тестовой выборке!")
        print(f"   IRLS хуже на: {degradation:.4f}")
    else:
        best_method = "оба метода"
        print("\n➖ Методы показывают одинаковые результаты на тестовой выборке")

    print(f"\n{'=' * 60}")
    print("ВЫВОДЫ:")
    print(f"{'=' * 60}")
    print(f"• Лучший метод: {best_method}")
    print(
        f"• OLS коэффициенты: const={ols_model.params['const']:.4f}, x={ols_model.params['x']:.4f}"
    )
    print(
        f"• IRLS коэффициенты: const={final_model.params['const']:.4f}, x={final_model.params['x']:.4f}"
    )

    if best_method == "IRLS":
        print("• IRLS лучше справляется с выбросами в данных")
        print("• Рекомендуется использовать IRLS для данных с потенциальными выбросами")
    elif best_method == "OLS":
        print("• Данные не содержат значимых выбросов или OLS достаточно робастен")
        print("• OLS предпочтительнее из-за простоты интерпретации")

    print("• Все задания лабораторной работы выполнены успешно!")


def main():
    """Главная функция для выполнения всех заданий."""
    print("ЛАБОРАТОРНАЯ РАБОТА: Итеративно взвешенные наименьшие квадраты (IRLS)")
    print("=" * 80)

    # Загрузка данных
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Размер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")

    # Задание 1: OLS модель
    ols_model, r2_train_ols, r2_test_ols, ols_pred_test = task_1_ols_model(
        X_train, X_test, y_train, y_test
    )

    # Задание 2: Анализ остатков OLS
    residuals_analysis = task_2_residuals_analysis(
        ols_model, X_train, X_test, y_train, y_test, ols_pred_test
    )

    # Задание 3: WLS модель (первая итерация IRLS)
    wls_model, weights, r2_train_wls, r2_test_wls = task_3_wls_model(
        X_train,
        X_test,
        y_train,
        y_test,
        ols_model.resid,
        residuals_analysis["scale_train"],
    )

    # Задание 4: Полный итеративный процесс IRLS
    final_model, models_history, convergence_info, r2_train_irls, r2_test_irls = (
        task_4_irls_process(X_train, X_test, y_train, y_test)
    )

    # Задание 5: Анализ остатков финальной модели
    outlier_analysis = task_5_outlier_analysis(final_model, X_train, y_train)

    # Задание 6: Сравнение методов
    task_6_comparison(
        ols_model, final_model, r2_train_ols, r2_test_ols, r2_train_irls, r2_test_irls
    )

    print("\n" + "=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА")
    print("=" * 80)


if __name__ == "__main__":
    main()
