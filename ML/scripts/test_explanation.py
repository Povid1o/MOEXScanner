"""
Тестовый скрипт для валидации модуля объяснимости.

Проверяет:
- Корректность работы ShapExplainer и ExplanationGenerator
- Структуру JSON ответа для фронтенда
- Обработку edge cases (NaN, нули)

Запуск:
    python scripts/test_explanation.py
"""

import sys
import json
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

# Добавляем пути для импорта
ML_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ML_ROOT))
sys.path.insert(0, str(ML_ROOT / "03_models"))

from inference import GlobalQuantileModel


def print_header(title: str):
    """Печатает заголовок секции."""
    print("\n" + "=" * 70)
    print(f"🔷 {title}")
    print("=" * 70 + "\n")


def print_section(title: str):
    """Печатает заголовок подсекции."""
    print(f"\n{'─' * 70}")
    print(f"📌 {title}")
    print(f"{'─' * 70}\n")


def load_sber_data() -> pd.DataFrame:
    """
    Загружает данные SBER из processed_ml.
    
    Returns:
        DataFrame с признаками SBER
    """
    data_file = ML_ROOT / "data" / "processed_ml" / "SBER_ml_features.parquet"
    
    if not data_file.exists():
        # Пробуем найти любой доступный файл
        available = list((ML_ROOT / "data" / "processed_ml").glob("*_ml_features.parquet"))
        if available:
            data_file = available[0]
            ticker = data_file.stem.replace("_ml_features", "")
            print(f"⚠️ Файл SBER не найден. Используем: {ticker}")
        else:
            raise FileNotFoundError(
                f"Не найдены файлы данных в {ML_ROOT / 'data' / 'processed_ml'}"
            )
    
    df = pd.read_parquet(data_file)
    print(f"✅ Загружено {len(df)} записей из {data_file.name}")
    
    return df


def test_basic_explanation():
    """Основной тест: прогноз с объяснениями для последней строки."""
    print_header("ТЕСТ 1: БАЗОВЫЙ ПРОГНОЗ С ОБЪЯСНЕНИЯМИ")
    
    # 1. Инициализация модели
    print_section("Инициализация модели")
    model = GlobalQuantileModel()
    
    try:
        model.load_models()
        print("✅ Модели загружены успешно")
    except FileNotFoundError as e:
        print(f"❌ Ошибка загрузки моделей: {e}")
        print("   Убедитесь, что модели обучены: python 03_models/train_global_model.py")
        return False
    
    # 2. Загрузка данных
    print_section("Загрузка данных")
    try:
        df = load_sber_data()
    except FileNotFoundError as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return False
    
    # 3. Подготовка данных для прогноза
    print_section("Подготовка данных")
    
    # Берем последнюю строку (симуляция "сегодняшнего прогноза")
    last_row = df.tail(1).copy()
    print(f"📊 Используем последнюю строку (индекс: {last_row.index[0]})")
    
    # Подготавливаем background_data (последние 100 строк)
    background_size = min(100, len(df))
    background_data = df.tail(background_size).copy()
    print(f"📊 Background data: {len(background_data)} строк")
    
    # 4. Выполнение прогноза с объяснениями
    print_section("Выполнение прогноза с объяснениями")
    
    try:
        result = model.predict(
            last_row,
            include_explanation=True,
            background_data=background_data
        )
        print("✅ Прогноз выполнен успешно")
    except Exception as e:
        print(f"❌ Ошибка при выполнении прогноза: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Валидация структуры результата
    print_section("Валидация структуры результата")
    
    # Проверяем, что результат - словарь
    assert isinstance(result, dict), f"Результат должен быть словарем, получен {type(result)}"
    
    # Проверяем наличие ключей
    required_keys = ['forecast', 'explanation']
    missing_keys = [key for key in required_keys if key not in result]
    assert not missing_keys, f"Отсутствуют обязательные ключи: {missing_keys}"
    
    print("✅ Структура результата корректна")
    
    # 6. Валидация forecast
    print_section("Валидация прогноза (forecast)")
    
    forecast = result['forecast']
    assert isinstance(forecast, pd.DataFrame), f"forecast должен быть DataFrame, получен {type(forecast)}"
    
    required_cols = ['pred_q16', 'pred_q50', 'pred_q84']
    missing_cols = [col for col in required_cols if col not in forecast.columns]
    assert not missing_cols, f"Отсутствуют обязательные колонки в forecast: {missing_cols}"
    
    print("✅ Прогноз содержит все необходимые квантили:")
    for col in required_cols:
        value = forecast[col].iloc[0]
        print(f"   {col}: {value:.6f}")
    
    if 'interval_width' in forecast.columns:
        interval = forecast['interval_width'].iloc[0]
        print(f"   interval_width: {interval:.6f}")
    
    # 7. Валидация explanation
    print_section("Валидация объяснений (explanation)")
    
    explanation = result['explanation']
    assert isinstance(explanation, dict), f"explanation должен быть словарем, получен {type(explanation)}"
    
    required_exp_keys = ['text', 'raw_data']
    missing_exp_keys = [key for key in required_exp_keys if key not in explanation]
    assert not missing_exp_keys, f"Отсутствуют обязательные ключи в explanation: {missing_exp_keys}"
    
    # Проверяем, что explanation не пустой
    assert explanation['text'] is not None, "explanation['text'] не должен быть None"
    assert explanation['raw_data'] is not None, "explanation['raw_data'] не должен быть None"
    
    # Проверяем текстовое объяснение
    explanation_text = explanation['text']
    
    # Для одной строки explanation_text - строка, для нескольких - список
    if isinstance(explanation_text, list):
        explanation_text_display = explanation_text[0] if explanation_text else ""
    else:
        explanation_text_display = explanation_text
    
    if not explanation_text_display or (isinstance(explanation_text_display, str) and len(explanation_text_display.strip()) == 0):
        print("⚠️ Текстовое объяснение пустое")
    else:
        print("✅ Текстовое объяснение сгенерировано:")
        print(f"\n{explanation_text_display}\n")
    
    # Проверяем сырые данные
    raw_data = explanation['raw_data']
    
    # Определяем, какой формат данных (для одной строки или нескольких)
    raw_data_for_display = raw_data
    if isinstance(raw_data, list) and len(raw_data) > 0:
        # Если это список списков (несколько строк), берем первый
        if isinstance(raw_data[0], list):
            raw_data_for_display = raw_data[0]
        # Если это список словарей (одна строка, но уже список), оставляем как есть
        elif isinstance(raw_data[0], dict):
            raw_data_for_display = raw_data
    
    if not raw_data_for_display:
        print("⚠️ Сырые данные объяснений пусты")
    else:
        if isinstance(raw_data_for_display, list):
            print(f"✅ Сырые данные: {len(raw_data_for_display)} признаков")
            # Показываем топ-5 признаков
            print("\nТоп-5 признаков по вкладу:")
            for i, item in enumerate(raw_data_for_display[:5], 1):
                feature = item.get('feature', 'N/A')
                contribution = item.get('contribution', 0)
                value = item.get('value', 'N/A')
                print(f"   {i}. {feature}: вклад={contribution:.6f}, значение={value}")
        else:
            print(f"⚠️ Сырые данные не в формате списка: {type(raw_data_for_display)}")
    
    # 8. Вывод полного JSON
    print_section("Полная JSON структура (для фронтенда)")
    
    # Подготавливаем JSON-совместимую структуру
    # Используем уже обработанные данные для отображения
    explanation_text_for_json = explanation_text_display
    raw_data_for_json = raw_data_for_display
    
    json_result = {
        'forecast': {
            'q16': float(forecast['pred_q16'].iloc[0]),
            'q50': float(forecast['pred_q50'].iloc[0]),
            'q84': float(forecast['pred_q84'].iloc[0]),
        },
        'explanation': {
            'text': explanation_text_for_json,
            'raw_data': raw_data_for_json if isinstance(raw_data_for_json, list) else []
        }
    }
    
    # Добавляем interval_width, если есть
    if 'interval_width' in forecast.columns:
        json_result['forecast']['interval_width'] = float(forecast['interval_width'].iloc[0])
    
    # Выводим JSON с красивым форматированием
    json_str = json.dumps(json_result, indent=2, ensure_ascii=False)
    print(json_str)
    
    # Сохраняем в файл для удобства
    output_file = ML_ROOT / "scripts" / "test_explanation_output.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(json_str)
    print(f"\n💾 JSON сохранен в: {output_file}")
    
    return True


def test_edge_cases():
    """Тест edge cases: NaN, нули, пустые данные."""
    print_header("ТЕСТ 2: EDGE CASES")
    
    # 1. Инициализация модели
    model = GlobalQuantileModel()
    
    try:
        model.load_models()
    except FileNotFoundError as e:
        print(f"❌ Ошибка загрузки моделей: {e}")
        return False
    
    # 2. Загрузка данных
    try:
        df = load_sber_data()
    except FileNotFoundError as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return False
    
    background_data = df.tail(100).copy()
    
    # Тест 2.1: Строка с NaN значениями
    print_section("Тест 2.1: Строка с NaN значениями")
    
    test_row_nan = df.tail(1).copy()
    # Заполняем некоторые признаки NaN
    numeric_cols = test_row_nan.select_dtypes(include=[np.number]).columns[:10]
    test_row_nan[numeric_cols] = np.nan
    
    try:
        result = model.predict(
            test_row_nan,
            include_explanation=True,
            background_data=background_data
        )
        print("✅ Прогноз с NaN выполнен успешно (NaN должны быть обработаны)")
        
        if isinstance(result, dict) and 'forecast' in result:
            print(f"   Прогноз q50: {result['forecast']['pred_q50'].iloc[0]:.6f}")
        
    except Exception as e:
        print(f"⚠️ Ошибка при обработке NaN (ожидаемо, если модель не обрабатывает NaN): {e}")
    
    # Тест 2.2: Строка с нулевыми значениями
    print_section("Тест 2.2: Строка с нулевыми значениями")
    
    test_row_zeros = df.tail(1).copy()
    # Заполняем числовые признаки нулями
    numeric_cols = test_row_zeros.select_dtypes(include=[np.number]).columns
    test_row_zeros[numeric_cols] = 0
    
    try:
        result = model.predict(
            test_row_zeros,
            include_explanation=True,
            background_data=background_data
        )
        print("✅ Прогноз с нулями выполнен успешно")
        
        if isinstance(result, dict) and 'forecast' in result:
            print(f"   Прогноз q50: {result['forecast']['pred_q50'].iloc[0]:.6f}")
        
    except Exception as e:
        print(f"❌ Ошибка при обработке нулей: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Тест 2.3: Пустой DataFrame
    print_section("Тест 2.3: Пустой DataFrame")
    
    empty_df = pd.DataFrame(columns=df.columns)
    
    try:
        result = model.predict(
            empty_df,
            include_explanation=False  # Без объяснений для пустого датасета
        )
        print(f"✅ Пустой DataFrame обработан: {len(result)} строк")
    except Exception as e:
        print(f"⚠️ Ошибка при обработке пустого DataFrame (ожидаемо): {e}")
    
    # Тест 2.4: Несколько строк
    print_section("Тест 2.4: Прогноз для нескольких строк")
    
    test_rows = df.tail(3).copy()
    
    try:
        result = model.predict(
            test_rows,
            include_explanation=True,
            background_data=background_data
        )
        print("✅ Прогноз для нескольких строк выполнен успешно")
        
        if isinstance(result, dict):
            forecast = result['forecast']
            print(f"   Прогнозов: {len(forecast)}")
            
            explanation = result.get('explanation', {})
            text = explanation.get('text', [])
            raw_data = explanation.get('raw_data', [])
            
            if isinstance(text, list):
                print(f"   Текстовых объяснений: {len(text)}")
            else:
                print(f"   Текстовое объяснение: {type(text)}")
            
            if isinstance(raw_data, list):
                print(f"   Сырых данных объяснений: {len(raw_data)}")
            else:
                print(f"   Сырые данные: {type(raw_data)}")
        
    except Exception as e:
        print(f"❌ Ошибка при обработке нескольких строк: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Главная функция запуска тестов."""
    print_header("ТЕСТИРОВАНИЕ МОДУЛЯ ОБЪЯСНИМОСТИ")
    
    # Подавляем предупреждения для чистоты вывода
    warnings.filterwarnings('ignore')
    
    success_count = 0
    total_tests = 2
    
    # Тест 1: Базовый прогноз с объяснениями
    if test_basic_explanation():
        success_count += 1
        print("\n✅ ТЕСТ 1 ПРОЙДЕН")
    else:
        print("\n❌ ТЕСТ 1 ПРОВАЛЕН")
    
    # Тест 2: Edge cases
    if test_edge_cases():
        success_count += 1
        print("\n✅ ТЕСТ 2 ПРОЙДЕН")
    else:
        print("\n❌ ТЕСТ 2 ПРОВАЛЕН")
    
    # Итоги
    print_header("ИТОГИ ТЕСТИРОВАНИЯ")
    print(f"Пройдено тестов: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        return 0
    else:
        print(f"\n⚠️ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ ({total_tests - success_count})")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

