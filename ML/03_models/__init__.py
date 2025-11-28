"""
Модуль моделей для прогнозирования волатильности.

Основные компоненты:
- train_global_model: Обучение глобальной LightGBM модели
- inference: Инференс и прогнозирование

Использование:
    # Обучение
    python 03_models/train_global_model.py
    
    # Инференс
    from models.inference import GlobalQuantileModel
    model = GlobalQuantileModel()
    model.load_models()
    predictions = model.predict(data)
"""

from .inference import GlobalQuantileModel, load_and_predict

__all__ = [
    'GlobalQuantileModel',
    'load_and_predict'
]

