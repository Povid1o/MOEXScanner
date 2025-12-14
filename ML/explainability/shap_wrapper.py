"""
Обертка для SHAP объяснений LightGBM моделей.

Предоставляет удобный интерфейс для вычисления SHAP значений
с оптимизацией производительности.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Union, Optional
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP не установлен. Установите: pip install shap")


class ShapExplainer:
    """
    Обертка для SHAP TreeExplainer с оптимизацией производительности.
    
    TreeExplainer создается один раз при инициализации и переиспользуется
    для всех последующих вызовов explain_local.
    
    Пример использования:
        from inference import GlobalQuantileModel
        
        # Загружаем модель
        model = GlobalQuantileModel()
        model.load_models()
        lgbm_model = model.models[0.50]  # Медианная модель
        
        # Подготавливаем фоновый датасет (например, выборка из X_train)
        background_data = X_train.sample(100)  # 100 образцов для фона
        
        # Создаем explainer
        explainer = ShapExplainer(lgbm_model, background_data)
        
        # Объясняем один вектор признаков
        result = explainer.explain_local(features_vector)
        
        # Получаем форматированный вывод
        formatted = explainer.format_explanation(result)
    """
    
    def __init__(
        self,
        model: lgb.Booster,
        background_data: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None
    ):
        """
        Инициализация SHAP explainer.
        
        Args:
            model: Обученная LightGBM модель (Booster)
            background_data: Фоновый датасет для инициализации TreeExplainer.
                            Может быть DataFrame или numpy array.
                            Рекомендуется использовать выборку из X_train (50-100 образцов).
            feature_names: Список имен признаков. Если None, берется из модели.
        
        Raises:
            ImportError: Если SHAP не установлен
            ValueError: Если данные несовместимы с моделью
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP не установлен. Установите: pip install shap"
            )
        
        self.model = model
        
        # Получаем имена признаков из модели, если не указаны
        if feature_names is None:
            self.feature_names = model.feature_name()
        else:
            self.feature_names = feature_names
        
        # Преобразуем background_data в правильный формат
        if isinstance(background_data, pd.DataFrame):
            # Проверяем, что все признаки присутствуют
            missing = set(self.feature_names) - set(background_data.columns)
            if missing:
                raise ValueError(
                    f"Отсутствуют признаки в background_data: {missing}"
                )
            # Берем только нужные признаки в правильном порядке
            background_array = background_data[self.feature_names].values
        elif isinstance(background_data, np.ndarray):
            background_array = background_data
            if background_array.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Несоответствие размерности: "
                    f"background_data имеет {background_array.shape[1]} признаков, "
                    f"модель ожидает {len(self.feature_names)}"
                )
        else:
            raise TypeError(
                f"background_data должен быть pd.DataFrame или np.ndarray, "
                f"получен {type(background_data)}"
            )
        
        # Создаем TreeExplainer один раз при инициализации
        # Это оптимизирует производительность - explainer не пересоздается
        # на каждом вызове explain_local
        print(f"🔧 Инициализация SHAP TreeExplainer с {len(background_array)} фоновыми образцами...")
        self.explainer = shap.TreeExplainer(
            model,
            background_array,
            feature_perturbation="tree_path_dependent"  # Быстрый режим для TreeExplainer
        )
        
        print(f"✅ SHAP Explainer готов. Признаков: {len(self.feature_names)}")
    
    def explain_local(
        self,
        features_vector: Union[pd.Series, pd.DataFrame, np.ndarray]
    ) -> Dict:
        """
        Вычисляет SHAP значения для одного вектора признаков.
        
        Args:
            features_vector: Вектор признаков для объяснения.
                            Может быть:
                            - pd.Series (индекс - имена признаков)
                            - pd.DataFrame (одна строка)
                            - np.ndarray (1D или 2D с одной строкой)
        
        Returns:
            Словарь с ключами:
                - 'shap_values': np.ndarray - массив SHAP значений
                - 'base_value': float - базовое значение (средний прогноз на фоне)
                - 'feature_names': List[str] - список имен признаков
                - 'prediction': float - прогноз модели для данного вектора
        """
        # Преобразуем входные данные в numpy array
        if isinstance(features_vector, pd.Series):
            # Проверяем, что все признаки присутствуют
            missing = set(self.feature_names) - set(features_vector.index)
            if missing:
                warnings.warn(
                    f"Отсутствуют признаки: {missing}. "
                    f"Будут использованы значения по умолчанию (0)."
                )
            # Создаем массив в правильном порядке
            feature_array = np.array([
                features_vector.get(name, 0.0) for name in self.feature_names
            ]).reshape(1, -1)
        
        elif isinstance(features_vector, pd.DataFrame):
            if len(features_vector) != 1:
                warnings.warn(
                    f"DataFrame содержит {len(features_vector)} строк. "
                    f"Будет использована первая строка."
                )
            # Проверяем наличие признаков
            missing = set(self.feature_names) - set(features_vector.columns)
            if missing:
                warnings.warn(
                    f"Отсутствуют признаки: {missing}. "
                    f"Будут использованы значения по умолчанию (0)."
                )
            # Берем первую строку в правильном порядке
            feature_array = np.array([
                features_vector.iloc[0].get(name, 0.0) 
                for name in self.feature_names
            ]).reshape(1, -1)
        
        elif isinstance(features_vector, np.ndarray):
            if features_vector.ndim == 1:
                feature_array = features_vector.reshape(1, -1)
            elif features_vector.ndim == 2:
                if features_vector.shape[0] != 1:
                    warnings.warn(
                        f"Массив содержит {features_vector.shape[0]} строк. "
                        f"Будет использована первая строка."
                    )
                feature_array = features_vector[:1, :]
            else:
                raise ValueError(
                    f"Неподдерживаемая размерность массива: {features_vector.ndim}"
                )
            
            if feature_array.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Несоответствие размерности: "
                    f"features_vector имеет {feature_array.shape[1]} признаков, "
                    f"модель ожидает {len(self.feature_names)}"
                )
        else:
            raise TypeError(
                f"features_vector должен быть pd.Series, pd.DataFrame или np.ndarray, "
                f"получен {type(features_vector)}"
            )
        
        # Вычисляем SHAP значения (быстро, т.к. explainer уже создан)
        shap_values = self.explainer.shap_values(feature_array)
        
        # TreeExplainer возвращает массив, берем первую строку
        if isinstance(shap_values, list):
            # Для мультиклассовых моделей shap_values - список
            shap_array = shap_values[0] if len(shap_values) > 0 else shap_values
        else:
            shap_array = shap_values
        
        # Для одной строки shap_array может быть 1D или 2D
        if shap_array.ndim == 2:
            shap_array = shap_array[0]
        
        # Получаем базовое значение
        base_value = float(self.explainer.expected_value)
        
        # Если expected_value - массив (мультикласс), берем первый элемент
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[0])
        
        # Вычисляем прогноз модели
        prediction = float(self.model.predict(feature_array)[0])
        
        return {
            'shap_values': shap_array,
            'base_value': base_value,
            'feature_names': self.feature_names.copy(),
            'prediction': prediction
        }
    
    def format_explanation(
        self,
        explanation: Dict,
        top_n: Optional[int] = None
    ) -> List[Dict]:
        """
        Форматирует объяснение в список словарей, отсортированный по абсолютной величине вклада.
        
        Args:
            explanation: Результат вызова explain_local()
            top_n: Если указано, возвращает только топ-N признаков по вкладу.
                   Если None, возвращает все признаки.
        
        Returns:
            Список словарей вида:
            [
                {
                    'feature': 'parkinson_vol',
                    'value': 0.05,        # Значение признака
                    'contribution': 0.02   # SHAP значение (вклад в прогноз)
                },
                ...
            ]
            Отсортирован по убыванию абсолютного вклада.
        """
        shap_values = explanation['shap_values']
        feature_names = explanation['feature_names']
        
        # Получаем значения признаков из explanation, если они есть
        # Иначе используем shap_values как есть (значения признаков недоступны)
        feature_values = explanation.get('feature_values', None)
        
        # Формируем список словарей
        result = []
        for i, feature_name in enumerate(feature_names):
            contribution = float(shap_values[i])
            
            # Значение признака (если доступно)
            if feature_values is not None:
                value = float(feature_values[i])
            else:
                # Если значения признаков не переданы, используем None
                value = None
            
            result.append({
                'feature': feature_name,
                'value': value,
                'contribution': contribution
            })
        
        # Сортируем по абсолютной величине вклада (по убыванию)
        result.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        # Возвращаем топ-N, если указано
        if top_n is not None:
            result = result[:top_n]
        
        return result
    
    def explain_and_format(
        self,
        features_vector: Union[pd.Series, pd.DataFrame, np.ndarray],
        top_n: Optional[int] = None
    ) -> List[Dict]:
        """
        Удобный метод: вычисляет объяснение и сразу форматирует его.
        
        Args:
            features_vector: Вектор признаков для объяснения
            top_n: Количество топ признаков для возврата (None = все)
        
        Returns:
            Отформатированный список словарей (см. format_explanation)
        """
        explanation = self.explain_local(features_vector)
        
        # Добавляем значения признаков в explanation для format_explanation
        if isinstance(features_vector, pd.Series):
            feature_values = np.array([
                features_vector.get(name, 0.0) for name in self.feature_names
            ])
        elif isinstance(features_vector, pd.DataFrame):
            feature_values = np.array([
                features_vector.iloc[0].get(name, 0.0) 
                for name in self.feature_names
            ])
        elif isinstance(features_vector, np.ndarray):
            if features_vector.ndim == 1:
                feature_values = features_vector
            else:
                feature_values = features_vector[0]
        else:
            feature_values = None
        
        if feature_values is not None:
            explanation['feature_values'] = feature_values
        
        return self.format_explanation(explanation, top_n=top_n)

