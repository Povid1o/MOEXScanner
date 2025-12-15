"""
Модуль инференса для прогнозирования волатильности.

Предоставляет функции для:
- Загрузки обученных квантильных моделей LightGBM
- Прогнозирования на новых данных
- Получения интервального прогноза [q16, q84]
- Ансамблевого прогноза (LightGBM + GARCH)
- Объяснений прогнозов через SHAP (опционально)

Использование (только LightGBM):
    from inference import GlobalQuantileModel
    
    model = GlobalQuantileModel()
    model.load_models()
    
    predictions = model.predict(new_data)
    
Использование (ансамбль LightGBM + GARCH):
    model = GlobalQuantileModel(use_ensemble=True)
    model.load_models()
    
    predictions = model.predict_ensemble(new_data)
    # predictions содержит колонки: pred_q16, pred_q50, pred_q84, ensemble_*

Использование (с объяснениями):
    model = GlobalQuantileModel()
    model.load_models()
    
    result = model.predict(new_data, include_explanation=True, background_data=X_train)
    # result содержит:
    #   - 'forecast': DataFrame с прогнозами
    #   - 'explanation': Dict с текстовым объяснением и сырыми данными для визуализации
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import Dict, Optional, List, Union
import warnings
import sys

warnings.filterwarnings('ignore')

# Добавляем путь для импорта из models/
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from models.ensemble import EnsembleModel, SimpleGARCH, EnsembleWeights
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    warnings.warn("Модуль ensemble не найден. Ансамблевые методы недоступны.")

# Импорты для объяснимости
try:
    from explainability.shap_wrapper import ShapExplainer
    from explainability.text_generator import ExplanationGenerator
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False
    warnings.warn("Модули explainability не найдены. Объяснимость недоступна.")


class GlobalQuantileModel:
    """
    Класс для загрузки и использования обученных квантильных моделей.
    
    Поддерживает:
    - Чистый LightGBM прогноз (по умолчанию)
    - Ансамблевый прогноз LightGBM + GARCH (use_ensemble=True)
    
    Атрибуты:
        models: Dict[float, lgb.Booster] - словарь моделей по квантилям
        feature_names: List[str] - список признаков модели
        ensemble: EnsembleModel - ансамблевая модель (если включена)
        garch: SimpleGARCH - GARCH модель для ансамбля
        explainer: ShapExplainer - объяснитель SHAP (ленивая инициализация)
        text_generator: ExplanationGenerator - генератор текстовых объяснений
    """
    
    # Категориальные признаки (должны совпадать с train_global_model.py)
    # ВАЖНО: Если модель обучена БЕЗ ticker_id, уберите его отсюда!
    # Проверьте config/training_config.py для актуального списка
    CATEGORICAL_FEATURES = [
        # 'ticker_id',  # Закомментировано для эксперимента без ticker_id
        'sector_id',
        'is_month_end',
        'is_month_start',
        'day_of_week',
        'vp_above_va',
        'volume_spike',
        'trend_signal',
        'price_position_ma'
    ]
    
    def __init__(
        self, 
        model_dir: Optional[Path] = None,
        use_ensemble: bool = False,
        ensemble_weights: Optional[Dict[str, float]] = None
    ):
        """
        Инициализация.
        
        Args:
            model_dir: Директория с сохранёнными моделями
            use_ensemble: Использовать ли ансамбль с GARCH
            ensemble_weights: Веса ансамбля {'lgbm': 0.7, 'garch': 0.3}
        """
        if model_dir is None:
            self.model_dir = Path(__file__).parent.parent / "data" / "models"
        else:
            self.model_dir = Path(model_dir)
        
        self.models: Dict[float, lgb.Booster] = {}
        self.feature_names: List[str] = []
        self.quantiles = [0.16, 0.50, 0.84]
        self._loaded = False
        
        # Ансамбль
        self.use_ensemble = use_ensemble and ENSEMBLE_AVAILABLE
        self.ensemble: Optional['EnsembleModel'] = None
        self.garch: Optional['SimpleGARCH'] = None
        
        if self.use_ensemble:
            if ensemble_weights is None:
                ensemble_weights = {'lgbm': 0.7, 'garch': 0.3}
            self.ensemble = EnsembleModel(weights=ensemble_weights)
            self.garch = SimpleGARCH()
            print(f"📦 Ансамблевый режим: LightGBM ({ensemble_weights['lgbm']}) + GARCH ({ensemble_weights['garch']})")
        elif use_ensemble and not ENSEMBLE_AVAILABLE:
            warnings.warn("Ансамбль запрошен, но модуль ensemble недоступен. Используется только LightGBM.")
        
        # Объяснимость (ленивая инициализация)
        self.explainer: Optional['ShapExplainer'] = None
        if EXPLAINABILITY_AVAILABLE:
            self.text_generator = ExplanationGenerator()
        else:
            self.text_generator = None
    
    def load_models(self) -> None:
        """
        Загружает все квантильные модели из директории.
        
        Raises:
            FileNotFoundError: Если модели не найдены
        """
        print("📥 Загрузка моделей...")
        
        for alpha in self.quantiles:
            filename = f"global_lgbm_q{int(alpha*100)}.txt"
            path = self.model_dir / filename
            
            if not path.exists():
                raise FileNotFoundError(f"Модель не найдена: {path}")
            
            self.models[alpha] = lgb.Booster(model_file=str(path))
            print(f"   ✅ Загружена: {filename}")
        
        # Получаем список признаков из первой модели
        self.feature_names = self.models[0.50].feature_name()
        self._loaded = True
        
        print(f"📋 Признаков в модели: {len(self.feature_names)}")
    
    def _init_explainer(
        self,
        background_data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Ленивая инициализация SHAP explainer.
        
        Инициализирует explainer только один раз, используя медианную модель (q50).
        Если background_data не предоставлен, TreeExplainer будет работать без фона
        (с меньшей точностью base_value, но все еще функционален).
        
        Args:
            background_data: Фоновый датасет для инициализации TreeExplainer.
                           Если None, explainer инициализируется без фона.
                           Рекомендуется использовать выборку из X_train (50-100 образцов).
        """
        if not EXPLAINABILITY_AVAILABLE:
            raise RuntimeError(
                "Объяснимость недоступна. Установите модули explainability."
            )
        
        if self.explainer is not None:
            # Уже инициализирован, пропускаем
            return
        
        if not self._loaded:
            raise RuntimeError(
                "Модели не загружены! Вызовите load_models() сначала."
            )
        
        # Используем медианную модель (q50) для объяснений
        median_model = self.models[0.50]
        
        try:
            if background_data is not None:
                # Используем выборку из фоновых данных (50-100 образцов для скорости)
                sample_size = min(100, len(background_data))
                background_sample = background_data.sample(
                    n=sample_size,
                    random_state=42
                ) if len(background_data) > sample_size else background_data
                
                # Подготавливаем данные так же, как в predict
                background_prepared = background_sample[self.feature_names].copy()
                
                # Конвертируем категориальные признаки
                for col in self.CATEGORICAL_FEATURES:
                    if col in background_prepared.columns:
                        background_prepared[col] = background_prepared[col].astype('category')
                
                # Заполняем NaN
                numeric_cols = background_prepared.select_dtypes(include=[np.number]).columns
                background_prepared[numeric_cols] = background_prepared[numeric_cols].fillna(0)
                
                self.explainer = ShapExplainer(
                    median_model,
                    background_prepared,
                    feature_names=self.feature_names
                )
            else:
                # Инициализация без фоновых данных (TreeExplainer поддерживает это)
                # Создаем минимальный фоновый массив (одна строка нулей)
                dummy_background = np.zeros((1, len(self.feature_names)))
                self.explainer = ShapExplainer(
                    median_model,
                    dummy_background,
                    feature_names=self.feature_names
                )
                warnings.warn(
                    "SHAP explainer инициализирован без фоновых данных. "
                    "base_value может быть менее точным. "
                    "Рекомендуется передать background_data при первом вызове predict с include_explanation=True."
                )
        except Exception as e:
            warnings.warn(
                f"Не удалось инициализировать SHAP explainer: {e}. "
                f"Объяснения будут недоступны."
            )
            self.explainer = None

    def _build_importance_explanations(
        self,
        X_prepared: pd.DataFrame,
        predictions: pd.DataFrame,
        top_n: int = 10,
    ):
        """
        Fallback-объяснения на основе feature importance LightGBM.

        Используется, когда SHAP недоступен или падает. Возвращает
        список объяснений по строкам и соответствующие текстовые описания.
        """
        try:
            imp_df = self.get_feature_importance(importance_type="gain", top_n=top_n)
        except Exception as e:
            warnings.warn(f"Не удалось получить feature importance для fallback-объяснений: {e}")
            return None, None

        if imp_df.empty:
            return None, None

        total_imp = imp_df["importance"].sum()
        if total_imp == 0:
            return None, None

        # Базовый список признаков с нормализованным вкладом (доля важности)
        base_explanation = []
        for _, row in imp_df.iterrows():
            feature = row["feature"]
            contribution = float(row["importance"] / total_imp)
            base_explanation.append(
                {
                    "feature": feature,
                    "value": None,  # заполним позже из X_prepared
                    "contribution": contribution,
                }
            )

        explanations_list = []
        explanation_texts = []

        for idx in X_prepared.index:
            row_expl = []
            for item in base_explanation:
                feature = item["feature"]
                value = (
                    X_prepared.loc[idx, feature]
                    if feature in X_prepared.columns
                    else None
                )
                row_expl.append(
                    {
                        "feature": feature,
                        "value": value,
                        "contribution": item["contribution"],
                    }
                )

            explanations_list.append(row_expl)

            if self.text_generator is not None:
                q50_value = float(predictions.loc[idx, "pred_q50"])
                text = self.text_generator.generate_detailed_text(
                    row_expl,
                    prediction_value=q50_value,
                    top_n=min(5, len(row_expl)),
                )
            else:
                text = ""

            explanation_texts.append(text)

        return explanations_list, explanation_texts
    
    def predict(
        self, 
        X: pd.DataFrame, 
        return_interval: bool = True,
        include_explanation: bool = False,
        background_data: Optional[pd.DataFrame] = None
    ) -> Union[pd.DataFrame, Dict]:
        """
        Делает прогноз на новых данных с опциональными объяснениями.
        
        Args:
            X: DataFrame с признаками (должны совпадать с feature_names)
            return_interval: Если True, возвращает также ширину интервала
            include_explanation: Если True, возвращает словарь с прогнозом и объяснениями
            background_data: Фоновый датасет для инициализации SHAP explainer
                           (используется только при первом вызове с include_explanation=True)
        
        Returns:
            Если include_explanation=False:
                DataFrame с колонками: pred_q16, pred_q50, pred_q84, [interval_width]
            
            Если include_explanation=True:
                Dict с ключами:
                    - 'forecast': DataFrame с прогнозами
                    - 'explanation': Dict с ключами:
                        - 'text': str - текстовое объяснение
                        - 'raw_data': List[Dict] - сырые данные для визуализации
        """
        if not self._loaded:
            raise RuntimeError("Модели не загружены! Вызовите load_models() сначала.")
        
        # Проверяем наличие признаков
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            warnings.warn(f"⚠️ Отсутствуют признаки: {missing_features}")
        
        # Подготовка данных
        X_prepared = X[self.feature_names].copy() if set(self.feature_names).issubset(X.columns) else X.copy()
        
        # Конвертируем категориальные признаки в category тип
        # КРИТИЧНО: LightGBM требует одинаковый тип данных при train и predict
        for col in self.CATEGORICAL_FEATURES:
            if col in X_prepared.columns:
                X_prepared[col] = X_prepared[col].astype('category')
        
        # Заполняем NaN только для числовых колонок
        numeric_cols = X_prepared.select_dtypes(include=[np.number]).columns
        X_prepared[numeric_cols] = X_prepared[numeric_cols].fillna(0)
        
        # Прогнозы для каждого квантиля
        predictions = pd.DataFrame(index=X.index)
        
        for alpha in self.quantiles:
            col_name = f"pred_q{int(alpha*100)}"
            predictions[col_name] = self.models[alpha].predict(X_prepared)
        
        # Ширина интервала (мера неопределённости)
        if return_interval:
            predictions['interval_width'] = predictions['pred_q84'] - predictions['pred_q16']
        
        # Если объяснения не запрошены, возвращаем стандартный формат (обратная совместимость)
        if not include_explanation:
            return predictions
        
        # Генерируем объяснения
        try:
            # Инициализируем explainer, если еще не инициализирован
            if self.explainer is None:
                self._init_explainer(background_data=background_data)
            
            # Если SHAP explainer недоступен — используем fallback на feature importance
            if self.explainer is None or self.text_generator is None:
                warnings.warn(
                    "SHAP explainer недоступен. Используем fallback-объяснения на основе feature importance."
                )
                explanations_list, explanation_texts = self._build_importance_explanations(
                    X_prepared,
                    predictions,
                    top_n=10,
                )
            else:
                # Вычисляем SHAP-объяснения для каждой строки
                explanations_list = []
                explanation_texts = []
                
                for idx in X.index:
                    # Получаем вектор признаков для текущей строки
                    features_vector = X_prepared.loc[idx]
                    
                    # Получаем объяснение
                    formatted_explanation = self.explainer.explain_and_format(
                        features_vector,
                        top_n=10  # Топ-10 признаков для объяснения
                    )
                    
                    # Если SHAP вернул пустой список, для этой строки используем fallback
                    if not formatted_explanation:
                        fallback_expl_list, _ = self._build_importance_explanations(
                            X_prepared.loc[[idx]],
                            predictions.loc[[idx]],
                            top_n=10,
                        )
                        formatted_explanation = fallback_expl_list[0] if fallback_expl_list else []
                    
                    # Получаем значение прогноза (q50)
                    q50_value = predictions.loc[idx, 'pred_q50']
                    
                    # Генерируем текстовое объяснение
                    explanation_text = self.text_generator.generate_detailed_text(
                        formatted_explanation,
                        prediction_value=q50_value,
                        top_n=5
                    )
                    
                    explanations_list.append(formatted_explanation)
                    explanation_texts.append(explanation_text)
            
            # Формируем результат
            if explanations_list is None or explanation_texts is None:
                # Полностью не удалось построить объяснения — возвращаем пустую структуру
                explanation_payload = {
                    'text': "",
                    'raw_data': []
                }
            else:
                explanation_payload = {
                    'text': explanation_texts if len(explanation_texts) > 1 else explanation_texts[0],
                    'raw_data': explanations_list if len(explanations_list) > 1 else explanations_list[0]
                }

            result = {
                'forecast': predictions,
                'explanation': explanation_payload
            }
            
            return result
            
        except Exception as e:
            # Если объяснения не удалось сгенерировать через SHAP, пробуем fallback
            warnings.warn(
                f"Не удалось сгенерировать SHAP-объяснения: {e}. "
                f"Пробуем fallback на основе feature importance."
            )
            explanations_list, explanation_texts = self._build_importance_explanations(
                X_prepared,
                predictions,
                top_n=10,
            )

            if explanations_list is None or explanation_texts is None:
                # Совсем не удалось получить объяснения
                return {
                    'forecast': predictions,
                    'explanation': {
                        'text': "",
                        'raw_data': []
                    }
                }

            return {
                'forecast': predictions,
                'explanation': {
                    'text': explanation_texts if len(explanation_texts) > 1 else explanation_texts[0],
                    'raw_data': explanations_list if len(explanations_list) > 1 else explanations_list[0]
                }
            }
    
    def predict_ensemble(
        self,
        X: pd.DataFrame,
        returns: Optional[pd.Series] = None,
        return_components: bool = False
    ) -> pd.DataFrame:
        """
        Ансамблевый прогноз: LightGBM + GARCH.
        
        Args:
            X: DataFrame с признаками для LightGBM
            returns: Series с log returns для GARCH (если None, берется из X['log_return'])
            return_components: Если True, возвращает также компоненты (lgbm, garch отдельно)
            
        Returns:
            DataFrame с колонками:
                - pred_q16, pred_q50, pred_q84: ансамблевые прогнозы
                - interval_width: ширина интервала
                - (опционально) lgbm_q50, garch_forecast: компоненты
        """
        if not self.use_ensemble or self.ensemble is None:
            warnings.warn("Ансамбль не инициализирован. Возвращаем только LightGBM прогноз.")
            return self.predict(X)
        
        if not self._loaded:
            raise RuntimeError("Модели не загружены! Вызовите load_models() сначала.")
        
        # 1. Прогноз LightGBM
        lgbm_predictions = self.predict(X, return_interval=True)
        
        # 2. Прогноз GARCH
        if returns is None:
            if 'log_return' in X.columns:
                returns = X['log_return']
            else:
                warnings.warn("log_return не найден в данных. GARCH будет использовать нули.")
                returns = pd.Series(np.zeros(len(X)), index=X.index)
        
        # Подгоняем GARCH и делаем прогноз
        returns_arr = returns.values
        
        # Rolling GARCH прогноз
        garch_forecasts = self.garch.forecast_rolling(returns_arr, window=20)
        
        # Заполняем NaN медианой LightGBM для первых значений
        nan_mask = np.isnan(garch_forecasts)
        if nan_mask.any():
            garch_forecasts[nan_mask] = lgbm_predictions['pred_q50'].values[nan_mask]
        
        # 3. Комбинируем через ансамбль
        ensemble_predictions = self.ensemble.predict(
            lgbm_predictions,
            garch_forecasts
        )
        
        # 4. Добавляем компоненты если нужно
        if return_components:
            ensemble_predictions['lgbm_q50'] = lgbm_predictions['pred_q50']
            ensemble_predictions['garch_forecast'] = garch_forecasts
        
        return ensemble_predictions
    
    def predict_with_uncertainty_ensemble(
        self,
        X: pd.DataFrame,
        returns: Optional[pd.Series] = None
    ) -> Dict:
        """
        Ансамблевый прогноз с расширенной статистикой неопределенности.
        
        Args:
            X: DataFrame с признаками
            returns: Series с log returns для GARCH
            
        Returns:
            Dict с прогнозами и метриками уверенности:
                - median: медианный прогноз
                - lower, upper: границы интервала
                - interval_width: ширина интервала
                - model_agreement: согласованность LightGBM и GARCH (0-1)
                - weights_used: использованные веса ансамбля
        """
        if not self.use_ensemble or self.ensemble is None:
            # Fallback на обычный predict_with_confidence
            return self.predict_with_confidence(X)
        
        # Получаем компоненты
        lgbm_preds = self.predict(X, return_interval=True)
        
        if returns is None and 'log_return' in X.columns:
            returns = X['log_return']
        elif returns is None:
            returns = pd.Series(np.zeros(len(X)), index=X.index)
        
        garch_forecasts = self.garch.forecast_rolling(returns.values, window=20)
        nan_mask = np.isnan(garch_forecasts)
        if nan_mask.any():
            garch_forecasts[nan_mask] = lgbm_preds['pred_q50'].values[nan_mask]
        
        # Используем метод ensemble для расширенной статистики
        result = self.ensemble.predict_with_uncertainty(
            lgbm_preds,
            garch_forecasts
        )
        
        return result
    
    def predict_with_confidence(
        self, 
        X: pd.DataFrame
    ) -> Dict:
        """
        Прогноз с дополнительной статистикой уверенности.
        
        Args:
            X: DataFrame с признаками
            
        Returns:
            Dict с прогнозами и метриками уверенности
        """
        preds = self.predict(X, return_interval=True)
        
        return {
            'median': preds['pred_q50'].values,
            'lower': preds['pred_q16'].values,
            'upper': preds['pred_q84'].values,
            'interval_width': preds['interval_width'].values,
            'mean_uncertainty': preds['interval_width'].mean()
        }
    
    def get_feature_importance(
        self, 
        importance_type: str = 'gain',
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Возвращает важность признаков для медианной модели.
        
        Args:
            importance_type: 'gain' или 'split'
            top_n: Количество топ признаков
            
        Returns:
            DataFrame с важностью признаков
        """
        if not self._loaded:
            raise RuntimeError("Модели не загружены!")
        
        # Используем медианную модель
        model = self.models[0.50]
        importance = model.feature_importance(importance_type=importance_type)
        
        imp_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return imp_df.head(top_n)


def load_and_predict(
    data_path: Path,
    model_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Удобная функция для быстрого прогноза на файле данных.
    
    Args:
        data_path: Путь к parquet файлу с признаками
        model_dir: Директория с моделями
        
    Returns:
        DataFrame с исходными данными и прогнозами
    """
    # Загружаем данные
    df = pd.read_parquet(data_path)
    
    # Загружаем модель
    model = GlobalQuantileModel(model_dir)
    model.load_models()
    
    # Прогноз
    predictions = model.predict(df)
    
    # Объединяем
    result = pd.concat([df, predictions], axis=1)
    
    return result


# === ЭКСПОРТ ===
__all__ = [
    'GlobalQuantileModel',
    'load_and_predict',
    'ENSEMBLE_AVAILABLE'
]


if __name__ == "__main__":
    # Тестовый запуск
    print("🚀 Тест модуля инференса")
    print(f"   Ансамбль доступен: {ENSEMBLE_AVAILABLE}")
    
    # Тест 1: Только LightGBM
    print("\n" + "="*50)
    print("📦 ТЕСТ 1: LightGBM")
    print("="*50)
    
    model = GlobalQuantileModel()
    
    try:
        model.load_models()
        
        # Тестовые данные
        ML_ROOT = Path(__file__).parent.parent
        test_file = list((ML_ROOT / "data" / "processed_ml").glob("*_ml_features.parquet"))[0]
        
        print(f"\n📊 Тестируем на: {test_file.name}")
        df = pd.read_parquet(test_file)
        
        predictions = model.predict(df.head(100))
        print(f"\n📈 LightGBM прогнозы:")
        print(predictions.head())
        
        print(f"\n📊 Feature Importance (топ-10):")
        print(model.get_feature_importance(top_n=10))
        
        # Тест 2: Ансамбль (если доступен)
        if ENSEMBLE_AVAILABLE:
            print("\n" + "="*50)
            print("📦 ТЕСТ 2: Ансамбль LightGBM + GARCH")
            print("="*50)
            
            ensemble_model = GlobalQuantileModel(
                use_ensemble=True,
                ensemble_weights={'lgbm': 0.7, 'garch': 0.3}
            )
            ensemble_model.load_models()
            
            # Ансамблевый прогноз
            ensemble_preds = ensemble_model.predict_ensemble(
                df.head(100),
                return_components=True
            )
            print(f"\n📈 Ансамблевые прогнозы:")
            print(ensemble_preds.head())
            
            # С неопределенностью
            uncertainty = ensemble_model.predict_with_uncertainty_ensemble(df.head(100))
            print(f"\n📊 Model Agreement: {uncertainty['model_agreement']:.3f}")
            print(f"   Weights: {uncertainty['weights_used']}")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Сначала обучите модели: python train_global_model.py")

