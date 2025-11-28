"""
Модуль инференса для обученных квантильных моделей LightGBM.

Предоставляет функции для:
- Загрузки обученных моделей
- Прогнозирования на новых данных
- Получения интервального прогноза [q16, q84]

Использование:
    from inference import GlobalQuantileModel
    
    model = GlobalQuantileModel()
    model.load_models()
    
    predictions = model.predict(new_data)
    # predictions содержит колонки: pred_q16, pred_q50, pred_q84
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import Dict, Optional, List
import warnings

warnings.filterwarnings('ignore')


class GlobalQuantileModel:
    """
    Класс для загрузки и использования обученных квантильных моделей.
    
    Атрибуты:
        models: Dict[float, lgb.Booster] - словарь моделей по квантилям
        feature_names: List[str] - список признаков модели
    """
    
    def __init__(self, model_dir: Optional[Path] = None):
        """
        Инициализация.
        
        Args:
            model_dir: Директория с сохранёнными моделями
        """
        if model_dir is None:
            self.model_dir = Path(__file__).parent.parent / "data" / "models"
        else:
            self.model_dir = Path(model_dir)
        
        self.models: Dict[float, lgb.Booster] = {}
        self.feature_names: List[str] = []
        self.quantiles = [0.16, 0.50, 0.84]
        self._loaded = False
    
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
    
    def predict(
        self, 
        X: pd.DataFrame, 
        return_interval: bool = True
    ) -> pd.DataFrame:
        """
        Делает прогноз на новых данных.
        
        Args:
            X: DataFrame с признаками (должны совпадать с feature_names)
            return_interval: Если True, возвращает также ширину интервала
            
        Returns:
            DataFrame с колонками: pred_q16, pred_q50, pred_q84, [interval_width]
        """
        if not self._loaded:
            raise RuntimeError("Модели не загружены! Вызовите load_models() сначала.")
        
        # Проверяем наличие признаков
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            warnings.warn(f"⚠️ Отсутствуют признаки: {missing_features}")
        
        # Подготовка данных
        X_prepared = X[self.feature_names].copy() if set(self.feature_names).issubset(X.columns) else X.copy()
        X_prepared = X_prepared.fillna(0)
        
        # Прогнозы для каждого квантиля
        predictions = pd.DataFrame(index=X.index)
        
        for alpha in self.quantiles:
            col_name = f"pred_q{int(alpha*100)}"
            predictions[col_name] = self.models[alpha].predict(X_prepared)
        
        # Ширина интервала (мера неопределённости)
        if return_interval:
            predictions['interval_width'] = predictions['pred_q84'] - predictions['pred_q16']
        
        return predictions
    
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
    'load_and_predict'
]


if __name__ == "__main__":
    # Тестовый запуск
    print("🚀 Тест модуля инференса")
    
    model = GlobalQuantileModel()
    
    try:
        model.load_models()
        
        # Тестовые данные
        ML_ROOT = Path(__file__).parent.parent
        test_file = list((ML_ROOT / "data" / "processed_ml").glob("*_ml_features.parquet"))[0]
        
        print(f"\n📊 Тестируем на: {test_file.name}")
        df = pd.read_parquet(test_file)
        
        predictions = model.predict(df.head(100))
        print(f"\n📈 Пример прогнозов:")
        print(predictions.head())
        
        print(f"\n📊 Feature Importance (топ-10):")
        print(model.get_feature_importance(top_n=10))
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Сначала обучите модели: python train_global_model.py")

