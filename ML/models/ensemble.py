"""
Ансамблевая модель для прогнозирования волатильности.

Объединяет прогнозы LightGBM (ML) и GARCH (статистическая модель)
для улучшения качества прогнозов.

Архитектура ансамбля:
1. LightGBM Quantile Model - глобальная ML модель на признаках
2. GARCH(1,1) - классическая эконометрическая модель волатильности

Методы комбинации:
- Weighted Average (по умолчанию)
- Stacking (опционально)

Автор: ML Pipeline v2.0 (Ensemble Model)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, Tuple, List
from dataclasses import dataclass
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnsembleWeights:
    """
    Веса для ансамбля моделей.
    
    Attributes:
        lgbm: Вес для LightGBM модели (по умолчанию 0.7)
        garch: Вес для GARCH модели (по умолчанию 0.3)
        
    Note:
        Веса должны суммироваться в 1.0
    """
    lgbm: float = 0.7
    garch: float = 0.3
    
    def __post_init__(self):
        """Валидация весов."""
        total = self.lgbm + self.garch
        if not np.isclose(total, 1.0, rtol=1e-3):
            warnings.warn(
                f"Веса ансамбля не суммируются в 1.0 (сумма={total}). "
                f"Нормализуем автоматически."
            )
            self.lgbm = self.lgbm / total
            self.garch = self.garch / total
    
    def to_dict(self) -> Dict[str, float]:
        """Преобразует в словарь."""
        return {'lgbm': self.lgbm, 'garch': self.garch}


class EnsembleModel:
    """
    Ансамблевая модель для комбинирования LightGBM и GARCH прогнозов.
    
    Основной метод - взвешенное среднее прогнозов волатильности.
    Поддерживает как скалярные веса, так и динамические (по условиям рынка).
    
    Attributes:
        weights: Веса для моделей ансамбля
        adaptive_weights: Использовать ли адаптивные веса
        
    Example:
        >>> ensemble = EnsembleModel(weights={'lgbm': 0.7, 'garch': 0.3})
        >>> combined = ensemble.predict(lgbm_forecasts, garch_forecasts)
    """
    
    def __init__(
        self,
        weights: Optional[Union[Dict[str, float], EnsembleWeights]] = None,
        adaptive_weights: bool = False
    ):
        """
        Инициализация ансамбля.
        
        Args:
            weights: Веса для моделей. Может быть:
                     - Dict: {'lgbm': 0.7, 'garch': 0.3}
                     - EnsembleWeights: dataclass с весами
                     - None: использует веса по умолчанию (0.7/0.3)
            adaptive_weights: Если True, веса адаптируются в зависимости
                             от режима волатильности (высокая/низкая)
        """
        if weights is None:
            self.weights = EnsembleWeights()
        elif isinstance(weights, dict):
            self.weights = EnsembleWeights(
                lgbm=weights.get('lgbm', 0.7),
                garch=weights.get('garch', 0.3)
            )
        else:
            self.weights = weights
        
        self.adaptive_weights = adaptive_weights
        
        logger.info(f"EnsembleModel инициализирован с весами: {self.weights.to_dict()}")
    
    def predict(
        self,
        lgbm_forecasts: Union[pd.Series, pd.DataFrame, np.ndarray],
        garch_forecasts: Union[pd.Series, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
        current_volatility: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Комбинирует прогнозы LightGBM и GARCH.
        
        Args:
            lgbm_forecasts: Прогнозы LightGBM. Может быть:
                           - Series: медианный прогноз
                           - DataFrame: квантильные прогнозы (pred_q16, pred_q50, pred_q84)
            garch_forecasts: Прогнозы GARCH (точечные)
            weights: Опциональные веса для этого вызова (переопределяют self.weights)
            current_volatility: Текущая волатильность для адаптивных весов
            
        Returns:
            Комбинированные прогнозы в том же формате, что и lgbm_forecasts
            
        Raises:
            ValueError: Если размерности прогнозов не совпадают
        """
        # Определяем веса
        if weights is not None:
            w = EnsembleWeights(
                lgbm=weights.get('lgbm', 0.7),
                garch=weights.get('garch', 0.3)
            )
        else:
            w = self.weights
        
        # Адаптивные веса
        if self.adaptive_weights and current_volatility is not None:
            w = self._compute_adaptive_weights(current_volatility)
        
        # Обработка разных типов входных данных
        if isinstance(lgbm_forecasts, pd.DataFrame):
            return self._combine_quantile_forecasts(lgbm_forecasts, garch_forecasts, w)
        else:
            return self._combine_point_forecasts(lgbm_forecasts, garch_forecasts, w)
    
    def _combine_point_forecasts(
        self,
        lgbm: Union[pd.Series, np.ndarray],
        garch: Union[pd.Series, np.ndarray],
        weights: EnsembleWeights
    ) -> Union[pd.Series, np.ndarray]:
        """
        Взвешенное среднее точечных прогнозов.
        
        Формула: combined = w_lgbm * lgbm + w_garch * garch
        """
        combined = weights.lgbm * np.asarray(lgbm) + weights.garch * np.asarray(garch)
        
        if isinstance(lgbm, pd.Series):
            return pd.Series(combined, index=lgbm.index, name='ensemble_forecast')
        
        return combined
    
    def _combine_quantile_forecasts(
        self,
        lgbm_df: pd.DataFrame,
        garch: Union[pd.Series, np.ndarray],
        weights: EnsembleWeights
    ) -> pd.DataFrame:
        """
        Комбинация квантильных прогнозов LightGBM с точечным GARCH.
        
        Для каждого квантиля:
        combined_qX = w_lgbm * lgbm_qX + w_garch * garch
        
        Интуиция: GARCH дает "базовый" уровень волатильности,
        LightGBM корректирует границы интервала.
        """
        garch_arr = np.asarray(garch).flatten()
        
        result = pd.DataFrame(index=lgbm_df.index)
        
        # Комбинируем каждый квантиль
        quantile_cols = [col for col in lgbm_df.columns if col.startswith('pred_q')]
        
        for col in quantile_cols:
            result[col] = weights.lgbm * lgbm_df[col].values + weights.garch * garch_arr
        
        # Сохраняем interval_width если есть
        if 'interval_width' in lgbm_df.columns:
            # Интервал масштабируется пропорционально весу LGBM
            result['interval_width'] = lgbm_df['interval_width'] * weights.lgbm
        
        # Пересчитываем interval_width из новых квантилей
        if 'pred_q84' in result.columns and 'pred_q16' in result.columns:
            result['interval_width'] = result['pred_q84'] - result['pred_q16']
        
        return result
    
    def _compute_adaptive_weights(
        self,
        current_volatility: Union[pd.Series, np.ndarray]
    ) -> EnsembleWeights:
        """
        Адаптивные веса в зависимости от режима волатильности.
        
        Логика:
        - При высокой волатильности: GARCH более надежен (mean reversion)
        - При низкой волатильности: LightGBM лучше улавливает паттерны
        
        Args:
            current_volatility: Текущая реализованная волатильность
            
        Returns:
            Адаптированные веса
        """
        vol = np.asarray(current_volatility)
        
        # Медианная волатильность как порог
        median_vol = np.nanmedian(vol)
        
        # Нормализованная волатильность (z-score like)
        vol_ratio = vol / median_vol if median_vol > 0 else np.ones_like(vol)
        
        # Веса: при высокой волатильности увеличиваем вес GARCH
        # Формула: garch_weight = base_weight + adjustment * (vol_ratio - 1)
        base_garch = self.weights.garch
        adjustment = 0.15  # Максимальное изменение веса
        
        # Ограничиваем vol_ratio для избежания экстремальных весов
        vol_ratio_clipped = np.clip(vol_ratio, 0.5, 2.0)
        
        # Адаптивный вес GARCH
        adaptive_garch = base_garch + adjustment * (vol_ratio_clipped.mean() - 1)
        adaptive_garch = np.clip(adaptive_garch, 0.1, 0.5)  # Ограничения
        
        adaptive_lgbm = 1.0 - adaptive_garch
        
        logger.debug(f"Адаптивные веса: LGBM={adaptive_lgbm:.3f}, GARCH={adaptive_garch:.3f}")
        
        return EnsembleWeights(lgbm=adaptive_lgbm, garch=adaptive_garch)
    
    def predict_with_uncertainty(
        self,
        lgbm_forecasts: pd.DataFrame,
        garch_forecasts: Union[pd.Series, np.ndarray],
        garch_std: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict:
        """
        Прогноз с расширенной статистикой неопределенности.
        
        Комбинирует неопределенность от обеих моделей.
        
        Args:
            lgbm_forecasts: DataFrame с квантильными прогнозами
            garch_forecasts: Точечные прогнозы GARCH
            garch_std: Стандартная ошибка прогноза GARCH (если доступна)
            
        Returns:
            Dict с ключами:
                - median: медианный прогноз ансамбля
                - lower: нижняя граница (q16)
                - upper: верхняя граница (q84)
                - interval_width: ширина интервала
                - model_agreement: согласованность моделей (0-1)
        """
        # Базовый ансамблевый прогноз
        combined = self.predict(lgbm_forecasts, garch_forecasts)
        
        # Согласованность моделей: насколько близки прогнозы
        lgbm_median = lgbm_forecasts['pred_q50'].values if 'pred_q50' in lgbm_forecasts.columns else lgbm_forecasts.values
        garch_arr = np.asarray(garch_forecasts)
        
        # Относительное расхождение
        mean_forecast = (lgbm_median + garch_arr) / 2
        relative_diff = np.abs(lgbm_median - garch_arr) / np.where(mean_forecast > 0, mean_forecast, 1)
        
        # Agreement: 1 = полное согласие, 0 = сильное расхождение
        agreement = 1 - np.clip(relative_diff, 0, 1)
        
        result = {
            'median': combined['pred_q50'].values if 'pred_q50' in combined.columns else combined.values,
            'lower': combined['pred_q16'].values if 'pred_q16' in combined.columns else None,
            'upper': combined['pred_q84'].values if 'pred_q84' in combined.columns else None,
            'interval_width': combined['interval_width'].values if 'interval_width' in combined.columns else None,
            'model_agreement': agreement.mean(),
            'weights_used': self.weights.to_dict()
        }
        
        return result
    
    def calibrate_weights(
        self,
        lgbm_forecasts: pd.DataFrame,
        garch_forecasts: np.ndarray,
        actual_volatility: np.ndarray,
        metric: str = 'mae'
    ) -> EnsembleWeights:
        """
        Калибрует веса ансамбля по историческим данным.
        
        Ищет оптимальные веса минимизирующие ошибку прогноза.
        
        Args:
            lgbm_forecasts: Исторические прогнозы LightGBM
            garch_forecasts: Исторические прогнозы GARCH
            actual_volatility: Фактическая реализованная волатильность
            metric: Метрика для оптимизации ('mae', 'mse', 'quantile')
            
        Returns:
            Оптимальные веса
        """
        logger.info("🔧 Калибровка весов ансамбля...")
        
        lgbm_arr = lgbm_forecasts['pred_q50'].values if 'pred_q50' in lgbm_forecasts.columns else lgbm_forecasts.values
        garch_arr = np.asarray(garch_forecasts)
        actual = np.asarray(actual_volatility)
        
        # Убираем NaN
        valid_mask = ~(np.isnan(lgbm_arr) | np.isnan(garch_arr) | np.isnan(actual))
        lgbm_arr = lgbm_arr[valid_mask]
        garch_arr = garch_arr[valid_mask]
        actual = actual[valid_mask]
        
        best_weight = 0.5
        best_error = float('inf')
        
        # Grid search по весам
        for w_lgbm in np.arange(0.1, 1.0, 0.05):
            w_garch = 1 - w_lgbm
            combined = w_lgbm * lgbm_arr + w_garch * garch_arr
            
            if metric == 'mae':
                error = np.mean(np.abs(combined - actual))
            elif metric == 'mse':
                error = np.mean((combined - actual) ** 2)
            else:
                error = np.mean(np.abs(combined - actual))
            
            if error < best_error:
                best_error = error
                best_weight = w_lgbm
        
        optimal_weights = EnsembleWeights(lgbm=best_weight, garch=1-best_weight)
        
        logger.info(f"✅ Оптимальные веса: LGBM={best_weight:.2f}, GARCH={1-best_weight:.2f}")
        logger.info(f"   Ошибка ({metric}): {best_error:.6f}")
        
        # Обновляем веса
        self.weights = optimal_weights
        
        return optimal_weights


class SimpleGARCH:
    """
    Упрощенная GARCH(1,1) модель для ансамбля.
    
    Реализует базовый GARCH без зависимостей от arch библиотеки.
    Подходит для быстрого прототипирования и случаев когда 
    arch не установлен.
    
    Модель: σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
    
    Где:
    - ω (omega): базовый уровень волатильности
    - α (alpha): реакция на шоки (ARCH term)
    - β (beta): персистентность волатильности (GARCH term)
    """
    
    def __init__(
        self,
        omega: float = 0.0001,
        alpha: float = 0.1,
        beta: float = 0.85
    ):
        """
        Инициализация параметров GARCH(1,1).
        
        Args:
            omega: Константа (базовый уровень)
            alpha: Коэффициент ARCH (реакция на шоки)
            beta: Коэффициент GARCH (персистентность)
            
        Note:
            Для стационарности требуется: alpha + beta < 1
        """
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        
        # Валидация стационарности
        if alpha + beta >= 1:
            warnings.warn(
                f"alpha + beta = {alpha + beta} >= 1. "
                f"Модель может быть нестационарной."
            )
    
    def fit(self, returns: np.ndarray) -> 'SimpleGARCH':
        """
        "Подгонка" модели (упрощенная - оценка параметров по данным).
        
        Использует метод моментов для грубой оценки параметров.
        Для production рекомендуется использовать arch библиотеку.
        
        Args:
            returns: Массив log returns
            
        Returns:
            self
        """
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]
        
        # Оценка unconditional variance
        var = np.var(returns)
        
        # Упрощенная оценка параметров через autocorrelation
        squared_returns = returns ** 2
        
        if len(squared_returns) > 1:
            # Autocorrelation of squared returns
            autocorr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
            autocorr = max(0, min(autocorr, 0.95))  # Ограничиваем
            
            # Грубая оценка: alpha + beta ≈ autocorr
            self.alpha = autocorr * 0.15
            self.beta = autocorr * 0.85
            
            # omega из unconditional variance: E[σ²] = ω / (1 - α - β)
            persistence = self.alpha + self.beta
            if persistence < 1:
                self.omega = var * (1 - persistence)
            else:
                self.omega = var * 0.05
        
        logger.info(f"SimpleGARCH параметры: ω={self.omega:.6f}, α={self.alpha:.4f}, β={self.beta:.4f}")
        
        return self
    
    def forecast(
        self,
        returns: np.ndarray,
        horizon: int = 1
    ) -> np.ndarray:
        """
        Прогноз волатильности на horizon шагов вперед.
        
        Args:
            returns: Исторические returns для инициализации
            horizon: Горизонт прогноза
            
        Returns:
            np.ndarray с прогнозами волатильности (annualized)
        """
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]
        
        n = len(returns)
        if n == 0:
            return np.array([np.nan] * horizon)
        
        # Вычисляем conditional variance по историческим данным
        sigma2 = np.zeros(n + horizon)
        sigma2[0] = np.var(returns)  # Начальная variance
        
        # Рекурсивный расчет
        for t in range(1, n):
            sigma2[t] = (
                self.omega + 
                self.alpha * returns[t-1]**2 + 
                self.beta * sigma2[t-1]
            )
        
        # Прогноз вперед (без новых returns)
        last_return = returns[-1]
        for h in range(horizon):
            t = n + h
            if h == 0:
                sigma2[t] = (
                    self.omega + 
                    self.alpha * last_return**2 + 
                    self.beta * sigma2[n-1]
                )
            else:
                # Для h > 0: E[ε²] = σ², поэтому
                sigma2[t] = (
                    self.omega + 
                    (self.alpha + self.beta) * sigma2[t-1]
                )
        
        # Извлекаем прогнозы и аннуализируем
        forecast_var = sigma2[n:n+horizon]
        forecast_vol = np.sqrt(forecast_var) * np.sqrt(252)
        
        return forecast_vol
    
    def forecast_rolling(
        self,
        returns: np.ndarray,
        window: int = 20
    ) -> np.ndarray:
        """
        Rolling прогноз на 1 шаг вперед для всего массива returns.
        
        Для каждой точки делает прогноз на основе предыдущих window наблюдений.
        
        Args:
            returns: Полный массив returns
            window: Размер окна для расчета
            
        Returns:
            np.ndarray с rolling прогнозами (длина = len(returns))
        """
        returns = np.asarray(returns)
        n = len(returns)
        
        forecasts = np.full(n, np.nan)
        
        for i in range(window, n):
            window_returns = returns[i-window:i]
            self.fit(window_returns)
            forecast = self.forecast(window_returns, horizon=1)
            forecasts[i] = forecast[0]
        
        return forecasts


# === ЭКСПОРТ ===
__all__ = [
    'EnsembleModel',
    'EnsembleWeights',
    'SimpleGARCH'
]


if __name__ == "__main__":
    # Тестовый запуск
    print("🧪 Тест ансамблевой модели")
    
    # Генерируем тестовые данные
    np.random.seed(42)
    n = 100
    
    # Симулированные прогнозы
    lgbm_q50 = np.random.uniform(0.15, 0.25, n)
    lgbm_q16 = lgbm_q50 - 0.03
    lgbm_q84 = lgbm_q50 + 0.03
    
    lgbm_df = pd.DataFrame({
        'pred_q16': lgbm_q16,
        'pred_q50': lgbm_q50,
        'pred_q84': lgbm_q84
    })
    
    garch_forecasts = np.random.uniform(0.18, 0.22, n)
    
    # Тест ансамбля
    ensemble = EnsembleModel(weights={'lgbm': 0.7, 'garch': 0.3})
    
    combined = ensemble.predict(lgbm_df, garch_forecasts)
    print(f"\n📊 Ансамблевый прогноз:")
    print(combined.head())
    
    # Тест с неопределенностью
    result = ensemble.predict_with_uncertainty(lgbm_df, garch_forecasts)
    print(f"\n📈 Model Agreement: {result['model_agreement']:.3f}")
    print(f"   Weights: {result['weights_used']}")
    
    # Тест SimpleGARCH
    print(f"\n🔧 Тест SimpleGARCH:")
    returns = np.random.normal(0, 0.02, 500)
    
    garch = SimpleGARCH()
    garch.fit(returns)
    
    forecast = garch.forecast(returns, horizon=5)
    print(f"   Прогноз на 5 дней: {forecast}")

