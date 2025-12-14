"""
Генератор текстовых объяснений на основе SHAP значений.

Переводит технические названия признаков в понятный русский текст
и формирует читаемые объяснения прогнозов волатильности.
"""

from typing import List, Dict, Optional, Tuple


# Словарь переводов технических названий признаков на русский язык
FEATURE_DESCRIPTIONS = {
    # === ВОЛАТИЛЬНОСТЬ ===
    'parkinson_vol_10d': 'внутридневного размаха цен (10 дней)',
    'parkinson_vol_20d': 'внутридневного размаха цен (20 дней)',
    'ewma_vol_10d': 'экспоненциальной волатильности (10 дней)',
    'ewma_vol_20d': 'инерции тренда (20 дней)',
    'gk_vol_10d': 'волатильности Гармана-Класса (10 дней)',
    'gk_vol_20d': 'волатильности Гармана-Класса (20 дней)',
    'rv_5d': 'реализованной волатильности (5 дней)',
    'rv_10d': 'реализованной волатильности (10 дней)',
    'rv_20d': 'реализованной волатильности (20 дней)',
    'rv_30d': 'реализованной волатильности (30 дней)',
    'rv_60d': 'реализованной волатильности (60 дней)',
    'vol_ratio_5_20': 'соотношения краткосрочной и среднесрочной волатильности',
    'vol_ratio_park_rv': 'соотношения внутридневной и реализованной волатильности',
    'vol_ratio_20_60': 'соотношения среднесрочной и долгосрочной волатильности',
    'up_vol_20d': 'волатильности восходящих движений (20 дней)',
    'down_vol_20d': 'волатильности нисходящих движений (20 дней)',
    'vol_asymmetry_20d': 'асимметрии волатильности (20 дней)',
    'vol_momentum_5d': 'импульса волатильности (5 дней)',
    'vol_momentum_10d': 'импульса волатильности (10 дней)',
    
    # === ОБЪЕМ ===
    'volume_zscore_20': 'аномального объема (20 дней)',
    'volume_zscore_60': 'аномального объема (60 дней)',
    'volume_ratio_20': 'относительного объема (20 дней)',
    'volume_spike': 'всплеска объема',
    'vp_position': 'позиции цены относительно Volume Profile',
    'vp_width_pct': 'ширины зоны Volume Profile',
    'vp_above_va': 'позиции выше зоны принятия решений',
    
    # === ТРЕНД ===
    'dist_to_sma_20': 'отклонения от скользящей средней (20 дней)',
    'dist_to_sma_50': 'отклонения от скользящей средней (50 дней)',
    'dist_to_sma_200': 'отклонения от скользящей средней (200 дней)',
    'dist_to_ema_20': 'отклонения от экспоненциальной средней (20 дней)',
    'dist_to_ema_50': 'отклонения от экспоненциальной средней (50 дней)',
    'sma_20_slope_norm': 'наклона тренда (20 дней)',
    'sma_50_slope_norm': 'наклона тренда (50 дней)',
    'momentum_10': 'импульса движения (10 дней)',
    'momentum_20': 'импульса движения (20 дней)',
    'rsi_14': 'индекса относительной силы (RSI)',
    'price_position_ma': 'позиции цены относительно скользящих средних',
    'trend_signal': 'сигнала тренда',
    'trend_strength': 'силы тренда',
    
    # === КАЛЕНДАРЬ ===
    'day_of_week': 'дня недели',
    'day_of_month': 'дня месяца',
    'week_of_month': 'недели месяца',
    'is_month_end': 'конца месяца',
    'is_month_start': 'начала месяца',
    'overnight_gap': 'гэпа открытия',
    'overnight_gap_zscore': 'нормализованного гэпа открытия',
    
    # === РЫНОЧНЫЕ ПРИЗНАКИ ===
    'beta_30d': 'беты к рынку (30 дней)',
    'beta_60d': 'беты к рынку (60 дней)',
    'beta_change': 'изменения беты',
    'correlation_30d': 'корреляции с рынком (30 дней)',
    'correlation_60d': 'корреляции с рынком (60 дней)',
    'index_vol_30d': 'волатильности индекса (30 дней)',
    'index_vol_60d': 'волатильности индекса (60 дней)',
    
    # === ВНУТРИДНЕВНЫЕ ПРИЗНАКИ ===
    'ivr': 'внутридневной реализованной волатильности',
    'opm': 'импульса открытия',
    'vds': 'асимметрии распределения объема',
    'pocs': 'смещения точки максимального объема',
    'irr': 'соотношения внутридневного диапазона',
    'hvc': 'количества периодов высокой волатильности',
    
    # === МЕТАДАННЫЕ ===
    'sector_id': 'секторного фактора',
    'sector_encoded': 'секторного фактора (кодированный)',
    'liquidity_rank': 'ранга ликвидности',
    'is_blue_chip': 'статуса голубой фишки',
    'lot_size_log': 'логарифма размера лота',
    
    # === БАЗОВЫЕ ===
    'log_return': 'дневной доходности',
}


class ExplanationGenerator:
    """
    Генератор текстовых объяснений на основе SHAP значений.
    
    Преобразует технические объяснения модели в понятный русский текст,
    выделяя основные драйверы и стабилизаторы волатильности.
    
    Пример использования:
        from explainability.shap_wrapper import ShapExplainer
        from explainability.text_generator import ExplanationGenerator
        
        # Получаем объяснение
        explainer = ShapExplainer(model, background_data)
        formatted = explainer.explain_and_format(features_vector, top_n=10)
        
        # Генерируем текст
        text_gen = ExplanationGenerator()
        text = text_gen.generate_text(formatted, prediction_value=0.15)
        print(text)
    """
    
    def __init__(self, feature_descriptions: Optional[Dict[str, str]] = None):
        """
        Инициализация генератора объяснений.
        
        Args:
            feature_descriptions: Словарь переводов признаков.
                                 Если None, используется FEATURE_DESCRIPTIONS.
        """
        self.feature_descriptions = (
            feature_descriptions if feature_descriptions is not None 
            else FEATURE_DESCRIPTIONS
        )
    
    def _get_feature_description(self, feature_name: str) -> str:
        """
        Получает описание признака на русском языке.
        
        Args:
            feature_name: Техническое название признака
        
        Returns:
            Описание на русском или исходное название, если описание не найдено
        """
        return self.feature_descriptions.get(
            feature_name,
            feature_name  # Возвращаем исходное название, если описание не найдено
        )
    
    def _split_drivers_stabilizers(
        self,
        formatted_explanation: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Разделяет признаки на драйверы (положительный вклад) 
        и стабилизаторы (отрицательный вклад).
        
        Args:
            formatted_explanation: Список словарей с объяснениями
        
        Returns:
            Кортеж (drivers, stabilizers), где каждый список отсортирован
            по абсолютной величине вклада
        """
        drivers = []
        stabilizers = []
        
        for item in formatted_explanation:
            contribution = item['contribution']
            if contribution > 0:
                drivers.append(item)
            elif contribution < 0:
                stabilizers.append(item)
            # Нулевой вклад игнорируем
        
        # Сортируем по абсолютной величине вклада (уже отсортированы из ShapExplainer)
        return drivers, stabilizers
    
    def generate_text(
        self,
        formatted_explanation: List[Dict],
        prediction_value: float,
        top_drivers: int = 3,
        include_stabilizers: bool = False
    ) -> str:
        """
        Генерирует текстовое объяснение прогноза волатильности.
        
        Args:
            formatted_explanation: Список словарей из ShapExplainer.format_explanation()
            prediction_value: Значение прогноза волатильности (например, 0.15 для 15%)
            top_drivers: Количество топ-драйверов для упоминания в тексте
            include_stabilizers: Включать ли информацию о стабилизаторах
        
        Returns:
            Текст объяснения на русском языке
        """
        if not formatted_explanation:
            return "Недостаточно данных для объяснения прогноза."
        
        # Разделяем на драйверы и стабилизаторы
        drivers, stabilizers = self._split_drivers_stabilizers(formatted_explanation)
        
        if not drivers:
            return (
                f"Прогноз волатильности ({prediction_value:.2%}) "
                f"сформирован преимущественно стабилизирующими факторами."
            )
        
        # Берем топ-N драйверов
        top_drivers_list = drivers[:top_drivers]
        
        # Формируем список описаний драйверов
        driver_descriptions = []
        for driver in top_drivers_list:
            feature_name = driver['feature']
            description = self._get_feature_description(feature_name)
            driver_descriptions.append(description)
        
        # Генерируем основное предложение
        if len(driver_descriptions) == 1:
            main_text = (
                f"Прогноз волатильности ({prediction_value:.2%}) "
                f"сформирован в основном за счет {driver_descriptions[0]}."
            )
        elif len(driver_descriptions) == 2:
            main_text = (
                f"Прогноз волатильности ({prediction_value:.2%}) "
                f"сформирован в основном за счет {driver_descriptions[0]} "
                f"и {driver_descriptions[1]}."
            )
        else:
            # 3 и более драйверов
            drivers_text = ", ".join(driver_descriptions[:-1])
            main_text = (
                f"Прогноз волатильности ({prediction_value:.2%}) "
                f"сформирован в основном за счет {drivers_text} "
                f"и {driver_descriptions[-1]}."
            )
        
        # Добавляем информацию о стабилизаторах, если запрошено
        if include_stabilizers and stabilizers:
            top_stabilizers = stabilizers[:2]  # Топ-2 стабилизатора
            stabilizer_descriptions = [
                self._get_feature_description(s['feature'])
                for s in top_stabilizers
            ]
            
            if len(stabilizer_descriptions) == 1:
                stabilizers_text = (
                    f" Снижению волатильности способствует "
                    f"{stabilizer_descriptions[0]}."
                )
            else:
                stabilizers_text = (
                    f" Снижению волатильности способствуют "
                    f"{stabilizer_descriptions[0]} и {stabilizer_descriptions[1]}."
                )
            
            main_text += stabilizers_text
        
        return main_text
    
    def generate_detailed_text(
        self,
        formatted_explanation: List[Dict],
        prediction_value: float,
        top_n: int = 5
    ) -> str:
        """
        Генерирует подробное текстовое объяснение с указанием конкретных вкладов.
        
        Args:
            formatted_explanation: Список словарей из ShapExplainer.format_explanation()
            prediction_value: Значение прогноза волатильности
            top_n: Количество топ-признаков для детального описания
        
        Returns:
            Подробный текст объяснения на русском языке
        """
        if not formatted_explanation:
            return "Недостаточно данных для объяснения прогноза."
        
        # Основное объяснение
        main_text = self.generate_text(
            formatted_explanation,
            prediction_value,
            top_drivers=3,
            include_stabilizers=True
        )
        
        # Добавляем детальную информацию о топ-N признаках
        top_features = formatted_explanation[:top_n]
        
        details = []
        for item in top_features:
            feature_name = item['feature']
            contribution = item['contribution']
            description = self._get_feature_description(feature_name)
            
            direction = "увеличивает" if contribution > 0 else "снижает"
            details.append(
                f"  • {description.capitalize()} {direction} "
                f"прогноз на {abs(contribution):.4f}"
            )
        
        if details:
            details_text = "\n" + "\n".join(details)
            return main_text + "\n\nДетализация вклада признаков:" + details_text
        
        return main_text

