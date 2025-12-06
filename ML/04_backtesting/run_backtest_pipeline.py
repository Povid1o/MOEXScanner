"""
Полный Pipeline Бэктестинга для Глобальной Модели Волатильности.

Stage 5: Signal Generation & Strict Backtesting

Этот скрипт выполняет:
1. Генерация сигналов (денормализация прогнозов в ценовые уровни)
2. Применение торговой стратегии "Mean Reversion in Trend"
3. Строгий аудит и расчёт метрик производительности

КРИТИЧНО:
- Вход на Close дня T (при сигнале на T)
- Комиссия 0.1% + Slippage 0.05%
- Нет look-ahead bias!

Автор: ML Pipeline v2.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import sys

warnings.filterwarnings('ignore')

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent / "03_models"))


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

@dataclass
class BacktestConfig:
    """Конфигурация бэктеста."""
    
    # Пути
    ML_ROOT: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    @property
    def ML_FEATURES_DIR(self) -> Path:
        return self.ML_ROOT / "data" / "processed_ml"
    
    @property
    def PRICE_DATA_DIR(self) -> Path:
        return self.ML_ROOT / "data" / "backtest"
    
    @property
    def MODELS_DIR(self) -> Path:
        return self.ML_ROOT / "data" / "models"
    
    @property
    def OUTPUT_DIR(self) -> Path:
        return self.ML_ROOT / "data" / "backtest"
    
    @property
    def REPORTS_DIR(self) -> Path:
        return self.ML_ROOT / "reports"
    
    # Торговые параметры
    COMMISSION_PCT: float = 0.001      # 0.1% комиссия
    SLIPPAGE_PCT: float = 0.0005       # 0.05% проскальзывание
    
    # Стоп-лосс множители
    LONG_STOP_MULT: float = 0.98       # Стоп для лонга: lower_band * 0.98
    SHORT_STOP_MULT: float = 1.02      # Стоп для шорта: upper_band * 1.02
    
    # Фильтры
    MIN_TRADES_THRESHOLD: int = 10     # Минимум сделок для статистики
    MIN_CONFIDENCE: float = 0.0        # Минимальная уверенность (ширина интервала)


# ============================================================================
# STEP 1: ГЕНЕРАЦИЯ СИГНАЛОВ (Денормализация)
# ============================================================================

class SignalGenerator:
    """
    Генератор сигналов: загружает модели, делает прогнозы, 
    восстанавливает ценовые уровни.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.model = None
    
    def load_models(self) -> None:
        """Загружает обученные квантильные модели."""
        from inference import GlobalQuantileModel
        
        # Коррекция bias: модель систематически завышает прогнозы на +4.14%
        self.model = GlobalQuantileModel(
            self.config.MODELS_DIR,
            bias_correction=-0.0414
        )
        self.model.load_models()
    
    def generate_signals_for_ticker(
        self, 
        ticker: str
    ) -> Optional[pd.DataFrame]:
        """
        Генерирует сигналы для одного тикера.
        
        Шаги:
        1. Загрузка ML features и price data
        2. Прогноз квантилей
        3. Восстановление ценовых уровней (денормализация)
        4. Объединение в единый DataFrame
        
        Args:
            ticker: Тикер акции
            
        Returns:
            DataFrame с сигналами или None если ошибка
        """
        try:
            # === 1. ЗАГРУЗКА ДАННЫХ ===
            ml_path = self.config.ML_FEATURES_DIR / f"{ticker}_ml_features.parquet"
            price_path = self.config.PRICE_DATA_DIR / f"{ticker}_price_data.parquet"
            
            if not ml_path.exists() or not price_path.exists():
                print(f"   ⚠️ {ticker}: Файлы не найдены")
                return None
            
            ml_df = pd.read_parquet(ml_path)
            price_df = pd.read_parquet(price_path)
            
            # === 2. ПРОГНОЗ КВАНТИЛЕЙ ===
            predictions = self.model.predict(ml_df, return_interval=True)
            
            # === 3. ОБЪЕДИНЕНИЕ С ЦЕНАМИ ===
            # Конвертируем даты
            ml_df['date'] = pd.to_datetime(ml_df['date'])
            price_df['date'] = pd.to_datetime(price_df['date'])
            
            # Добавляем прогнозы к ML данным
            ml_df = pd.concat([ml_df.reset_index(drop=True), 
                              predictions.reset_index(drop=True)], axis=1)
            
            # Мержим с ценами
            signals_df = pd.merge(
                ml_df,
                price_df[['date', 'open', 'high', 'low', 'close']],
                on='date',
                how='inner'
            )
            
            # === 4. ДЕНОРМАЛИЗАЦИЯ: ВОССТАНОВЛЕНИЕ ЦЕНОВЫХ УРОВНЕЙ ===
            # ВАЖНО: Модель предсказывает ВОЛАТИЛЬНОСТЬ (std dev), которая ВСЕГДА >= 0
            # Волатильность определяет ШИРИНУ канала, а не направление движения
            # Используем pred_q84 (верхний квантиль волатильности) для симметричного канала
            
            # Верхняя граница: добавляем волатильность к цене
            signals_df['upper_band'] = signals_df['close'] * (1 + signals_df['pred_q84'])
            
            # Нижняя граница: ВЫЧИТАЕМ волатильность (симметричный канал)
            # Это гарантирует, что lower_band НИЖЕ цены закрытия
            signals_df['lower_band'] = signals_df['close'] * (1 - signals_df['pred_q84'])
            
            # Средняя линия: используем медианную волатильность для расчёта целевого уровня
            # Для LONG: цель выше входа (half-way к upper_band)
            # Для SHORT: цель ниже входа (half-way к lower_band)
            # Используем pred_q50 для адекватного целевого уровня
            signals_df['median_band'] = signals_df['close'] * (1 + signals_df['pred_q50'] * 0.5)
            
            # Уверенность прогноза (уже ширина интервала)
            signals_df['prediction_confidence'] = signals_df['interval_width']
            
            # Добавляем тикер
            signals_df['ticker'] = ticker
            
            return signals_df
            
        except Exception as e:
            print(f"   ❌ {ticker}: Ошибка - {e}")
            return None
    
    def generate_all_signals(self) -> pd.DataFrame:
        """
        Генерирует сигналы для всех доступных тикеров.
        
        Returns:
            Объединённый DataFrame всех сигналов
        """
        print("\n" + "=" * 60)
        print("📡 STEP 1: ГЕНЕРАЦИЯ СИГНАЛОВ")
        print("=" * 60)
        
        # Загружаем модели
        self.load_models()
        
        # Находим все тикеры
        ml_files = list(self.config.ML_FEATURES_DIR.glob("*_ml_features.parquet"))
        tickers = [f.stem.replace('_ml_features', '') for f in ml_files]
        
        print(f"📋 Найдено тикеров: {len(tickers)}")
        
        all_signals = []
        
        for ticker in tickers:
            print(f"   🔄 {ticker}...", end=" ")
            signals = self.generate_signals_for_ticker(ticker)
            if signals is not None:
                all_signals.append(signals)
                print(f"✅ {len(signals)} строк")
            else:
                print("❌")
        
        if not all_signals:
            raise ValueError("Не удалось сгенерировать сигналы ни для одного тикера!")
        
        # Объединяем все сигналы
        full_signals = pd.concat(all_signals, ignore_index=True)
        
        print(f"\n📊 Всего сигналов: {len(full_signals):,} строк")
        print(f"   Тикеров: {full_signals['ticker'].nunique()}")
        
        return full_signals


# ============================================================================
# STEP 2: ТОРГОВАЯ ЛОГИКА (Mean Reversion in Trend)
# ============================================================================

class TradingStrategy:
    """
    Реализация стратегии "Mean Reversion in Trend".
    
    Логика:
    - LONG: Восходящий тренд (dist_to_sma_50 > 0) + цена касается нижней границы
    - SHORT: Нисходящий тренд (dist_to_sma_50 < 0) + цена касается верхней границы
    - Выход: на медиане или стоп-лоссе
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
    
    def apply_strategy(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет торговую логику к сигналам.
        
        Args:
            signals_df: DataFrame с сигналами и ценами
            
        Returns:
            DataFrame с добавленными торговыми сигналами
        """
        print("\n" + "=" * 60)
        print("📈 STEP 2: ПРИМЕНЕНИЕ ТОРГОВОЙ ЛОГИКИ")
        print("=" * 60)
        
        df = signals_df.copy()
        
        # === ИДЕНТИФИКАЦИЯ ТРЕНДА ===
        # Используем dist_to_sma_50 из ML features
        if 'dist_to_sma_50' not in df.columns:
            print("   ⚠️ dist_to_sma_50 не найден, создаём на основе dist_to_sma_20")
            df['dist_to_sma_50'] = df.get('dist_to_sma_20', 0)
        
        df['trend'] = np.where(df['dist_to_sma_50'] > 0, 1, -1)  # 1=uptrend, -1=downtrend
        
        # === ENTRY SIGNALS ===
        
        # LONG: Uptrend + Low касается Lower Band (покупка на откате)
        df['signal_long'] = (
            (df['trend'] == 1) &                       # Восходящий тренд
            (df['low'] <= df['lower_band']) &          # Low <= нижняя граница
            (df['prediction_confidence'] > self.config.MIN_CONFIDENCE)  # Достаточная уверенность
        ).astype(int)
        
        # SHORT: Downtrend + High касается Upper Band (продажа на росте)
        df['signal_short'] = (
            (df['trend'] == -1) &                      # Нисходящий тренд
            (df['high'] >= df['upper_band']) &         # High >= верхняя граница
            (df['prediction_confidence'] > self.config.MIN_CONFIDENCE)
        ).astype(int)
        
        # === EXIT LEVELS ===
        
        # Цель выхода (Take Profit)
        # LONG: цена должна ВЫРАСТИ к медиане (median_band > close для прибыли)
        df['take_profit_long'] = df['median_band']
        
        # SHORT: цена должна УПАСТЬ для прибыли, поэтому используем симметричную цель
        # TP SHORT = close * (1 - volatility), зеркально к median_band
        df['take_profit_short'] = df['close'] * (1 - df['pred_q50'] * 0.5)
        
        # Стоп-лосс
        df['stop_loss_long'] = df['lower_band'] * self.config.LONG_STOP_MULT
        df['stop_loss_short'] = df['upper_band'] * self.config.SHORT_STOP_MULT
        
        # === СТАТИСТИКА ===
        n_long = df['signal_long'].sum()
        n_short = df['signal_short'].sum()
        
        print(f"   📊 Сигналы LONG: {n_long:,}")
        print(f"   📊 Сигналы SHORT: {n_short:,}")
        print(f"   📊 Всего сигналов: {n_long + n_short:,}")
        
        # Проверка по тикерам
        signals_by_ticker = df.groupby('ticker').agg({
            'signal_long': 'sum',
            'signal_short': 'sum'
        }).rename(columns={'signal_long': 'longs', 'signal_short': 'shorts'})
        signals_by_ticker['total'] = signals_by_ticker['longs'] + signals_by_ticker['shorts']
        
        print(f"\n   📋 Распределение по тикерам:")
        for ticker, row in signals_by_ticker.iterrows():
            print(f"      {ticker}: L={row['longs']}, S={row['shorts']}, Total={row['total']}")
        
        return df


# ============================================================================
# STEP 3: БЭКТЕСТ-ДВИЖОК (Симуляция сделок)
# ============================================================================

@dataclass
class Trade:
    """Структура для хранения информации о сделке."""
    ticker: str
    direction: str  # 'LONG' или 'SHORT'
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'TP', 'SL', 'END'
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_days: int = 0


class BacktestEngine:
    """
    Движок бэктеста со строгим аудитом.
    
    Особенности:
    - Реалистичная симуляция с комиссией и slippage
    - Нет look-ahead bias (вход на Close дня сигнала)
    - Расчёт стандартных метрик (Sharpe, MaxDD, Win Rate)
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades: List[Trade] = []
    
    def _apply_slippage(self, price: float, direction: str, is_entry: bool) -> float:
        """
        Применяет проскальзывание к цене.
        
        Логика:
        - Entry LONG: платим больше (price + slippage)
        - Entry SHORT: получаем меньше (price - slippage)
        - Exit LONG: получаем меньше
        - Exit SHORT: платим больше
        """
        slippage = price * self.config.SLIPPAGE_PCT
        
        if direction == 'LONG':
            return price + slippage if is_entry else price - slippage
        else:  # SHORT
            return price - slippage if is_entry else price + slippage
    
    def _calculate_pnl(
        self, 
        entry_price: float, 
        exit_price: float, 
        direction: str
    ) -> Tuple[float, float]:
        """
        Рассчитывает P&L с учётом комиссии.
        
        Returns:
            Tuple[pnl_absolute, pnl_pct]
        """
        commission = (entry_price + exit_price) * self.config.COMMISSION_PCT
        
        if direction == 'LONG':
            gross_pnl = exit_price - entry_price
        else:  # SHORT
            gross_pnl = entry_price - exit_price
        
        net_pnl = gross_pnl - commission
        pnl_pct = net_pnl / entry_price
        
        return net_pnl, pnl_pct
    
    def simulate_trades_for_ticker(
        self, 
        df: pd.DataFrame, 
        ticker: str
    ) -> List[Trade]:
        """
        Симулирует сделки для одного тикера.
        
        Логика:
        - При сигнале на день T, входим по Close дня T
        - Выходим когда цена касается TP или SL
        - Максимум одна позиция одновременно
        
        Args:
            df: DataFrame с сигналами для тикера
            ticker: Тикер
            
        Returns:
            Список сделок
        """
        trades = []
        df = df.sort_values('date').reset_index(drop=True)
        
        in_position = False
        current_trade: Optional[Trade] = None
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            if not in_position:
                # === ПРОВЕРКА ВХОДА ===
                
                if row['signal_long'] == 1:
                    # Вход в LONG на Close дня сигнала
                    entry_price = self._apply_slippage(row['close'], 'LONG', is_entry=True)
                    
                    current_trade = Trade(
                        ticker=ticker,
                        direction='LONG',
                        entry_date=row['date'],
                        entry_price=entry_price
                    )
                    in_position = True
                    
                elif row['signal_short'] == 1:
                    # Вход в SHORT на Close дня сигнала
                    entry_price = self._apply_slippage(row['close'], 'SHORT', is_entry=True)
                    
                    current_trade = Trade(
                        ticker=ticker,
                        direction='SHORT',
                        entry_date=row['date'],
                        entry_price=entry_price
                    )
                    in_position = True
            
            else:
                # === ПРОВЕРКА ВЫХОДА ===
                
                exit_price = None
                exit_reason = None
                
                if current_trade.direction == 'LONG':
                    # ПЕССИМИСТИЧНЫЙ ПОРЯДОК: сначала проверяем Stop Loss!
                    # При конфликте (TP и SL в одном баре) приоритет у убытка
                    
                    # Проверка Stop Loss ПЕРВЫМ (Low <= SL)
                    if row['low'] <= row['stop_loss_long']:
                        exit_price = self._apply_slippage(
                            row['stop_loss_long'], 'LONG', is_entry=False
                        )
                        exit_reason = 'SL'
                    
                    # Проверка Take Profit ВТОРЫМ (High >= TP)
                    elif row['high'] >= row['take_profit_long']:
                        exit_price = self._apply_slippage(
                            row['take_profit_long'], 'LONG', is_entry=False
                        )
                        exit_reason = 'TP'
                
                else:  # SHORT
                    # ПЕССИМИСТИЧНЫЙ ПОРЯДОК: сначала проверяем Stop Loss!
                    
                    # Проверка Stop Loss ПЕРВЫМ (High >= SL для шорта)
                    if row['high'] >= row['stop_loss_short']:
                        exit_price = self._apply_slippage(
                            row['stop_loss_short'], 'SHORT', is_entry=False
                        )
                        exit_reason = 'SL'
                    
                    # Проверка Take Profit ВТОРЫМ (Low <= TP для шорта)
                    elif row['low'] <= row['take_profit_short']:
                        exit_price = self._apply_slippage(
                            row['take_profit_short'], 'SHORT', is_entry=False
                        )
                        exit_reason = 'TP'
                
                # Если есть выход
                if exit_price is not None:
                    current_trade.exit_date = row['date']
                    current_trade.exit_price = exit_price
                    current_trade.exit_reason = exit_reason
                    
                    pnl, pnl_pct = self._calculate_pnl(
                        current_trade.entry_price,
                        exit_price,
                        current_trade.direction
                    )
                    current_trade.pnl = pnl
                    current_trade.pnl_pct = pnl_pct
                    current_trade.holding_days = (
                        current_trade.exit_date - current_trade.entry_date
                    ).days
                    
                    trades.append(current_trade)
                    in_position = False
                    current_trade = None
        
        # Закрываем открытую позицию в конце
        if in_position and current_trade is not None:
            last_row = df.iloc[-1]
            exit_price = self._apply_slippage(
                last_row['close'], current_trade.direction, is_entry=False
            )
            
            current_trade.exit_date = last_row['date']
            current_trade.exit_price = exit_price
            current_trade.exit_reason = 'END'
            
            pnl, pnl_pct = self._calculate_pnl(
                current_trade.entry_price,
                exit_price,
                current_trade.direction
            )
            current_trade.pnl = pnl
            current_trade.pnl_pct = pnl_pct
            current_trade.holding_days = (
                current_trade.exit_date - current_trade.entry_date
            ).days
            
            trades.append(current_trade)
        
        return trades
    
    def run_backtest(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Запускает полный бэктест по всем тикерам.
        
        Args:
            signals_df: DataFrame с торговыми сигналами
            
        Returns:
            DataFrame с логом сделок
        """
        print("\n" + "=" * 60)
        print("⚙️ STEP 3: СИМУЛЯЦИЯ СДЕЛОК")
        print("=" * 60)
        
        self.trades = []
        
        tickers = signals_df['ticker'].unique()
        
        for ticker in tickers:
            ticker_df = signals_df[signals_df['ticker'] == ticker].copy()
            ticker_trades = self.simulate_trades_for_ticker(ticker_df, ticker)
            self.trades.extend(ticker_trades)
            
            if ticker_trades:
                print(f"   {ticker}: {len(ticker_trades)} сделок")
        
        # Конвертируем в DataFrame
        if not self.trades:
            print("   ⚠️ Нет сделок для анализа!")
            return pd.DataFrame()
        
        trade_log = pd.DataFrame([
            {
                'ticker': t.ticker,
                'direction': t.direction,
                'entry_date': t.entry_date,
                'entry_price': t.entry_price,
                'exit_date': t.exit_date,
                'exit_price': t.exit_price,
                'exit_reason': t.exit_reason,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'holding_days': t.holding_days
            }
            for t in self.trades
        ])
        
        print(f"\n📊 Всего сделок: {len(trade_log)}")
        print(f"   Прибыльных: {(trade_log['pnl'] > 0).sum()}")
        print(f"   Убыточных: {(trade_log['pnl'] < 0).sum()}")
        
        return trade_log


# ============================================================================
# STEP 4: РАСЧЁТ МЕТРИК
# ============================================================================

class PerformanceAnalyzer:
    """Анализатор производительности стратегии."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
    
    def calculate_metrics(self, trade_log: pd.DataFrame) -> Dict:
        """
        Рассчитывает метрики производительности.
        
        Returns:
            Dict с метриками
        """
        if trade_log.empty:
            return {}
        
        pnl_series = trade_log['pnl_pct']
        
        # Базовые метрики
        total_trades = len(trade_log)
        winning_trades = (trade_log['pnl'] > 0).sum()
        losing_trades = (trade_log['pnl'] < 0).sum()
        
        # Win Rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Total Return (накопленный)
        cumulative_return = (1 + pnl_series).prod() - 1
        
        # Sharpe Ratio (годовой, ~252 торговых дней)
        if pnl_series.std() > 0:
            # Предполагаем примерно 1 сделку в неделю для аннуализации
            trades_per_year = 52
            sharpe = (pnl_series.mean() / pnl_series.std()) * np.sqrt(trades_per_year)
        else:
            sharpe = 0
        
        # Max Drawdown
        cumulative = (1 + pnl_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Profit Factor
        gross_profit = trade_log[trade_log['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trade_log[trade_log['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Expectancy (средняя прибыль на сделку)
        expectancy = trade_log['pnl'].mean()
        
        # Avg Win / Avg Loss
        avg_win = trade_log[trade_log['pnl'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = trade_log[trade_log['pnl'] < 0]['pnl_pct'].mean() if losing_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': cumulative_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss
        }
    
    def calculate_per_ticker_metrics(self, trade_log: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает метрики для каждого тикера.
        
        Returns:
            DataFrame с метриками по тикерам
        """
        results = []
        
        for ticker in trade_log['ticker'].unique():
            ticker_trades = trade_log[trade_log['ticker'] == ticker]
            metrics = self.calculate_metrics(ticker_trades)
            metrics['ticker'] = ticker
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def generate_sanity_check(
        self, 
        signals_df: pd.DataFrame
    ) -> Dict:
        """
        Проверка адекватности прогнозов.
        
        Проверяет:
        - Находятся ли прогнозируемые bands в разумном диапазоне от close
        - Есть ли аномальные значения
        
        Returns:
            Dict с результатами проверки
        """
        # Отклонение bands от close
        upper_deviation = (signals_df['upper_band'] / signals_df['close'] - 1).abs()
        lower_deviation = (1 - signals_df['lower_band'] / signals_df['close']).abs()
        
        # Статистика
        return {
            'upper_band_mean_deviation': upper_deviation.mean(),
            'upper_band_max_deviation': upper_deviation.max(),
            'lower_band_mean_deviation': lower_deviation.mean(),
            'lower_band_max_deviation': lower_deviation.max(),
            'bands_within_10pct': ((upper_deviation < 0.1) & (lower_deviation < 0.1)).mean(),
            'nan_predictions': signals_df['pred_q50'].isna().sum()
        }


# ============================================================================
# STEP 5: ГЕНЕРАЦИЯ ОТЧЁТА
# ============================================================================

class ReportGenerator:
    """Генератор отчётов бэктеста."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
    
    def generate_audit_report(
        self,
        signals_df: pd.DataFrame,
        trade_log: pd.DataFrame,
        portfolio_metrics: Dict,
        ticker_metrics: pd.DataFrame,
        sanity_check: Dict
    ) -> str:
        """
        Генерирует полный аудиторский отчёт.
        
        Returns:
            Текст отчёта
        """
        report = []
        report.append("=" * 70)
        report.append("   BACKTEST AUDIT REPORT")
        report.append(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        # === SANITY CHECK ===
        report.append("\n" + "=" * 70)
        report.append("   SECTION 1: SANITY CHECK (Проверка адекватности прогнозов)")
        report.append("=" * 70)
        
        report.append(f"\n   Upper Band:")
        report.append(f"      Среднее отклонение от Close: {sanity_check['upper_band_mean_deviation']:.2%}")
        report.append(f"      Макс. отклонение: {sanity_check['upper_band_max_deviation']:.2%}")
        
        report.append(f"\n   Lower Band:")
        report.append(f"      Среднее отклонение от Close: {sanity_check['lower_band_mean_deviation']:.2%}")
        report.append(f"      Макс. отклонение: {sanity_check['lower_band_max_deviation']:.2%}")
        
        report.append(f"\n   Доля прогнозов в пределах 10%: {sanity_check['bands_within_10pct']:.1%}")
        report.append(f"   NaN прогнозов: {sanity_check['nan_predictions']}")
        
        if sanity_check['upper_band_max_deviation'] > 0.5:
            report.append("\n   ⚠️ WARNING: Обнаружены экстремальные отклонения upper_band (>50%)!")
        
        if sanity_check['bands_within_10pct'] < 0.9:
            report.append("\n   ⚠️ WARNING: Менее 90% прогнозов в разумном диапазоне!")
        
        # === PORTFOLIO PERFORMANCE ===
        report.append("\n" + "=" * 70)
        report.append("   SECTION 2: PORTFOLIO PERFORMANCE")
        report.append("=" * 70)
        
        if portfolio_metrics:
            report.append(f"\n   Общая статистика:")
            report.append(f"      Всего сделок: {portfolio_metrics['total_trades']}")
            report.append(f"      Прибыльных: {portfolio_metrics['winning_trades']}")
            report.append(f"      Убыточных: {portfolio_metrics['losing_trades']}")
            
            report.append(f"\n   Ключевые метрики:")
            report.append(f"      Win Rate: {portfolio_metrics['win_rate']:.1%}")
            report.append(f"      Total Return: {portfolio_metrics['total_return']:.2%}")
            report.append(f"      Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}")
            report.append(f"      Max Drawdown: {portfolio_metrics['max_drawdown']:.2%}")
            report.append(f"      Profit Factor: {portfolio_metrics['profit_factor']:.2f}")
            report.append(f"      Expectancy: {portfolio_metrics['expectancy']:.4f}")
            
            report.append(f"\n   Средние значения:")
            report.append(f"      Avg Win: {portfolio_metrics['avg_win_pct']:.2%}")
            report.append(f"      Avg Loss: {portfolio_metrics['avg_loss_pct']:.2%}")
        else:
            report.append("\n   ❌ Нет данных для портфолио")
        
        # === PER-TICKER PERFORMANCE ===
        report.append("\n" + "=" * 70)
        report.append("   SECTION 3: PER-TICKER PERFORMANCE")
        report.append("=" * 70)
        
        if not ticker_metrics.empty:
            # Сортируем по Sharpe
            ticker_metrics_sorted = ticker_metrics.sort_values('sharpe_ratio', ascending=False)
            
            report.append("\n   {:<8} {:>8} {:>10} {:>10} {:>12}".format(
                'Ticker', 'Trades', 'Win Rate', 'Sharpe', 'Total Ret'
            ))
            report.append("   " + "-" * 50)
            
            for _, row in ticker_metrics_sorted.iterrows():
                report.append("   {:<8} {:>8} {:>10.1%} {:>10.2f} {:>12.2%}".format(
                    row['ticker'],
                    int(row['total_trades']),
                    row['win_rate'],
                    row['sharpe_ratio'],
                    row['total_return']
                ))
        
        # === WARNINGS ===
        report.append("\n" + "=" * 70)
        report.append("   SECTION 4: WARNINGS & ALERTS")
        report.append("=" * 70)
        
        warnings_found = False
        
        if not ticker_metrics.empty:
            # Тикеры с малым количеством сделок
            low_trades = ticker_metrics[
                ticker_metrics['total_trades'] < self.config.MIN_TRADES_THRESHOLD
            ]
            if not low_trades.empty:
                report.append(f"\n   ⚠️ Тикеры с <{self.config.MIN_TRADES_THRESHOLD} сделками:")
                for ticker in low_trades['ticker']:
                    trades_count = low_trades[low_trades['ticker'] == ticker]['total_trades'].values[0]
                    report.append(f"      - {ticker}: {int(trades_count)} сделок")
                warnings_found = True
            
            # Тикеры с отрицательной expectancy
            negative_exp = ticker_metrics[ticker_metrics['expectancy'] < 0]
            if not negative_exp.empty:
                report.append(f"\n   ⚠️ Тикеры с отрицательной Expectancy:")
                for ticker in negative_exp['ticker']:
                    exp = negative_exp[negative_exp['ticker'] == ticker]['expectancy'].values[0]
                    report.append(f"      - {ticker}: {exp:.4f}")
                warnings_found = True
            
            # Тикеры с win rate < 40%
            low_wr = ticker_metrics[ticker_metrics['win_rate'] < 0.4]
            if not low_wr.empty:
                report.append(f"\n   ⚠️ Тикеры с Win Rate <40%:")
                for ticker in low_wr['ticker']:
                    wr = low_wr[low_wr['ticker'] == ticker]['win_rate'].values[0]
                    report.append(f"      - {ticker}: {wr:.1%}")
                warnings_found = True
        
        if not warnings_found:
            report.append("\n   ✅ Критических предупреждений не обнаружено")
        
        # === FOOTER ===
        report.append("\n" + "=" * 70)
        report.append("   END OF REPORT")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def save_report(
        self,
        report_text: str,
        signals_df: pd.DataFrame,
        trade_log: pd.DataFrame
    ) -> None:
        """Сохраняет все выходные файлы."""
        
        # Создаём директории
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # 1. Сохраняем сигналы
        signals_path = self.config.OUTPUT_DIR / "full_signals.parquet"
        signals_df.to_parquet(signals_path, index=False)
        print(f"   💾 Сигналы: {signals_path}")
        
        # 2. Сохраняем лог сделок
        if not trade_log.empty:
            trades_path = self.config.OUTPUT_DIR / "trade_log.parquet"
            trade_log.to_parquet(trades_path, index=False)
            print(f"   💾 Сделки: {trades_path}")
        
        # 3. Сохраняем отчёт
        report_path = self.config.REPORTS_DIR / "backtest_audit.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"   💾 Отчёт: {report_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_full_backtest_pipeline():
    """Запуск полного pipeline бэктестинга."""
    
    print("\n" + "=" * 70)
    print("🚀 BACKTEST PIPELINE - Stage 5: Signal Generation & Backtesting")
    print("=" * 70)
    print(f"📅 Запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Инициализация
    config = BacktestConfig()
    
    print(f"\n⚙️ Конфигурация:")
    print(f"   Комиссия: {config.COMMISSION_PCT:.2%}")
    print(f"   Slippage: {config.SLIPPAGE_PCT:.2%}")
    print(f"   Stop Loss (Long): {(1-config.LONG_STOP_MULT):.1%} от lower_band")
    print(f"   Stop Loss (Short): {(config.SHORT_STOP_MULT-1):.1%} от upper_band")
    
    # === STEP 1: ГЕНЕРАЦИЯ СИГНАЛОВ ===
    signal_generator = SignalGenerator(config)
    signals_df = signal_generator.generate_all_signals()
    
    # === STEP 2: ТОРГОВАЯ ЛОГИКА ===
    strategy = TradingStrategy(config)
    signals_df = strategy.apply_strategy(signals_df)
    
    # === STEP 3: СИМУЛЯЦИЯ СДЕЛОК ===
    engine = BacktestEngine(config)
    trade_log = engine.run_backtest(signals_df)
    
    # === STEP 4: РАСЧЁТ МЕТРИК ===
    print("\n" + "=" * 60)
    print("📊 STEP 4: РАСЧЁТ МЕТРИК")
    print("=" * 60)
    
    analyzer = PerformanceAnalyzer(config)
    
    # Sanity check
    sanity_check = analyzer.generate_sanity_check(signals_df)
    
    # Portfolio metrics
    portfolio_metrics = analyzer.calculate_metrics(trade_log) if not trade_log.empty else {}
    
    # Per-ticker metrics
    ticker_metrics = analyzer.calculate_per_ticker_metrics(trade_log) if not trade_log.empty else pd.DataFrame()
    
    if portfolio_metrics:
        print(f"\n📈 Portfolio Summary:")
        print(f"   Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}")
        print(f"   Win Rate: {portfolio_metrics['win_rate']:.1%}")
        print(f"   Total Return: {portfolio_metrics['total_return']:.2%}")
        print(f"   Max Drawdown: {portfolio_metrics['max_drawdown']:.2%}")
    
    # === STEP 5: ГЕНЕРАЦИЯ ОТЧЁТА ===
    print("\n" + "=" * 60)
    print("📝 STEP 5: ГЕНЕРАЦИЯ ОТЧЁТА")
    print("=" * 60)
    
    report_generator = ReportGenerator(config)
    
    report_text = report_generator.generate_audit_report(
        signals_df,
        trade_log,
        portfolio_metrics,
        ticker_metrics,
        sanity_check
    )
    
    report_generator.save_report(report_text, signals_df, trade_log)
    
    # Выводим отчёт в консоль
    print("\n" + report_text)
    
    print("\n" + "=" * 70)
    print("✅ BACKTEST PIPELINE ЗАВЕРШЁН!")
    print("=" * 70)
    
    return {
        'signals': signals_df,
        'trades': trade_log,
        'portfolio_metrics': portfolio_metrics,
        'ticker_metrics': ticker_metrics
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    results = run_full_backtest_pipeline()

