import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class AdjustedReturnsCalculator:
    def __init__(self, data_dir: str, cache_dir: str = None):
        self.data_dir = data_dir
        self.cache_dir = cache_dir or os.path.join(data_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.corp_actions_path = os.path.join(data_dir, "corporate_actions.csv")
        self.corp_df = pd.read_csv(self.corp_actions_path)
        self.corp_df["action_date"] = pd.to_datetime(self.corp_df["action_date"], errors="coerce")

    def process_ticker(self, ticker: str, show_plot: bool = True) -> pd.DataFrame:
        print(f"\n{'='*60}")
        print(f"📊 Обрабатываем тикер: {ticker}")
        print(f"{'='*60}")

        # === 1. Загружаем OHLCV ===
        file_path = self._find_ticker_file(ticker)
        df = pd.read_csv(file_path)
        df = self._prepare_ohlcv(df)
        
        print(f"✅ Загружено {len(df)} дней данных")
        print(f"   Период: {df['date'].min().date()} → {df['date'].max().date()}")

        # === 2. Фильтруем корпоративные действия ===
        actions = self.corp_df[self.corp_df["ticker"] == ticker].copy()
        
        if actions.empty:
            print("⚠️  Корпоративных действий не найдено — возврат исходных log_return.")
            df["close_adj"] = df["close"]
            df["adjustment_factor"] = 1.0
            df["log_return"] = np.log(df["close"] / df["close"].shift(1))
            df["log_return_adj"] = df["log_return"]
            return df

        print(f"\n📋 Найдено {len(actions)} корпоративных действий:")
        for _, action in actions.iterrows():
            action_type = action['action_type']
            if action_type == 'dividend':
                print(f"   {action['action_date'].date()}: дивиденд {action['dividend_amount']}₽")
            elif action_type == 'split':
                # Правильно определяем split_ratio
                split_ratio = action.get('split_ratio', None)
                if pd.isna(split_ratio):
                    # Если split_ratio пустой, берем из dividend_amount (костыль в данных)
                    split_ratio = action.get('dividend_amount', 1)
                print(f"   {action['action_date'].date()}: сплит 1:{split_ratio}")

        # === 3. ПРАВИЛЬНАЯ корректировка цен ===
        df = self._apply_corporate_actions_correct(df, actions)

        # === 4. Вычисляем log_returns ===
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["log_return_adj"] = np.log(df["close_adj"] / df["close_adj"].shift(1))

        # === 5. Рассчёт волатильности ===
        window = 30
        df["volatility_raw"] = df["log_return"].rolling(window).std() * np.sqrt(252)
        df["volatility_adj"] = df["log_return_adj"].rolling(window).std() * np.sqrt(252)

        # === 6. Статистика ===
        self._print_statistics(df, actions)

        # === 7. Сохраняем результат ===
        cache_path = os.path.join(self.cache_dir, f"{ticker}_adjusted.csv")
        df.to_csv(cache_path, index=False)
        print(f"\n💾 Сохранено в {cache_path}")

        # === 8. Графики проверки ===
        if show_plot:
            self._plot_comprehensive_check(df, actions, ticker)

        return df

    def _apply_corporate_actions_correct(self, df: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
        """
        ПРАВИЛЬНАЯ корректировка цен на дивиденды и сплиты.
        
        Алгоритм:
        1. Идем от ПОЗДНЕГО к РАННЕМУ событию (обратный хронологический порядок)
        2. Для дивидендов: ratio = (close_before - dividend) / close_before
        3. Для сплитов: ratio = 1 / split_ratio (например, 1/8 = 0.125 для сплита 1:8)
        4. Все цены ДО события умножаются на cumulative_ratio
        """
        df = df.copy()
        df["close_adj"] = df["close"].astype(float)
        df["adjustment_factor"] = 1.0
        
        # Сортируем действия от ПОЗДНЕГО к РАННЕМУ
        actions_sorted = actions.sort_values("action_date", ascending=False).copy()

        # Кумулятивный adjustment factor (накапливаем все корректировки)
        cumulative_factor = 1.0

        for _, action in actions_sorted.iterrows():
            action_date = pd.to_datetime(action["action_date"])
            action_type = action["action_type"]

            # Находим индекс ex-date в данных
            ex_date_mask = df["date"] == action_date
            
            if not ex_date_mask.any():
                # Если точной даты нет, берем ближайшую следующую
                future_dates = df["date"] > action_date
                if not future_dates.any():
                    print(f"  ⚠️  Дата {action_date.date()} после всех данных, пропускаем")
                    continue
                ex_idx = df[future_dates].index[0]
            else:
                ex_idx = df[ex_date_mask].index[0]

            # Вычисляем adjustment ratio для ЭТОГО события
            event_ratio = 1.0

            if action_type == "dividend":
                dividend = action.get("dividend_amount", 0)
                
                if pd.notna(dividend) and dividend > 0:
                    # Получаем цену закрытия ДО экс-дивидендной даты
                    if ex_idx > 0:
                        close_before = df.loc[ex_idx - 1, "close"]
                    else:
                        close_before = df.loc[ex_idx, "open"]
                    
                    # Проверка: дивиденд не должен быть больше цены
                    if dividend >= close_before:
                        print(f"  ⚠️  АНОМАЛИЯ: дивиденд {dividend}₽ >= цены {close_before:.2f}₽ на {action_date.date()}")
                        print(f"      Возможно, это особая выплата. Пропускаем.")
                        continue
                    
                    # Вычисляем ratio
                    event_ratio = (close_before - dividend) / close_before
                    
                    print(f"  ✅ Дивиденд {dividend:.2f}₽ на {action_date.date()}: "
                          f"close_before={close_before:.2f}₽, ratio={event_ratio:.4f}")

            elif action_type == "split":
                # Определяем split_ratio из данных
                split_ratio = action.get("split_ratio", None)
                
                if pd.isna(split_ratio):
                    # Костыль: если split_ratio пустой, берем из dividend_amount
                    split_ratio = action.get("dividend_amount", None)
                
                if pd.notna(split_ratio) and split_ratio != 1:
                    # Для сплита 1:N (например, 1:8) цены ДО сплита делим на N
                    # Это эквивалентно умножению на (1/N)
                    event_ratio = 1.0 / split_ratio
                    
                    print(f"  ✅ Сплит 1:{split_ratio} на {action_date.date()}: ratio={event_ratio:.4f}")
                else:
                    print(f"  ⚠️  Некорректный split_ratio: {split_ratio}, пропускаем")
                    continue

            # Обновляем кумулятивный фактор
            cumulative_factor *= event_ratio

            # Применяем корректировку ко ВСЕМ ценам ДО ex_idx
            if ex_idx > 0:
                df.loc[:ex_idx-1, "close_adj"] *= event_ratio
                df.loc[:ex_idx-1, "adjustment_factor"] *= event_ratio

        return df

    def _print_statistics(self, df: pd.DataFrame, actions: pd.DataFrame):
        """Выводит статистику по корректировке"""
        print(f"\n{'='*60}")
        print("📈 СТАТИСТИКА КОРРЕКТИРОВКИ")
        print(f"{'='*60}")
        
        # Волатильность до/после
        raw_vol = df["log_return"].std() * np.sqrt(252)
        adj_vol = df["log_return_adj"].std() * np.sqrt(252)
        
        print(f"\n1️⃣  Волатильность (annualized):")
        print(f"   До коррекции:    {raw_vol:.2%}")
        print(f"   После коррекции: {adj_vol:.2%}")
        
        vol_change_abs = adj_vol - raw_vol
        vol_change_pct = (adj_vol / raw_vol - 1) * 100 if raw_vol > 0 else 0
        
        print(f"   Изменение:       {vol_change_abs:+.2%} ({vol_change_pct:+.1f}%)")
        
        # Проверка эффективности
        if abs(vol_change_pct) < 2:
            print(f"   ⚠️  ВНИМАНИЕ: изменение < 2%, возможно корректировка не работает!")
        elif vol_change_pct < -10:
            print(f"   ✅ Хорошо: волатильность снизилась на {abs(vol_change_pct):.1f}%")
        
        # Проверка на аномальные returns
        print(f"\n2️⃣  Выбросы в доходностях:")
        
        raw_outliers = df[df["log_return"].abs() > 0.10]
        adj_outliers = df[df["log_return_adj"].abs() > 0.10]
        
        print(f"   До коррекции:  {len(raw_outliers)} дней с |return| > 10%")
        print(f"   После:         {len(adj_outliers)} дней с |return| > 10%")
        print(f"   Устранено:     {len(raw_outliers) - len(adj_outliers)} выбросов")
        
        if len(raw_outliers) - len(adj_outliers) == 0:
            print(f"   ⚠️  ВНИМАНИЕ: ни один выброс не устранен!")
        
        # Показываем топ-5 оставшихся выбросов
        if len(adj_outliers) > 0:
            print(f"\n   Топ-5 оставшихся выбросов ПОСЛЕ корректировки:")
            top_outliers = adj_outliers.nlargest(5, "log_return_adj", keep="first")
            for _, row in top_outliers.iterrows():
                print(f"   {row['date'].date()}: raw={row['log_return']:+.2%} → adj={row['log_return_adj']:+.2%}")
        
        # Проверка дивидендных дней
        print(f"\n3️⃣  Проверка корректировки в дивидендные/сплит дни:")
        
        for _, action in actions.iterrows():
            action_date = action["action_date"]
            action_type = action["action_type"]
            
            # Находим доходность в день события и на следующий день
            mask = (df["date"] >= action_date) & (df["date"] <= action_date + pd.Timedelta(days=5))
            df_window = df[mask].head(2)
            
            if len(df_window) > 0:
                first_row = df_window.iloc[0]
                
                if action_type == "dividend":
                    dividend = action["dividend_amount"]
                    print(f"\n   📅 {action_date.date()} (дивиденд {dividend:.2f}₽):")
                else:
                    split_ratio = action.get("split_ratio") or action.get("dividend_amount")
                    print(f"\n   📅 {action_date.date()} (сплит 1:{split_ratio}):")
                
                print(f"      Raw return:  {first_row['log_return']:+.2%}")
                print(f"      Adj return:  {first_row['log_return_adj']:+.2%}")
                print(f"      Разница:     {abs(first_row['log_return'] - first_row['log_return_adj']):.2%}")
                
                # Оценка качества корректировки
                if abs(first_row['log_return']) > 0.05 and abs(first_row['log_return_adj']) < 0.03:
                    print(f"      ✅ Корректировка работает!")
                elif abs(first_row['log_return'] - first_row['log_return_adj']) < 0.01:
                    print(f"      ⚠️  Корректировка НЕ применилась!")

        # Финальная проверка: сравнение цен
        print(f"\n4️⃣  Проверка цен (последний день):")
        last_row = df.iloc[-1]
        print(f"   Raw close:  {last_row['close']:.2f}₽")
        print(f"   Adj close:  {last_row['close_adj']:.2f}₽")
        print(f"   Cumulative adjustment factor: {last_row['adjustment_factor']:.4f}")

    def _plot_comprehensive_check(self, df: pd.DataFrame, actions: pd.DataFrame, ticker: str):
        """Комплексный набор графиков для проверки"""
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))
        
        # === График 1: Сравнение цен ===
        ax1 = axes[0]
        ax1.plot(df["date"], df["close"], label="Raw close", linewidth=1.5, alpha=0.7, color="blue")
        ax1.plot(df["date"], df["close_adj"], label="Adjusted close", linewidth=1.5, color="orange")
        
        # Отмечаем события
        for _, action in actions.iterrows():
            color = "red" if action["action_type"] == "dividend" else "purple"
            ax1.axvline(action["action_date"], color=color, alpha=0.3, linestyle="--", linewidth=1.5)
        
        ax1.set_title(f"{ticker}: Сравнение цен (raw vs adjusted)", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Цена закрытия, ₽")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # === График 2: Log returns ===
        ax2 = axes[1]
        ax2.plot(df["date"], df["log_return"], label="Raw returns", linewidth=0.8, alpha=0.6, color="blue")
        ax2.plot(df["date"], df["log_return_adj"], label="Adjusted returns", linewidth=0.8, alpha=0.9, color="orange")
        
        for _, action in actions.iterrows():
            color = "red" if action["action_type"] == "dividend" else "purple"
            ax2.axvline(action["action_date"], color=color, alpha=0.3, linestyle="--", linewidth=1.5)
        
        ax2.set_title("Log Returns: до и после корректировки", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Log Return")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color="black", linewidth=0.5)
        
        # === График 3: Rolling volatility ===
        ax3 = axes[2]
        ax3.plot(df["date"], df["volatility_raw"], label="Raw volatility", linewidth=1.5, alpha=0.7, color="blue")
        ax3.plot(df["date"], df["volatility_adj"], label="Adjusted volatility", linewidth=1.5, color="orange")
        
        for _, action in actions.iterrows():
            color = "red" if action["action_type"] == "dividend" else "purple"
            ax3.axvline(action["action_date"], color=color, alpha=0.3, linestyle="--", linewidth=1.5)
        
        ax3.set_title("Rolling Volatility (30d, annualized)", fontsize=14, fontweight="bold")
        ax3.set_ylabel("Volatility")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # === График 4: Гистограммы ===
        ax4 = axes[3]
        
        raw_rets = df["log_return"].dropna()
        adj_rets = df["log_return_adj"].dropna()
        
        ax4.hist(raw_rets, bins=60, alpha=0.5, label=f"Raw (σ={raw_rets.std():.4f})", color="blue", range=(-0.3, 0.3))
        ax4.hist(adj_rets, bins=60, alpha=0.5, label=f"Adj (σ={adj_rets.std():.4f})", color="orange", range=(-0.3, 0.3))
        
        ax4.set_title("Распределение log returns", fontsize=14, fontweight="bold")
        ax4.set_xlabel("Log Return")
        ax4.set_ylabel("Частота")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def _prepare_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in ["end", "date", "datetime", "begin", "tradedate"]:
            if c in df.columns:
                df["date"] = pd.to_datetime(df[c], errors="coerce")
                break
        else:
            raise ValueError("❌ Не найдена колонка даты")

        if "close" not in df.columns:
            raise ValueError("❌ Нет колонки 'close'")

        df = df.sort_values("date").reset_index(drop=True)
        return df

    def _find_ticker_file(self, ticker: str) -> str:
        for f in os.listdir(self.data_dir):
            if ticker in f and f.endswith(".csv") and "corporate_actions" not in f:
                return os.path.join(self.data_dir, f)
        raise FileNotFoundError(f"❌ Файл для {ticker} не найден")


# ============================================
# ИСПОЛЬЗОВАНИЕ
# ============================================
if __name__ == "__main__":
    calc = AdjustedReturnsCalculator(data_dir="/content/moex_analysis_results")
    
    # Тест на BELU (есть и дивиденды, и сплит)
    print("\n" + "="*70)
    print("ТЕСТ 1: BELU (дивиденды + сплит 1:8)")
    print("="*70)
    df_belu = calc.process_ticker("BELU", show_plot=True)
    
    # Тест на VTBR (гигантский дивиденд)
    print("\n" + "="*70)
    print("ТЕСТ 2: VTBR (экстремальный дивиденд 25.58₽)")
    print("="*70)
    df_vtbr = calc.process_ticker("VTBR", show_plot=True)