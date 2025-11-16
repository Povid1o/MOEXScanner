import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# Создаем папку для сохранения данных
output_folder = "moex_analysis_results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Создана папка для результатов: {output_folder}")

# Запрос параметров у пользователя
print("=" * 50)
print("СБОР ДАННЫХ С МОСКОВСКОЙ БИРЖИ")
print("=" * 50)

ticker = input("Введите тикер акции (например: SBER, GAZP, MTSS) - ").upper().strip()

timeframe_choice = input("Таймфрейм (1 - часы, 2 - дни) - ").strip()

# Определяем интервал на основе выбора и устанавливаем базовую дату начала
if timeframe_choice == "1":
    interval = "60"  # часовой
    timeframe_name = "hourly"
    default_days = 548  # примерно 1.5 года (365 * 1.5)
    default_period = "1.5 года назад"
    print("✓ Выбран часовой таймфрейм")
else:
    interval = "24"  # дневной
    timeframe_name = "daily"
    default_days = 5 * 365  # 5 лет
    default_period = "5 лет назад"
    print("✓ Выбран дневной таймфрейм")

date_from_input = input(f"Начиная с даты (в формате гггг-мм-дд, Enter для {default_period}) - ").strip()
if not date_from_input:
    # Если ничего не введено, ставим дату в зависимости от таймфрейма
    date_from = (datetime.now() - timedelta(days=default_days)).strftime("%Y-%m-%d")
    print(f"✓ Используется дата по умолчанию ({default_period}): {date_from}")
else:
    date_from = date_from_input

# Автоматически устанавливаем сегодняшнюю дату как дату окончания
date_to = datetime.now().strftime("%Y-%m-%d")

print("\n" + "=" * 50)
print("ПАРАМЕТРЫ СБОРА ДАННЫХ:")
print(f"Тикер: {ticker}")
print(f"Период: с {date_from} по {date_to}")
print(f"Таймфрейм: {timeframe_name}")
print(f"Папка сохранения: {output_folder}")
print("=" * 50)

def get_complete_ticker_data(ticker, date_from, date_to, interval="24"):
    """
    Получает полные данные с пагинацией
    """
    all_data = []
    start = 0
    page_size = 500

    print(f"\nЗагрузка данных для {ticker}...")

    while True:
        url = (f'http://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}'
               f'/candles.json?from={date_from}&till={date_to}&interval={interval}&start={start}')

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            data = response.json()

            if 'candles' not in data or not data['candles']['data']:
                break

            page_data = data['candles']['data']
            all_data.extend(page_data)

            print(f"  Загружено страниц: {len(all_data)} записей...")

            if len(page_data) < page_size:
                break

            start += page_size

        except Exception as e:
            print(f"  Ошибка при загрузке данных: {e}")
            break

    return all_data, data['candles']['columns'] if 'candles' in data else []

try:
    # Получаем данные с пагинацией
    all_data, columns = get_complete_ticker_data(ticker, date_from, date_to, interval)

    if not all_data:
        print(f"\n❌ Не удалось получить данные для тикера {ticker}")
        print("Возможные причины:")
        print("  - Тикер введен неверно")
        print("  - Нет данных за указанный период")
        print("  - Проблемы с подключением к API MOEX")
        exit()

    # Создаем DataFrame
    print("\n" + "=" * 50)
    print("ОБРАБОТКА ДАННЫХ")
    print("=" * 50)

    df = pd.DataFrame(all_data, columns=columns)

    # Конвертируем даты
    df['begin'] = pd.to_datetime(df['begin'])
    df['end'] = pd.to_datetime(df['end'])

    # Сортируем по дате
    df = df.sort_values('end')

    # Выводим информацию о DataFrame
    print("\nИНФОРМАЦИЯ О ДАТАФРЕЙМЕ:")
    print("=" * 30)
    df.info()

    # Основная статистика
    print(f"\nОСНОВНАЯ СТАТИСТИКА ДАННЫХ:")
    print("=" * 30)
    print(f"Период данных: {df['end'].min().strftime('%d.%m.%Y')} - {df['end'].max().strftime('%d.%m.%Y')}")
    print(f"Всего записей: {len(df)}")
    print(f"Диапазон цен закрытия: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"Средняя цена закрытия: {df['close'].mean():.2f}")

    # Сохраняем CSV файл
    csv_filename = f"{ticker}_{timeframe_name}_{date_from}_to_{date_to}.csv"
    csv_path = os.path.join(output_folder, csv_filename)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ Данные сохранены в CSV файл: {csv_filename}")

    # Строим график
    print("\n" + "=" * 50)
    print("ПОСТРОЕНИЕ ГРАФИКА")
    print("=" * 50)

    plt.figure(figsize=(14, 8))

    # Основной график цен
    plt.subplot(2, 1, 1)
    plt.plot(df['end'], df['close'], linewidth=2, color='blue', label='Цена закрытия')
    plt.plot(df['end'], df['high'], linewidth=1, color='green', alpha=0.7, label='Максимум')
    plt.plot(df['end'], df['low'], linewidth=1, color='red', alpha=0.7, label='Минимум')

    plt.title(f'График цен акций {ticker} ({timeframe_name})\nПериод: {df["end"].min().strftime("%d.%m.%Y")} - {df["end"].max().strftime("%d.%m.%Y")}',
              fontsize=14, fontweight='bold')
    plt.ylabel('Цена (руб.)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # График объема торгов
    plt.subplot(2, 1, 2)
    plt.bar(df['end'], df['volume'], color='orange', alpha=0.7, label='Объем торгов')
    plt.title('Объемы торгов', fontsize=12, fontweight='bold')
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Объем', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Сохраняем график
    plot_filename = f"{ticker}_{timeframe_name}_{date_from}_to_{date_to}.png"
    plot_path = os.path.join(output_folder, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ График сохранен в файл: {plot_filename}")

    # Показываем график
    print("✓ График отображен на экране")
    plt.show()

    # Создаем файл с информацией о сборе
    info_content = f"""ОТЧЕТ О СБОРЕ ДАННЫХ
Тикер: {ticker}
Период: с {date_from} по {date_to}
Таймфрейм: {timeframe_name}
Дата сбора: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

СТАТИСТИКА ДАННЫХ:
Период данных: {df['end'].min().strftime('%d.%m.%Y')} - {df['end'].max().strftime('%d.%m.%Y')}
Всего записей: {len(df)}
Первая цена закрытия: {df['close'].iloc[0]:.2f}
Последняя цена закрытия: {df['close'].iloc[-1]:.2f}
Минимальная цена: {df['close'].min():.2f}
Максимальная цена: {df['close'].max():.2f}
Средняя цена: {df['close'].mean():.2f}

СОХРАНЕННЫЕ ФАЙЛЫ:
Данные: {csv_filename}'
График: {plot_filename}
"""

    info_filename = f"report_{ticker}_{date_from}_to_{date_to}.txt"
    info_path = os.path.join(output_folder, info_filename)
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(info_content)

    print("\n" + "=" * 50)
    print("ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
    print("=" * 50)
    print(f"✓ Данные сохранены в папку: {output_folder}")
    print(f"✓ CSV файл с данными: {csv_filename}")
    print(f"✓ PNG файл с графиком: {plot_filename}")
    print(f"✓ TXT файл с отчетом: {info_filename}")
    print(f"\nВсего записей: {len(df)}")
    print(f"Период данных: {df['end'].min().strftime('%d.%m.%Y')} - {df['end'].max().strftime('%d.%m.%Y')}")

except requests.exceptions.RequestException as e:
    print(f"\n❌ Ошибка при запросе данных: {e}")
    print("Проверьте подключение к интернету и правильность введенных параметров")
except Exception as e:
    print(f"\n❌ Неизвестная ошибка: {e}")
    print("Попробуйте запустить скрипт еще раз с другими параметрами")