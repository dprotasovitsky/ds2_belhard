import sqlite3

import pandas as pd
from config.settings import config


class DatabaseManager:
    """Менеджер базы данных SQLite"""

    def __init__(self, db_path=config.DATABASE_PATH):
        self.db_path = db_path

    def create_database(self, df):
        """Создание и заполнение базы данных"""
        conn = self._get_connection()

        # Подготовка данных для базы
        df_db = self._prepare_data_for_db(df)

        # Создание таблицы
        df_db.to_sql("energy_consumption", conn, if_exists="replace", index=False)

        # Создание индексов (только если колонки существуют)
        self._create_indexes(conn, df_db.columns)

        conn.commit()
        conn.close()

    def run_analysis_queries(self):
        """Выполнение аналитических SQL запросов"""
        conn = self._get_connection()
        queries = {}

        try:
            # Почасовое потребление
            queries["hourly_avg"] = self._execute_query(
                conn,
                """
                SELECT hour, AVG(global_active_power) as avg_power
                FROM energy_consumption
                GROUP BY hour
                ORDER BY hour
            """,
            )

            # Потребление по дням недели
            queries["daily_avg"] = self._execute_query(
                conn,
                """
                SELECT day_of_week, AVG(global_active_power) as avg_power
                FROM energy_consumption
                GROUP BY day_of_week
                ORDER BY day_of_week
            """,
            )

            # Ежемесячное потребление (только если есть колонки year и month)
            try:
                queries["monthly_avg"] = self._execute_query(
                    conn,
                    """
                    SELECT year, month, AVG(global_active_power) as avg_power
                    FROM energy_consumption
                    GROUP BY year, month
                    ORDER BY year, month
                """,
                )
            except:
                queries["monthly_avg"] = []

            # Сравнение будни/выходные - ИСПРАВЛЕННЫЙ ЗАПРОС
            try:
                queries["weekday_weekend"] = self._execute_query(
                    conn,
                    """
                    SELECT
                        CASE WHEN weekend = 1 THEN 'Выходные' ELSE 'Будни' END as day_type,
                        AVG(global_active_power) as avg_power,
                        AVG(sub_metering_1) as avg_kitchen,
                        AVG(sub_metering_2) as avg_laundry,
                        AVG(sub_metering_3) as avg_water_heater
                    FROM energy_consumption
                    GROUP BY day_type
                """,
                )
            except Exception as e:
                print(f"Ошибка в запросе weekday_weekend: {e}")
                # Альтернативный запрос если основные колонки не существуют
                try:
                    queries["weekday_weekend"] = self._execute_query(
                        conn,
                        """
                        SELECT
                            CASE WHEN weekend = 1 THEN 'Выходные' ELSE 'Будни' END as day_type,
                            AVG(global_active_power) as avg_power
                        FROM energy_consumption
                        GROUP BY day_type
                        """,
                    )
                except:
                    queries["weekday_weekend"] = []

        except Exception as e:
            print(f"Error executing queries: {e}")
        finally:
            conn.close()

        return queries

    def _get_connection(self):
        """Получение соединения с базой данных"""
        return sqlite3.connect(self.db_path)

    def _prepare_data_for_db(self, df):
        """Подготовка данных для базы данных"""
        df_db = df.copy()

        # Переименовываем колонки в нижний регистр для SQL
        column_mapping = {
            "Global_active_power": "global_active_power",
            "Global_reactive_power": "global_reactive_power",
            "Voltage": "voltage",
            "Global_intensity": "global_intensity",
            "Sub_metering_1": "sub_metering_1",
            "Sub_metering_2": "sub_metering_2",
            "Sub_metering_3": "sub_metering_3",
            "DayOfWeek": "day_of_week",
            "Year": "year",
            "Month": "month",
            "Hour": "hour",
            "Weekend": "weekend",
            "Season": "season",
        }

        # Применяем переименование только для существующих колонок
        for old_name, new_name in column_mapping.items():
            if old_name in df_db.columns:
                df_db = df_db.rename(columns={old_name: new_name})

        return df_db

    def _create_indexes(self, conn, available_columns):
        """Создание индексов для ускорения запросов (только для существующих колонок)"""
        cursor = conn.cursor()

        # Всегда создаем индекс для datetime
        if "DateTime" in available_columns:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_datetime ON energy_consumption(datetime)"
            )

        # Создаем индексы только если колонки существуют
        if all(col in available_columns for col in ["year", "month"]):
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_year_month ON energy_consumption(year, month)"
            )

        if "hour" in available_columns:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_hour ON energy_consumption(hour)"
            )

        if "day_of_week" in available_columns:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_day_of_week ON energy_consumption(day_of_week)"
            )

    def _execute_query(self, conn, query):
        """Выполнение SQL запроса"""
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()
