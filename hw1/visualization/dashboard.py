import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config.settings import config
from plotly.subplots import make_subplots


class DashboardVisualizer:
    """Визуализатор для основного дашборда"""

    def __init__(self, df, queries):
        self.df = df
        self.queries = queries

    def _plot_time_series(self, sample_df):
        """Визуализация временного ряда"""
        fig = px.line(
            sample_df,
            x="DateTime",
            y="Global_active_power",
            title="Потребление электроэнергии",
        )
        fig.update_layout(template=config.PLOT_THEME)
        return fig

    def _plot_distribution(self):
        """Визуализация распределения"""
        fig = px.histogram(
            self.df,
            x="Global_active_power",
            nbins=50,
            title="Распределение активной мощности",
        )
        fig.update_layout(template=config.PLOT_THEME)
        return fig

    def _plot_seasonal_consumption(self):
        """Потребление по сезонам"""
        # Проверяем наличие колонки Season
        if "Season" not in self.df.columns:
            return self._create_empty_plot("Данные о сезонах недоступны")

        seasonal_data = (
            self.df.groupby("Season")["Global_active_power"].mean().reset_index()
        )
        fig = px.bar(
            seasonal_data,
            x="Season",
            y="Global_active_power",
            title="Среднее потребление по сезонам",
        )
        fig.update_layout(template=config.PLOT_THEME)
        return fig

    def _plot_hourly_consumption(self):
        """Почасовое потребление"""
        if "hourly_avg" not in self.queries:
            return self._create_empty_plot("Данные не доступны")

        hourly_data = pd.DataFrame(
            self.queries["hourly_avg"], columns=["hour", "avg_power"]
        )
        fig = px.bar(
            hourly_data, x="hour", y="avg_power", title="Среднее потребление по часам"
        )
        fig.update_layout(template=config.PLOT_THEME)
        return fig

    def _plot_weekly_consumption(self):
        """Потребление по дням недели"""
        if "daily_avg" not in self.queries:
            return self._create_empty_plot("Данные не доступны")

        daily_data = pd.DataFrame(
            self.queries["daily_avg"], columns=["day", "avg_power"]
        )
        days = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
        daily_data["day_name"] = [days[i] for i in daily_data["day"]]
        fig = px.bar(
            daily_data, x="day_name", y="avg_power", title="Потребление по дням недели"
        )
        fig.update_layout(template=config.PLOT_THEME)
        return fig

    def _plot_monthly_consumption(self):
        """Ежемесячное потребление"""
        if "monthly_avg" not in self.queries:
            return self._create_empty_plot("Данные не доступны")

        monthly_data = pd.DataFrame(
            self.queries["monthly_avg"], columns=["year", "month", "avg_power"]
        )
        monthly_data["period"] = (
            monthly_data["year"].astype(str)
            + "-"
            + monthly_data["month"].astype(str).str.zfill(2)
        )
        fig = px.line(
            monthly_data, x="period", y="avg_power", title="Ежемесячное потребление"
        )
        fig.update_layout(template=config.PLOT_THEME)
        return fig

    def _plot_correlation_matrix(self):
        """Корреляционная матрица"""
        numeric_cols = [
            "Global_active_power",
            "Global_reactive_power",
            "Voltage",
            "Global_intensity",
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3",
        ]
        # Проверяем наличие всех колонок
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        if len(available_cols) < 2:
            return self._create_empty_plot(
                "Недостаточно данных для корреляционной матрицы"
            )

        corr_matrix = self.df[available_cols].corr()
        fig = px.imshow(
            corr_matrix, title="Корреляционная матрица", aspect="auto", text_auto=".2f"
        )
        fig.update_layout(template=config.PLOT_THEME)
        return fig

    def _plot_weekday_weekend_comparison(self):
        """Сравнение будни/выходные"""
        if "weekday_weekend" not in self.queries:
            return self._create_empty_plot("Данные не доступны")

        weekday_data = pd.DataFrame(
            self.queries["weekday_weekend"],
            columns=[
                "day_type",
                "avg_power",
                "avg_kitchen",
                "avg_laundry",
                "avg_water_heater",
            ],
        )

        # Проверяем наличие данных для графика
        available_columns = []
        column_names = []
        display_names = []

        if (
            "avg_kitchen" in weekday_data.columns
            and not weekday_data["avg_kitchen"].isna().all()
        ):
            available_columns.append("avg_kitchen")
            column_names.append("Кухня")
            display_names.append("Кухня")

        if (
            "avg_laundry" in weekday_data.columns
            and not weekday_data["avg_laundry"].isna().all()
        ):
            available_columns.append("avg_laundry")
            column_names.append("Прачечная")
            display_names.append("Прачечная")

        if (
            "avg_water_heater" in weekday_data.columns
            and not weekday_data["avg_water_heater"].isna().all()
        ):
            available_columns.append("avg_water_heater")
            column_names.append("Водонагреватель")
            display_names.append("Водонагреватель")

        if not available_columns:
            return self._create_empty_plot("Нет данных для сравнения помещений")

        # Создаем данные для графика
        plot_data = []
        for day_type in weekday_data["day_type"].unique():
            day_data = weekday_data[weekday_data["day_type"] == day_type].iloc[0]
            for i, col in enumerate(available_columns):
                if not pd.isna(day_data[col]):
                    plot_data.append(
                        {
                            "day_type": day_type,
                            "room_type": display_names[i],
                            "power": day_data[col],
                        }
                    )

        if not plot_data:
            return self._create_empty_plot("Нет данных для отображения")

        plot_df = pd.DataFrame(plot_data)

        fig = px.bar(
            plot_df,
            x="day_type",
            y="power",
            color="room_type",
            title="Потребление по типам помещений (будни/выходные)",
            barmode="group",
            labels={
                "power": "Мощность (kW)",
                "day_type": "Тип дня",
                "room_type": "Помещение",
            },
        )

        fig.update_layout(template=config.PLOT_THEME)
        return fig

    def _plot_submeter_analysis(self):
        """Анализ суб-метров"""
        sub_meter_cols = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
        available_cols = [col for col in sub_meter_cols if col in self.df.columns]

        if not available_cols:
            return self._create_empty_plot("Нет данных по суб-метрам")

        # Вычисляем общее потребление по каждому суб-метру
        sub_meter_data = []
        names = ["Кухня", "Прачечная", "Водонагреватель"]

        for i, col in enumerate(available_cols):
            if i < len(names):
                total_consumption = self.df[col].sum()
                if total_consumption > 0:  # Только если есть данные
                    sub_meter_data.append(
                        {"name": names[i], "value": total_consumption}
                    )

        if not sub_meter_data:
            return self._create_empty_plot("Нет данных для анализа суб-метров")

        plot_df = pd.DataFrame(sub_meter_data)

        fig = px.pie(
            plot_df,
            values="value",
            names="name",
            title="Распределение потребления по суб-метрам",
        )
        fig.update_layout(template=config.PLOT_THEME)
        return fig

    def _plot_room_consumption_details(self):
        """Детальная информация по потреблению помещений"""
        # Проверяем наличие данных о суб-метрах
        sub_meter_cols = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
        available_cols = [col for col in sub_meter_cols if col in self.df.columns]

        if not available_cols:
            return self._create_empty_plot("Нет данных по помещениям")

        # Собираем статистику по каждому помещению
        room_stats = []
        room_names = ["Кухня", "Прачечная", "Водонагреватель"]

        for i, col in enumerate(available_cols):
            if i < len(room_names):
                room_data = self.df[col]
                if room_data.sum() > 0:  # Только если есть потребление
                    room_stats.append(
                        {
                            "Помещение": room_names[i],
                            "Среднее потребление": room_data.mean(),
                            "Максимальное потребление": room_data.max(),
                            "Минимальное потребление": room_data.min(),
                            "Общее потребление": room_data.sum(),
                        }
                    )

        if not room_stats:
            return self._create_empty_plot("Нет данных по потреблению помещений")

        stats_df = pd.DataFrame(room_stats)

        # Создаем график среднего потребления
        fig = px.bar(
            stats_df,
            x="Помещение",
            y="Среднее потребление",
            title="Среднее потребление по помещениям",
            color="Помещение",
        )
        fig.update_layout(template=config.PLOT_THEME)
        return fig

    def _create_empty_plot(self, message):
        """Создание пустого графика с сообщением"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template=config.PLOT_THEME,
            height=400,
        )
        return fig
