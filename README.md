# ds
forecast model for Lenta Hackaton
## Структура репозитория

- /scripts - заготовки исполняемых файлов с инференсом модели
- /notebooks - воспроизводимый код модели в jupiter и результаты предсказаний для файла sales_submission.csv
_________

## Исследованные гипотезы

В ходе исследования для модели дополнительно гененерировались новые признаки, такие как сдвиг и запаздывающие значения, а также принадлежность товара к тому или иному кластеру.

|**Модель**|**Метрика на тестовой выборке**|Примечание|
|---|---|---|
|RandomForest|0,45 |  |
|CatBoost Regressor|0,48 |  |
|LinearRegression|0,48 |  |
|LinearRegression с дополнительными признаками, сгенерированными VARMAX|0,48 | Поведение модели менее стабильно|

Необходимые предсказания были получены, но на данный момент качество модели оставляет желать лучшего, 
необходимы дальнейшие эксперименты по предобработке данных и настройкам обучения моделей.


## Предполагаемый сценарий работы с БД
- запрос: id товара, id магазина, дата = последняя дата обучающего датасета + 1
- из базы данных товаров загружаются значения полей ('pr_group_id',	'pr_cat_id', 'pr_subcat_id', 'pr_uom_id'), из базы данных магазинов - ('st_city_id', 'st_division_code',
       'st_type_format_id', 'st_type_loc_id', 'st_type_size_id')
- из архива продаж загружаются данные по продажам этойпары "товар-магазин" за предыдущие дни с 21-го по 14-й
- данные указанные выше объединяются и в формате (???) передаются на вход ml и сортируются по дате
- в model.py рассчитываются средние значения и лаги ('lag_14',
       'lag_15', 'lag_16', 'lag_17', 'lag_18', 'lag_19', 'lag_20', 'lag_21',
       'rolling_mean')
- в model.py дата преобразуется в номер дня недели, дня в месяце, привязка к выходным и праздникам ('holiday', 'day_of_week', 'day_of_month', 'is_weekend')
- в model.py на основании сохраненной матицы корреляции расчитывается кластер товара ('cluster')
- данные кодируются, масштабируются и передаются в модель для предсказаний
- предсказанные значения сохраняются в словарь и записываются в json файл
