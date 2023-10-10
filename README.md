# Прогноз спроса товаров сети гипермаркетов "Лента"

## Структура репозитория

- /scripts/src - исполняемые файлы с инференсом модели
  - /tmp - папка для хранения исходных данных и результата расчета прогноза
  - encoder.save, scaler.save - энкодер и скейлер для предобработки данных
  - model.py - скрипт предобработки данных и выполнения прогноза
  - app.py - FastAPI сервис для автономного тестирования модели, заготовка для сервисов бэкенда
  - обученная модель лежит [здесь](https://drive.google.com/file/d/1_hg6Ik4bL5PoKDJzi39wJ1R5K67L0H8K/view?usp=sharing)
- /notebooks - jupiter-ноутбуки с экспериментами и результаты предсказаний для файла sales_submission.csv
  - /drafts - черновики ноутбуков с различными рассмотренными моделями
  -  demand_forecast_for_lenta_skus.ipynb - финальный ноутбук (без запуска можно увидеть на [nbviewer](https://nbviewer.org/github/Hackathon-Lenta-Team2/ds/blob/9e2f47aff284f2b6ab2937f56f12302dc867285a/notebooks/demand_forecast_for_lenta_skus.ipynb))
  -  sales_submission.csv - результаты тестовых предсказаний
_________

## Запуск модели (автономный)

- склонировать ds репозиторий
```bash
  git clone git@github.com:Hackathon-Lenta-Team2/ds.git
```
- модель находится в папке /scripts
- в папку scripts/scr нужно скачать модель по [ссылке](https://drive.google.com/file/d/1_hg6Ik4bL5PoKDJzi39wJ1R5K67L0H8K/view?usp=sharing)
- данные для прогноза лежат в scripts/src/tmp/ds_data.csv
- сборка образа в папке scripts/
  ```bash
  docker build --tag ds_lenta .
  ```
- запуск контейнера
   ```bash
  docker run --rm -it -p 8001:8000 --name ds ds_lenta
  ```
- после этого модель работает по адресу ```http://localhost:8001/ds/start```
- выводится статус, что выполняется прогноз, в это время в фоновом режиме выполняется функция make_forecast() из app.py
- по окончании прогноза результат сохраняется в scripts/scr/tmp/forecast_archive.json
- отправляется get-запрос по адресу ```http://localhost:8001/ds/ready```
____________________
## Особенности данных

- недельная сезонность
- характерные всплески спроса перед выходными и праздниками
- много пропусков в данных: нами была принята гипотеза, что отсутствие информации о продажах неравносильно отсутствию спроса, то есть пропущенные значения не заполнялись "0", а вместо них брались более ранние значения для данной пары "товар-магазин"
_____

## Рассмотренные модели

В ходе исследования для модели дополнительно генерировались новые признаки, такие как запаздывающие значения и скользящее среднее, флаг праздника даты, учитывающий и несколько дней до него, номер дня в неделе и в месяце, а также принадлежность товара к тому или иному кластеру. (От признака с кластером в последствии отказались, поскольку он практически не учитывался моделями).

|**Модель**|**Метрика WAPE на тестовой выборке**|Примечание|
|---|---|---|
|**RandomForest**|**0,45** |  |
|CatBoost Regressor|0,48 |  |
|LinearRegression|0,48 |  |
|LinearRegression с дополнительными признаками, сгенерированными VARMAX|0,48 | Поведение модели на кросс-валидации менее стабильно. Подробнее результаты исследования приведены [здесь](https://github.com/Hackathon-Lenta-Team2/ds/blob/main/notebooks/drafts/demand_forecast_for_lenta_skus.ipynb)|
|[LSTM](https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/)|0,76||
|DummyRegressor (константная модель, предсказывает предыдущее значение)|0,62||

Для дальнейшей работы выбрана модель с лучшей мерикой - RandomForest.
Проанализирована важность признаков. 
Топ-5 признаков:
-  скользящее среднее
-  лаг-14
-  лаг-21
-  день недели
-  лаг-20
____

## Сценарий работы с БД и бэкендом

|**Этап**|**Что происходит**|**Где происходит**|
|---|---|---|
|1|запрос: id товара, id магазина|backend|
|2|из базы данных товаров загружаются значения полей ('pr_group_id',	'pr_cat_id', 'pr_subcat_id', 'pr_uom_id'), из базы данных магазинов - ('st_city_id', 'st_division_code','st_type_format_id', 'st_type_loc_id', 'st_type_size_id')|backend|
|3|из архива продаж загружаются данные по продажам этой пары "товар-магазин" за предыдущие дни с 0-го по 21-й (или 22 предыдущих значения, что, в общем случае, не одно и то же) |backend|
|4|данные, указанные выше, объединяются в файл .csv с количеством строк, равным количеству запросов, и передаются на вход ml|backend|
|5|на основани таблицы по продажам рассчитываются лаги ('lag_14','lag_15', 'lag_16', 'lag_17', 'lag_18', 'lag_19', 'lag_20', 'lag_21'), при этом сдвигаясь для разных дат прогноза. Так, для прогноза на завтра - lag_14 = 2 недели назад (14 значений назад), а для прогноза на дату через 2 недели - lag_14 = вчера (одно значение назад). Рассчитывается скользящее среднее ('rolling_mean'). Число строк становится в 14 раз больше, чем пришло с бэкенда, число столбцов уменьшается на 14|ML|
|6|дата преобразуется в номер дня недели, дня в месяце, выполняется привязка к выходным и праздникам ('holiday', 'day_of_week', 'day_of_month', 'is_weekend')|ML|
|7|данные кодируются, масштабируются и передаются в модель для предсказаний|ML|
|8|предсказанные значения сохраняются в словарь и записываются в json-файл|ML|
_____
## Выводы и пути совершенствования
В результате работы над проектом создано работоспособное приложение, функционирующее как автономно, так и в составе набора сервисов бэкенда.

Надо отметить пути дальнейшего развития и повышения качества предсказаний:
- более детальное исследование возможностей для генерации новых признаков
- изменение концепции работы с пропусками (требуются дополнительные данные от заказчика для понимания природы пропусков)
- использование в модели не только сезонных, но и климатических признаков (требуются данные от заказчика по географии продаж и интеграция со сторонним погодным сервисом)
- оптимизация обработки данных до передачи в модель для ускорения расчета
- реализация возможности дообучения модели по мере накопления данных по новым продажам.

