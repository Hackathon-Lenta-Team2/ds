**Клонирование репозитория**
```bash
git clone
```

**Сборка**
```bash
docker build --t metal .
```
**Запуск**
```bash
docker run --rm -it -p 8501:8501 --name LGBMContainer metal
```

После этого модель доступна по адресу 
```bash
localhost:8501/
```
