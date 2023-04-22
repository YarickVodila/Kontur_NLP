# Описание решения кейса

Данный репозиторий содержит решение тестового задания на стажировку от компании Контур.

## Описание задачи

В Контуре мы много работаем с документами: арбитражные иски, госзакупки, исполнительные производства. В данном задании мы предлагаем вам сделать модель, которая поможет отделу госзакупок извлекать 
нужный кусок текста из документа для того, чтобы сформировать анкету заявки. То, какой именно фрагмент текста нужно извлечь, зависит от пункта анкеты, соответствующего документу.
Всего в каждом документе, с которыми вы будет работать, есть 1 из 2-х пунктов анкеты, по которым необходимо извлекать кусочки из текста:
- обеспечение исполнения контракта
- обеспечение гарантийных обязательств

Соответственно, ваша модель, принимая на вход `текст документа` и `наименование одного из двух пунктов`, должна возвращать `соответствующий кусочек текста из текста документа`.

## Анализ типов задач NLP

В самом начале исследования данных необходимо было понять под какую задачу NLP подходит данное задание. Всего было выделено на начальном этапе 3 потенциально возможные задачи:
1) `NER` - named entity recognition (распознавания именованных объектов);
2) `SpanCategorizer` - span categorizer (классификация интервалов);
3) `QA` - Question Answering (ответы на вопросы).

После изучения данных типов задач NLP, были выявлены некоторые особенности их реализации:

1) `SpanCategorizer` - при разработке решения использовалась библиотека Spacy. Но при обучении пайплайна, модель начинает деградировать и в конце концов вовсе не предсказывает ничего. Также при настройке модели необходимо было указать параметр n-gram, который при обучении было решено использовать N = range(3, 40), так как максимальное количество слов в ответе может быть примерно 40. Из-за этого параметра модель стала слишком 'большой' из за чего возникали ошибки. Поэтому этот тип задачи не использовался.
2) `QA` - при разработке решения использовалась библиотека DeepPavlov. Обучалась модель "squad_ru_convers_distilrubert_6L", которая была обучена на вопросах sberquad. Модель обучалась на данных контура около двух часов и в конечном итоге метрика accuracy на валидационной выборке получилась примерно 0.33. После тестирования трёх архитектур точность не сильно выросла и поэтому развитие данный тип задачи не получила.
3) `NER` - при разработке решения использовалась библиотека Spacy. Но в документации было написано, что не желательно, когда сущности длинные и характеризуются токенами в середине. Несмотря на это, пайплайн обучился довольно неплохо и было решено использовать этот тип задачи.

## Разработка решения
После того как тип задачи был выбран, необходимо было для удобства использования модели и её обучения необходимо было создать отдельный модуль (`ner_spacy_model.py`), который содержит класс с необходимыми методами для инициализации, обучения и предсказания модели.

Данный модуль использует файл в папке `Config` для инициализации и настройки модели. При необходимости можно дополнительно настроить пайплайн для обучения новой модели.

После обучения, лучшая модель будет сохранена с папке `best_model`. Сохранение модели происходит исходя из наилучшей метрики на валидацонной выборки, если она присутствует или на основе тренировочной выборки.  
Модуль также предполагает дообучение и использование ранее обученной модели. Для этого необходимо во время создания объекта в параметре `model_dir` указать путь до модели. 

Например: 
```python 
model_dir = best_model 
```

## Обзор файлов в директории проекта

Файл `Decision.ipynb` - данный файл содержит решение данного кейса. В нём производится необходимая обработка и очистка данных для тренировки, а также предсказание и запись в файл `predictions.json`

Файл `prepross.ipynb` - данный файл содержит тестирование предобработки данных.

Файл `NER_spacy.ipynb` - данный файл содержит предварительного решения и тестирование модели `ner` на этапе разработки.

Также в директории содержится папка `Testing` в которой хранятся тестовые файлы для исследования различных типов задач.

## Архитектура 
Компонент `tok2vek` переводит токены в вектор 

Компонент `ner` используется для поика именованных сущностей в тексте.