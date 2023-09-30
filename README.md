# Classification_of_images_from_camera_traps
# RZD_hackathon
## Классификация снимков с фотоловушек (задача от Алтайский государственный университет, «Цифровой прорыв», 2023, г. Новосибирск)

**Разработчик:** Команда "Деревяшки"

### Описание задачи
В нашей стране большое количество заповедных мест. Многие из них труднодоступны или совсем закрыты для туристов. В заповедниках живут разные виды животных, за которыми ученые наблюдают через фотоловушки:  высчитывают популяцию и разные другие важные показатели. Проблема заключается в невозможности контролировать качество каждого снимка фотоловушки. Т.к. количество снимков с каждой из них насчитывает несколько тысяч в год, просматривать каждую фотографию проблематично.

### Задача участников
Участникам предлагается разработать программный модуль с использованием технологий искусственного интеллекта, позволяющий классифицировать фотографии на качественные и некачественные. В результате создания такого решения существенно сократится количество времени, затрачиваемое на анализ полученных снимков.

### Подход к решению задачи
- Разметка данных
- Детекция: модель определения наличия животных на снимках (на базе YOLO v8)
- Классификация: Модель определения некачественных снимков (на базе EfficientNet-B4)

### Описание файлов решения
**Ноутбук обучения классификаций:** ` Classification.ipynb`
**Файл запуска классификации:** `animal_classification_V2.0.py`
**Файл весов модели классификации:** `efficient_weights_2class.pth`
**Ноутбук обучения детекции:** ` xxx`

Используемые библиотеки:
- ultralytics
- opencv-python
- pandas
- numpy
- Pytorch
- openCV
- torchmetrics

### Использованные данные
В работе использовались обезличенные данные от Алтайского государственного университета:
- ` train_dataset_altai__.zip `

### Контакты
Для связи с нами обращайтесь к [Dimk_88](https://t.me/Dimk_88).
