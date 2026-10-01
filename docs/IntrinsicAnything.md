# Проверка IntrinsicAnything для albedo спрайтов

Проверка выполнена 27 сентября 2026 года на двух изображениях из `example_sprites/`. Исходный код модели: [zju3dv/IntrinsicAnything](https://github.com/zju3dv/IntrinsicAnything), checkpoint albedo: [LittleFrog/IntrinsicAnything](https://huggingface.co/LittleFrog/IntrinsicAnything). Результаты эксперимента перенесены в локальную папку `generated/albedo_intrinsicanything/`, которую Git игнорирует; в репозитории хранятся только исходные тестовые спрайты.

## Среда и входные данные

- Windows, NVIDIA RTX 3060 Laptop с 6 ГБ VRAM.
- Локальная копия IntrinsicAnything на коммите `5fa5ec0`, Python 3.11, PyTorch `2.6.0+cu126`, torchvision `0.21.0+cpu`, PyTorch Lightning `1.9.5`, diffusers `0.31.0`.
- Checkpoint `weights/albedo/checkpoints/last.ckpt` размером 15 458 840 153 байта (около 14,4 ГиБ). Код модели, окружение и веса размещены в игнорируемом Git каталоге `models/IntrinsicAnything` внутри проекта Sprite Soul.
- Для запуска с этой версией PyTorch в *локальной копии модели* потребовались три совместимые правки: `torch.load(..., weights_only=False, mmap=True)` для старого checkpoint; отложенный импорт неиспользуемого `carvekit`; создание архитектуры CLIP из кода вместо повторного скачивания её весов (нужные веса уже есть в checkpoint). Эти правки не вносились в Sprite Soul.

| Входной спрайт | Размер | Результат с DDIM 100 |
| --- | ---: | --- |
| [treesV2.png](../example_sprites/treesV2.png) | 319 × 362 | `generated/albedo_intrinsicanything/treesV2_albedo.png` |
| [well_spring.png](../example_sprites/well_spring.png) | 1192 × 1319 | `generated/albedo_intrinsicanything/well_spring_albedo.png` |

У модели выход RGB на белом фоне. В сохранённых результатах Sprite Soul исходный alpha-канал возвращён без изменений; сравнение масок с входными PNG показало полное совпадение. Размеры файлов также сохранены.

## Прогоны

| Проверка | Параметры | Результат |
| --- | --- | --- |
| Быстрая проверка обоих спрайтов | `--ddim 20 --batch_size 1` | Модель запустилась на 6 ГБ без ошибки нехватки памяти. |
| Основной проход обоих спрайтов | `--ddim 100 --batch_size 1` | Получены файлы по ссылкам выше. На дереве освещение стало ровнее, но листья и кора потеряли детали. На колодце смягчились блики, но размылись черепица, камни и линии верёвки. Потребление VRAM во время загрузки и inference наблюдалось около 5115 МиБ. |
| Направляемая обработка колодца по плиткам | `--ddim 20 --batch_size 1 --guidance 3 --splits_vertical 4 --splits_horizontal 4 --splits_overlap 0`, за основу взят результат DDIM 100 | `generated/albedo_intrinsicanything/well_spring_tiled_artifacts_albedo.png`. За ~16 минут обработаны 16 плиток, пик VRAM — около 5985 из 6144 МиБ, без OOM. Появились заметные прямоугольные швы, тёмное пятно на крыше и светлый прямоугольник в области ворота/колодца. |

Команды для повторения из каталога клона IntrinsicAnything (сначала скопировать два входных PNG в отдельный каталог `in_sprites`, а `well_spring.png` — в `in_well`):

```powershell
python inference.py --input_dir in_sprites --model_dir weights/albedo --output_dir out/albedo_100 --ddim 100 --batch_size 1
python inference.py --input_dir in_well --model_dir weights/albedo --output_dir out/albedo_well_4x4 --guidance_dir out/albedo_100 --guidance 3 --splits_vertical 4 --splits_horizontal 4 --splits_overlap 0 --ddim 20 --batch_size 1
```

Исходная реализация уменьшает изображение для основного прохода до 256 пикселей, а затем увеличивает результат. Это заметный источник потери мелких деталей на большом спрайте колодца. Проверка качества была визуальной: эталонных albedo для этих спрайтов нет, поэтому количественная оценка точности не проводилась.

**Вывод тестового прохода:** IntrinsicAnything технически запускается на RTX 3060 с 6 ГБ, но прямой вывод модели не даёт готовую albedo-карту для этих спрайтов. Основной проход стирает часть нарисованных деталей; режим плиток без перекрытия даёт выраженные артефакты. Теперь Sprite Soul использует предсказание только как низкочастотное поле освещения и применяет его к исходным RGB в полном разрешении; запуск доступен через GUI и `--maps albedo` в CLI.

## Исправление загрузки на Windows, 1 октября 2026

На `example_sprites/house.png` воспроизведено аварийное завершение с кодом `0xC0000005`: `faulthandler` показал сбой в `LitEma.__init__` при клонировании параметров UNet. Во время сбоя свободная системная память с учётом файла подкачки снизилась примерно до 145 МиБ. Предупреждение `LightningDeprecationWarning`, которое раньше попадало в окно ошибки, не объясняло завершение процесса.

Теперь Sprite Soul запускает `smg/intrinsic_worker.py` в окружении модели. Worker извлекает собственные копии весов в fp16, подставляет обученные EMA-параметры UNet и освобождает отображение большого training checkpoint до создания архитектуры. Модель создаётся в fp16 с `use_ema=False`: дополнительная копия UNet для обучения больше не выделяется. Имена и полнота EMA-весов проверяются, а недостающие веса модели вызывают ошибку. Основной checkpoint и зависимости остаются прежними.

После исправления дом прошёл DDIM 100, batch size 1. Результат сохранён в игнорируемом Git каталоге `generated/albedo-house-fixed/house_albedo.png`; размер 845 × 886 и исходный alpha-канал сохранены. Полный вывод каждого запуска теперь записывается в `generated/logs/intrinsic-*.log`. При ошибке приложение показывает код возврата, путь к журналу и его последние строки; для Windows access violation и нехватки памяти добавляется пояснение.
