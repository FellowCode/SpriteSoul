# Проверка SuperMat на спрайтах

Проверка выполнена 1 октября 2026 года на `treesV2.png` и `well.png` из `example_sprites/`.
Использована одиночная версия [SuperMat](https://github.com/hyj542682306/SuperMat),
которая предсказывает Albedo, Roughness и Metallic по одному изображению.
После эксперимента SuperMat встроена в Sprite Soul: меню **Генерировать → Карта шероховатости**,
режим просмотра **Roughness**, **Все карты**, экспорт `_roughness.png` и подготовка AI.
В CLI доступны `--maps roughness` и `setup --models roughness`; оба режима `all` включают SuperMat.

```powershell
.\.venv\Scripts\python.exe -m smg setup --models roughness
.\.venv\Scripts\python.exe -m smg generate example_sprites/treesV2.png --maps roughness -o generated
```

Интеграция использует перечисленные ниже фиксированные версии кода и весов.
Зелёный канал ORM сохраняется как float до восстановления исходного размера, затем
экспортируется в 8-битный RGBA без gamma, нормализации контраста и изменений alpha.
Модель запускается в отдельном процессе; его завершение освобождает CUDA-память.
Полный PNG, включая атлас, подаётся целиком. Roughness хранится в текущем сеансе и PNG,
без добавления в формат `.ssoul`. Ниже приведены измерения первоначального эксперимента.

## Результат

Модель успешно работает на NVIDIA RTX 3060 Laptop с 6 ГБ VRAM.

| Спрайт | Исходный размер | Inference | Roughness, 5–95-й перцентили |
| --- | ---: | ---: | ---: |
| Дерево (`treesV2.png`) | 319 × 362 | 1,197 с | 0,616–0,682 |
| Колодец (`well.png`) | 1236 × 1272 | 0,541 с | 0,549–0,729 |

Время измерено вокруг вызова модели с CUDA synchronization, без загрузки модели,
записи PNG и изменения размеров. Первый вызов включает первоначальную подготовку CUDA.
Пик выделенной памяти PyTorch — около 3137 МиБ, пик зарезервированной — 3494 МиБ
(около 3,4 ГиБ). Это показатели PyTorch, а не общее потребление всего GPU.
Перцентили рассчитаны по пикселям исходного спрайта с alpha ≥ 128.

На дереве Roughness почти однородная: мелкие листья и кора не выражены в карте.
На колодце видны границы крыши, деревянной конструкции и камней, но мелкий рисунок
сглажен. В Metallic модели присутствуют заметные серые значения на камнях,
поэтому эту карту стоит проверять отдельно, прежде чем использовать как готовый материал.
Albedo визуально смягчает исходные тени, но также теряет детали, особенно у большого колодца.

**Вывод:** SuperMat технически пригодна для быстрого получения начальной карты
шероховатости на этой машине. Для художественных спрайтов результат требует проверки
и возможности ручной настройки. Эталонных карт для этих двух изображений нет;
физическая точность предсказания не измерялась. Почти однородная карта сама по себе
не доказывает ошибку: цветовая детализация не обязана соответствовать вариациям Roughness.

## Файлы

Результаты находятся в игнорируемом Git каталоге `generated/supermat/`:

- `roughness_comparison.png` — исходник и Roughness для обоих спрайтов.
- `comparison.png` — исходник, Roughness, Metallic и Albedo.
- `run.json` — версии, время, память, перцентили и результаты проверки alpha.
- `treesV2/treesV2_roughness.png` и `well/well_roughness.png` — итоговые карты.
- В подпапках также есть итоговые Albedo, Metallic и ORM, исходники и сырые выходы
  модели `*_512.png`.

Итоговые карты имеют исходные размеры и исходный alpha-канал; проверено точное
совпадение alpha у всех восьми итоговых PNG. RGB-каналы Roughness и Metallic одинаковы.
Контраст карты и её диапазон не изменялись. Roughness и Metallic следует импортировать
как линейные данные, без преобразования sRGB. Прозрачность добавлена при сохранении; сама модель
не предсказывает alpha.

## Метод и окружение

- Код SuperMat: commit `4fe25bc6cb8cb7a3ed81ce74512f760dd33f80f4`.
- Checkpoint: [oyiya/SuperMat](https://huggingface.co/oyiya/SuperMat), revision
  `91ffb8edaf259a1e5bbcbcd53b702ad44d70b709`, файл `supermat.pth`,
  3 541 172 771 байт. SHA-256:
  `21d171726e8170e1ed566985be68ef3c5009d762800926ad85c1f288747b7b29`.
- Базовые компоненты: [sd2-community/stable-diffusion-2-1](https://huggingface.co/sd2-community/stable-diffusion-2-1),
  revision `bb2154823665391b4fb29b0b9cf82a198964ee05`.
- Python 3.11.9, PyTorch `2.6.0+cu126`, torchvision `0.21.0+cpu`, diffusers `0.31.0`,
  accelerate `1.1.1`, transformers `4.57.6`, huggingface-hub `0.36.2`.
- Inference: 512 × 512, один шаг DDIM с `timestep_spacing="trailing"`, FP16.

Вход подготовлен официальной функцией SuperMat: RGBA уменьшается/увеличивается до
512 × 512 bilinear-фильтром, прозрачный фон композится на серый RGB `(0.5, 0.5, 0.5)`.
Такой resize временно меняет пропорции неквадратного спрайта. На выходе предсказания
возвращаются к исходным размерам bilinear-фильтром, затем копируется исходная alpha.
Увеличение карты колодца обратно до 1236 × 1272 не восстанавливает потерянные детали.

Для уменьшения объёма загрузки базовый UNet создан из конфигурации и полностью
заполнен checkpoint SuperMat. Проверка `load_state_dict(strict=True)` прошла;
пропущенных или лишних ключей нет. Базовые веса UNet, многовидовая модель и UV Refiner
не скачивались. Text encoder и VAE загружены в FP16; исходный код модели не изменялся.
Импортируется напрямую одиночная архитектура, которая использует внимание PyTorch
и не требует xformers.

Обычная загрузка большого файла и загрузка через Xet зависали до передачи данных.
В экспериментальном скрипте используется возобновляемая загрузка HTTP Range частями
по 16 МиБ с четырьмя соединениями. SHA-256 всех трёх больших файлов проверен по
метаданным соответствующего фиксированного revision на Hugging Face.

## Повторный запуск

Скрипт эксперимента: `scripts/try_supermat.py`. Код, окружение и веса хранятся в
игнорируемом Git каталоге `models/SuperMat/`.

Установленное окружение повторно использует PyTorch и другие общие зависимости
основного `.venv` через `.pth`, сохраняя собственные diffusers и accelerate.
Для повторного inference на двух спрайтах:

```powershell
.\models\SuperMat\.venv-supermat\Scripts\python.exe scripts/try_supermat.py
```

Для другого изображения и отдельного каталога результатов:

```powershell
.\models\SuperMat\.venv-supermat\Scripts\python.exe scripts/try_supermat.py --output generated/supermat_house example_sprites/house.png
```

Подготовка с нуля при наличии основного `.venv` Sprite Soul с CUDA PyTorch:

```powershell
git clone https://github.com/hyj542682306/SuperMat.git models/SuperMat
git -C models/SuperMat checkout 4fe25bc6cb8cb7a3ed81ce74512f760dd33f80f4
.\.venv\Scripts\python.exe -m venv models/SuperMat/.venv-supermat
$supermatBaseSite = (Resolve-Path .venv/Lib/site-packages).Path
Set-Content -LiteralPath models/SuperMat/.venv-supermat/Lib/site-packages/sprite_soul_base.pth -Value $supermatBaseSite -Encoding ascii
.\models\SuperMat\.venv-supermat\Scripts\python.exe -m pip install --no-cache-dir diffusers==0.31.0 accelerate==1.1.1
.\models\SuperMat\.venv-supermat\Scripts\python.exe scripts/try_supermat.py --download-only
.\models\SuperMat\.venv-supermat\Scripts\python.exe scripts/try_supermat.py
```

Веса и базовые компоненты занимают около 4,1 ГиБ на диске; скрипт перед загрузкой
проверяет наличие места и оставляет резерв 1 ГБ.
