# Sprite Soul CLI

`--maps all` создаёт Depth, Normal, Albedo, AO и Roughness за один запуск. `--maps depth|normal|albedo|ao|roughness` создаёт одну выбранную карту; `both` оставлен для Depth и Normal и используется по умолчанию. AO рассчитывается из Depth и Normal; недостающие карты генерируются Depth Anything V2 и DSINE. Готовую Depth передайте через `--depth-map` или откройте `.ssoul`; готовую Normal — через `--normal-map`. При наличии обеих карт AI-модели не запускаются. `--maps ao` экспортирует только AO, вспомогательные карты не записываются. `--ao-depth-only` использует только Depth. Roughness создаётся SuperMat независимо от остальных карт.

CLI генерирует карты из PNG без запуска графического интерфейса. После установки из корня проекта:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\sprite-soul.exe --help
.\.venv\Scripts\sprite-soul.exe generate --help
.\.venv\Scripts\sprite-soul.exe setup --help
```

На других системах команда — `sprite-soul`; эквивалентный запуск — `python -m smg`. CLI не создаёт `QApplication` и не открывает окно. Для генерации Depth и AI Normal нужны видеокарта и драйвер NVIDIA. При первом запуске CLI сам установит CUDA-сборку PyTorch и скачает нужную модель; потребуется доступ к интернету. Экспорт только готовой Depth PNG или проекта `.ssoul` с `--maps depth` не требует CUDA.

## Подготовка моделей

```powershell
# Проверить CUDA и скачать Depth, DSINE, CLIPSeg, IntrinsicAnything и SuperMat
.\.venv\Scripts\sprite-soul.exe setup

# Читать события установки как JSON Lines
.\.venv\Scripts\sprite-soul.exe setup --progress json

# Подготовить только Depth Anything V2
.\.venv\Scripts\sprite-soul.exe setup --models depth

# Подготовить только DSINE
.\.venv\Scripts\sprite-soul.exe setup --models ai

# Подготовить только CLIPSeg для определения кроны в GUI
.\.venv\Scripts\sprite-soul.exe setup --models clipseg

# Подготовить только IntrinsicAnything для Albedo
.\.venv\Scripts\sprite-soul.exe setup --models albedo

# Подготовить SuperMat для Roughness
.\.venv\Scripts\sprite-soul.exe setup --models roughness

# Только установить CUDA-сборку PyTorch
.\.venv\Scripts\sprite-soul.exe setup --models none

# Скачать модели на другой машине без NVIDIA GPU
.\.venv\Scripts\sprite-soul.exe setup --skip-cuda
```

`setup` повторно использует уже установленные компоненты и кэш моделей. Режим `all` теперь включает и checkpoint IntrinsicAnything (около 14,4 ГиБ). Установка [CUDA-сборки PyTorch](https://docs.pytorch.org/get-started/previous-versions/) запускается только если её нет; для этого проекта выбирается CUDA 12.6 или 12.4 по версии драйвера. Отдельный CUDA Toolkit не требуется. Системный драйвер NVIDIA должен быть установлен: при его отсутствии или слишком старой версии команда покажет [официальную страницу драйверов](https://www.nvidia.com/Download/index.aspx). Установка драйвера может потребовать прав администратора и перезагрузки Windows.

В UI команда **Подготовить AI** выполняет такую же подготовку в фоновом потоке. При генерации Normal CLIPSeg автоматически создаёт маску кроны и определяет дерево: расширенная маска должна занимать строго больше 40% непрозрачных пикселей. В этом случае форма кроны корректирует Normal DSINE; иначе Normal остаётся обычной. CLI также создаёт маску и сохраняет её в `.ssoul` при `--save-project`. Сохранённая в проекте или подтверждённая в UI маска используется без повторного запуска CLIPSeg. Кнопка **Определить крону** позволяет вручную настроить порог и проверить маску. Прогресс отображается в строке состояния.

## Быстрый старт

Генерация и экспорт выполняются одной командой: достаточно передать исходный PNG, выбрать карты и каталог результата. Сохранять или предварительно создавать проект не нужно.

```powershell
.\.venv\Scripts\sprite-soul.exe generate sprite.png --maps all -o exported --no-save-project
```

Команда сразу записывает `sprite_depth.png`, `sprite_normal.png`, `sprite_albedo.png`, `sprite_ao.png` и `sprite_roughness.png` в `exported/`. `.ssoul` не создаётся; промежуточная Depth и маска кроны остаются в памяти до завершения обработки. `--no-save-project` явно выключает сохранение проекта; это также поведение по умолчанию. Для одной карты замените `all` на `depth`, `normal`, `albedo`, `ao` или `roughness`. Отдельная команда экспорта не требуется.

```powershell
# Depth Anything V2 и DSINE: Depth и Normal
.\.venv\Scripts\sprite-soul.exe generate .\assets\tree.png

# Все пять карт за один запуск
.\.venv\Scripts\sprite-soul.exe generate .\assets\tree.png --maps all

# Только Roughness; SuperMat без Depth, DSINE и IntrinsicAnything
.\.venv\Scripts\sprite-soul.exe generate .\assets\tree.png --maps roughness

# Только AO; можно добавить --depth-map для пропуска Depth AI
.\.venv\Scripts\sprite-soul.exe generate .\assets\tree.png --maps ao --ao-radius 32 --ao-strength 1.5

# AO из двух готовых карт, без запуска моделей
.\.venv\Scripts\sprite-soul.exe generate .\assets\tree.png --maps ao --depth-map .\maps\tree_depth.png --normal-map .\maps\tree_normal.png

# AO только из Depth, без DSINE
.\.venv\Scripts\sprite-soul.exe generate .\assets\tree.png --maps ao --depth-map .\maps\tree_depth.png --ao-depth-only

# Только Albedo с настройками коррекции
.\.venv\Scripts\sprite-soul.exe generate .\assets\tree.png --maps albedo --albedo-strength 1.2 --albedo-smooth 2 --albedo-shadows 0.8

# Несколько спрайтов за один запуск
.\.venv\Scripts\sprite-soul.exe generate .\assets\tree.png .\assets\house.png --save-project

# DirectX, инверсия Depth и сглаживание
.\.venv\Scripts\sprite-soul.exe generate .\assets\tree.png --convention directx --invert-depth --depth-smooth 1.5

# Depth из готовой карты глубины — без Depth AI
.\.venv\Scripts\sprite-soul.exe generate .\assets\tree.png --maps depth --depth-map .\maps\tree_depth.png

# AO для нескольких спрайтов из соответствующих карт в каталоге
.\.venv\Scripts\sprite-soul.exe generate .\assets\tree.png .\assets\house.png --maps ao --depth-map .\maps

# Сгенерировать AI Normal из сохранённого проекта
.\.venv\Scripts\sprite-soul.exe generate .\generated\tree.ssoul --maps normal --overwrite

# AI Normal (DSINE) и обе карты; сглаживание подавляет ореолы
.\.venv\Scripts\sprite-soul.exe generate .\assets\tree.png --dsine-fov 60 --ai-smoothing 1.5

# Только AI Normal: Depth-модель не загружается
.\.venv\Scripts\sprite-soul.exe generate .\assets\tree.png --maps normal --normal-source ai
```

Старый синтаксис без слова `generate` также поддерживается. `sprite-soul --help` показывает команды; `generate --help` и `setup --help` содержат все параметры соответствующей команды.

## Прогресс для других программ

Обе команды принимают `--progress text|json|none`. По умолчанию этапы и вывод `pip` идут в stderr, а в stdout печатаются пути созданных файлов. `--progress none` и прежний `--quiet` скрывают этапы, сохраняя ошибки и пути. В режиме `--progress json` **stdout содержит только JSON Lines**: один завершённый JSON-объект на строку, с немедленным flush. Предупреждения сторонних библиотек могут идти в stderr. Пути результатов и ошибки тоже передаются JSON-событиями; обычных строк с путями в stdout в этом режиме нет.

```powershell
.\.venv\Scripts\sprite-soul.exe setup --progress json
.\.venv\Scripts\sprite-soul.exe generate .\assets\tree.png --progress json
```

Пример чтения из Python:

```python
import json
import subprocess

with subprocess.Popen(
    ["sprite-soul", "setup", "--progress", "json"],
    stdout=subprocess.PIPE,
    text=True,
    encoding="ascii",  # JSON экранирует не-ASCII символы
) as process:
    for line in process.stdout:
        event = json.loads(line)
        if event["event"] == "progress" and event.get("total"):
            print(event["phase"], event["current"], "/", event["total"], event["unit"])
        elif event["event"] in ("result", "error", "complete"):
            print(event)
    if process.wait() != 0:
        raise RuntimeError("Sprite Soul завершился с ошибкой")
```

Формат событий, версия `schema_version: 1`:

| `event` | Поля | Смысл |
| --- | --- | --- |
| `progress` | `phase`, `status`, `message` | Начало, обновление, использование кэша или завершение этапа |
| `result` | `path`, `input` | Созданный файл |
| `error` | `message`, опционально `input` | Ошибка подготовки или обработки файла |
| `complete` | `status`, `processed`, `failed` | Итог команды |

Для `pip_install` числовые `current` и `total` имеют `unit: "bytes"` и относятся к **текущему скачиваемому пакету**; поле `package` указывает пакет. Это приблизительные значения из вывода `pip`. Для `model_download` `unit: "files"` показывает число завершённых файлов данной модели; при скачивании исходного кода DSINE приходят дополнительные события с `unit: "bytes"`. Если размер скачивания неизвестен, `total` отсутствует. Не следует считать `current/total` общим процентом всей установки. События `progress` также содержат `component` или `model`/`file`, когда это применимо. Поле `input` добавляется при генерации.

## Вход и результат

Позиционные аргументы — один или несколько исходных `.png`, проектов `.ssoul`, каталогов или шаблонов файлов. Каталог включает PNG и `.ssoul` непосредственно в нём, без вложенных папок; шаблоны в PowerShell передавайте в кавычках. Результаты по умолчанию сохраняются в `./generated/`, который игнорируется Git; `-o/--output` позволяет выбрать другой каталог. Для `sprite.png` команда по умолчанию создаёт `sprite_depth.png` и `sprite_normal.png`. `--save-project` дополнительно создаёт `sprite.ssoul` с итоговой глубиной; его можно открыть в UI или использовать как вход CLI. Проект ссылается на исходный PNG, поэтому исходник должен оставаться доступным.

```powershell
sprite-soul generate tree.png well.png rock.png --maps all -o exported
sprite-soul generate .\sprites\ --maps all -o exported
sprite-soul generate ".\sprites\*.png" --maps both -o exported
```

Для нескольких входов генерация выполняется проходами по всему пакету: Depth, CLIPSeg для спрайтов без сохранённой маски, DSINE Normal, AO из итоговой геометрии, IntrinsicAnything Albedo, SuperMat Roughness, затем сохранение проектов при `--save-project`. Выполняются только нужные этапы. Модель остаётся загруженной на протяжении своего прохода и выгружается перед загрузкой следующей. Albedo и Roughness работают в отдельных процессах, каждый из которых обслуживает весь свой проход и завершается до перехода к следующему. Модели подготавливаются один раз на пакет.

Для каждого выбранного типа карт выводится отдельный счётчик, например `Normal: сгенерировано 3 из 10 карт`: сначала 0, затем обновление после каждого успешного экспорта. В JSON это события `progress` с `phase: "batch"`, `map: "depth"|"normal"|"albedo"|"ao"|"roughness"`, `current`, `total` и `unit: "maps"`. `total` — фиксированное число входов после раскрытия каталогов и шаблонов; входы с ошибкой остаются в общем количестве, но не увеличивают число созданных карт. Depth и AO считаются отдельно, вспомогательные этапы и `.ssoul` в эти счётчики не входят. `--progress none` и `--quiet` скрывают счётчики.

Исходники и доступные Depth/маски проверяются до запуска моделей. Данные каждого спрайта временно сохраняются на диск, поэтому весь пакет не удерживается в RAM. Временные файлы удаляются при завершении, ошибке или Ctrl+C. При `--albedo-debug DIR` каждый спрайт получает отдельный подкаталог `DIR/<имя_спрайта>/`.

Depth PNG сохраняется в 16-битном RGBA: одинаковое значение глубины в R/G/B, alpha исходного PNG, расширенная до 16 бит. Normal PNG — 8-битный RGBA с исходной alpha. Размеры и положение пикселей не меняются. По умолчанию экспортируется OpenGL +Y. Normal создаётся моделью DSINE независимо от Depth.

`--depth-map` принимает готовый серый PNG 8 или 16 бит (в том числе экспортированный Sprite Soul RGBA PNG). Для одного входа укажите файл; для нескольких — каталог, содержащий `<имя_спрайта>_depth.png` для каждого исходника. Каталог можно указать и для одного входа. Каждая карта проверяется на совпадение ширины и высоты со своим спрайтом до запуска моделей; отсутствующая или неподходящая карта вызывает ошибку для соответствующего входа. Переданная карта имеет приоритет над Depth внутри `.ssoul`. Глубина используется напрямую в диапазоне 0–1, без повторной нормализации. Прозрачность всегда берётся из исходного PNG. Без `--depth-map` вход `.ssoul` использует сохранённую Depth; настройки обработки Depth применяются и к ней. При генерации Normal из PNG CLIPSeg автоматически определяет крону. Если маска занимает больше 40% непрозрачных пикселей или проект уже содержит маску кроны, генерация Normal накладывает мелкие детали DSINE на выпуклую форму кроны. Остальные пиксели сохраняют обычную AI Normal.

Если выходной файл уже существует, команда сообщает об ошибке и не запускает модель для этого входа. Для перезаписи укажите `--overwrite`. При пакетной обработке ошибка одного входа исключает его из следующих этапов, остальные входы продолжают обрабатываться; общий код выхода будет `1`. Уже экспортированные карты сохраняются. В JSON-режиме каждый файл получает событие `result` сразу после экспорта; результаты идут в порядке проходов по моделям. `processed` считает полностью завершённые спрайты, `failed` — входы с ошибкой. Совпадающие имена выходных файлов в одном запуске считаются ошибкой, даже с `--overwrite`.

## Параметры

| Параметр | Назначение | Значение по умолчанию |
| --- | --- | --- |
| `--maps both\|depth\|normal\|albedo\|ao\|roughness\|all` | Какие карты записывать | `both` |
| `--depth-map PNG\|DIR` | Готовая Depth для AO/Depth: один PNG или каталог карт для пакета | не задана |
| `--normal-map PNG\|DIR` | Готовая Normal для AO/Normal: PNG или каталог `<имя>_normal.png`, ориентация `--convention` | не задана |
| `--ao-depth-only` | Рассчитывать AO из Depth, без AI Normal | выключено |
| `--normal-source ai` | Источник Normal; поддерживается только DSINE | `ai` |
| `--convention opengl\|directx` | Конвенция Normal | `opengl` |
| `--invert-depth` | Инвертировать глубину | выключено |
| `--depth-range 0.01..3` | Диапазон глубины | `1` |
| `--depth-contrast 0.25..3` | Контраст глубины | `1` |
| `--depth-smooth 0..8` | Радиус Gaussian blur (sigma) | `0` |
| `--ai-smoothing 0..8` | Alpha-aware сглаживание AI Normal | `1.5` |
| `--ai-details 0..1.5` | Сила восстановления мелкого рельефа из исходного RGB | `0.35` |
| `--dsine-fov 20..120` | Поле зрения DSINE в градусах | `60` |
| `--invert-ai-x` / `--no-invert-ai-x` | Направление X у DSINE | инверсия включена |
| `--invert-ai-y` / `--no-invert-ai-y` | Направление Y у DSINE | инверсия включена |
| `--crown-mode auto\|off` | Автоматическая коррекция Normal для кроны; `off` не применяет сохранённую маску, но сохраняет её в `.ssoul` | `auto` |
| `--crown-threshold 0..1` | Порог CLIPSeg при автоматическом поиске кроны | `0.5` |
| `--ao-radius 1..128` | Радиус AO в пикселях | `24` |
| `--ao-strength 0..4` | Степень итоговой видимости AO; `0` даёт белую карту, геометрия не меняется | `2` |
| `--albedo-strength 0..4` | Сила коррекции Albedo | `1` |
| `--albedo-smooth 0..16` | Сглаживание поля освещения Albedo (sigma) | `2` |
| `--albedo-shadows 0..4` | Сила коррекции теней Albedo | `1` |
| `--albedo-debug DIR` | Сохранить промежуточные карты Albedo | выключено |
| `--save-project` / `--no-save-project` | Включить / выключить дополнительную запись `.ssoul`; PNG экспортируются в обоих случаях | выключено |
| `--overwrite` | Разрешить перезапись | выключено |
| `--progress text\|json\|none` | Формат прогресса | `text` |
| `--quiet` | Синоним `--progress none` | выключено |

`--maps normal` запускает только DSINE и не загружает Depth Anything V2. `--save-project` требует глубину. `--maps depth` не запускает DSINE. Флаги инверсии, `--ai-smoothing` и `--ai-details` влияют на итоговые нормали так же, как настройки UI. Детализация использует только высокочастотную яркость исходного изображения, сохраняя крупные наклоны DSINE; значение `0` полностью отключает этот этап.

`--normal-map` принимает RGB/RGBA PNG 8 бит. Для нескольких входов нужен каталог с `<имя>_normal.png` для каждого спрайта; размеры проверяются до запуска моделей. `--convention directx` переводит входную Normal в OpenGL для расчёта AO и задаёт ориентацию экспортируемой Normal. Готовая карта используется напрямую, без повторной постобработки, инверсии AI-осей или коррекции кроны; alpha всегда берётся из исходного спрайта. `--maps normal --normal-map ...` также экспортирует готовую карту без DSINE. `--ao-depth-only` несовместим с `--normal-map`. Алгоритм и ограничения AO описаны в [AO](AO.md).

`--maps roughness` подготавливает только SuperMat (около 4,1 ГиБ весов и базовых компонентов) и CUDA. Инференс выполняется в отдельном процессе, в 512×512, за один шаг FP16. Результат — `<имя>_roughness.png` в 8-битном RGBA с исходным размером и alpha. Чёрный = гладкая поверхность, белый = шероховатая. Импортируйте карту как линейные данные без sRGB. Atlas обрабатывается целиком; мелкие детали могут сглаживаться. Roughness экспортируется в PNG и не сохраняется в `.ssoul`.

В текстовом режиме прогресс и ошибки идут в stderr, имена записанных файлов — в stdout. Код выхода: `0` — все входы обработаны, `1` — ошибка хотя бы одного входа, `2` — неверные аргументы, `130` — прерывание Ctrl+C. Синтаксические ошибки аргументов выводятся текстом до выбора режима прогресса.
