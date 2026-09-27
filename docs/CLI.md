# Sprite Soul CLI

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
# Проверить CUDA и скачать Depth, DSINE и SAM до первой генерации
.\.venv\Scripts\sprite-soul.exe setup

# Читать события установки как JSON Lines
.\.venv\Scripts\sprite-soul.exe setup --progress json

# Подготовить только Depth Anything V2
.\.venv\Scripts\sprite-soul.exe setup --models depth

# Подготовить только DSINE
.\.venv\Scripts\sprite-soul.exe setup --models ai

# Подготовить только SAM 2.1 Base+ для интерактивного выделения листвы в GUI
.\.venv\Scripts\sprite-soul.exe setup --models sam

# Только установить CUDA-сборку PyTorch
.\.venv\Scripts\sprite-soul.exe setup --models none

# Скачать модели на другой машине без NVIDIA GPU
.\.venv\Scripts\sprite-soul.exe setup --skip-cuda
```

`setup` повторно использует уже установленные компоненты и кэш моделей. Установка [CUDA-сборки PyTorch](https://docs.pytorch.org/get-started/previous-versions/) запускается только если её нет; для этого проекта выбирается CUDA 12.6 или 12.4 по версии драйвера. Отдельный CUDA Toolkit не требуется. Системный драйвер NVIDIA должен быть установлен: при его отсутствии или слишком старой версии команда покажет [официальную страницу драйверов](https://www.nvidia.com/Download/index.aspx). Установка драйвера может потребовать прав администратора и перезагрузки Windows.

В UI команда **Подготовить AI** выполняет такую же подготовку в фоновом потоке. Кнопка **Выделить листву** сама подготовит SAM 2.1 при первом запуске; CLI не создаёт маски, поскольку для SAM нужны интерактивные клики. Прогресс отображается в строке состояния.

## Быстрый старт

```powershell
# Depth Anything V2 → 16-битная Depth → Normal из Depth
.\.venv\Scripts\sprite-soul.exe generate .\assets\tree.png

# Несколько спрайтов за один запуск
.\.venv\Scripts\sprite-soul.exe generate .\assets\tree.png .\assets\house.png --save-project

# DirectX, инверсия Depth и сглаживание
.\.venv\Scripts\sprite-soul.exe generate .\assets\tree.png --convention directx --invert-depth --depth-smooth 1.5

# Depth из готовой карты глубины — без Depth AI
.\.venv\Scripts\sprite-soul.exe generate .\assets\tree.png --maps depth --depth-map .\maps\tree_depth.png

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

Позиционные аргументы — один или несколько исходных `.png` или проектов `.ssoul`. Результаты по умолчанию сохраняются в `./generated/`, который игнорируется Git; `-o/--output` позволяет выбрать другой каталог. Для `sprite.png` команда по умолчанию создаёт `sprite_depth.png` и `sprite_normal.png`. `--save-project` дополнительно создаёт `sprite.ssoul` с итоговой глубиной; его можно открыть в UI или использовать как вход CLI. Проект ссылается на исходный PNG, поэтому исходник должен оставаться доступным.

Depth PNG сохраняется в 16-битном RGBA: одинаковое значение глубины в R/G/B, alpha исходного PNG, расширенная до 16 бит. Normal PNG — 8-битный RGBA с исходной alpha. Размеры и положение пикселей не меняются. По умолчанию экспортируется OpenGL +Y. Нормали из Depth рассчитываются из обработанной итоговой глубины.

`--depth-map` принимает готовый серый PNG 8 или 16 бит (в том числе экспортированный Sprite Soul RGBA PNG), размер которого равен размеру исходного спрайта. Это значение используется напрямую в диапазоне 0–1, без повторной нормализации. Прозрачность всегда берётся из исходного PNG. Опцию можно указать только для одного исходного PNG. Вход `.ssoul` также пропускает Depth AI; настройки обработки Depth применяются к глубине проекта.

Если выходной файл уже существует, команда сообщает об ошибке и не запускает модель для этого входа. Для перезаписи укажите `--overwrite`. При пакетной обработке ошибки одного входа печатаются в stderr, остальные входы продолжают обрабатываться; общий код выхода будет `1`. Совпадающие имена выходных файлов в одном запуске считаются ошибкой, даже с `--overwrite`.

## Параметры

| Параметр | Назначение | Значение по умолчанию |
| --- | --- | --- |
| `--maps both\|depth\|normal\|albedo\|all` | Какие карты записывать | `both` |
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
| `--save-project` | Дополнительно записать `.ssoul` | выключено |
| `--overwrite` | Разрешить перезапись | выключено |
| `--progress text\|json\|none` | Формат прогресса | `text` |
| `--quiet` | Синоним `--progress none` | выключено |

`--maps normal` запускает только DSINE и не загружает Depth Anything V2. `--save-project` требует глубину. `--maps depth` не запускает DSINE. Флаги инверсии, `--ai-smoothing` и `--ai-details` влияют на итоговые нормали так же, как настройки UI. Детализация использует только высокочастотную яркость исходного изображения, сохраняя крупные наклоны DSINE; значение `0` полностью отключает этот этап.

В текстовом режиме прогресс и ошибки идут в stderr, имена записанных файлов — в stdout. Код выхода: `0` — все входы обработаны, `1` — ошибка хотя бы одного входа, `2` — неверные аргументы, `130` — прерывание Ctrl+C. Синтаксические ошибки аргументов выводятся текстом до выбора режима прогресса.
