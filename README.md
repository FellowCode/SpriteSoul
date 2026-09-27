# Sprite Soul

Локальный редактор Depth и Normal карт для PNG-спрайтов и атласов. При первой генерации приложение скачивает [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) и, при выборе AI Normal, [DSINE](https://github.com/baegwangbin/DSINE) с [весами](https://huggingface.co/dylanebert/DSINE). Код DSINE хранится в пользовательском кэше `%LOCALAPPDATA%\SpriteSoul`, веса — в кэше Hugging Face. Для генерации нужна NVIDIA CUDA; редактирование сохранённого проекта работает без загрузки модели.

## Установка на Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe app.py
```

Откройте PNG, нажмите **Генерировать** для Depth, затем **Генерировать AI Normal** или выберите `AI`/`Hybrid` в `Normal Source`. Нормали DSINE сразу переводятся в OpenGL и в таком виде кэшируются; DirectX получается инверсией Y при выводе. `Invert AI X` и `Invert AI Y` включены по умолчанию для спрайтов, на которых DSINE воспринимает рельеф наоборот. Их можно переключать независимо без повторного inference; они влияют на AI, Hybrid, Lighting Preview и экспорт. `AI Influence` смешивает направления нормалей в режиме Hybrid; `DSINE FOV` задаёт предположение о камере (60° по умолчанию). Результат DSINE кэшируется в памяти до замены исходника, изменения FOV или явной регенерации. DSINE обучалась на перспективных изображениях, поэтому для ортографических спрайтов её результат стоит сравнивать с Depth. Колесо мыши меняет масштаб, перетаскивание в режиме Pan двигает холст. В режиме Lighting Preview перетаскивание двигает свет. **Экспорт** создаёт `_depth.png` (16-битный grayscale в RGB с исходной alpha), `_normal.png` (RGBA) для выбранного Normal Source и `.ssoul` для продолжения редактирования. Файл `.ssoul` можно открыть через **Открыть**. AI-кэш хранится только в текущем сеансе и после повторного открытия проекта генерируется заново.

Проверка: `.\.venv\Scripts\python.exe -m pytest -q`.
