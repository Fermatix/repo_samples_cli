# repo-sampler

Утилита для автоматического сбора репрезентативных кодовых сэмплов из git-репозиториев. Агент клонирует репозиторий, самостоятельно изучает его структуру и отбирает ~5 000 строк кода, отражающих типичное качество кодовой базы.

---

## Быстрый старт

### 1. Установка

```bash
uv sync
```

### 2. Настройка API-ключа

Создайте файл `.env` в корне проекта:

```
OPENROUTER_API_KEY=sk-or-...
```

### 3. Список репозиториев

Создайте файл `repos.txt` — по одному URL на строку:

```
https://github.com/owner/repo1
https://gitlab.com/org/group/repo2
# строки начинающиеся с # — игнорируются
```

### 4. Запуск

```bash
uv run repo-sampler run repos.txt
```

Результаты сохраняются в `./output/`.

---

## Структура вывода

```
output/
├── run.log                          # полный лог прогона
├── samples.jsonl                    # метаданные всех репозиториев
└── gitlab.com__owner__repo/
    ├── repo_summary.md              # обзор репозитория
    ├── agent_log.json               # лог действий агента
    └── samples/                     # отобранные файлы (verbatim)
        └── <оригинальный/путь/файла>
```

---

## Дополнительные опции

```bash
# Пропустить уже обработанные репозитории (проверяет samples.jsonl)
uv run repo-sampler run repos.txt --resume

# Только клонирование без LLM (проверка доступности репозиториев)
uv run repo-sampler run repos.txt --dry-run

# Изменить язык (по умолчанию python)
uv run repo-sampler run repos.txt --language typescript

# Полный прогон одного репозитория с выводом в консоль
uv run repo-sampler show-sample https://github.com/owner/repo

# Параллелизм (по умолчанию 10)
uv run repo-sampler run repos.txt --workers 20
```

---

## Настройки (`.env`)

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `OPENROUTER_API_KEY` | — | **Обязательно.** Ключ OpenRouter |
| `AGENT_MODEL` | `deepseek/deepseek-v4-flash` | Модель для агента |
| `TARGET_LOC` | `5000` | Целевой объём сэмпла (строк кода) |
| `CLONE_WORKERS` | `10` | Параллельных клонирований |
| `CLONE_DIR` | `/tmp/repo-sampler/clones` | Директория для клонов |
| `OUTPUT_DIR` | `./output` | Директория вывода |
