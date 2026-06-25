# NewsDigestBot — Telegram RAG + дайджесты

**Telegram-бот для сбора постов из каналов, индексации в векторной БД, RAG-ответов на вопросы и автоматической генерации дайджестов.**

Бот мониторит выбранные Telegram-чаты и каналы, сохраняет контент локально, отвечает на вопросы пользователей с помощью RAG (Retrieval-Augmented Generation) и публикует структурированные дайджесты через Telegraph.

---

## Функциональность

| Функция | Описание |
|---------|----------|
| **Мониторинг каналов** | Сбор постов из настроенных Telegram-каналов и чатов |
| **RAG-ответы** | Вопросы по проиндексированному контенту через локальную LLM |
| **Дайджесты** | Формирование ссылок на дайджесты за 7 и 30 дней через Telegraph |
| **Два режима авторизации** | По токену бота (aiogram) или по сессии пользователя (Telethon) |
| **Векторные хранилища** | ChromaDB (локально) или Astra DB (облако) |
| **Автодайджесты** | Опциональная отправка по расписанию (cron) |

---

## Стек технологий

| Слой | Технологии |
|------|------------|
| **Фреймворк бота** | aiogram 3.x |
| **Telegram-клиент** | Telethon |
| **LLM / эмбеддинги** | Ollama (llama3, mxbai-embed-large) |
| **Векторная БД** | ChromaDB / Astra DB |
| **RAG** | LangChain + собственный pipeline |
| **Публикация** | Telegraph API |
| **Планировщик** | APScheduler |

---

## Архитектура

```
Telegram-каналы и чаты
        │
        ▼
┌───────────────────┐
│  bot.py           │  обработчики aiogram + watcher Telethon
│  (aiogram/Telethon)│
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐     ┌──────────────────┐
│  rag_system.py    │────▶│  Chroma / Astra  │
│  database.py      │     │  векторное хран. │
└─────────┬─────────┘     └──────────────────┘
          │
          ▼
┌───────────────────┐     ┌──────────────────┐
│  Ollama (локально)│     │  digest_generator│
│  LLM + эмбеддинги │     │  → Telegraph     │
└───────────────────┘     └──────────────────┘
```

---

## Быстрый старт

### Требования

- Python 3.10+
- [Ollama](https://ollama.com/) запущен локально (`ollama serve`)
- Модели: `llama3`, `mxbai-embed-large:latest`
- Telegram Bot Token + API ID/HASH с [my.telegram.org](https://my.telegram.org)

### Установка

```powershell
git clone https://github.com/Sergey051291/TgBot.git
cd TgBot

python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt
```

### Настройка

```powershell
copy .env.example .env
# Заполните .env: токены, API credentials, мониторимые каналы
```

Основные переменные в `.env`:
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`
- `AUTH_MODE=bot` или `user`
- `VECTOR_BACKEND=chroma` или `astra`
- `MONITORED_CHANNELS` — JSON-словарь id каналов → категории
- `OLLAMA_MODEL`, `OLLAMA_EMBEDDINGS`

### Запуск

```powershell
python bot.py
```

---

## Структура проекта

```
TgBot/
├── bot.py                 # Точка входа бота
├── config.py              # Загрузка конфигурации
├── database.py            # Слой хранения данных
├── rag_system.py          # RAG-пipeline (поиск + генерация)
├── digest_generator.py    # Формирование дайджестов Telegraph
├── connect_astra.py       # Проверка подключения к Astra DB
├── script/                # Вспомогательные скрипты
├── .env.example           # Шаблон окружения
├── requirements.txt
```

---

## Вспомогательные скрипты

В папке `script/`:
- `get_channel_ids.py` — получение ID каналов
- `check_all.py` — проверка конфигурации
- `whoami.py` — проверка сессии Telethon
- `quick_fetch.py` — тестовая выборка сообщений

---

## Важно

- **Не коммитьте** `.env`, `*.session` и `chroma_storage/` — там секреты и локальные данные.
- При смене модели эмбеддингов нужна переиндексация: очистите `chroma_storage/` или выполните `/reindex`.
- Для Astra DB установите `VECTOR_BACKEND=astra` в `.env`.

---

## Автор

Агрегация новостей из Telegram с локальной LLM и RAG.

**Стек:** Python · aiogram · Telethon · Ollama · ChromaDB · LangChain
