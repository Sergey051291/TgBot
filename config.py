# config.py  # имя файла/модуля конфигурации

import os  # доступ к переменным окружения и утилитам ОС
import sys  # работа со стандартными потоками и выходом из программы
import json  # разбор/сериализация JSON-строк
from pathlib import Path  # удобная работа с путями/файловой системой
from typing import Dict, Any  # типы словаря и произвольного значения для аннотаций

# ──────────────────────────────────────────────────────────────────────────────  # визуальный разделитель блока
# КАК ЭТО РАБОТАЕТ  # заголовок пояснения
# - VECTOR_BACKEND управляет векторным хранилищем: "chroma" (по умолчанию) или "astra".  # краткое описание выбора БД
# - Если выбран "astra", проверяются ASTRA_DB_TOKEN/ASTRA_DB_ENDPOINT и т.д.  # условия для Astra
# - Если выбран "chroma", проверяется только CHROMA_DIR (каталог создаётся автоматически).  # условия для Chroma
# ──────────────────────────────────────────────────────────────────────────────  # конец разделителя

# --- Параметры форматирования ответов  # комментарий-раздел для настроек форматирования
SUMMARY_NUMBERED_BY_CATEGORY = {  # словарь: включать ли нумерацию в сводке по категориям
    "real_estate": False,  # для категории «недвижимость» нумерация отключена
    "auto": False,  # для категории «авто» нумерация отключена
}  # конец словаря настроек форматирования

def _to_bool(v: Any, default: bool = False) -> bool:  # утилита: привести значение из переменной окружения к bool
    if isinstance(v, bool):  # если уже булево значение
        return v  # возвращаем как есть
    if v is None:  # если значение отсутствует
        return default  # возвращаем значение по умолчанию
    s = str(v).strip().lower()  # нормализуем в строку и приводим к нижнему регистру
    return s in ("1", "true", "yes", "y", "on")  # набор строк, трактуемых как истина

# Обязательные ВСЕГДА (для запуска бота)  # список ключевых переменных окружения
REQUIRED_ALWAYS = [  # перечень обязательных переменных
    "TELEGRAM_BOT_TOKEN",  # токен Telegram-бота
    "TELEGRAM_API_ID",  # числовой API ID (my.telegram.org)
    "TELEGRAM_API_HASH",  # API hash (my.telegram.org)
]  # конец списка обязательных

# Astra — теперь опционально (требуется только при VECTOR_BACKEND=astra)  # пояснение к блоку Astra
ASTRA_VARS = [  # перечень переменных для Astra DB
    "ASTRA_DB_TOKEN",  # токен доступа к Astra DB
    "ASTRA_DB_ENDPOINT",  # HTTP(S) endpoint Astra DB
    # остальные — с дефолтами  # прочие параметры задаются ниже через дефолты
]  # конец списка Astra-переменных

OPTIONAL_VARS_DEFAULTS: Dict[str, Any] = {  # словарь: опциональные переменные и их значения по умолчанию
    # LLM  # блок настроек моделей
    "OLLAMA_MODEL": "llama3",  # имя модели для генерации (Ollama)
    "OLLAMA_EMBEDDINGS": "mxbai-embed-large:latest",  # модель эмбеддингов для индексации/поиска

    # Telegraph  # блок настроек Telegraph
    "TELEGRAPH_SHORT_NAME": "NewsDigestBot",  # short_name автора для Telegraph-страниц

    # Astra (опционально)  # дефолты для Astra DB
    "ASTRA_DB_KEYSPACE": "default_keyspace",  # keyspace по умолчанию
    "ASTRA_DB_COLLECTION": "news_digest",  # коллекция для дайджестов
    "ASTRA_POSTS_COLLECTION": "news_posts",  # коллекция для сырых постов

    # Локальная Chroma  # настройки локального бэкенда
    "VECTOR_BACKEND": "chroma",              # "chroma" | "astra"  # тип векторного бэкенда по умолчанию
    "CHROMA_DIR": "./chroma_storage",  # каталог для хранения данных Chroma

    # Прочее  # разные полезные флаги/идентификаторы
    "USE_FAISS_FALLBACK": "true",  # включить fallback на FAISS при ошибках/отсутствии основной БД
    "DIGEST_CHANNEL_ID": None,  # ID канала/чата, куда публиковать дайджест (если нужен автопостинг)
    "MONITORED_CHANNELS": "{}",  # JSON-словарь каналов для мониторинга: {chat_id_or_username: "category"}
}  # конец словаря опциональных дефолтов


class Config:  # класс-обёртка конфигурации с загрузкой/нормализацией/валидацией
    """Простой загрузчик .env с валидацией для выбранного векторного бэкенда."""  # докстрока класса
    _config: Dict[str, Any] = {}  # внутреннее хранилище конфигурации (после чтения)

    # Доступ к атрибутам класса после load()  # блок аннотаций для автодоступа/IDE
    TELEGRAM_BOT_TOKEN: str  # токен бота (строка)
    TELEGRAM_API_ID: int  # API ID (целое)
    TELEGRAM_API_HASH: str  # API hash (строка)

    VECTOR_BACKEND: str  # выбранный векторный бэкенд ("chroma"/"astra")
    CHROMA_DIR: str  # путь к каталогу Chroma

    ASTRA_DB_TOKEN: str | None  # токен Astra (может быть None при chroma)
    ASTRA_DB_ENDPOINT: str | None  # endpoint Astra (может быть None при chroma)
    ASTRA_DB_KEYSPACE: str  # keyspace Astra
    ASTRA_DB_COLLECTION: str  # коллекция для дайджестов в Astra
    ASTRA_POSTS_COLLECTION: str  # коллекция для постов в Astra

    OLLAMA_MODEL: str  # имя модели Ollama для генерации
    OLLAMA_EMBEDDINGS: str  # модель эмбеддингов
    TELEGRAPH_SHORT_NAME: str  # short_name для Telegraph

    USE_FAISS_FALLBACK: bool  # логический флаг fallback на FAISS
    DIGEST_CHANNEL_ID: int | None  # целевой канал для автодайджестов (или None)
    MONITORED_CHANNELS: Dict[str, str]  # карта: chat_id/username -> категория

    @classmethod  # метод класса (работает на уровне класса, не экземпляра)
    def load(cls) -> "Config":  # главная точка загрузки конфигурации
        cls._load_dotenv()  # подгружаем переменные из .env
        cls._read_all()  # читаем все переменные окружения/дефолты в словарь
        cls._normalize()  # приводим типы/значения к корректному виду
        cls._validate()  # проверяем обязательные параметры/согласованность
        # прокинем в атрибуты класса  # переносим ключи словаря в атрибуты класса
        for k, v in cls._config.items():  # итерируемся по всем парам ключ-значение
            setattr(cls, k, v)  # задаём одноимённый атрибут класса
        return cls  # возвращаем сам класс как удобный объект-конфиг

    @staticmethod  # статический метод (не требует cls/self)
    def _env_path() -> Path:  # вычисление пути к .env
        # .env лежит рядом с проектом (там же, где и этот файл)  # пояснение расположения
        return Path(__file__).parent / ".env"  # путь: директория файла + ".env"

    @classmethod  # метод класса
    def _load_dotenv(cls) -> None:  # загрузка переменных из .env в окружение
        env_path = cls._env_path()  # получаем путь к .env
        if not env_path.exists():  # если файла нет
            print(f"⛔ Файл .env не найден: {env_path}", file=sys.stderr)  # печатаем ошибку в stderr
            sys.exit(1)  # аварийно завершаем выполнение
        from dotenv import load_dotenv  # импорт функции загрузки .env
        load_dotenv(env_path, override=True)  # подставляем значения из .env (перезаписывая имеющиеся)

    @classmethod  # метод класса
    def _read_all(cls) -> None:  # чтение всех релевантных переменных окружения
        cfg: Dict[str, Any] = {}  # временный словарь конфигурации

        # Всегда обязательные  # заголовок блока обязательных переменных
        for var in REQUIRED_ALWAYS:  # пробегаем по списку обязательных ключей
            cfg[var] = os.getenv(var)  # читаем значение из окружения (может быть None)

        # Опциональные с дефолтами  # блок опциональных параметров
        for var, default in OPTIONAL_VARS_DEFAULTS.items():  # идём по парам ключ/дефолт
            cfg[var] = os.getenv(var, default)  # берём из окружения или дефолт

        # Astra опционально (могут быть None)  # параметры Astra читаются без дефолтов
        for var in ASTRA_VARS:  # идём по списку переменных Astra
            cfg[var] = os.getenv(var)  # читаем из окружения (может вернуть None)

        cls._config = cfg  # сохраняем собранную конфигурацию в поле класса

    @classmethod  # метод класса
    def _normalize(cls) -> None:  # нормализация типов/значений в конфиге
        c = cls._config  # локальная ссылка на словарь конфигурации

        # числовые поля  # далее приводим строковые значения к числам
        try:  # безопасная попытка преобразовать API_ID к int
            c["TELEGRAM_API_ID"] = int(c.get("TELEGRAM_API_ID") or 0)  # преобразуем, пустое -> 0
        except Exception:  # если преобразование не удалось
            c["TELEGRAM_API_ID"] = 0  # жёстко выставляем 0 (будет поймано валидатором)

        if c.get("DIGEST_CHANNEL_ID"):  # если ID канала задан (не пусто/None)
            try:  # пробуем привести к int
                c["DIGEST_CHANNEL_ID"] = int(c["DIGEST_CHANNEL_ID"])  # нормализуем тип
            except Exception:  # при ошибке преобразования
                c["DIGEST_CHANNEL_ID"] = None  # сбрасываем в None

        # MONITORED_CHANNELS — JSON  # парсим JSON-строку в словарь
        mc = c.get("MONITORED_CHANNELS")  # исходное значение поля
        if isinstance(mc, str):  # если это строка
            try:  # пробуем распарсить JSON
                c["MONITORED_CHANNELS"] = json.loads(mc) if mc.strip() else {}  # пустая строка -> {}
            except Exception:  # если JSON некорректен
                c["MONITORED_CHANNELS"] = {}  # подставляем пустой словарь
        if not isinstance(c["MONITORED_CHANNELS"], dict):  # если после всех попыток это не словарь
            c["MONITORED_CHANNELS"] = {}  # принудительно делаем пустой словарь

        # булевы флаги  # приведение строковых флагов к bool
        c["USE_FAISS_FALLBACK"] = _to_bool(c.get("USE_FAISS_FALLBACK"), True)  # true/false -> bool, дефолт True

        # backend  # нормализация значения бэкенда
        vb = str(c.get("VECTOR_BACKEND") or "chroma").strip().lower()  # строка, обрезка пробелов, нижний регистр
        if vb not in ("chroma", "astra"):  # проверяем допустимые значения
            vb = "chroma"  # если что-то иное — откатываемся к chroma
        c["VECTOR_BACKEND"] = vb  # сохраняем нормализованное значение

        # Chroma dir — нормализуем путь  # блок подготовки каталога Chroma
        c["CHROMA_DIR"] = str(c.get("CHROMA_DIR") or "./chroma_storage")  # приводим к строке, дефолт если пусто
        if c["VECTOR_BACKEND"] == "chroma":  # только для локального хранилища
            try:  # создаём каталог при необходимости
                Path(c["CHROMA_DIR"]).mkdir(parents=True, exist_ok=True)  # mkdir -p, не падает если уже есть
            except Exception as e:  # если создать не удалось (нет прав/некорректный путь)
                print(f"⚠ Не удалось создать каталог CHROMA_DIR={c['CHROMA_DIR']}: {e}", file=sys.stderr)  # предупреждение

    @classmethod  # метод класса
    def _validate(cls) -> None:  # валидация конфигурации (обязательные поля и совместимость)
        c = cls._config  # ссылка на собранную конфигурацию
        errors = []  # список ошибок валидации

        # Базовые  # проверяем обязательные всегда
        for var in REQUIRED_ALWAYS:  # пробегаем по списку обязательных
            if not c.get(var):  # если переменная отсутствует/пустая
                errors.append(f"{var} не задан")  # фиксируем ошибку

        # Astra — только если выбран этот бэкенд  # условная проверка переменных Astra
        if c.get("VECTOR_BACKEND") == "astra":  # если выбран Astra-бэкенд
            for var in ASTRA_VARS:  # проверяем необходимые поля
                if not c.get(var):  # если какое-то не задано
                    errors.append(f"{var} не задан (требуется при VECTOR_BACKEND=astra)")  # добавляем ошибку

        if errors:  # если есть ошибки конфигурации
            print("⛔ Ошибка конфигурации:", " | ".join(errors), file=sys.stderr)  # печатаем сводку ошибок
            print("\nПроверьте .env. Обязательные переменные:", file=sys.stderr)  # подсказка пользователю
            print("\n".join(REQUIRED_ALWAYS), file=sys.stderr)  # список обязательных переменных
            if c.get("VECTOR_BACKEND") == "astra":  # если бэкенд — Astra
                print("\nТакже требуются для Astra:", file=sys.stderr)  # заголовок для Astra-переменных
                print("\n".join(ASTRA_VARS), file=sys.stderr)  # список требуемых параметров Astra
            sys.exit(1)  # прерываем выполнение с ненулевым кодом

    @classmethod  # метод класса
    def get(cls, key: str, default: Any = None) -> Any:  # безопасное получение значения по ключу
        return cls._config.get(key, default)  # вернуть значение или дефолт

    @classmethod  # метод класса
    def to_dict(cls) -> Dict[str, Any]:  # получить копию всей конфигурации
        return dict(cls._config)  # возвращаем новый словарь (чтобы внешне не мутировать оригинал)


# Автозагрузка  # нижеследующая строка автоматически загружает конфиг при импорте модуля
Config.load()  # вызываем загрузку: читаем .env, нормализуем, валидируем и выставляем атрибуты класса
