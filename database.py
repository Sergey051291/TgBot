# database.py — локальные/внешние сторы для постов и векторов  # заголовок/назначение файла

from __future__ import annotations  # отложенная оценка аннотаций типов (строки вместо реальных типов)

# --- отключаем телеметрию Chroma ещё до импорта chromadb ---  # важно установить переменную до импорта
import os  # стандартный модуль ОС: пути, переменные окружения, файловые операции
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # отключаем анонимную телеметрию chroma/posthog по умолчанию

import json  # сериализация/десериализация JSON
import logging  # логирование
import time  # время/таймеры/метки
from functools import lru_cache  # кэширование результатов функций
from pathlib import Path  # объектно-ориентированная работа с путями
from typing import Any, Dict, List, Optional, Tuple  # аннотации типов

import numpy as np  # численные массивы/линейная алгебра
import requests  # HTTP-запросы к Ollama API
from requests.exceptions import RequestException  # базовое исключение requests
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings  # типы для совместимости с chroma

# Логирование  # настройка логгера для модуля
logger = logging.getLogger("database")  # берём именованный логгер "database"
if not logger.handlers:  # чтобы не дублировать обработчики при повторных импортов
    logging.basicConfig(  # конфигурируем базовую схему логирования
        level=logging.INFO,  # уровень логов INFO
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # формат строки лога
    )
# приглушаем шумную телеметрию Chroma/PostHog  # уменьшаем болтливость сторонних модулей
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)  # подавляем телеметрию chroma
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)  # подавляем posthog

# --- Astra (опционально) ---  # попытка импортировать клиент Astra; если не установлен — работаем без него
try:  # блок пробного импорта
    from astrapy import DataAPIClient  # type: ignore  # основной клиент Data API
    from astrapy.exceptions import DataAPIException  # type: ignore  # исключение уровня Data API
    _ASTRA_OK = True  # Astra доступна
except Exception:  # если любой импорт не удался
    DataAPIClient = None  # type: ignore  # заглушка, чтобы не падать при ссылках
    DataAPIException = Exception  # type: ignore  # используем базовое исключение как подмену
    _ASTRA_OK = False  # Astra недоступна

# --- Chroma ---  # подготовка к работе с локальной базой Chroma
try:  # пробуем импортировать chromadb и настройки
    import chromadb  # type: ignore  # основной пакет chroma
    from chromadb.config import Settings as ChromaSettings  # type: ignore  # класс настроек клиента
    from chromadb.api.types import EmbeddingFunction, Documents, Embeddings  # type: ignore  # типы API
    _CHROMA_OK = True  # chroma доступна
except Exception:  # если импорт не удался
    chromadb = None  # type: ignore  # заглушка
    ChromaSettings = None  # type: ignore  # заглушка
    EmbeddingFunction = object  # type: ignore  # подмена типа для аннотаций
    Documents = List[str]  # type: ignore  # подмена типа документов
    Embeddings = List[List[float]]  # type: ignore  # подмена типа эмбеддингов
    _CHROMA_OK = False  # chroma недоступна

from langchain_ollama import OllamaEmbeddings  # обёртка эмбеддингов Ollama для LangChain
from config import Config  # конфигурация проекта (модели, пути, ключи и пр.)


# ---------------------------------------------------------------------  # раздел: утилиты
# Утилиты
# ---------------------------------------------------------------------

def _ensure_dir(path: str) -> None:  # гарантируем существование каталога
    os.makedirs(path, exist_ok=True)  # создаём каталог(и) при отсутствии (аналог mkdir -p)

def _normalize_model_name(name: str) -> str:  # приводим имя модели к базовому виду
    return (name or "").split(":", 1)[0].strip()  # отсекаем тег версии после двоеточия и пробелы


# ---------------------------------------------------------------------  # раздел: эмбеддинги Ollama — проверка/кэш
# Ollama embeddings: проверка и кэш
# ---------------------------------------------------------------------

def verify_ollama_embeddings(model_name: str) -> int:  # проверка доступности модели эмбеддингов в Ollama
    """
    Проверяем доступность Ollama и возвращаем размерность эмбеддинга.
    Бросаем исключение, если что-то не так.
    """  # докстрока функции
    try:  # пробуем опросить локальный Ollama
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=10)  # запрашиваем список моделей от Ollama
        r.raise_for_status()  # поднимаем исключение при не-2xx ответе
        models = r.json().get("models", []) or []  # получаем список моделей из JSON

        want_raw = model_name  # исходное имя из настроек
        want_norm = _normalize_model_name(want_raw)  # нормализованное имя без тега

        api_names = [m.get("name", "") for m in models]  # все имена как есть
        api_norms = {_normalize_model_name(n) for n in api_names}  # нормализованные имена для сравнения

        if want_norm not in api_norms:  # если нужной модели нет среди загруженных
            raise RuntimeError(  # возбуждаем понятную ошибку
                f"Модель '{want_raw}' не найдена в Ollama. "
                f"Доступные: {', '.join(api_names)}. "
                f"Сначала выполните: ollama pull {want_raw}"
            )

        vec = OllamaEmbeddings(model=want_raw).embed_query("dimension_probe")  # создаём эмбеддинг зонда «dimension_probe»
        if not isinstance(vec, list) or not vec:  # проверка, что пришёл непустой список
            raise RuntimeError("Ollama вернул пустой эмбеддинг")  # сигнализируем о проблеме
        logger.info("Ollama OK, эмбеддинг %s, dim=%d", want_raw, len(vec))  # логируем успешную проверку и размерность
        return len(vec)  # возвращаем размерность вектора
    except RequestException as e:  # сетевые/HTTP ошибки requests
        logger.error("Не удалось обратиться к Ollama: %s", e)  # логируем причину
        raise  # пробрасываем дальше
    except Exception as e:  # прочие ошибки (неверная модель и т.п.)
        logger.error("Проблема с моделью эмбеддингов Ollama: %s", e)  # логируем
        raise  # пробрасываем далее

@lru_cache(maxsize=2048)  # кэшируем результаты эмбеддинга для одинаковых (text, model_name)
def _embed_cached(text: str, model_name: str) -> List[float]:  # приватная обёртка над OllamaEmbeddings
    emb = OllamaEmbeddings(model=model_name)  # создаём объект эмбеддингов для указанной модели
    return emb.embed_query(text)  # считаем вектор для запроса и возвращаем

def embed_text(text: str) -> List[float]:  # публичная функция эмбеддинга текста с моделью из конфига
    return _embed_cached(text, Config.OLLAMA_EMBEDDINGS)  # берём модель из Config и используем LRU-кэш


# ---------------------------------------------------------------------  # раздел: 1) локальный стор сырых постов
# 1) Локальный стор «сырых» постов (JSONL)
# ---------------------------------------------------------------------

class LocalPostsStore:  # простой JSONL-стор для хранения исходных сообщений
    """JSONL-стор для «сырых» постов с дедупом по (channel, message_id)."""  # докстрока класса
    def __init__(self, path: str = "posts_store.jsonl"):  # инициализация с путём к файлу JSONL
        self.path = Path(path)  # Path-объект для файла
        self.path.parent.mkdir(parents=True, exist_ok=True)  # создаём каталог(и), если их нет

    def upsert_posts(self, items: List[Dict[str, Any]]) -> int:  # вставка новых постов с дедупликацией
        seen = set()  # множество уже встреченных ключей (channel, message_id)
        out: List[Dict[str, Any]] = []  # результирующий список документов для записи
        if self.path.exists():  # если файл уже существует
            with self.path.open("r", encoding="utf-8") as f:  # открываем на чтение
                for line in f:  # читаем по строкам JSONL
                    try:
                        doc = json.loads(line)  # парсим JSON-объект
                    except Exception:
                        continue  # пропускаем битые строки
                    key = (str(doc.get("channel") or ""), int(doc.get("message_id") or 0))  # составной ключ
                    seen.add(key)  # запоминаем существующую запись
                    out.append(doc)  # добавляем в текущий буфер

        inserted = 0  # счётчик добавленных новых элементов
        for it in items:  # обрабатываем входящие элементы
            key = (str(it.get("channel") or ""), int(it.get("message_id") or 0))  # формируем ключ
            if not key[0] or not key[1] or key in seen:  # если пустые поля или уже есть
                continue  # пропускаем
            d = it.get("date")  # вытаскиваем дату
            if hasattr(d, "isoformat"):  # если это объект datetime
                it = {**it, "date": d.isoformat()}  # записываем дату как ISO-строку
            out.append(it)  # добавляем в буфер
            seen.add(key)  # помечаем как увиденный
            inserted += 1  # увеличиваем счётчик

        if inserted:  # если есть что записывать
            with self.path.open("w", encoding="utf-8") as f:  # открываем файл на перезапись
                for doc in out:  # пишем все документы по одному в JSONL
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")  # JSON-строка + перенос
        return inserted  # возвращаем количество добавленных записей


# ---------------------------------------------------------------------  # раздел: 2) простой локальный векторный стор
# 2) Простой локальный векторный стор (NumPy)
# ---------------------------------------------------------------------

class LocalVectorStore:  # минималистичное векторное хранилище на numpy + jsonl метаданных
    """
    Простейшее векторное хранилище:
      - embeddings.npy  (shape: [N, D])
      - meta.jsonl      (по строке на документ; хранит {content, metadata, channel, message_id})
    Поиск: косинусная близость.
    """  # докстрока с описанием форматов

    def __init__(self, dir_path: Optional[str] = None):  # инициализация стора
        self.dir_path = dir_path or getattr(Config, "FAISS_DIR", "./faiss_index")  # корневой каталог индекса
        _ensure_dir(self.dir_path)  # гарантируем наличие каталога
        self.emb_path = os.path.join(self.dir_path, "embeddings.npy")  # путь к файлу матрицы эмбеддингов
        self.meta_path = os.path.join(self.dir_path, "meta.jsonl")  # путь к файлу метаданных

        self.dim = verify_ollama_embeddings(Config.OLLAMA_EMBEDDINGS)  # проверяем Ollama и узнаём размерность векторов

        self._emb: Optional[np.ndarray] = None  # ссылка на загруженный mmap-массив эмбеддингов
        self._load()  # пытаемся подгрузить существующие данные

    def _load(self) -> None:  # внутренняя загрузка матрицы эмбеддингов из файла
        if os.path.exists(self.emb_path):  # если файл присутствует
            try:
                self._emb = np.load(self.emb_path, mmap_mode="r+")  # memory-map для экономии памяти
                logger.info("LocalVectorStore: загружено %s (shape=%s)", self.emb_path, self._emb.shape)  # лог
            except Exception as e:  # на случай несовместимости/битого файла
                logger.warning("Не удалось загрузить embeddings.npy: %s", e)  # предупреждение
                self._emb = None  # сбрасываем

    def _append_meta(self, docs: List[Dict[str, Any]]) -> None:  # дозаписываем метаданные в JSONL
        with open(self.meta_path, "a", encoding="utf-8") as f:  # открываем на добавление
            for d in docs:  # каждую запись по строке
                f.write(json.dumps(d, ensure_ascii=False) + "\n")  # сериализация + перенос

    def _iter_meta(self):  # генератор по объектам метаданных
        if not os.path.exists(self.meta_path):  # если файла нет
            return  # ничего не итерируем
        with open(self.meta_path, "r", encoding="utf-8") as f:  # открываем на чтение
            for line in f:  # построчно
                line = line.strip()  # убираем пробелы/переводы
                if not line:  # пустые строки пропускаем
                    continue
                try:
                    yield json.loads(line)  # отдаём распарсенный объект
                except Exception:
                    continue  # пропускаем битые JSON-строки

    def _existing_keys(self) -> set[Tuple[str, int]]:  # множество ключей уже в индексе
        keys: set[Tuple[str, int]] = set()  # инициализация множества
        for doc in self._iter_meta() or []:  # проходим по всем метаданным
            ch = str(doc.get("channel") or "")  # канал
            mid = doc.get("message_id")  # ID сообщения
            if ch and mid is not None:  # только валидные ключи
                keys.add((ch, mid))  # добавляем в множество
        return keys  # возвращаем набор

    # --- API ---  # публичные методы стора

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> bool:  # добавление документов
        if len(texts) != len(metadatas):  # проверяем согласованность списков
            logger.error("LocalVectorStore.add_texts: len(texts)!=len(metadatas)")  # лог ошибки
            return False  # несоответствие размеров — прерываем

        to_embed: List[str] = []  # буфер текстов для эмбеддинга
        to_meta: List[Dict[str, Any]] = []  # буфер метаданных для записи

        existing = self._existing_keys()  # набор уже известных ключей

        for t, md in zip(texts, metadatas):  # параллельный обход
            ch = str((md or {}).get("channel") or "")  # канал из метаданных
            mid = (md or {}).get("message_id")  # ID сообщения
            if ch and mid is not None and (ch, mid) in existing:  # если такой ключ уже есть
                continue  # дедуп: пропускаем
            to_embed.append(t)  # добавляем текст для эмбеддинга
            to_meta.append({  # формируем запись метаданных
                "content": t,  # сам текст
                "metadata": md,  # исходные метаданные
                "channel": ch,  # канал в явном виде
                "message_id": mid,  # ID сообщения
            })

        if not to_embed:  # если нет новых документов
            logger.info("LocalVectorStore: нет новых документов")  # информируем
            return True  # считаем операцию успешной (ничего не делать)

        vecs = np.array([embed_text(x) for x in to_embed], dtype=np.float32)  # эмбеддим все новые тексты в массив
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:  # проверяем форму матрицы [N, D]
            logger.error("Размерность эмбеддингов не совпала: got %s, want %d", vecs.shape, self.dim)  # лог ошибки
            return False  # не записываем

        if self._emb is None:  # если ещё не было файла эмбеддингов
            np.save(self.emb_path, vecs)  # сохраняем новый массив
            self._emb = np.load(self.emb_path, mmap_mode="r+")  # пересоздаём mmap-ссылку
        else:  # если уже есть существующие эмбеддинги
            new = np.concatenate([np.asarray(self._emb), vecs], axis=0)  # конкатенируем по строкам
            np.save(self.emb_path, new.astype(np.float32))  # перезаписываем файл
            self._emb = np.load(self.emb_path, mmap_mode="r+")  # обновляем mmap-ссылку

        self._append_meta(to_meta)  # дозаписываем метаданные к индексам
        logger.info("LocalVectorStore: добавлено %d векторов (итого %d)", len(to_embed), int(self._emb.shape[0]))  # лог
        return True  # успешное добавление

    def similarity_search(  # поиск ближайших документов по косинусной близости
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        if self._emb is None or self._emb.size == 0:  # если индекс пуст
            return []  # нечего возвращать

        q = np.asarray(embed_text(query), dtype=np.float32)  # эмбеддинг запроса
        qn = q / (np.linalg.norm(q) + 1e-9)  # нормализация вектора запроса
        X = np.asarray(self._emb)  # матрица эмбеддингов документов
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)  # нормализация каждой строки
        sims = Xn @ qn  # shape [N]  # косинусная близость как скалярное произведение

        idx = np.argsort(-sims)[: max(1, k)]  # индексы k наибольших схождений
        results: List[Dict[str, Any]] = []  # список результатов
        meta_list = list(self._iter_meta() or [])  # соответствующие метаданные (в том же порядке)
        for i in idx:  # для каждого кандидата
            score = float(sims[int(i)])  # оценка близости
            if score_threshold is not None and score < score_threshold:  # фильтр по порогу
                continue  # отбрасываем слабые
            doc = meta_list[int(i)] if int(i) < len(meta_list) else {}  # безопасно берём мета
            results.append({  # формируем элемент результата
                "content": doc.get("content", ""),  # исходный текст
                "metadata": doc.get("metadata", {}),  # метаданные
                "score": score,  # близость
            })
        return results  # возвращаем список найденных

    def clear_cache(self) -> None:  # очистка LRU-кэша эмбеддингов
        _embed_cached.cache_clear()  # сброс кэша
        logger.info("LocalVectorStore: очищен кэш эмбеддингов")  # логируем


# ---------------------------------------------------------------------  # раздел: 3) ChromaVectorStore — персистентная Chroma
# 3) ChromaVectorStore — локальная персистентная Chroma
# ---------------------------------------------------------------------

# --- ChromaVectorStore --------------------------------------------------------  # заголовок класса
class ChromaVectorStore:  # обёртка над chromadb.PersistentClient
    """
    Персистентная Chroma (SQLite), коллекция из .env:
      CHROMA_DIR=./chroma_storage
      CHROMA_COLLECTION=news_digest
    """  # докстрока с описанием переменных окружения
    def __init__(self):  # конструктор
        if not _CHROMA_OK:  # если chroma не импортировалась
            raise RuntimeError("chromadb не установлен. Установи: pip install 'chromadb>=0.5,<0.6'")  # подсказка

        _ensure_dir(Config.CHROMA_DIR or "./chroma_storage")  # гарантируем наличие каталога для хранения

        # Встроенная обёртка эмбеддингов под новый API Chroma  # адаптер, чтобы Chroma могла вызывать эмбеддинги
        class _OllamaEmbeddingFunction(EmbeddingFunction):  # класс-обёртка, реализующий __call__
            def __call__(self, input: Documents) -> Embeddings:  # реализация интерфейса
                return [embed_text(t) for t in input]  # эмбеддим каждый документ списком

        self.client = chromadb.PersistentClient(  # создаём персистентного клиента Chroma
            path=Config.CHROMA_DIR,  # путь к каталогу БД
            settings=ChromaSettings(anonymized_telemetry=False)  # отключаем телеметрию
        )
        self.collection = self.client.get_or_create_collection(  # берём/создаём коллекцию
            name=getattr(Config, "CHROMA_COLLECTION", "news_digest"),  # имя коллекции
            embedding_function=_OllamaEmbeddingFunction(),  # функция эмбеддинга
            metadata={"hnsw:space": (getattr(Config, "CHROMA_DISTANCE", "cosine") or "cosine")}  # метрика
        )
        # Проверим размерность эмбеддингов (заодно и доступ к Ollama)  # валидация модели
        self.dim = verify_ollama_embeddings(Config.OLLAMA_EMBEDDINGS)  # узнаём D

    def count(self) -> int:  # количество документов в коллекции
        try:
            return int(self.collection.count())  # возвращаем число из chroma
        except Exception:
            return 0  # в случае ошибки — 0

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> bool:  # пакетная вставка документов
        if not texts:  # если пустой список
            return True  # ничего вставлять — тоже успех
        ids: List[str] = []  # будущие ID документов
        safe_meta: List[Dict[str, Any]] = []  # очищенные метаданные
        for i, md in enumerate(metadatas or []):  # по каждому документу
            ch = str((md or {}).get("channel") or "")  # канал
            mid = (md or {}).get("message_id")  # ID сообщение
            ids.append(f"{ch}:{mid}:{time.time_ns()}:{i}")  # формируем уникальный id
            safe_meta.append(md or {})  # метаданные как есть либо пустые
        try:
            self.collection.add(documents=texts, metadatas=safe_meta, ids=ids)  # вставляем в chroma
            logger.info("Chroma: добавлено %d документов", len(texts))  # лог о количестве
            return True  # успех
        except Exception as e:  # при ошибке вставки
            logger.error("Chroma add_texts: %s", e)  # логируем
            return False  # неудача

    def similarity_search(self, query: str, k: int = 5, score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:  # поиск по chroma
        try:
            res = self.collection.query(  # выполняем запрос
                query_texts=[query],  # один текст запроса
                n_results=k,  # сколько результатов вернуть
                include=["documents", "metadatas", "distances"],  # просим документы/метаданные/дистанции
            )
            docs = res.get("documents", [[]])[0]  # извлекаем список документов (0-й батч)
            mets = res.get("metadatas", [[]])[0]  # метаданные соответствующих документов
            dists = res.get("distances", [[]])[0]  # расстояния (меньше — ближе)
            out: List[Dict[str, Any]] = []  # результаты
            for doc, md, dist in zip(docs, mets, dists):  # итерируем по возвращённым данным
                score = float(1.0 - dist)  # чем больше score, тем ближе (конвертируем из distance)
                if score_threshold is not None and score < score_threshold:  # фильтр по порогу
                    continue  # пропускаем слабые соответствия
                out.append({"content": doc, "metadata": md or {}, "score": score})  # упаковываем результат
            return out  # отдаём результаты
        except Exception as e:  # если запрос к chroma упал
            logger.error("Chroma similarity_search: %s", e)  # лог ошибки
            return []  # пустой ответ

    # ↓↓↓ нужно для /digest и отладки  # доп. выборки по времени/категории
    def recent(self, category: Optional[str] = None, since_ts: Optional[int] = None, limit: int = 200) -> List[Dict[str, Any]]:  # свежие документы по фильтрам
        """Выборка по метаданным: category и timestamp>$gt. Chroma требует один оператор: используем $and."""  # докстрока
        # соберём клаузы  # подготавливаем условия where
        clauses: List[Dict[str, Any]] = []  # список условий
        if category:  # если задана категория
            clauses.append({"category": category})  # добавляем условие точного совпадения
        if since_ts is not None:  # если задан порог времени
            clauses.append({"timestamp": {"$gt": int(since_ts)}})  # добавляем условие по timestamp

        # склеим в where  # комбинируем условия в один словарь
        if len(clauses) == 0:  # без условий
            where = {}  # пустой фильтр
        elif len(clauses) == 1:  # одно условие
            where = clauses[0]  # берём его напрямую
        else:  # несколько условий
            where = {"$and": clauses}  # объединяем через $and (как требует chroma)

        try:
            res = self.collection.get(where=where, include=["documents", "metadatas"], limit=limit)  # запрос к коллекции
            docs = res.get("documents") or []  # документы
            mets = res.get("metadatas") or []  # метаданные
            if docs and isinstance(docs[0], list):  # Chroma 0.4/0.5 иногда даёт [[...]]  # выравниваем форму
                docs = docs[0]  # берём вложенный список
            if mets and isinstance(mets[0], list):  # аналогично для метаданных
                mets = mets[0]  # разворачиваем
            out = [{"content": d, "metadata": m or {}} for d, m in zip(docs, mets)]  # объединяем пары
            # сортировка по timestamp, если есть  # хотим последние сверху
            out.sort(key=lambda x: int((x.get("metadata") or {}).get("timestamp", 0)), reverse=True)  # сортировка
            return out  # возвращаем список
        except Exception as e:  # если при выборке произошла ошибка
            logger.error("Chroma recent(): %s", e)  # логируем
            return []  # отдаём пустой список

        # конец recent  # (метод завершён)

    def last_by_category(self, category: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:  # последние по категории (без порога времени)
        """Просто последние по категории (если timestamp отсутствует)."""  # докстрока
        where = {"category": category} if category else {}  # формируем фильтр по категории либо пустой
        try:
            res = self.collection.get(where=where, include=["documents", "metadatas"], limit=limit)  # запрос
            docs = res.get("documents") or []  # документы
            mets = res.get("metadatas") or []  # метаданные
            if docs and isinstance(docs[0], list):  # выравниваем вложенность
                docs = docs[0]  # разворачиваем
            if mets and isinstance(mets[0], list):  # выравниваем вложенность
                mets = mets[0]  # разворачиваем
            out = [{"content": d, "metadata": m or {}} for d, m in zip(docs, mets)]  # объединяем пары
            out.sort(key=lambda x: int((x.get("metadata") or {}).get("timestamp", 0)), reverse=True)  # сортируем по ts
            return out[:limit]  # возвращаем не больше limit
        except Exception as e:  # при ошибке
            logger.error("Chroma last_by_category(): %s", e)  # логируем
            return []  # отдаём пусто



# ---------------------------------------------------------------------  # раздел: 4) AstraDB — на будущее
# 4) AstraDB — на будущее (не используется, если VECTOR_BACKEND != 'astra')
# ---------------------------------------------------------------------

class AstraDBVectorStore:  # векторное хранилище на Astra DB c локальными эмбеддингами
    """
    Векторное хранилище поверх Astra DB (Data API) c локальными эмбеддингами (Ollama).
    С дедупликацией по (channel, message_id).
    """  # докстрока
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):  # конструктор с ретраями
        if not _ASTRA_OK:  # если библиотека astrapy недоступна
            raise RuntimeError("Astra-библиотека не установлена")  # сообщаем пользователю
        self.max_retries = max_retries  # максимум попыток подключения
        self.retry_delay = retry_delay  # задержка между попытками (сек.)

        # проверим Ollama и узнаем размерность  # валидация модели эмбеддингов
        self.dim = verify_ollama_embeddings(Config.OLLAMA_EMBEDDINGS)  # размерность векторов

        self.client, self.db, self.collection = self._initialize_astra_connection()  # устанавливаем подключение

    def _initialize_astra_connection(self):  # внутренний метод подключения к Astra
        last_exc = None  # последняя ошибка для отчёта
        for attempt in range(1, self.max_retries + 1):  # несколько попыток
            try:
                logger.info("Подключение к AstraDB (%d/%d)", attempt, self.max_retries)  # логируем попытку
                client = DataAPIClient(Config.ASTRA_DB_TOKEN)  # type: ignore  # создаём клиент Data API
                db = client.get_database_by_api_endpoint(Config.ASTRA_DB_ENDPOINT)  # type: ignore  # выбираем базу

                name = Config.ASTRA_DB_COLLECTION  # имя коллекции для векторов
                names = db.list_collection_names()  # список коллекций
                if name in names:  # если уже есть
                    collection = db.get_collection(name)  # берём существующую
                    logger.info("Astra: используется коллекция '%s'", name)  # лог
                else:  # если нет — создаём
                    collection = db.create_collection(  # создаём новую коллекцию
                        collection_name=name,  # имя
                        dimension=self.dim,  # размерность векторов
                        metric="cosine",  # метрика косинусная
                        options={"indexing": {"deny": ["embedding"]}},  # не индексируем поле embedding
                    )
                    logger.info("Astra: создана коллекция '%s' (dim=%d, cosine)", name, self.dim)  # лог
                return client, db, collection  # возвращаем три объекта
            except Exception as e:  # при ошибке подключения/создания
                last_exc = e  # сохраняем последнюю ошибку
                if attempt < self.max_retries:  # если будут ещё попытки
                    logger.warning("Astra ошибка: %s. Повтор через %.1f c…", e, self.retry_delay)  # предупреждаем
                    time.sleep(self.retry_delay)  # ждём перед новой попыткой
        raise DataAPIException(f"Окончательная ошибка подключения к AstraDB: {last_exc}")  # type: ignore  # сдаёмся

    @lru_cache(maxsize=1024)  # кэшируем эмбеддинги для ускорения
    def _get_embedding(self, text: str) -> List[float]:  # обёртка над embed_text
        return embed_text(text)  # возвращаем вектор

    def _exists(self, channel: str, message_id: Optional[int]) -> bool:  # проверяем наличие записи по ключу
        if not channel or message_id is None:  # если ключ некорректный
            return False  # считаем, что нет
        try:
            doc = self.collection.find_one({"channel": channel, "message_id": message_id})  # ищем документ
            return bool(doc)  # True если найден
        except Exception:
            return False  # при ошибке считаем, что не существует

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> bool:  # вставка пакета документов
        if len(texts) != len(metadatas):  # проверяем соответствие размеров
            logger.error("Количество текстов и метаданных не совпадает")  # лог ошибки
            return False  # прерываем
        try:
            to_insert = []  # буфер документов к вставке
            ts = int(time.time())  # текущая метка времени (секунды)
            for t, md in zip(texts, metadatas):  # перебираем пары текст/метаданные
                channel = (md or {}).get("channel")  # канал
                message_id = (md or {}).get("message_id")  # ID сообщения
                if channel and message_id is not None and self._exists(channel, message_id):  # дедупликация
                    continue  # уже есть — пропускаем
                vec = self._get_embedding(t)  # считаем эмбеддинг
                doc = {  # формируем документ для Astra
                    "content": t,  # текст
                    "embedding": vec,  # вектор
                    "metadata": md,  # метаданные
                    "channel": channel,  # канал
                    "message_id": message_id,  # ID
                    "timestamp": ts,  # время индексации
                }
                to_insert.append(doc)  # добавляем в буфер

            if not to_insert:  # если нечего вставлять
                logger.info("Astra: нет новых документов для вставки")  # информируем
                return True  # считаем успехом

            _ = self.collection.insert_many(to_insert)  # массовая вставка в коллекцию
            logger.info("Astra: добавлено %d документов", len(to_insert))  # лог
            return True  # успех

        except Exception as e:  # ошибка при вставке
            logger.error("Astra add_texts: %s", e)  # логируем
            return False  # неудача

    def similarity_search(self, query: str, k: int = 5,
                          score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:  # поиск по схожести в Astra
        try:
            qv = self._get_embedding(query)  # эмбеддинг запроса
            options: Dict[str, Any] = {  # параметры поиска
                "limit": k,  # максимум результатов
                "includeSimilarity": True,  # включить схожесть ($similarity) в ответ
                "fields": ["content", "metadata", "channel", "message_id"],  # какие поля вернуть
            }
            if score_threshold is not None:  # если задан порог
                options["similarityThreshold"] = score_threshold  # добавляем фильтр

            result = self.collection.vector_find(vector=qv, **options)  # выполняем vector search
            docs = result.get("data", {}).get("documents", [])  # извлекаем документы
            out: List[Dict[str, Any]] = []  # список результатов
            for d in docs:  # формируем каждый результат
                out.append({
                    "content": d.get("content", ""),  # текст
                    "metadata": d.get("metadata", {}),  # метаданные
                    "score": d.get("$similarity", None),  # оценка схожести (если присутствует)
                })
            return out  # отдаём результаты
        except Exception as e:  # при ошибке поиска
            logger.error("Astra search: %s", e)  # логируем
            return []  # пустой ответ

    def clear_cache(self) -> None:  # очистка кэша эмбеддингов
        _embed_cached.cache_clear()  # сбрасываем LRU-кэш
        logger.info("Astra: кэш эмбеддингов очищен")  # логируем


class AstraPostsStore:  # стор «сырых» постов в Astra DB (для совместимости)
    """Сырые посты в Astra DB. Оставлено для совместимости."""  # докстрока
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):  # конструктор с ретраями
        if not _ASTRA_OK:  # если astrapy не установлен
            raise RuntimeError("Astra-библиотека не установлена")  # сообщаем
        self.max_retries = max_retries  # максимум попыток подключения
        self.retry_delay = retry_delay  # задержка между попытками
        self.client, self.db, self.collection = self._init()  # инициализируем подключение/коллекцию

    def _init(self):  # внутренний метод подготовки коллекции постов
        last_exc = None  # последняя ошибка
        for attempt in range(1, self.max_retries + 1):  # несколько попыток
            try:
                client = DataAPIClient(Config.ASTRA_DB_TOKEN)  # type: ignore  # клиент Data API
                db = client.get_database_by_api_endpoint(Config.ASTRA_DB_ENDPOINT)  # type: ignore  # база
                name = getattr(Config, "ASTRA_POSTS_COLLECTION", "news_posts")  # имя коллекции постов
                names = db.list_collection_names()  # список коллекций
                if name in names:  # если существует
                    coll = db.get_collection(name)  # берём её
                    logger.info("Astra Posts: используется коллекция '%s'", name)  # лог
                else:  # иначе
                    coll = db.create_collection(collection_name=name)  # создаём пустую (без векторов)
                    logger.info("Astra Posts: создана коллекция '%s'", name)  # лог
                return client, db, coll  # возвращаем клиент/БД/коллекцию
            except Exception as e:  # при ошибке
                last_exc = e  # сохраняем
                if attempt < self.max_retries:  # если будут ещё попытки
                    logger.warning("Astra Posts: ошибка %s. Повтор…", e)  # предупреждаем
                    time.sleep(self.retry_delay)  # ждём
        raise DataAPIException(f"Не удалось инициализировать PostsStore: {last_exc}")  # type: ignore  # сдаёмся

    def upsert_posts(self, items: List[Dict[str, Any]]) -> int:  # вставка новых постов с дедупликацией
        inserted = 0  # счётчик вставленных
        for it in items:  # перебираем входные элементы
            try:
                channel = str(it.get("channel") or "")  # канал как строка
                mid = it.get("message_id")  # ID сообщения
                if not channel or mid is None:  # валидируем ключ
                    continue  # пропускаем
                exists = self.collection.find_one({"channel": channel, "message_id": mid})  # проверяем наличие
                if exists:  # если уже есть
                    continue  # пропускаем
                d = it.get("date")  # дата из элемента
                if hasattr(d, "isoformat"):  # если datetime
                    d = d.isoformat()  # переводим в ISO
                doc = {  # собираем документ для вставки
                    "channel": channel,  # канал
                    "message_id": mid,  # ID сообщения
                    "date": d,  # дата (ISO/строка/None)
                    "category": it.get("category"),  # категория
                    "text": it.get("text", ""),  # текст поста
                    "timestamp": int(time.time()),  # время записи
                }
                self.collection.insert_one(doc)  # вставляем документ
                inserted += 1  # увеличиваем счётчик
            except Exception as e:  # при сбое вставки конкретного элемента
                logger.warning("Astra Posts: upsert не удался: %s", e)  # логируем предупреждение
        if inserted:  # если что-то вставили
            logger.info("Astra Posts: добавлено %d документов", inserted)  # логируем итог
        return inserted  # возвращаем количество вставленных
