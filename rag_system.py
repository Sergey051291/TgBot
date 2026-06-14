# rag_system.py — гибрид: период → сводка из Chroma, иначе обычный RAG  # краткое описание назначения модуля
from __future__ import annotations  # поддержка отложенной оценки аннотаций типов (строки вместо реальных типов)

import re  # регулярные выражения
from typing import List, Dict, Any, Optional  # типовые аннотации для списков/словарей/опциональных значений
from datetime import datetime, timedelta  # работа с датами/временем и интервалами

from config import Config  # доступ к конфигурации (модели, бэкенды и пр.)
from langchain_core.prompts import PromptTemplate  # шаблон промпта для LLM
from langchain_ollama import OllamaLLM  # LLM-клиент через Ollama

from database import ChromaVectorStore, AstraDBVectorStore  # векторные сторы: локальная Chroma или Astra
from digest_generator import DigestGenerator  # генератор HTML-дайджестов (Telegraph)


# --- служебные паттерны для вырезания мусора из индекса/дайджестов  # раздел с фильтрами «служебного мусора»
NOISE_PATTERNS = [  # список регэкспов строк, которые не надо индексировать/показывать
    r"^запускаю\s+сбор",  # служебная фраза бота
    r"^дайджест\s+за\s*\d+\s*дн",  # заголовки дайджестов
    r"^дайджест\s*:",  # ещё вариант заголовка
    r"^в\s*режиме\s*бота\s+чтение\s+истории",  # сообщение об ограничении bot-режима
    r"^в\s*bot-режиме\s+я\s+не\s+могу\s+собрать\s+дайджест",  # аналогичная служебка
    r"^мой\s+индекс\s+пока\s+пуст",  # служебные статусы
    r"^debug:",  # отладочные выводы
    r"^в\s*базе\s+документов:",  # статус размера индекса
    r"^краткий\s+ответ",  # фразы-шаблоны
    r"^в\s*контексте\s+не\s+упоминаются",  # шаблонные отрицательные ответы
    r"^за\s*\d+\s*дн.*ничего\s+не\s+найдено",  # «ничего не найдено за N дней»
    r"^нет\s+информации\s+о\s+том",  # ещё отрицательная формулировка
    r"^привет!\s*я\s*собираю\s*посты",  # приветствие бота
    r"^что\s+по\s+(?:авто|недвижим)",  # примерные подсказки команд
]  # конец списка паттернов
NOISE_RX = re.compile("|".join(NOISE_PATTERNS), flags=re.IGNORECASE)  # объединяем в один регэксп (без учета регистра)

# — регэкс, чтобы отлавливать «нет информации/сведений …»  # комментарий к следующему шаблону
NO_DATA_RX = re.compile(  # шаблон «нет информации / ничего не найдено» для пост-обработки выводов LLM
    r"^\s*(нет\s+(информации|сведений)|ничего\s+не\s+найдено)\b",
    re.IGNORECASE
)  # завершаем компиляцию


class RAGSystem:  # основной класс RAG-системы (индексация, поиск, ответы, дайджесты)
    def __init__(self) -> None:  # конструктор
        backend = (getattr(Config, "VECTOR_BACKEND", "chroma") or "chroma").lower()  # читаем тип бэкенда из конфига
        if backend == "astra":  # если выбран Astra
            self.vector_db = AstraDBVectorStore()  # используем векторный стор Astra
        else:  # иначе — по умолчанию Chroma
            self.vector_db = ChromaVectorStore()  # локальная персистентная Chroma

        self.llm = OllamaLLM(model=Config.OLLAMA_MODEL, temperature=0.2)  # инициализируем LLM с «холодной» температурой
        self.dg = DigestGenerator()  # готовим генератор дайджестов (Telegraph)

        # Усиленный QA-промпт: «нет информации» — только при пустом контексте.  # пояснение логики
        self.qa_prompt = PromptTemplate(  # создаём шаблон промпта для ответов на вопросы
            input_variables=["context", "question"],  # переменные, подставляемые в шаблон
            template=(  # сам текст промпта
                "Контекст для ответа (могут быть выдержки из нескольких заметок):\n{context}\n\n"
                "Вопрос пользователя: {question}\n\n"
                "Правила:\n"
                "1) Отвечай кратко и по делу ТОЛЬКО на основе фактов из контекста.\n"
                "2) Если в контексте есть хоть какие-то релевантные факты — дай краткий ответ по ним.\n"
                "3) Пиши «нет информации» ТОЛЬКО если контекст пуст или полностью нерелевантен вопросу.\n"
                "4) Никаких фраз вроде «вот краткий ответ», «here is…». Просто ответ.\n"
                "5) Отвечай только на РУССКОМ ЯЗЫКЕ.\n"
                "6) Не используй markdown и символы **.\n"
            ),
        )  # конец PromptTemplate

    # ---------- общий фильтр «мусора» ----------  # раздел: эвристики для отбраковки нерелевантных текстов
    @staticmethod  # метод не использует состояние класса
    def _looks_like_garbage(s: str) -> bool:  # проверка: строка похожа на «мусор»?
        s = (s or "").strip()  # нормализация строки
        if (not s) or len(s) < 40 or s.endswith("?") or s.startswith(("/", "!", ".")):  # пусто/коротко/вопрос/команда
            return True  # считаем мусором
        if "источник" in s.lower() and "?" in s[:120].lower():  # короткие «что за источник?» и т.п.
            return True  # тоже мусор
        return False  # иначе — годится

    # ---------------- Индексация ----------------  # раздел: добавление текстов в векторное хранилище
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> bool:  # прокси к vector_db.add_texts
        return self.vector_db.add_texts(texts, metadatas)  # делегируем вставку сторам

    def index_posts(self, posts: List[Dict[str, Any]]) -> int:  # индексация постов с очисткой и метаданными
        texts: List[str] = []  # буфер текстов
        metas: List[Dict[str, Any]] = []  # буфер метаданных
        for p in posts:  # обходим входные записи
            txt = (p.get("text") or "").strip()  # берём и нормализуем текст
            if not txt:  # если пусто
                continue  # пропускаем
            if NOISE_RX.search(txt) or self._looks_like_garbage(txt):  # фильтруем «мусор»
                continue  # пропускаем

            md = {  # собираем метаданные
                "category": p.get("category"),  # категория (auto/real_estate/…)
                "channel": p.get("channel"),  # id канала/чата
                "username": p.get("username"),  # username канала (если есть)
                "date": p.get("date"),  # дата (ISO-строка)
                "message_id": p.get("message_id"),  # id сообщения
                "author_id": p.get("author_id"),  # id автора
                "author_is_bot": bool(p.get("author_is_bot")),  # флаг «автор — бот»
            }  # конец словаря метаданных
            ts = p.get("timestamp")  # пробуем взять готовый timestamp
            if ts is None and p.get("date"):  # если ts нет, но есть date
                try:  # пытаемся распарсить ISO-дату
                    dt = datetime.fromisoformat(str(p["date"]).replace("Z", "+00:00"))  # поддержка Z-суффикса
                    ts = int(dt.timestamp())  # переводим в UNIX-время
                except Exception:  # при ошибке парсинга
                    ts = None  # оставляем None
            if ts is not None:  # если timestamp получился
                md["timestamp"] = int(ts)  # кладём в метаданные как int

            texts.append(txt)  # добавляем текст в буфер
            metas.append(md)  # добавляем метаданные в буфер

        if not texts:  # если ничего не прошло фильтры
            return 0  # возвращаем 0 добавленных
        return len(texts) if self.add_texts(texts, metas) else 0  # вставляем и возвращаем число добавленных или 0

    def count(self) -> int:  # получить размер индекса (если стор поддерживает)
        return getattr(self.vector_db, "count", lambda: 0)()  # безопасно вызываем count() или 0

    # ---------------- Поиск / Ответ ----------------  # раздел: поиск документов и формирование ответа
    def search(self, query: str, k: int = 8, category: Optional[str] = None):  # поиск ближайших документов
        need = k * 2 if category else k  # если задана категория — берём запас кандидатов
        docs = self.vector_db.similarity_search(query, need)  # вызываем поиск у стора
        if category:  # при наличии категории
            cat = str(category).strip().lower()  # нормализуем ключ категории
            filtered: List[Dict[str, Any]] = []  # буфер отфильтрованных
            for d in docs:  # проходим по найденным
                md = d.get("metadata", {}) or {}  # извлекаем метаданные
                if str(md.get("category", "")).strip().lower() == cat:  # оставляем совпадающие категории
                    filtered.append(d)  # добавляем в буфер
            docs = filtered[:k] if filtered else docs[:k]  # возвращаем k лучших по категории, иначе просто k
        return docs  # отдаём список документов

    def _fallback_summary_from_hits(self, docs: List[Dict[str, Any]], max_items: int = 6) -> str:  # запасной мини-ответ из самих документов
        def _one_sentence(s: str, limit: int = 220) -> str:  # взять одно предложение/усечь
            s = (s or "").strip()  # нормализуем
            cut = re.split(r"(?<=[\.\!\?])\s+", s, maxsplit=1)[0]  # первое законченное предложение
            if len(cut) > limit:  # если длинное
                cut = cut[:limit].rstrip() + "…"  # обрезаем и ставим многоточие
            return cut  # возвращаем

        out: List[str] = []  # буфер для пунктов
        for d in docs:  # обходим документы
            txt = d.get("content", "")  # извлекаем текст
            if not txt:  # пропускаем пустые
                continue
            sent = _one_sentence(txt)  # берём одну фразу
            if sent and not NOISE_RX.search(sent):  # если не «мусор»
                out.append(f"— {sent}")  # добавляем строку с тире
            if len(out) >= max_items:  # ограничиваем количество
                break  # выходим из цикла
        return "\n".join(out) if out else "Есть несколько релевантных заметок, см. источники ниже."  # склеиваем или даём заглушку

    def query(self, question: str, k: int = 8, category: Optional[str] = None) -> str:  # основной метод ответа на вопрос
        # Если пользователь спросил «за неделю/за месяц» — делаем тематическую сводку  # объяснение ветвления
        days = self._detect_horizon_days(question)  # пытаемся извлечь временной горизонт из вопроса
        if days:  # если удалось
            ans = self._summarize_for_period(question, days=days, category=category)  # строим сводку за период
            if ans:  # если есть результат
                return ans  # сразу отдаём его

        # Иначе обычный RAG-ответ  # ветка стандартного ответа по найденным документам
        docs = self.search(question, k=k, category=category)  # ищем релевантные документы
        if not docs:  # если ничего не найдено
            cnt = self.count()  # смотрим размер индекса
            if cnt == 0:  # если база пустая
                return "Мой индекс пока пуст. Запусти /digest или /reindex, чтобы я наполнил базу и смог ответить."  # совет по наполнению
            return "В найденных материалах нет ответа. Попробуй переформулировать или уточнить вопрос."  # просьба уточнить

        context = "\n\n".join(f"- {d.get('content', '')}" for d in docs)  # формируем контекст из выдержек
        prompt = self.qa_prompt.format(context=context, question=question)  # подставляем контекст и вопрос в шаблон
        summary = self.llm.invoke(prompt)  # вызываем модель для ответа

        # Страховка: если LLM сказал «нет информации», а документы есть, склеим минимальную сводку из фактов  # пояснение fallback
        if NO_DATA_RX.search(summary or "") and docs:  # проверяем отрицание «нет информации»
            summary = self._fallback_summary_from_hits(docs)  # используем короткий конспект

        return summary.replace("**", "")  # возвращаем ответ, убрав возможные **

    # ---------------- Дайджест (HTML через DigestGenerator) ----------------  # раздел: сбор и публикация дайджеста
    def recent_posts(self, days: int, category: Optional[str], limit: int = 200) -> List[Dict[str, Any]]:  # получить свежие посты
        items: List[Dict[str, Any]] = []  # инициализация списка
        if hasattr(self.vector_db, "recent"):  # если стор поддерживает выборку по времени
            since_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp())  # вычисляем порог времени
            items = self.vector_db.recent(category=category, since_ts=since_ts, limit=limit)  # берём свежие записи
        if not items and hasattr(self.vector_db, "last_by_category"):  # если не удалось — берём просто последние
            items = self.vector_db.last_by_category(category=category, limit=limit) or \
                    self.vector_db.last_by_category(category=None, limit=limit)  # или без категории
        return items  # отдаём список

    def make_digest(self, days: int, category: Optional[str]) -> Optional[str]:  # построить и опубликовать дайджест
        posts = self.recent_posts(days=days, category=category, limit=600)  # забираем до 600 свежих постов
        if not posts:  # если пусто
            return None  # не из чего делать дайджест

        clean: List[Dict[str, Any]] = []  # «очищенные» посты
        seen_keys = set()  # множество ключей для дедупликации
        for p in posts:  # обходим сырые записи
            t = (p.get("content") or "").strip()  # берём текст
            if not t:  # пропускаем пустые
                continue
            if NOISE_RX.search(t) or self._looks_like_garbage(t):  # фильтруем мусор/служебку
                continue

            md = p.get("metadata") or {}  # метаданные поста
            if md.get("author_is_bot"):  # исключаем сообщения, созданные ботами
                continue  # не берём

            mid = md.get("message_id")  # id сообщения
            ch = str(md.get("channel") or "")  # канал
            if mid is not None and ch:  # если есть пара (канал, id)
                key = ("mid", ch, int(mid))  # используем её как ключ
            else:  # иначе — текстовая дедупликация
                norm = re.sub(r"\W+", " ", t.lower()).strip()  # нормализуем текст (только буквы/цифры/пробелы)
                key = ("txt", norm[:140])  # ключ по началу нормализованного текста

            if key in seen_keys:  # если такой уже был
                continue  # пропускаем
            seen_keys.add(key)  # помечаем как встреченный
            clean.append(p)  # добавляем в итоговый список

        if not clean:  # если после очистки ничего не осталось
            return None  # прекращаем

        items: List[Dict[str, Any]] = []  # подготовим компактные записи для генератора
        for p in clean:  # проходим по очищенным
            md = p.get("metadata") or {}  # метаданные
            items.append({  # формируем элемент для DigestGenerator
                "text": p.get("content", ""),  # текст поста
                "date": md.get("date"),  # дата ISO
                "channel": md.get("channel"),  # id канала
                "username": md.get("username"),  # username канала
                "category": md.get("category") or category,  # категория (из md или заданная)
                "message_id": md.get("message_id"),  # id сообщения
            })  # конец элемента

        topic = "Автомобильные новости" if (category or "") == "auto" else "Новости недвижимости"  # заголовок по категории
        period_label = f"{days} дней" if days not in (7, 30) else ("7 дней" if days == 7 else "30 дней")  # подпись периода
        title = f"{topic} за {period_label}"  # конечный заголовок страницы

        url = self.dg.generate_digest(items, title=title)  # генерируем и публикуем дайджест в Telegraph
        return url or None  # возвращаем URL или None, если публикация не удалась

    # ---------------- Хелперы для сводки по периоду (ответ в чате) ----------------  # раздел: парсинг периода и сводка
    PERIOD_RE = re.compile(  # регэксп для извлечения периода («за N дней», «за неделю/месяц»)
        r"за\s+(?:(\d+)\s*(?:дн|дней|дня)|("  # вариант с числом дней
        r"недел(?:ю|и)|нед|месяц|мес(?:яц|)|месяца|месяцев"  # словесные формы недели/месяца
        r"))",
        flags=re.IGNORECASE
    )  # конец компиляции

    def _detect_horizon_days(self, question: str) -> Optional[int]:  # попытка извлечь количество дней из вопроса
        q = (question or "").lower()  # нормализуем строку
        m = self.PERIOD_RE.search(q)  # пытаемся найти период регэкспом
        if not m:  # если не нашли
            if "за неделю" in q or "на неделе" in q:  # эвристика для недели
                return 7  # 7 дней
            if "за месяц" in q or "за прошедший месяц" in q:  # эвристика для месяца
                return 30  # 30 дней
            return None  # периода нет
        if m.group(1):  # если поймано число дней
            try:
                d = int(m.group(1))  # приводим к int
                if 1 <= d <= 365:  # разумный диапазон
                    return d  # отдаём число
            except Exception:  # если ошибка преобразования
                return None  # возвращаем None
        token = m.group(2) or ""  # словесный токен периода
        if token.startswith("нед"):  # «неделя»
            return 7  # 7 дней
        if token.startswith("мес") or token.startswith("месяц"):  # «месяц»
            return 30  # 30 дней
        return None  # иначе не распознали

    def _keywords_from_question(self, question: str) -> List[str]:  # выделяем «смысловые» слова из вопроса
        stop = {"что","как","по","про","за","на","в","и","или","о","об","из","для",  # набор стоп-слов
                "неделю","неделя","нед","месяц","мес","дней","дня","день","последнюю","последний"}  # формы времени
        q = (question or "").lower()  # нормализуем
        q = re.sub(r"\d+\s*(?:дн|дней|дня)", " ", q)  # убираем «N дней»
        words = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9\-]+", q)  # вытаскиваем токены (слова/цифры/дефисы)
        return [w for w in words if w and w not in stop]  # фильтруем стоп-слова

    def _sanitize_period_answer(self, text: str) -> str:  # пост-обработка ответа LLM для периодов
        """
        Убираем префиксы нумерации/маркеры, строки с «нет информации»,
        пустой пункт «Отсутствие событий», «1–2 предложения с сутью: …» и т.п.
        """  # докстрока с описанием целей очистки
        out: List[str] = []  # буфер строк результата
        for raw in (text or "").splitlines():  # построчно обрабатываем ответ
            s = raw.strip()  # убираем пробелы
            # снос нумерации/буллетов  # удаляем префиксы списков и нумерацию
            s = re.sub(r'^\s*(?:[\-\*•]+|\d+[.)–\-:])\s*', '', s)  # маркеры и цифры с пунктуацией
            s = re.sub(r'^\s*\d+\)\s*', '', s)  # ещё вариант нумерации «1)»
            # убираем «нет информации»/«ничего не найдено»  # фильтр негативных заглушек
            if NO_DATA_RX.search(s):
                continue  # пропускаем такие строки
            # удаляем одиночный заголовок «Отсутствие событий»  # бесполезный пункт
            if re.match(r'^\s*отсутствие\s+событий[.:]?\s*$', s, re.I):
                continue  # пропускаем
            # убираем «1–2 предложения с сутью:»  # шаблонную подсказку
            s = re.sub(r'^\s*1[\u2013\-–]\s*2\s+предложения\s+с\s+сутью:\s*', '', s, flags=re.I)
            if s:  # если строка осталась содержательной
                out.append(s)  # добавляем в результат
        text = "\n".join(out)  # склеиваем обратно
        text = text.replace("**", "")  # убираем возможные ** (markdown-артефакты)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()  # схлопываем лишние пустые строки и тримим
        return text  # отдаём очищенный текст

    def _summarize_for_period(self, question: str, days: int, category: Optional[str]) -> Optional[str]:  # построить ответ-«сводку за период»
        # берём последние заметки по категории (или вообще, если пусто)  # пояснение стратегии подбора контекста
        posts = self.recent_posts(days=days, category=category, limit=300) or \
                self.recent_posts(days=days, category=None, limit=300)  # затем без категории
        if not posts:  # если ничего не нашли
            return None  # сводку не построить

        # фильтруем мусор  # убираем служебку/пустые/короткие
        filtered_posts = []  # буфер для чистых постов
        for p in posts:  # обходим всё, что достали
            t = (p.get("content") or "").strip()  # текст
            if not t:  # пропускаем пустые
                continue
            if NOISE_RX.search(t) or self._looks_like_garbage(t):  # фильтр мусора
                continue
            filtered_posts.append(p)  # оставляем валидные

        if not filtered_posts:  # если после очистки пусто
            return None  # выходим

        # небольшой таргетинг по ключевым словам из вопроса  # сузим контекст, если есть явные ключи
        keys = self._keywords_from_question(question)  # извлекаем ключевые слова
        if keys:  # если они есть
            narrowed = [p for p in filtered_posts if any(k in (p.get("content","").lower()) for k in keys)]  # фильтр по словам
            if len(narrowed) >= 10:  # используем только если контекста достаточно
                filtered_posts = narrowed  # сужаем выборку

        # Контекст — краткие выжимки  # подготовим список коротких фрагментов
        snippets = []  # буфер сниппетов
        for p in filtered_posts[:200]:  # ограничение на количество
            t = (p.get("content") or "").strip()  # текст
            if len(t) > 1200:  # не даём слишком длинные куски
                t = t[:1200] + "…"  # обрезаем и ставим многоточие
            snippets.append(f"- {t}")  # добавляем как пункт
        context = "\n".join(snippets)  # собираем контекст

        period_label = f"{days} дней" if days not in (7, 30) else ("неделю" if days == 7 else "месяц")  # подпись периода в тексте
        topic = {"auto": "по авто", "real_estate": "по недвижимости"}.get(str(category or ""), "")  # локальный суффикс темы

        # Жёсткий запрет нумерации и «нет информации» внутри блоков.  # пояснение к промпту
        prompt = (  # собираем промпт для LLM, чтобы получить структурированный список блоков
            f"Ниже заметки {topic} за {period_label}:\n{context}\n\n"
            "Сделай на Русском языке краткую сводку-ответ на вопрос пользователя.\n"
            "Формат вывода — список блоков; у каждого блока РОВНО две строки:\n"
            "1) <b>Короткий заголовок</b>\n"
            "2) Краткий вывод (1–2 предложения) без любых префиксов.\n"
            "Между блоками — пустая строка. Не используй **, маркеры и НУМЕРАЦИЮ "
            "(нельзя 1), 1., 1–, 1—, • и т.п. Разрешён только тег <b>…</b> для заголовков).\n"
            "Запрещено использовать слова «нет информации»/«нет сведений»/«ничего не найдено» "
            "внутри блоков, если в контексте есть хоть один факт. "
            "Эти слова допустимы только при ПОЛНОМ отсутствии релевантных фактов.\n"
            f"Вопрос пользователя: «{question}»"
        )  # конец текста промпта
        answer = self.llm.invoke(prompt)  # вызываем модель для генерации сводки
        return self._sanitize_period_answer(answer)  # чистим ответ и возвращаем готовый текст
