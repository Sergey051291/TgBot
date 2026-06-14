# bot.py — бот делает дайджест из локального индекса (даже в bot-режиме)  # заголовок/описание файла

from __future__ import annotations  # включает отложенную оценку аннотаций типов (строки вместо реальных типов)

import asyncio  # работа с асинхронными задачами/циклами событий
import logging  # логирование событий
import re  # регулярные выражения
import os  # доступ к переменным окружения и файловой системе
from datetime import datetime, timedelta, timezone  # даты/время и таймзоны
from typing import Dict, List, Optional, Union  # типовые аннотации

from aiogram import Bot, Dispatcher, F  # основные классы aiogram: бот, диспетчер, фильтры
from aiogram.filters import Command  # фильтр для команд (/start, /status, …)
from aiogram.types import Message  # тип сообщения Telegram
from aiogram.client.default import DefaultBotProperties  # дефолтные свойства бота
from aiogram.enums import ParseMode  # режим парсинга (HTML/Markdown)

from telethon import TelegramClient, events  # Telethon-клиент и события
from telethon.tl.types import PeerChannel  # тип сущности «канал/чат» в Telethon

from config import Config  # конфиг с токенами/ID/настройками
from rag_system import RAGSystem  # собственная система RAG (поиск+суммаризация)
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # планировщик задач (cron) на asyncio

logger = logging.getLogger(__name__)  # создаём логгер для текущего модуля
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")  # базовая настройка логов

bot = Bot(  # инициализация объекта бота aiogram
    token=Config.TELEGRAM_BOT_TOKEN,  # токен бота из конфига
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),  # по умолчанию формат ответов — HTML
)
dp = Dispatcher()  # диспетчер обработчиков сообщений

AUTH_MODE = (  # режим авторизации Telethon: 'user' (по номеру) или 'bot' (по токену)
    getattr(Config, "AUTH_MODE", None)  # сначала берём из Config.AUTH_MODE, если есть
    or os.getenv("AUTH_MODE")  # иначе из переменной окружения AUTH_MODE
    or os.getenv("TELETHON_AUTH_MODE")  # либо TELETHON_AUTH_MODE
    or "bot"  # по умолчанию — режим бота
).lower()  # нормируем в нижний регистр

SESSION_NAME = "parser_user_session" if AUTH_MODE == "user" else "bot_session"  # имя файла сессии Telethon
tg_client = TelegramClient(SESSION_NAME, Config.TELEGRAM_API_ID, Config.TELEGRAM_API_HASH)  # клиент Telethon с API ID/HASH

rag = RAGSystem()  # создаём экземпляр RAG-системы (векторная БД и логика)
_watchers_started = False  # флаг, чтобы не запускать watcher несколько раз

# ──────────────────────────────────────────────────────────────────────────────
# Утилиты/категории
# ──────────────────────────────────────────────────────────────────────────────
def mapping_category(raw: Union[int, str]) -> Optional[str]:  # сопоставление chat_id/юзернейма с категорией из конфига
    m: Dict[str, str] = getattr(Config, "MONITORED_CHANNELS", {}) or {}  # карта отслеживаемых каналов -> категория
    s = str(raw)  # нормализуем идентификатор в строку
    if s in m:  # если ключ есть напрямую
        return m[s]  # возвращаем категорию
    if s.lstrip("-").isdigit():  # если похоже на числовой id (возможно без -100 префикса)
        if not s.startswith("-100"):  # не имеет tg-префикса supergroup/channel
            cat = m.get(f"-100{s.lstrip('-')}")  # пытаемся добавить префикс -100
            if cat:
                return cat  # нашли категорию по варианту с -100
    return None  # иначе категория не определена


def guess_category(title: str) -> Optional[str]:  # грубая эвристика категории по названию чата
    t = (title or "").lower()  # нижний регистр и защита от None
    if "авто" in t or "auto" in t or "автом" in t:  # ключи, указывающие на авто
        return "auto"  # категория авто
    if "недвиж" in t or "real" in t:  # ключи, связанные с недвижимостью
        return "real_estate"  # категория недвижимость
    return None  # не распознали


def resolve_category(chat_title: str, chat_id: int | str) -> Optional[str]:  # финальное определение категории
    return mapping_category(chat_id) or guess_category(chat_title)  # сначала точное сопоставление, потом эвристика


async def ensure_telethon_ready() -> None:  # гарантируем готовность Telethon-клиента
    if not tg_client.is_connected():  # если не подключён
        await tg_client.connect()  # подключаемся к серверам Telegram
    global _watchers_started  # используем глобальный флаг
    if AUTH_MODE == "user":  # режим пользователя (по номеру)
        await tg_client.start()                 # спросит телефон/код только однажды  # авторизация интерактивно
    else:  # режим бота по токену
        await tg_client.start(bot_token=Config.TELEGRAM_BOT_TOKEN)  # старт по токену бота
        if not _watchers_started:  # если не запускали watcher
            start_channel_watchers()  # запускаем слежение за входящими постами
            _watchers_started = True  # помечаем как запущенный


# ──────────────────────────────────────────────────────────────────────────────
# Watcher новых постов (входящие сообщения в рабочих чатах)
# ──────────────────────────────────────────────────────────────────────────────
def start_channel_watchers() -> None:  # настройка обработчика новых сообщений Telethon
    mapping: Dict[str, str] = getattr(Config, "MONITORED_CHANNELS", {}) or {}  # карта каналов для индексирования
    if not mapping:  # если ничего не задано
        logger.warning("MONITORED_CHANNELS пуст — watcher не запущен.")  # предупреждаем в лог
        return  # выходим

    # кто я (чтобы не индексировать собственные ответы)
    my_id_box = {"id": None}  # контейнер для ID бота/аккаунта
    async def _init_me():  # асинхронно получаем собственный аккаунт
        try:
            me = await tg_client.get_me()  # запрос информации о себе
            my_id_box["id"] = getattr(me, "id", None)  # сохраняем ID
            logger.info("Watcher: my bot id = %s", my_id_box["id"])  # лог ID
        except Exception as e:  # на случай ошибки API
            logger.warning("Watcher: can't get me(): %s", e)  # предупреждаем
    tg_client.loop.create_task(_init_me())  # запускаем корутину без ожидания

    # базовый фильтр "не вопрос/не команда/не коротыш"
    def _indexable_text(t: str) -> bool:  # решаем, стоит ли индексировать текст
        if not t:  # пусто
            return False  # не индексируем
        t = t.strip()  # убираем пробелы по краям
        if not t:  # после трима пусто
            return False  # не индексируем
        if t.startswith(("/", "!", ".")):          # команды  # начинаются с символов команд
            return False  # игнорируем команды
        if t.endswith("?"):                        # вопросы  # заканчиваются вопросительным знаком
            return False  # игнорируем вопросы
        if len(t) < 40:                            # слишком коротко  # короткие сообщения
            return False  # не индексируем короткие
        return True  # пригодно для индексирования

    # эвристика: похоже ли на «ручной» новостной пост
    NEWS_RX = re.compile(  # регулярка, ловящая «новостные» термины (рус/eng, авто/недвиж)
        r"(крт|ипотек|ставк|процент|руб\.?|₽|кв\.?\s*м|квартир|продаж|"
        r"проект|запустил|представил|концепт|модель|релиз|компания|рынок|"
        r"авто|дилер|поставк|реновац|дду|жк|комплекс|кластер)",
        re.I
    )
    def _looks_like_manual_news(t: str) -> bool:  # проверка «похоже ли на новость, написанную вручную»
        t = (t or "").strip()  # нормализация
        if len(t) >= 160:  # длинный текст
            return True  # скорее новость
        if t.count("\n") >= 1:  # есть переносы строк
            return True  # скорее новость
        if "http://" in t or "https://" in t or "t.me/" in t:  # есть ссылки/пересылки
            return True  # похоже на пост
        if NEWS_RX.search(t):  # совпадение по ключевым словам
            return True  # новостной контент
        sentences = [s for s in re.split(r"[.!?]\s+", t) if s.strip()]  # считаем предложения
        if len(sentences) >= 2 and not t.endswith("?"):  # 2+ предложений и не вопрос
            return True  # похоже на новость
        return False  # не похоже

    @tg_client.on(events.NewMessage(incoming=True))  # регистрируем обработчик новых входящих сообщений
    async def _on_new_post(event: events.NewMessage.Event):  # корутина на каждое новое сообщение
        try:
            msg: Message = event.message  # получаем сам объект сообщения
            if not msg:  # если пусто (редкий случай)
                return  # выходим

            # 0) не индексируем свои исходящие
            if getattr(msg, "out", False):  # если сообщение исходящее (от нас)
                return  # пропускаем

            # 1) не индексируем ботов (и самого себя)
            try:
                sender = await event.get_sender()  # получаем отправителя
                sid = getattr(sender, "id", None)  # его ID
                if sid and my_id_box["id"] and sid == my_id_box["id"]:  # если это мы сами
                    return  # пропускаем
                if getattr(sender, "bot", False):  # если отправитель — бот
                    return  # не индексируем ботов
            except Exception:
                sender = None  # на всякий случай, если не удалось получить отправителя

            text = (msg.message or "")  # основной текст сообщения
            if not _indexable_text(text):  # быстрый фильтр пригодности
                return  # не индексируем

            # 2) распознаём форвард из канала — такие индексируем всегда
            is_forward = False  # флаг «переслано из канала»
            fwd = getattr(msg, "fwd_from", None)  # информация о пересылке
            if fwd and getattr(fwd, "from_id", None) and isinstance(fwd.from_id, PeerChannel):  # источник — канал
                is_forward = True  # помечаем как форвард

            # 3) если НЕ форвард — это ручной пост. Пропускаем только если он «похож на новость»
            if not is_forward and not _looks_like_manual_news(text):  # ручной, но не новостной
                return  # игнорируем

            # 4) сопоставляем категорию по карте/названию чата
            chat = await event.get_chat()  # получаем чат/канал, откуда пришло сообщение
            raw_id = getattr(chat, "id", None)  # его числовой ID
            username = getattr(chat, "username", None)  # юзернейм канала (если есть)

            keys = [f"-100{raw_id}", str(raw_id)]  # варианты ключей сопоставления
            if username:
                keys.append(f"@{username}")  # добавляем @username как возможный ключ

            category = None  # сюда положим найденную категорию
            for k in keys:  # перебираем ключи
                if k in mapping:  # если ключ есть в карте
                    category = mapping[k]  # берём категорию
                    break  # прекращаем поиск
            if not category:  # если не нашли напрямую
                category = guess_category(getattr(chat, "title", "") or "")  # пытаемся угадать по названию
            if not category:  # если всё ещё нет
                return  # не индексируем без категории

            # 5) пишем документ
            doc = {  # формируем документ для индексации
                "text": text,  # текст сообщения
                "category": category,  # распознанная категория
                "channel": raw_id,  # ID канала/чата
                "username": username,  # username (если есть)
                "date": msg.date.isoformat() if msg.date else None,  # дата ISO
                "timestamp": int(msg.date.timestamp()) if msg.date else None,  # unix-метка
                "message_id": msg.id,  # ID сообщения
                "forwarded": bool(is_forward),  # признак форварда
                "author_id": getattr(sender, "id", None) if sender else None,  # ID автора
                "author_is_bot": bool(getattr(sender, "bot", False)) if sender else None,  # признак «автор — бот»
            }
            added = rag.index_posts([doc])  # индексируем в RAG/векторную БД
            if added:  # если что-то добавилось
                logger.info(  # логируем успешную индексацию
                    "Watcher: +1 пост (%s) (%s), msg_id=%s",
                    category, "forward" if is_forward else "manual", msg.id
                )
        except Exception as e:  # отлавливаем любые сбои в обработчике
            logger.warning("Watcher error: %s", e)  # пишем предупреждение

    logger.info("Watcher новых сообщений запущен (индексируем форварды и ручные посты-новости).")  # сообщаем о запуске


# ──────────────────────────────────────────────────────────────────────────────
# Команды
# ──────────────────────────────────────────────────────────────────────────────
@dp.message(Command("start"))  # обработчик команды /start
async def cmd_start(message: Message):  # приветствие и подсказки
    cat = resolve_category(getattr(message.chat, "title", "") or "", message.chat.id)  # определяем категорию чата
    sample = "что по недвижимости за неделю?\nчто по недвижимости за месяц?" if cat != "auto" else "что по авто за неделю?\nчто по авто за месяц?"  # пример вопросов
    mode_txt = "user" if AUTH_MODE == "user" else "bot"  # текст о режиме Telethon
    await message.answer(  # отправляем приветственное сообщение
        "Привет! Я собираю посты в локальную векторную БД (Chroma) и отвечаю на вопросы.\n"
        f"Режим Telethon: <b>{mode_txt}</b>.\n"
        f"Попробуй спросить: <i>{sample}</i>\n\n"
        "Команды:\n"
        "• /digest — сгенерировать дайджест (из локального индекса)\n"
        "• /reindex 30 — переиндексация истории (только user-режим)\n"
        "• /status — размер индекса\n"
    )  # конец формирования приветствия


@dp.message(Command("status"))  # обработчик команды /status
async def cmd_status(message: Message):  # показывает размер индекса
    await message.answer(f"В базе документов: {rag.count()}")  # строка с количеством документов


@dp.message(Command("debug"))  # обработчик команды /debug
async def cmd_debug(message: Message):  # выводит несколько последних записей для отладки
    cat = resolve_category(getattr(message.chat, "title", "") or "", message.chat.id)  # текущая категория
    items = rag.recent_posts(days=30, category=cat, limit=3)  # берём 3 свежих поста за 30 дней
    if not items:  # если пусто
        await message.answer("debug: recent_posts пусто. Попробую взять просто последние по категории/вообще.")  # уведомление
        try:
            if hasattr(rag.vector_db, "last_by_category"):  # если БД умеет выдавать последние по категории
                items = rag.vector_db.last_by_category(category=cat, limit=3) \
                        or rag.vector_db.last_by_category(category=None, limit=3)  # резервный вариант — без категории
        except Exception:
            items = []  # защищаемся от ошибкок доступа к БД
    if not items:  # если всё равно пусто
        await message.answer("debug: вообще ничего не нашёл.")  # сообщаем
        return  # выходим
    lines = []  # сюда соберём строки отчёта
    for it in items:  # перебор элементов
        md = it.get("metadata") or {}  # метаданные объекта (из векторной БД)
        lines.append(f"- cat={md.get('category')} ts={md.get('timestamp')} ch={md.get('channel')} msg={md.get('message_id')}")  # краткая строка
    await message.answer("debug sample:\n" + "\n".join(lines))  # отправляем отчёт


@dp.message(Command("reindex"))  # обработчик команды /reindex
async def cmd_reindex(message: Message):  # ручная переиндексация истории каналов
    if AUTH_MODE != "user":  # в режиме бота чтение истории запрещено
        await message.answer(
            "В режиме бота чтение истории каналов запрещено Telegram — поэтому /reindex недоступен.\n"
            "Но я индексирую все новые посты автоматически, и по ним можно сделать /digest."
        )  # объясняем ограничение
        return  # выходим

    try:
        parts = (message.text or "").strip().split()  # парсим аргументы
        days = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 30  # число дней (по умолчанию 30)
    except Exception:
        days = 30  # на ошибке — значение по умолчанию

    await ensure_telethon_ready()  # убеждаемся, что Telethon готов
    since = datetime.now(tz=timezone.utc) - timedelta(days=days)  # нижняя граница по времени

    lines: List[str] = []  # собираем отчёт по каналам
    total_added = 0  # сколько всего добавлено документов
    for raw_id, category in (getattr(Config, "MONITORED_CHANNELS", {}) or {}).items():  # перебираем настроенные каналы
        chan_ref = str(raw_id).strip()  # нормализуем идентификатор
        try:
            entity = await tg_client.get_entity(PeerChannel(int(chan_ref)))  # получаем сущность канала
            batch: List[Dict] = []  # пакет документов для индексации
            async for m in tg_client.iter_messages(entity, offset_date=None):  # итерируем сообщения (от новых к старым)
                if m.date and m.date.replace(tzinfo=timezone.utc) < since:  # если сообщение старше порога
                    break  # останавливаем итерацию
                if not m.message:  # пропускаем пустые
                    continue  # дальше
                batch.append({  # добавляем запись в пакет
                    "text": m.message,  # текст
                    "category": category,  # категория из настроек
                    "channel": int(getattr(entity, 'id', 0) or 0),  # числовой ID канала
                    "date": m.date.isoformat() if m.date else None,  # дата ISO
                    "timestamp": int(m.date.timestamp()) if m.date else None,  # unix-метка
                    "message_id": m.id,  # ID сообщения
                })
            if batch:  # если пакет непустой
                added = rag.index_posts(batch)  # индексируем пакет
                total_added += added  # суммируем добавленное
                lines.append(f"• {chan_ref}: добавлено {added}")  # строка отчёта
            else:
                lines.append(f"• {chan_ref}: за {days} дней ничего не найдено")  # канал пуст за период
        except Exception as e:  # ошибки доступа к каналу
            lines.append(f"• {chan_ref}: ошибка доступа ({e.__class__.__name__})")  # фиксируем в отчёте

    await message.answer(  # отправляем сводный отчёт по реиндексации
        "Реиндексация завершена.\n" + "\n".join(lines) + f"\n\nИтого добавлено: {total_added}. Всего: {rag.count()}."
    )  # конец ответа


@dp.message(Command("digest"))  # обработчик команды /digest
async def cmd_digest(message: Message):  # формирует дайджест по уже индексированным данным
    """Генерим дайджест из уже проиндексированных постов (Chroma)."""  # докстрока функции
    category = resolve_category(getattr(message.chat, "title", "") or "", message.chat.id)  # определяем категорию чата
    await message.answer("Запускаю сбор: 7 и 30 дней…")  # уведомляем о старте
    links: List[str] = []  # сюда будем собирать ссылки/тексты по периодам

    for days in (7, 30):  # два периода: неделя и месяц (условно)
        try:
            url_or_text = rag.make_digest(days=days, category=category)  # генерим дайджест через RAGSystem
            if not url_or_text:  # ничего не вернулось
                links.append(f"За {days} дней ничего не найдено.")  # добавляем сообщение об отсутствии данных
            elif url_or_text.startswith("http"):  # если это ссылка
                links.append(f"Дайджест за {days} дней: {url_or_text}")  # добавляем ссылку
            else:
                links.append(f"<b>Дайджест за {days} дней</b>\n{url_or_text}")  # иначе это текст — форматируем
        except Exception as e:  # ошибка генерации дайджеста
            links.append(f"Ошибка формирования дайджеста за {days} дней: {e.__class__.__name__}")  # фиксируем

    await message.answer("\n\n".join(links))  # отправляем объединённый ответ


# ──────────────────────────────────────────────────────────────────────────────
# Ответы на обычные вопросы (RAG)
# ──────────────────────────────────────────────────────────────────────────────

# Показывать «Сводка ответа» только если она НЕ отрицательная:
NO_DATA_RX = re.compile(  # регэксп, распознающий «нет информации/ничего не найдено»
    r"^\s*(нет\s+(информации|сведений)|ничего\s+не\s+найдено)",
    re.I
)

def _looks_like_manual_news_for_q(t: str) -> bool:  # эвристика: сообщение похоже на новостной пост (не вопрос)
    """Эвристика: похоже ли сообщение на «ручной» новостной пост (чтобы не отвечать на него)."""  # докстрока
    t = (t or "").strip()  # нормализация строки
    if not t:  # пустой текст
        return False  # не считаем новостью
    if "http://" in t or "https://" in t or "t.me/" in t:  # наличие ссылок
        return True  # вероятно новость
    if "\n" in t and not t.endswith("?"):  # есть переносы и не вопрос
        return True  # похоже на новость
    if len(t) >= 140 and "?" not in t:  # длинно и без вопросительных знаков
        return True  # вероятно пост/новость
    # 2+ предложений без вопросительного тона
    sentences = [s for s in re.split(r"[.!;]\s+", t) if s.strip()]  # делим на предложения
    if len(sentences) >= 2 and not t.endswith("?"):  # минимум два предложения и не вопрос
        return True  # новостной контент
    return False  # иначе — не новость


@dp.message(F.text)  # обработчик для всех текстовых сообщений (не команд)
async def answer_questions(message: Message):  # ответ по RAG на обычные вопросы
    q = (message.text or "").strip()  # берём текст вопроса
    if not q:  # если пусто
        return  # выходим
    # игнорируем команды/служебные реплики
    if q.startswith("/"):  # команды не обрабатываем здесь
        return  # выходим
    if q.lower().startswith("запускаю сбор") or q.lower().startswith("дайджест за"):  # служебные ответы бота
        return  # не отвечаем
    # не отвечаем на форварды (это новости), только индексируем их watcher'ом
    try:
        if getattr(message, "forward_origin", None):  # если сообщение — форвард
            return  # не отвечаем
    except Exception:
        pass  # игнорируем проблемы с атрибутом
    # не отвечаем на «ручные» новостные посты
    if _looks_like_manual_news_for_q(q):  # если похоже на новость/пост, а не вопрос
        return  # выходим

    category = resolve_category(getattr(message.chat, "title", "") or "", message.chat.id)  # определяем категорию
    try:
        # rag.query возвращает уже отформатированный HTML (с «Сводкой» и источниками)
        answer = rag.query(q, k=6, category=category)  # делаем запрос к RAG: 6 ближайших фактов
        # Если «Сводка» отрицательная — вырезаем её блок, оставляем только источники.
        if "Сводка ответа" in (answer or ""):  # если в ответе есть блок сводки
            m = re.search(r"(?s)<b>Сводка ответа:</b>\s*(.+?)(?:\n\s*\n|$)", answer)  # вытаскиваем текст сводки
            if m and NO_DATA_RX.search(m.group(1) or ""):  # если сводка говорит «ничего нет»
                answer = answer.replace(m.group(0), "", 1).lstrip()  # удаляем сводку, оставляем источники
        if not answer:  # если вообще ничего
            answer = "Нет релевантных фактов в индексе за запрошенный период."  # ответ по умолчанию
    except Exception as e:  # если что-то пошло не так при RAG-запросе
        logger.exception("RAG error: %s", e)  # логируем стек
        answer = "Что-то пошло не так при поиске по базе."  # пользовательское сообщение об ошибке
    await message.answer(answer, disable_web_page_preview=True)  # отправляем HTML-ответ, без предпросмотра ссылок


# ──────────────────────────────────────────────────────────────────────────────
async def main():  # основная точка входа приложения
    logger.info("Старт… режим Telethon: %s", AUTH_MODE)  # логируем режим
    await ensure_telethon_ready()  # подготавливаем Telethon (подключение/авторизация/ watcher)

    # --- Автодайджесты по воскресеньям 09:00 (Europe/Moscow) ---
    scheduler = AsyncIOScheduler(timezone="Europe/Moscow")  # создаём планировщик с московской таймзоной

    async def _post_digest_for_chat(chat_id: int, category: str, days: int) -> None:  # отправка дайджеста в чат
        url = rag.make_digest(days=days, category=category)  # генерируем дайджест (ожидаем ссылку)
        if url:  # если получили ссылку/контент
            label = "7 дней" if days == 7 else "30 дней"  # подпись периода
            await bot.send_message(chat_id, f"Дайджест за {label}: {url}", disable_web_page_preview=True)  # отправляем

    async def _weekly_job():  # еженедельная задача по всем отслеживаемым чатам
        mapping = getattr(Config, "MONITORED_CHANNELS", {}) or {}  # берём карту каналов/категорий
        for chat_id_str, category in mapping.items():  # перебираем записи
            try:
                chat_id = int(chat_id_str)  # приводим ключ к int
            except Exception:
                continue  # пропускаем, если ключ нечисловой
            # сначала 7 дней, затем 30 дней
            await _post_digest_for_chat(chat_id, category, 7)  # отправляем недельный дайджест
            await _post_digest_for_chat(chat_id, category, 30)  # затем месячный

    # каждое воскресенье в 09:00 по Москве
    scheduler.add_job(_weekly_job, "cron", day_of_week="sun", hour=9, minute=0)  # настраиваем cron-задачу
    scheduler.start()  # запускаем планировщик
    # --- конец блока расписания ---

    await dp.start_polling(bot)  # запускаем поллинг aiogram (обработка апдейтов бота)

if __name__ == "__main__":  # запуск скрипта как основной программы
    try:
        asyncio.run(main())  # запускаем main() в новом event loop
    except (KeyboardInterrupt, SystemExit):  # корректная остановка по Ctrl+C/сигналам
        logger.info("Бот остановлен.")  # логируем завершение
