# digest_generator.py — компактный JSON→HTML генератор для Telegraph (Ollama)  # краткое описание назначения файла
from __future__ import annotations  # позволяет использовать строковые аннотации типов (отложенная оценка)

import json  # работа с JSON-данными (dump/loads)
import re  # регулярные выражения для парсинга/очистки текста
import logging  # логирование ошибок и служебной информации
from typing import List, Dict, Any, Optional  # аннотации типов для читаемости и проверок

from config import Config  # загрузка настроек (модель, short_name Telegrаph и т.п.)
from langchain_ollama import OllamaLLM  # LLM-интерфейс к локальному Ollama через LangChain
from telegraph import Telegraph  # лёгкий API-клиент для публикации страниц на telegra.ph


# сколько заметок отдаём модели (чтобы не «перекармливать» контекст)  # пояснение константы
MAX_RECORDS = 120  # ограничение на число записей, подаваемых в LLM
# обрезка одной заметки перед отправкой в LLM  # пояснение следующей константы
SNIPPET_LEN = 900  # максимальная длина текста одной заметки (символов)


def safe_abs_chat_id(raw: Any) -> Optional[str]:  # безопасно получить «короткий» chat_id для t.me/c ссылок
    """-100123 → '123' (для t.me/c/123/456). Возвращает None, если не получилось."""  # докстрока с примером
    try:  # основной путь преобразования
        s = str(raw)  # приводим вход к строке
        if s.startswith("-100"):  # если это формат супергуруппы/канала с префиксом -100
            return s.replace("-100", "")  # отбрасываем префикс
        n = int(s)  # иначе пробуем как целое число
        return str(abs(n))  # возвращаем абсолютное значение как строку
    except Exception:  # если не удалось преобразовать (например, None/мусор)
        try:  # вторая попытка — напрямую из исходного raw
            return str(abs(int(raw)))  # приводим к int и берём модуль
        except Exception:  # снова не удалось
            return None  # сигнализируем о неуспехе


def guess_post_url(item: Dict[str, Any]) -> Optional[str]:  # попытаться построить URL поста из метаданных
    """Построить ссылку на сообщение, если есть username или channel+message_id."""  # докстрока
    url = item.get("url")  # берём явный url, если уже присутствует
    if url:  # если есть
        return url  # сразу возвращаем

    msg_id = item.get("message_id")  # id сообщения в чате/канале
    username = item.get("username")  # username канала вида @name
    channel = item.get("channel")  # числовой id канала/супергруппы

    if username and msg_id:  # если есть username и id сообщения
        u = str(username).lstrip("@")  # удаляем ведущий '@' на всякий случай
        return f"https://t.me/{u}/{msg_id}"  # конструируем публичную ссылку вида t.me/name/123

    if channel and msg_id:  # если есть numeric channel id и message id
        chat_short = safe_abs_chat_id(channel)  # получаем «короткий» id без -100
        if chat_short:  # если получилось
            return f"https://t.me/c/{chat_short}/{msg_id}"  # формируем t.me/c/<id>/<msg_id>
    return None  # иначе ссылку построить не удалось


def sanitize_html(html: str) -> str:  # минимальная санитация HTML для Telegraph
    """Минимальная санитация под Telegraph: оставляем только нужные теги и чистим пустое + префиксы переводчика."""  # докстрока
    # срезаем возможные «Here is the translation …» / «Вот перевод …» в начале  # удаляем типичные префиксы перевода
    html = re.sub(r"^\s*(?:here\s+is\s+the\s+translation[^:<]*:|вот\s+перевод[^:<]*:)\s*",
                  "", html, flags=re.I)  # регистронезависимая чистка начала строки

    allowed = {"h3", "p", "a", "b", "strong", "i", "em", "u", "code", "pre", "blockquote", "br", "hr"}  # белый список тегов
    html = re.sub(r"<\s*(script|style|iframe|video|source|svg|math)[^>]*>.*?</\s*\1\s*>", "", html, flags=re.I | re.S)  # вырезаем опасные/лишние блоки

    def _strip(m):  # внутренняя функция: пропустить/убрать тег
        tag = m.group(1).lower()  # имя тега
        return m.group(0) if tag in allowed else ""  # оставляем только разрешённые
    html = re.sub(r"</?\s*([a-zA-Z0-9]+)\b[^>]*>", _strip, html)  # фильтруем все теги по whitelist

    html = re.sub(r"<(p|h3)>\s*</\1>", "", html, flags=re.I)  # удаляем пустые <p> и <h3>
    return html  # отдаём очищенный HTML


def looks_english(text: str) -> bool:  # грубая эвристика: «похоже на английский?»
    """Грубая проверка: если латиницы заметно больше кириллицы — считаем англ."""  # докстрока
    t = re.sub(r"<[^>]+>", " ", text or "")  # удаляем теги, оставляя текст
    cyr = len(re.findall(r"[А-Яа-яЁё]", t))  # считаем кириллические символы
    lat = len(re.findall(r"[A-Za-z]", t))  # считаем латиницу
    return lat > cyr * 1.2 and lat > 30  # латиницы заметно больше и есть минимальная длина


class DigestGenerator:  # основной класс генератора дайджестов
    def __init__(self) -> None:  # конструктор
        self.llm = OllamaLLM(model=Config.OLLAMA_MODEL, temperature=0.2)  # создаём LLM-клиент с нужной моделью и «холодной» температурой
        self.telegraph = Telegraph()  # инициализируем клиент telegra.ph
        try:  # пробуем зарегистрировать аккаунт-автора
            self.telegraph.create_account(short_name=getattr(Config, "TELEGRAPH_SHORT_NAME", "NewsDigestBot") or "NewsDigestBot")  # создаём/получаем аккаунт по short_name
        except Exception:  # если не получилось (например, сеть)
            pass  # продолжаем без падения

        self.style_hint = (  # текстовая подсказка стилю для LLM
            "Сгруппируй заметки в 4–8 тем. Для каждой темы сделай 2–6 пунктов. "  # требования к структуре секций
            "Каждый пункт: короткий <title> и 1–2 предложения <summary> на русском. "  # формат пункта
            "Если в исходной заметке есть поле url — ОБЯЗАТЕЛЬНО перенеси его в массив <urls> "  # жёсткое требование про ссылки
            "(даже если он один). Никаких вводных фраз/комментариев."  # запрет на преамбулы
        )  # конец style_hint

    # ---------- JSON-проход ----------  # раздел: генерация JSON-структуры через LLM
    def _prompt_json(self, title: str, records: List[Dict[str, Any]]) -> str:  # собрать промпт для строгого JSON-ответа
        # чуть подсократим тексты  # предварительная усадка каждого текста до SNIPPET_LEN
        for r in records:  # итерация по заметкам
            t = (r.get("text") or "").strip()  # берём текст и нормализуем пробелы по краям
            if len(t) > SNIPPET_LEN:  # если длиннее лимита
                r["text"] = t[:SNIPPET_LEN] + "…"  # обрезаем и ставим многоточие

        data = json.dumps(records, ensure_ascii=False)  # сериализуем список заметок как JSON (с Unicode)
        return (  # формируем промпт многострочной строкой
            "Ты получишь перечень РУССКОЯЗЫЧНЫХ новостных заметок (JSON). "  # пояснение задачи
            "Верни СТРОГО JSON без пояснений в формате:\n"  # требование формата
            "{\n"  # начало схемы
            '  "sections": [\n'  # ключевой массив секций
            '    {"title": "…", "items":[{"title":"…","summary":"…","urls":["…", …]}, …]}, …\n'  # пример структуры секции/пункта
            "  ]\n"  # закрываем массив
            "}\n\n"  # закрываем объект
            f"{self.style_hint}\n"  # вставляем стиль-«хинт»
            "ПИШИ ТОЛЬКО НА РУССКОМ. Никакого markdown и HTML — только JSON. "  # ограничения на формат вывода
            "Если не можешь соблюсти формат — верни пустой объект {}.\n\n"  # указание fallback-формата
            f"Тема дайджеста: «{title}».\n"  # передаём тему
            f"Заметки (JSON-список объектов с полями text и url):\n{data}"  # прикладываем данные заметок
        )  # возвращаем готовый промпт

    def _llm_json(self, prompt: str) -> Optional[Dict[str, Any]]:  # попытка получить и распарсить строгий JSON от LLM
        raw = self.llm.invoke(prompt)  # вызываем LLM с промптом
        text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))  # приводим ответ к строке
        m = re.search(r"\{.*\}", text, flags=re.S)  # пытаемся вырезать JSON-объект из текста
        if not m:  # если ничего похожего не нашли
            return None  # сигнализируем об ошибке
        try:  # пробуем распарсить JSON
            return json.loads(m.group(0))  # возвращаем словарь при успехе
        except Exception:  # если парсинг упал
            return None  # возвращаем None

    # ---------- HTML-рендер ----------  # раздел: сбор HTML из структуры секций
    @staticmethod  # не использует состояние экземпляра
    def _render_html(sections: List[Dict[str, Any]]) -> str:  # преобразуем секции в HTML-строку
        out: List[str] = []  # аккумулятор HTML-фрагментов
        for s in sections:  # итерируем по секциям
            title = (s.get("title") or "").strip()  # заголовок секции
            if title:  # если он не пуст
                out.append(f"<h3>{title}</h3>")  # добавляем <h3>
            for it in (s.get("items") or []):  # проходим по пунктам секции
                it_title = (it.get("title") or "").strip()  # заголовок пункта
                it_sum = (it.get("summary") or "").strip()  # краткое описание/вывод
                urls = [u for u in (it.get("urls") or []) if u]  # список ссылок без пустых значений
                links = ""  # строка для «источник» ссылок
                if urls:  # если есть ссылки
                    al = [f'<a href="{u}">источник</a>' for u in urls[:3]]  # берём до 3 ссылок и оформляем
                    links = " " + " · ".join(al)  # объединяем точкой по центру
                if it_title and it_sum:  # если есть и заголовок, и саммари
                    out.append(f"<p><b>{it_title}</b>: {it_sum}{links}</p>")  # генерируем абзац с жирным заголовком
                elif it_sum:  # если заголовка нет, но есть текст
                    out.append(f"<p>{it_sum}{links}</p>")  # просто абзац
        return sanitize_html("\n".join(out))  # склеиваем и прогоняем через санитайзер


    # ---------- Нерушимый фолбэк (без LLM) ----------  # раздел: построение HTML без участия модели
    @staticmethod  # статический метод
    def _fallback_from_records(records: List[Dict[str, Any]]) -> str:  # простой дайджест из исходных записей
        items = []  # сюда соберём пункты секции
        for r in records[:40]:  # ограничимся 40 заметками для компактности
            t = re.sub(r"\s+", " ", (r.get("text") or "").strip())  # нормализуем пробелы в тексте
            if not t:  # пустые пропускаем
                continue  # к следующей записи
            # заголовок — первое предложение (или до 90 символов)  # эвристика для тайткла
            title = t.split(".")[0]  # берём текст до первой точки
            if len(title) > 90:  # если слишком длинный
                title = title[:90].rstrip() + "…"  # обрезаем и добавляем многоточие
            summary = t if len(t) <= 280 else t[:280].rstrip() + "…"  # короткое саммари до 280 символов
            url = r.get("url") or guess_post_url(r)  # URL берём из записи или строим эвристикой
            urls = [url] if url else []  # приводим к списку ссылок
            items.append({"title": title, "summary": summary, "urls": urls})  # добавляем пункт секции
        return sanitize_html(DigestGenerator._render_html([{"title": "Главное", "items": items}]))  # отдаём HTML одной секции


    # ---------- Публичный метод ----------  # раздел: внешний интерфейс генератора
    def generate_digest(self, items: List[Dict[str, Any]], title: str) -> Optional[str]:  # сгенерировать дайджест и вернуть URL
        if not items:  # если вход пуст
            return None  # возвращаем None

        # подготовим записи: text + url (пытаемся собрать ссылку)  # нормализация входных данных
        records: List[Dict[str, Any]] = []  # подготовленный список для LLM
        for it in items:  # идём по исходным объектам
            txt = (it.get("text") or "").strip()  # берём текст поста
            if not txt:  # пустой текст пропускаем
                continue  # к следующему элементу
            url = it.get("url") or guess_post_url(it)  # строим/берём URL, если возможно
            records.append({"text": txt, "url": url})  # добавляем в нормализованном формате

        if not records:  # если после фильтрации ничего не осталось
            return None  # выходим

        records = records[:MAX_RECORDS]  # ограничиваем общий объём для модели

        # 1) Пытаемся получить строгий JSON  # основной сценарий — JSON-структура секций
        prompt = self._prompt_json(title, records)  # собираем промпт
        obj = self._llm_json(prompt)  # просим модель вернуть и парсим JSON

        # 2) Если JSON не распарсился — просим краткий список маркерами  # запасной путь
        html: Optional[str] = None  # буфер для будущего HTML
        if not obj or "sections" not in obj:  # если структура невалидна
            joined = "\n".join(f"- {r['text']}" for r in records[:300])  # готовим список маркеров из текстов
            fb_prompt = (  # альтернативный промпт для краткого списка
                "Сделай короткий русскоязычный дайджест по пунктам. "  # инструкция по формату
                "Каждый пункт начинай с дефиса «- », далее «Заголовок: краткий вывод». "  # шаблон пункта
                "Не используй markdown. ТОЛЬКО текст.\n\n" + joined  # запрет markdown + данные
            )
            fb = self.llm.invoke(fb_prompt)  # вызываем модель ещё раз
            fb_text = fb if isinstance(fb, str) else getattr(fb, "content", str(fb))  # приводим к строке
            lines = [m.strip() for m in re.findall(r"^[\-•]\s*(.+)", fb_text, flags=re.M)]  # вытягиваем пункты по началу строки
            if lines:  # если удалось извлечь список
                sec = {"title": "Главное", "items": []}  # создаём одну секцию
                for line in lines[:60]:  # ограничиваем количество пунктов
                    if ": " in line:  # если соблюдён формат «Заголовок: текст»
                        title_part, sum_part = line.split(": ", 1)  # делим на две части
                    else:  # иначе делаем эвристическое разделение
                        title_part, sum_part = line[:80], line  # заголовок до 80 символов
                    sec["items"].append({"title": title_part.strip(), "summary": sum_part.strip(), "urls": []})  # добавляем пункт без ссылок
                html = self._render_html([sec])  # рендерим HTML одной секции

        # 3) Если и это не сработало — делаем бэкап прямо из исходных записей  # нерушимый fallback
        if not html:  # если HTML всё ещё не получили
            if obj and "sections" in obj:  # если JSON есть и валиден
                html = self._render_html(obj.get("sections") or [])  # рендерим по заданной структуре
            else:  # иначе строим дайджест без модели
                html = self._fallback_from_records(records)  # простой HTML из исходников

        # 4) На всякий случай переведём на русский, если внезапно англ.  # страховка на случай английского вывода
        if looks_english(html):  # если эвристика определила «английский»
            tr = self.llm.invoke(  # просим модель перевести HTML
                "Переведи на русский СЛЕДУЮЩИЙ HTML. Верни ТОЛЬКО HTML, без префиксов, без комментариев:\n\n" + html
            )  # формируем промпт на перевод
            html = tr if isinstance(tr, str) else getattr(tr, "content", str(tr))  # приводим к строке
            html = sanitize_html(html)  # на всякий случай санитизируем

        # 5) Публикуем в Telegraph  # финальный этап — публикация страницы
        try:  # оформляем публикацию
            page = self.telegraph.create_page(  # создаём страницу на telegra.ph
                title=title,  # заголовок страницы
                html_content=html,  # HTML-содержимое
                author_name="NewsDigestBot",  # имя автора
            )  # вызов API Telegraph
            return f"https://telegra.ph/{page['path']}"  # собираем и возвращаем URL опубликованной страницы
        except Exception as e:  # если публикация не удалась
            logging.error(f"Telegraph error: {e}")  # логируем ошибку публикации
            return None  # сигнализируем вызывающему, что URL получить не удалось
