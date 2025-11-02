import os
import sys
import argparse
import time
from typing import Optional, Tuple

import mwclient
from requests import HTTPError
from mwclient.errors import (
    APIError,
    InvalidResponse,
    MaximumRetriesExceeded,
    LoginError,
    ProtectedPageError,
    InsufficientPermission,
)

from openai import OpenAI


DEFAULT_SITE = "ru.wikipedia.org"
DEFAULT_PATH = "/w/"
DEFAULT_MODEL = os.getenv("CHAT_MODEL") or "gpt-4o-mini"
MAX_WIKITEXT_CHARS = int(os.getenv("MAX_WIKITEXT_CHARS", "120000"))


class WikiConnectionError(Exception):
    """Ошибка подключения или взаимодействия с MediaWiki API."""
    def __init__(self, message: str | Exception):
        super().__init__(f"Ошибка соединения с Wiki: {message}")

class PageNotFoundError(Exception):
    """Статья не найдена на указанном Wiki-сайте."""
    def __init__(self, title: str):
        super().__init__(f"Статья '{title}' не найдена.")

class OpenAIError(Exception):
    """Ошибка при обращении к API OpenAI."""
    def __init__(self, message: str | Exception):
        super().__init__(f"Ошибка OpenAI API: {message}")



def make_openai_client(api_key: str, api_base: Optional[str]) -> OpenAI:
    if api_base:
        return OpenAI(api_key=api_key, base_url=api_base)
    return OpenAI(api_key=api_key)


def connect_wiki(site_host: str, path: str = DEFAULT_PATH, ua: str = "mwclient/summary-bot") -> mwclient.Site:
    return mwclient.Site(
        host=(site_host if "://" not in site_host else site_host.split("://", 1)[1]),
        path=path,
        scheme="https",
        clients_useragent=ua,
    )


def fetch_wikitext(site: mwclient.Site, title: str, retries: int = 3, backoff: float = 1.0) -> Tuple[str, str]:
    """
    Возвращает (нормализованное_имя_страницы, wikitext).
    """
    last_err = None
    for i in range(retries):
        try:
            page = site.pages[title]
            if not page.exists:
                raise RuntimeError(f"Страница '{title}' не существует на {site.host}")
            # нормализованное имя (учитывает регистр/перенаправления)
            norm_title = page.name
            # если это редирект — перейдём по нему
            if page.redirect:
                page = page.resolve_redirect()
                norm_title = page.name
            text = page.text()
            if not text:
                raise RuntimeError(f"Пустой wikitext у страницы '{norm_title}'")
            return norm_title, text
        except (APIError, InvalidResponse, HTTPError, MaximumRetriesExceeded) as e:
            last_err = e
            time.sleep(backoff * (2 ** i))
        except Exception as e:
            last_err = e
            break
    raise RuntimeError(f"Не удалось получить статью '{title}': {last_err}")


def build_prompt(wikitext: str, title: str) -> str:
    wt = wikitext if len(wikitext) <= MAX_WIKITEXT_CHARS else (wikitext[:MAX_WIKITEXT_CHARS] + "\n…")
    return (
        "Ты — редактор статей. Возьми информацию из большой сложной статьи и сократи её, "
        "сделав понятной школьнику 8–9 класса. Сохраняй важные факты, даты, определения. "
        f"Дай ссылку на исходную статью в формате {{{{основная статья|{title}}}}}. "
        "Пиши в wiki-формате (ориентируйся на исходную статью). Для жирного текста используй '''тут текст'''. "
        "В ответ дай только саму статью без дополнительных комментариев. "
        f"\n\nИсходная статья: {title}:\n"
        f"{wt}\n\nСтатья для школьника:\n"
    )

def run_summarization(
    title: str,
    site_host: str = DEFAULT_SITE,
    path: str = DEFAULT_PATH,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Выполняет суммаризацию статьи MediaWiki через OpenAI.
    Возвращает итоговый текст.
    """
    api_key = os.getenv("CHAT_API_KEY") or os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("CHAT_API_BASE")

    if not api_key:
        raise OpenAIError("Переменная окружения OPENAI_API_KEY или CHAT_API_KEY не установлена.")

    #1 Подключение к MediaWiki
    try:
        site = connect_wiki(site_host, path)
    except Exception as e:
        raise WikiConnectionError(e)

    #2 Получение wikitext
    try:
        norm_title, wikitext = fetch_wikitext(site, title)
    except Exception as e:
        msg = str(e)
        if "не существует" in msg or "Пустой wikitext" in msg:
            raise PageNotFoundError(title)
        raise WikiConnectionError(e)

    #3 Генерация prompt
    prompt = build_prompt(wikitext, norm_title)

    #4 Вызов OpenAI
    try:
        client = make_openai_client(api_key, api_base)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise OpenAIError(e)


def publish_draft(
    content: str,
    article_title: str,
    wiki_host: str = "ru.ruwiki.ru",
    wiki_path: str = "/w/",
    username: str | None = None,
    password: str | None = None,
    summary: str = "Публикация черновика автоматически (бот)",
    minor: bool = False,
    overwrite: bool = True,
) -> str:
    """
    Публикует `content` на страницу:
    Участник:<username>/Черновик/<article_title>
    Возвращает нормализованное имя созданной/обновлённой страницы.
    """

    username = username or os.getenv("WIKI_USERNAME")
    password = password or os.getenv("WIKI_PASSWORD")
    if not username or not password:
        raise RuntimeError("Не заданы креды: WIKI_USERNAME / WIKI_PASSWORD")

    site = mwclient.Site(
        host=(wiki_host if "://" not in wiki_host else wiki_host.split("://", 1)[1]),
        path=wiki_path,
        scheme="https",
        clients_useragent=f"wiki-summarizer-bot/1.0 ({username})"
    )

    try:
        site.login(username, password)
    except (LoginError, APIError, HTTPError) as e:
        raise RuntimeError(f"Логин не удался: {e}")

    # Страница вида "Участник:Логин/Черновик/<название статьи>"
    page_title = f"Участник:{username}/Черновик/{article_title}"
    page = site.pages[page_title]

    # Опции сохранения
    save_kwargs = {"summary": summary, "minor": minor}
    if not overwrite:
        save_kwargs["createonly"] = True

    try:
        page.save(text=content, **save_kwargs)
        return page.name
    except ProtectedPageError as e:
        raise RuntimeError(f"Страница защищена от записи: {e}")
    except InsufficientPermission as e:
        raise RuntimeError(f"Недостаточно прав для правки: {e}")
    except APIError as e:
        code = getattr(e, "code", "")
        if code and "captcha" in code.lower():
            raise RuntimeError("Правка требует CAPTCHA (используйте бот-пароль с правом edit/skipcaptcha).")
        raise RuntimeError(f"Ошибка API при сохранении: {e}")
    except Exception as e:
        raise RuntimeError(f"Не удалось сохранить страницу: {e}")

def main():
    parser = argparse.ArgumentParser(description="Суммаризация статьи MediaWiki через OpenAI")
    parser.add_argument("title", help="Название статьи (например, 'Изотопы')")
    parser.add_argument("--site", default=DEFAULT_SITE, help=f"Хост MediaWiki (по умолчанию {DEFAULT_SITE})")
    parser.add_argument("--path", default=DEFAULT_PATH, help=f"Путь API/страниц (по умолчанию {DEFAULT_PATH})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Модель OpenAI (по умолчанию {DEFAULT_MODEL})")
    args = parser.parse_args()

    try:
        summary_text = run_summarization(args.title, args.site, args.path, args.model)
        print(summary_text)

        """page_name = publish_draft(
            content=summary_text,
            article_title=f"{args.title}_draft",
            wiki_host=args.site,
            wiki_path=args.path,
            username="my_user",
            password="py_password",
            summary=f"Публикация черновика для «{args.title}_draft»",
            overwrite=True,
        )"""
        
    except PageNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(3)
    except WikiConnectionError as e:
        print(f"❌ {e}")
        sys.exit(2)
    except OpenAIError as e:
        print(f"❌ {e}")
        sys.exit(4)
    except Exception as e:
        print(f"❌ Неизвестная ошибка: {e}")
        sys.exit(99)


if __name__ == "__main__":
    main()