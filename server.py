import os
import sys
import asyncio
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel

import wiki_summarize

app = FastAPI(title="Wiki Summarizer Service", version="1.0")

class SummaryResponse(BaseModel):
    title: str
    summary: str


@app.get("/summarize", response_model=SummaryResponse)
async def summarize(
    title: str = Query(..., description="Название статьи (например, 'Изотопы')"),
    site: str = Query("ru.wikipedia.org", description="Домен MediaWiki"),
    path: str = Query("/w/", description="Путь к API (по умолчанию /w/)"),
):
    """Вызывает wiki_summarize.main() и возвращает текст"""
    try:
        summary_text = await asyncio.to_thread(
            wiki_summarize.run_summarization, title, site, path
        )
    except wiki_summarize.PageNotFoundError:
        raise HTTPException(status_code=404, detail="Статья не найдена")
    except wiki_summarize.WikiConnectionError as e:
        raise HTTPException(status_code=502, detail=f"Ошибка подключения к Wiki: {e}")
    except wiki_summarize.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"Ошибка OpenAI API: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Неизвестная ошибка: {e}")

    return SummaryResponse(title=title, summary=summary_text)
