<h1 align="center">Second Brain</h1>

<p align="center">
<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python" />
<img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
<img src="https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB" alt="React" />
<img src="https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite" />
<img src="https://img.shields.io/badge/chromadb-FFA500?style=for-the-badge&logoColor=white" alt="ChromaDB" />
<img src="https://img.shields.io/badge/ollama-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama" />
</p>

Двуязычный (русский/английский) инструмент для поиска по личным заметкам на основе RAG (Retrieval-Augmented Generation). Индексирует документы (`.docx`, `.pdf`, `.txt` и другие форматы) и позволяет задавать по ним вопросы на естественном языке.

## Основной функционал и особенности:

*   **Двуязычный кросс-языковой поиск** - вопрос на одном языке находит и обрабатывает заметки на другом за счет мультиязычной модели эмбеддингов и расширения запроса через LLM;
*   **Гибридный поиск** - комбинация семантического поиска (векторы) и точного поиска по ключевым словам, с приоритизацией сигналов: URL/домены, имена файлов, годы, тематические папки;
*   **Реранк через cross-encoder** - кандидаты от векторного поиска переранжируются точной мультиязычной моделью, что выправляет порядок выдачи на размытых запросах;
*   **Структурный чанкинг** - разбиение по границам заметок (заголовки в .docx, даты в .txt); длинная однотемная заметка остаётся цельной единицей поиска и сводки;
*   **Смена LLM через конфиг** - локальная Ollama, Claude API или любой OpenAI-совместимый endpoint;
*   **Чат-сессии** - переписка сохраняется в SQLite: можно вернуться к чату, продолжить уточнениями, удалить ненужный;
*   **Синтез ответа с указанием источников** - ответ собирается только из найденных фрагментов, с ссылками на документы-источники; для больших наборов используется map-reduce по секциям документов;
*   **Поддержка множества форматов** - `.docx` (в т. ч. зашифрованные), `.doc`, `.pdf` (с OCR для сканов), `.txt`, `.csv`, `.html`, `.epub`, `.pptx`;
*   **CLI для индексации и отладки** - команды `index`, `watch`, `reindex`, `prune`, `stats`, `grep`, `search-test` и другие;
*   **Веб-интерфейс** - чат на React + Vite + Tailwind с потоковой выдачей ответа и отображением источников.

<div><img src="resources/screenshot.png" width="1000" alt="Скриншот окна программы" /></div>

## Архитектура

Пайплайн разделен на два этапа - индексацию и поиск:

**Индексация:** документы парсятся под каждый формат (расшифровка `.docx`, OCR для сканов `.pdf`), разбиваются на чанки, кодируются мультиязычной моделью эмбеддингов и сохраняются в двух хранилищах - векторы в **ChromaDB**, а сырой текст и метаданные в **SQLite** (через SQLAlchemy). Инкрементальная индексация по контрольным суммам не обрабатывает неизмененные файлы.

**Поиск:** запрос проходит через `Retriever`, который выбирает стратегию по сигналам в тексте (точный поиск по ключевым словам, поиск по имени файла, по году, по тематической папке, либо семантический поиск с расширением запроса). Результат при необходимости переранжируется через cross-encoder. Найденные фрагменты передаются в `Synthesiser`, который через LLM собирает из них ответ на языке вопроса; переписка ведётся в сессиях, сохраняющихся в SQLite.

Бэкенд на **FastAPI** отдает результат потоком (streaming), фронтенд на **React** отображает ответ по мере генерации.

## Стек

**Бэкенд:** Python, FastAPI, ChromaDB, SQLite / SQLAlchemy, sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2` + cross-encoder реранк), Ollama (`qwen2.5:7b`) или Claude/OpenAI-совместимые API, Typer (CLI)

**Фронтенд:** React, Vite, Tailwind CSS

**Парсинг:** python-docx, pdfminer, pytesseract (OCR), python-pptx, ebooklib, html2text

## Установка и запуск

Требуется установленный [Ollama](https://ollama.com) и Python 3.11+.

```bash
# 1. Установить зависимости
pip install -e .

# 2. Загрузить модель (по умолчанию - qwen2.5:7b)
ollama pull qwen2.5:7b

# 3. Настроить окружение
cp .env.example .env
# при необходимости отредактировать .env

# 4. Проиндексировать папку с документами
brain index "/path/to/your/documents"

# 5. Запустить бэкенд (без --reload)
uvicorn brain.api.main:app --port 8000

# 6. Запустить фронтенд
cd ui && npm install && npm run dev
```
## Проверка качества поиска

Искусственный корпус и набор запросов для регрессионного прогона:

```bash
python tests/make_fixtures.py --corpus            # генерация корпуса
python -m tests.eval_retrieval                    # быстрый прогон без LLM
python -m tests.eval_retrieval --with-rerank      # с реранком
python -m tests.eval_retrieval --with-expansion   # с LLM-расширением запроса
```
Метрики: доля прошедших кейсов, MRR, доля пустых выдач. База - `tests/eval_baseline.md`.

## Примеры запросов

Программа лучше всего работает с конкретными запросами - где есть год, имя файла, домен сайта, название или другое опорное слово:

*   *«What articles have I read from `<site>`?»* - находит сохраненные ссылки с заметками и собирает их в список;
*   *«Чем я занимался в `<год>` году? Дай краткую справку.»* - собирает сводку из заметок за конкретный год;
*   *«Sum up my notes on `<тема/название>`.»* - извлекает и суммирует упоминания конкретной темы или произведения по всем документам.
