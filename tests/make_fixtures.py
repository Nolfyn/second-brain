"""
Генерация фикстур:
    python tests/make_fixtures.py              — парсерные фикстуры + eval-корпус
    python tests/make_fixtures.py --corpus     — только eval-корпус

Создает:
  tests/fixtures/sample_plain.docx      — обычный Word doc
  tests/fixtures/sample_protected.docx  — запароленный Word doc
  tests/fixtures/sample_text.pdf        — text-based PDF
  tests/fixtures/corpus/                — искусственный корпус для eval harness
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


def make_plain_docx(output: Path) -> None:
    from docx import Document

    doc = Document()
    doc.add_heading("Test Document", level=1)
    doc.add_paragraph(
        "This is a sample document for testing the second-brain parser. "
        "It contains English and some Russian text."
    )
    doc.add_paragraph(
        "Это тестовый абзац на русском языке. "
        "Семантический поиск должен находить этот текст."
    )
    doc.add_paragraph(
        "A second paragraph with different content about philosophy and stoicism."
    )
    doc.save(output)
    print(f"Created: {output}")


def make_protected_docx(output: Path, password: str) -> None:
    import io
    import msoffcrypto
    from docx import Document

    # Сначала собираем обычный docx в памяти
    doc = Document()
    doc.add_heading("Protected Document", level=1)
    doc.add_paragraph(
        "This document is password-protected. If you can read this, decryption worked."
    )
    doc.add_paragraph("Этот документ защищён паролем.")

    plain_buffer = io.BytesIO()
    doc.save(plain_buffer)
    plain_buffer.seek(0)

    # Шифруем
    encrypted_buffer = io.BytesIO()
    office_file = msoffcrypto.OfficeFile(plain_buffer)
    office_file.encrypt(password, encrypted_buffer)
    encrypted_buffer.seek(0)

    with open(output, "wb") as f:
        f.write(encrypted_buffer.read())
    print(f"Created: {output} (password: {password})")


def make_text_pdf(output: Path) -> None:
    try:
        from reportlab.pdfgen import canvas as rl_canvas

        c = rl_canvas.Canvas(str(output))
        c.drawString(72, 750, "Second Brain — Test PDF")
        c.drawString(
            72,
            720,
            "This is a text-based PDF for testing pdfminer extraction.",
        )
        c.drawString(
            72,
            700,
            "It contains readable text that pdfminer can extract directly.",
        )
        c.save()
        print(f"Created: {output}")
    except ImportError:
        # Fallback: минимальный PDF вручную (без внешних библиотек)
        pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 120>>
stream
BT /F1 12 Tf 72 750 Td (Second Brain Test PDF) Tj 0 -20 Td (This is a text-based PDF for parser testing.) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
0000000446 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
528
%%EOF"""
        with open(output, "wb") as f:
            f.write(pdf_content)
        print(
            f"Created: {output} (minimal PDF — install reportlab for a better one)"
        )


def make_corpus(fixtures_dir: Path) -> None:
    """Искусственный двуязычный корпус с известной структурой для eval."""
    from docx import Document

    corpus = fixtures_dir / "corpus"

    def docx(rel_path: str, *paragraphs: str) -> None:
        path = corpus / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = Document()
        for text in paragraphs:
            doc.add_paragraph(text)
        doc.save(path)
        print(f"Created: {path}")

    # --- Notes.docx: лог ссылок с заметками о прочитанном ---
    docx(
        "Notes.docx",
        "Лог прочитанных статей за зиму.",
        "https://gwern.net/doc/psychology/2021-smith.pdf — исследование сна и памяти: полисомнография, выборка 200 человек. Полезная методология для своих экспериментов.",
        "https://gwern.net/zeo — трекинг сна через Zeo, анализ дневников. Решил попробовать вести свой журнал сна.",
        "https://vk.com/wall-123456_789 — пост про советскую инфраструктру: метро, дворцы культуры, плановые города.",
        "https://vk.com/@historymath — длинная статья про историю советской математической школы.",
        "https://example.org/soviets.html — 'The Economics of the Soviet Collapse'. Конспект: причины распада плановой экономики, хронический дефицит, реформы 1987-1991, гиперинфляция. Очень полезная статья, нашёл через рекомендации.",
        "Отдельно: список книг к покупке и мысли о продуктивности.",
    )

    # --- Games.docx: игры, выводы по Honkai, вскользь про Shadow Slave ---
    docx(
        "Games.docx",
        "Журнал игр за год.",
        "Honkai: Star Rail — выводы после двух месяцев: сюжет второго акта сильный, боёвка монотонная, гача щедрее, чем в Genshin. Дропнул после планеты Ярило, вернулся на ивент.",
        "Disco Elysium — прошёл второй раз, альтернативная концовка за коммуниста. Шедевр диалогов.",
        "Shadow Slave — читал новеллу по совету, первые сто глав затянуты, бросил. Упоминание вскользь, продолжать не планирую.",
    )

    # --- china/China.docx: поездка с секцией выводов ---
    docx(
        "china/China.docx",
        "Поездка в Китай, октябрь 2016.",
        "Шанхай: метро дешевле такси в разы, маглев из аэропорта — отдельное впечатление. Небоскрёбы Луцзяцзуй.",
        "Пекин: запретный город занял целый день, стена на Мутяньюй менее людная, чем Бадалин.",
        "Выводы по поездке в Китай: виза оформлялась долго, но поездка стоила того; еда дешевле, чем ожидал; языковой барьер решается переводчиком; следующий раз — Сиань.",
    )

    # --- Egypt.docx ---
    docx(
        "Egypt.docx",
        "Поездка в Египет, сентябрь 2015. Пирамиды Гизы рано утром без толп, жара выше сорока. Дайвинг в Шарм-эль-Шейхе: кораллы у Рас-Мохаммеда. Общее впечатление: страна контрастов, торговцы утомляют, но история того стоит.",
    )

    # --- J 2017.docx: дневник с датами и редким топонимом ---
    docx(
        "J 2017.docx",
        "15.03.2017. Работа над проектом идёт медленно. Читал про казахстанский город Кызылорду — старый форт на Сырдарье, история кочевой торговли.",
        "02.07.2017. День рождения, ужин с друзьями. Обсуждали переезд, пока ничего решили.",
        "19.11.2017. Закончил курс по статистике. Начал вести бюджет в таблице.",
    )

    # --- Celebration 2016-2017.docx ---
    docx(
        "Celebration 2016-2017.docx",
        "Новый год 2016-2017: встречали дома, фондю, фильмы до утра. Подарки — книги и настолки. Резолюция: бегать три раза в неделю (продержался месяц).",
    )

    # --- Music 2016 Pt I.docx ---
    docx(
        "Music 2016 Pt I.docx",
        "Музыкальный лог, первая половина 2016. В основном слушал пост-рок: Godspeed You! Black Emperor, Explosions in the Sky. Из русского — Панк-фракция красных бригад. Концерт Мумий Тролля в марте.",
    )

    # --- Bookshelf.docx: длинная заметка об одной книге с заголовками-секциями,
    # ключевые факты — в середине ---
    glass_bead_middle = (
        "Середина романа — это портрет провинциального интеллектуального уклада: "
        "Кнехт среди монахов-бенедиктинцев, затем миссия в Духовном царстве. "
        "Ключевой тезис книги: власть интеллекта требует отречения, и Магистр Люди "
        "в финале выбирает жизнь за пределами ордена. Орден Касталии — метафора "
        "отчуждённой интеллектуальной элиты. Критика отвлечённого образования — "
        "главная нить второй половины."
    )
    bookshelf = corpus / "Bookshelf.docx"
    bookshelf.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    doc.add_heading("Вступление", level=1)
    doc.add_paragraph(
        "Заметка о романе Германа Гессе 'The Glass Bead Game' (Magister Ludi). "
        "Читал три месяца, выписывал цитаты. Игра Гесс описывается как синтез "
        "математики, музыки и культурной истории. Герой — Иосиф Кнехт, "
        "восходящая звезда ордена."
    )
    doc.add_heading("Середина", level=1)
    doc.add_paragraph(glass_bead_middle)
    doc.add_heading("Финал", level=1)
    doc.add_paragraph(
        "Кнехт покидает орден ради воспитания юного Тито и гибнет в горах. "
        "Трактовка: отказ от абстракции ради конкретной жизни. Перечитывать "
        "в паре с 'Волхвом' Фаулза."
    )
    doc.save(bookshelf)
    print(f"Created: {bookshelf}")

    # --- Sweden.docx: русскоязычная заметка (кросс-языковые кейсы) ---
    docx(
        "Sweden.docx",
        "Поездка в Швецию, июнь 2017. Музеи Стокгольма: Васа — флагман, который затонул через двадцать минут после спуска; Скансен под открытым небом; Moderna с коллекцией современного искусства.",
        "Выводы по Швеции: дорого, но всё организовано идеально; кофе-брейк fika — отдельная культура; общественный транспорт без контролёров; вернусь ради севера и северного сияния.",
    )

    # --- reference/Grammar Reference.docx: материал под .brainignore ---
    docx(
        "reference/Grammar Reference.docx",
        "Past perfect: had + past participle. Употребляется для действия, завершившегося до другого действия в прошлом: 'I had left before she arrived'.",
        "Present perfect vs past simple: результат vs факт. Таблица неправильных глаголов прилагается.",
    )

    # --- films.txt: plain-text файл ---
    txt_path = corpus / "films.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(
        "Films watched, backlog notes.\n"
        "Blade Runner 2049 — кинотеатр, визуал лучшее за год.\n"
        "Dunkirk — звук невероятный, но холодный фильм.\n"
        "Your Name — аниме, пересматривал дважды.\n"
        "Левиафан — тяжёлый, но нужный.\n",
        encoding="utf-8",
    )
    print(f"Created: {txt_path}")

    # --- .docx для OCR/legacy тестов не генерируем: копируются вручную ---


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        action="store_true",
        help="генерировать только eval-корпус",
    )
    args = parser.parse_args()

    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)

    make_corpus(fixtures_dir)

    if not args.corpus:
        password = os.getenv("DOC_PASSWORD", "test123")
        make_plain_docx(fixtures_dir / "sample_plain.docx")
        make_protected_docx(fixtures_dir / "sample_protected.docx", password)
        make_text_pdf(fixtures_dir / "sample_text.pdf")

        print()
        print("Done! Remaining fixtures to add manually:")
        print(
            "  tests/fixtures/sample_scanned.pdf  — copy any scanned PDF from your collection"
        )
        print(
            "  tests/fixtures/sample_legacy.doc   — copy any old .doc file from your collection"
        )


if __name__ == "__main__":
    main()
