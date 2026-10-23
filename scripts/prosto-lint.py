#!/usr/bin/env python3
"""prosto-lint — checks Russian text against the rules of the `prosto` skill.

Standard library only. Read-only: it never edits the text, it only reports.

Usage:
    python prosto-lint.py text.md [more.md ...]
    cat text.md | python prosto-lint.py
    python prosto-lint.py text.md --json
    python prosto-lint.py text.md --baseline 5
    python prosto-lint.py text.md --disable passive,noun-chain
    python prosto-lint.py --rules
    python prosto-lint.py --selftest

Exit codes: 0 — clean, 1 — hard findings above --baseline, 2 — bad command line
or unreadable file. Soft findings are advisory and never fail the run.

To exempt a block that quotes bad wording on purpose, wrap it in markers:

    <!-- prosto-lint: off -->
    ...examples of what not to write...
    <!-- prosto-lint: on -->
"""

import argparse
import bisect
import contextlib
import io
import json
import os
import re
import sys
import tempfile
from collections import namedtuple

HARD = "hard"
SOFT = "soft"

Rule = namedtuple("Rule", "name severity regex message fixes scope")

# --- word lists ------------------------------------------------------------

# Bureaucratic wording with a plain replacement.
KANTSELYARIT = {
    "в целях": "чтобы",
    "с целью": "чтобы",
    "при наличии": "если есть",
    "в случае если": "если",
    "в случае, если": "если",
    "по причине того что": "потому что",
    "в связи с тем что": "потому что",
    "в связи с тем, что": "потому что",
    "на сегодняшний день": "сейчас",
    "в настоящее время": "сейчас",
    "в рамках": "в",
    "в части": "про",
    "в соответствии с": "по",
    "имеет место": "есть",
    "является": "это",
    # Only the forms that cannot be the plural noun «данные» (= data).
    "данный": "этот",
    "данная": "эта",
    "данное": "это",
    "данную": "эту",
    "данной": "этой",
    "данном": "этом",
    "вышеуказанный": "этот",
    "вышеописанный": "этот",
    "необходимо отметить": "убрать",
    "следует отметить": "убрать",
    "стоит отметить": "убрать",
    "обращаем внимание": "убрать",
    "осуществляется": "сказать, кто делает",
}

MARKETING = [
    "мощный", "мощная", "мощное", "передовой", "передовая", "инновационный",
    "инновационная", "уникальный", "уникальная", "бесшовный", "бесшовная",
    "молниеносный", "молниеносно", "революционный", "легендарный",
    "непревзойдённый", "непревзойденный", "лучший в своём классе",
    "лучший в своем классе",
]

# The skill forbids figurative language: say what actually happens.
METAPHORS = [
    "под капотом", "из коробки", "на лету", "в два клика", "в один клик",
    "камень преткновения", "узкое место", "подводные камни", "ложка дёгтя",
    "ложка дегтя", "палка о двух концах", "краеугольный камень",
    "серебряная пуля", "выстрелить себе в ногу", "на блюдечке",
]

VAGUE_QUANTITY = [
    "много", "немного", "несколько", "ряд", "некоторые", "достаточно",
    "значительно", "существенно", "большинство",
]

# Mixing names for one thing makes the reader count entities that do not exist.
SYNONYM_GROUPS = [
    ("папка", "директория", "каталог"),
    ("файл", "документ"),
    ("ошибка", "сбой", "неполадка"),
    ("проверить", "сверить", "убедиться", "удостовериться"),
    ("запустить", "стартовать", "включить"),
    ("удалить", "стереть", "снести"),
    ("сервер", "машина", "хост"),
]

LONG_HARD = 20   # above this — a hard finding
LONG_SOFT = 15   # above this — advisory

# --- patterns --------------------------------------------------------------


def alternation(words):
    """One regex for a word list, with letter boundaries that work in Cyrillic."""
    body = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<![^\W\d_])({body})(?![^\W\d_])", re.IGNORECASE)


NOMINALIZATION_RE = re.compile(
    r"\b(осуществ|производ|произвест|выполн|провед|провод|обеспеч|реализ)\w*\s+"
    r"\w*(ение|ения|ению|ении|ание|ания|анию|ании|аци[июяе]|ировани\w*)\b",
    re.IGNORECASE,
)
PASSIVE_RE = re.compile(
    r"\b\w{3,}(ется|ются|ался|алась|ались)\b"
    r"|\b(был|была|было|были)\s+\w{3,}(ен|ена|ено|ены|н|на|но|ны|т|та|то|ты)\b",
    re.IGNORECASE,
)
PARTICIPLE_RE = re.compile(
    r"\b\w{4,}(ующий|ающий|ующая|ающая|ующее|ающее|ующих|ающих|ивший|авший|"
    r"ившая|авшая|анный|енный|ённый|анная|енная|ённая|вшись|ившись|авшись)\w*\b"
    r"|\b(используя|применяя|учитывая|опираясь|исходя)\b",
    re.IGNORECASE,
)
# Inline `code`, paths, flags and setting keys: needed only when the human types them.
TECH_TOKEN_RE = re.compile(r"`[^`]*([\\/]|--|::|\{\}|\.py|\.json|\.md|=)[^`]*`")

# Three or more long nouns in a row: "повышение качества обслуживания клиентов".
CHAIN_WORD_RE = re.compile(
    r"^[а-яё]{6,}(ия|ии|ью|ости|ения|ений|аний|ования|ов|ев|ам|ами|ой)$",
    re.IGNORECASE,
)

WORD_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)
DIGIT_RE = re.compile(r"\d")
SENTENCE_END_RE = re.compile(r"[.!?…]+[\s\"»)]|[.!?…]+$")
LIST_MARKER_RE = re.compile(r"^([-*+]|\d+[.)])\s+")
FENCE_RE = re.compile(r"^(```|~~~)")
OFF_RE = re.compile(r"<!--\s*prosto-lint:\s*off\s*-->", re.IGNORECASE)
ON_RE = re.compile(r"<!--\s*prosto-lint:\s*on\s*-->", re.IGNORECASE)

# --- rule registry ---------------------------------------------------------
# One table answers "which rules exist, how severe are they, what do they say".

RULES = [
    Rule("kantselyarit", HARD, alternation(KANTSELYARIT), "Канцелярит.",
         KANTSELYARIT, "text"),
    Rule("marketing", HARD, alternation(MARKETING),
         "Рекламное слово. Убрать или поставить измеримый факт.", None, "text"),
    Rule("metaphor", HARD, alternation(METAPHORS),
         "Образное выражение. Сказать прямо, что происходит.", None, "text"),
    Rule("nominalization", SOFT, NOMINALIZATION_RE,
         "Похоже на действие в существительном. Проверить, нужен ли глагол.",
         None, "text"),
    Rule("passive", SOFT, PASSIVE_RE,
         "Похоже на страдательный залог. Назвать, кто делает.", None, "text"),
    Rule("participle", SOFT, PARTICIPLE_RE,
         "Причастие или деепричастие. Часто проще отдельным предложением.",
         None, "text"),
    Rule("tech-token", SOFT, TECH_TOKEN_RE,
         "Путь, ключ настройки или код. Нужен, только если человек будет вводить это сам.",
         None, "text"),
    Rule("long-sentence", HARD, None,
         f"Длинное предложение, больше {LONG_HARD} слов. Одна мысль — одно предложение.",
         None, "sentence"),
    Rule("sentence-length", SOFT, None,
         f"Предложение длиннее {LONG_SOFT} слов. Можно разделить.", None, "sentence"),
    Rule("vague-quantity", SOFT, alternation(VAGUE_QUANTITY),
         "Расплывчатое количество. Нужно число.", None, "sentence"),
    Rule("noun-chain", SOFT, None,
         "Три существительных подряд. Разбить на два предложения.", None, "sentence"),
    Rule("synonym-rotation", SOFT, None,
         "Одно и то же названо по-разному. Выбрать одно слово.", None, "file"),
]

SEVERITY = {rule.name: rule.severity for rule in RULES}
BY_NAME = {rule.name: rule for rule in RULES}

# --- text preparation ------------------------------------------------------


def strip_ignored(text):
    """Blank out what is not the author's own prose, keeping line numbers intact.

    Dropped: YAML frontmatter, fenced code, blockquotes, and any block between
    `<!-- prosto-lint: off -->` and `<!-- prosto-lint: on -->`. A document that
    teaches these rules quotes bad wording on purpose and marks it off.
    """
    lines = text.split("\n")
    front = 0
    if lines and lines[0].strip() == "---":
        for number in range(1, len(lines)):
            if lines[number].strip() == "---":
                front = number + 1
                break
    out, fenced, off = [], False, False
    for number, line in enumerate(lines):
        stripped = line.strip()
        if OFF_RE.search(stripped):
            off = True
        elif ON_RE.search(stripped):
            off = False
            out.append("")
            continue
        if FENCE_RE.match(stripped):
            fenced = not fenced
            out.append("")
            continue
        ignored = number < front or fenced or off or stripped.startswith(">")
        out.append("" if ignored else line)
    return "\n".join(out)


def line_starts(text):
    starts = [0]
    for match in re.finditer("\n", text):
        starts.append(match.end())
    return starts


def iter_sentences(text):
    """(sentence, offset, words) for every sentence, list markers removed."""
    out = []
    for line, line_start in zip(text.split("\n"), line_starts(text)):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        body = LIST_MARKER_RE.sub("", stripped)
        shift = line_start + line.index(stripped) + (len(stripped) - len(body))
        start = 0
        for match in SENTENCE_END_RE.finditer(body):
            chunk = body[start:match.end()]
            if chunk.strip():
                out.append((chunk.strip(), shift + start, WORD_RE.findall(chunk)))
            start = match.end()
        tail = body[start:]
        if tail.strip():
            out.append((tail.strip(), shift + start, WORD_RE.findall(tail)))
    return out


# --- checks ----------------------------------------------------------------


def make_finding(starts, pos, rule_name, match, message=None):
    line = bisect.bisect_right(starts, pos)
    return {
        "line": line,
        "col": pos - starts[line - 1] + 1,
        "rule": rule_name,
        "severity": SEVERITY[rule_name],
        "message": message or BY_NAME[rule_name].message,
        "match": match,
    }


def check_text_rules(text, starts):
    out = []
    for rule in RULES:
        if rule.scope != "text":
            continue
        for match in rule.regex.finditer(text):
            found = match.group(0)
            message = rule.message
            if rule.fixes:
                fix = rule.fixes.get(found.lower())
                message = f"{rule.message} Проще: «{fix}»." if fix else rule.message
            out.append(make_finding(starts, match.start(), rule.name, found, message))
    return out


def check_sentence_rules(starts, sentences):
    out = []
    vague = BY_NAME["vague-quantity"]
    for chunk, pos, chunk_words in sentences:
        count = len(chunk_words)
        if count > LONG_HARD:
            out.append(make_finding(starts, pos, "long-sentence", f"{count} слов"))
        elif count > LONG_SOFT:
            out.append(make_finding(starts, pos, "sentence-length", f"{count} слов"))
        if not DIGIT_RE.search(chunk):
            match = vague.regex.search(chunk)
            if match:
                out.append(make_finding(starts, pos + match.start(),
                                        "vague-quantity", match.group(0)))
        out += noun_chains(starts, chunk, pos)
    return out


def noun_chains(starts, chunk, pos):
    """Three or more chained nouns inside one sentence."""
    out, run, run_start = [], [], 0
    for match in list(WORD_RE.finditer(chunk)) + [None]:
        if match and CHAIN_WORD_RE.match(match.group(0)):
            if not run:
                run_start = match.start()
            run.append(match.group(0))
            continue
        if len(run) >= 3:
            out.append(make_finding(starts, pos + run_start, "noun-chain", " ".join(run)))
        run = []
    return out


def check_synonyms(text, starts):
    """One thing — one name, inside one file."""
    out = []
    low = text.lower()
    for group in SYNONYM_GROUPS:
        used = []
        for word in group:
            # Cut the ending so "папка" also matches "папке", "папки".
            stem = re.sub(r"[аяьоыие]$", "", word)
            match = re.search(rf"(?<![^\W\d_]){re.escape(stem)}", low)
            if match:
                used.append((word, match.start()))
        if len(used) >= 2:
            out.append(make_finding(starts, used[1][1], "synonym-rotation",
                                    " / ".join(word for word, _ in used)))
    return out


def lint(text, disabled=()):
    clean = strip_ignored(text)
    starts = line_starts(clean)
    sentences = iter_sentences(clean)
    found = (check_text_rules(clean, starts)
             + check_sentence_rules(starts, sentences)
             + check_synonyms(clean, starts))
    found = [f for f in found if f["rule"] not in disabled]
    found.sort(key=lambda f: (f["line"], f["col"], f["rule"]))
    return found, sum(len(w) for _, _, w in sentences)


# --- command line ----------------------------------------------------------


def non_negative_int(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"нужно целое число, а не {value!r}")
    if number < 0:
        raise argparse.ArgumentTypeError("не может быть отрицательным")
    return number


def rule_list(value):
    names = {name.strip() for name in value.split(",") if name.strip()}
    unknown = names - set(SEVERITY)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"неизвестные правила: {', '.join(sorted(unknown))}")
    return names


def print_rules():
    width = max(len(name) for name in SEVERITY)
    print(f"{'правило':<{width}}  строгость  что проверяет")
    for rule in RULES:
        kind = "жёсткое" if rule.severity == HARD else "совет  "
        print(f"{rule.name:<{width}}  {kind}    {rule.message}")
    print()
    print("Жёсткие роняют проверку, советы — нет.")
    print("Смягчения — «возможно», «похоже», «предполагаю» — не помечаются никогда.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Проверка русского текста по правилам скилла prosto.",
        allow_abbrev=False)
    parser.add_argument("paths", nargs="*", help="файлы; без них читается stdin")
    parser.add_argument("--json", action="store_true", help="машинный вывод")
    parser.add_argument("--baseline", type=non_negative_int, default=0,
                        help="сколько жёстких находок терпеть (по умолчанию 0)")
    parser.add_argument("--disable", type=rule_list, default=set(),
                        help="выключить правила через запятую")
    parser.add_argument("--rules", action="store_true", help="показать таблицу правил")
    parser.add_argument("--selftest", action="store_true", help="самопроверка")
    args = parser.parse_intermixed_args(argv)

    if args.rules:
        return print_rules()
    if args.selftest:
        return selftest()

    sources = []
    try:
        for path in args.paths:
            with open(path, encoding="utf-8") as handle:
                sources.append((path, handle.read()))
    except OSError as error:
        parser.error(f"не читается {error.filename}: {error.strerror}")
    if not args.paths:
        sources.append(("<stdin>", sys.stdin.read()))

    findings, hard, soft, total_words = [], 0, 0, 0
    for name, text in sources:
        file_findings, count = lint(text, args.disable)
        total_words += count
        for item in file_findings:
            item["file"] = name
            findings.append(item)
            if item["severity"] == HARD:
                hard += 1
            else:
                soft += 1

    if args.json:
        print(json.dumps({"findings": findings, "words": total_words,
                          "hard": hard, "soft": soft, "baseline": args.baseline},
                         ensure_ascii=False, indent=2))
    else:
        for item in findings:
            print(f"{item['file']}:{item['line']}:{item['col']} {item['rule']}: "
                  f"{item['message']} [{item['match']}]")
        print()
        print(f"{hard + soft} находок ({hard} жёстких, допуск {args.baseline}; "
              f"{soft} советов), {total_words} слов")
        print("«Возможно», «похоже», «предполагаю» не помечаются: это содержание, а не стиль.")
    return 1 if hard > args.baseline else 0


# --- self-check ------------------------------------------------------------


def rules_in(text, **kwargs):
    return {f["rule"] for f in lint(text, **kwargs)[0]}


def selftest():
    assert "kantselyarit" in rules_in("В целях проверки я открыл файл.")
    assert "kantselyarit" in rules_in("Данная папка пуста."), "женский род «данный»"
    assert "kantselyarit" not in rules_in("Я выгрузил данные из базы."), "«данные» — не канцелярит"
    assert "marketing" in rules_in("Это мощный инструмент.")
    assert "metaphor" in rules_in("Работает из коробки.")

    # Guillemets must not hide a hard rule: that was an escape hatch, now closed.
    assert "marketing" in rules_in("Это «мощный» инструмент."), "кавычки не прячут правило"
    # A table row is ordinary prose: the "Стало" column has to be graded too.
    assert "kantselyarit" in rules_in("| Правило | в целях проверки | чтобы проверить |")

    # Only an explicit marker exempts a block.
    off = ("<!-- prosto-lint: off -->\nВ целях проверки мощный инструмент.\n"
           "<!-- prosto-lint: on -->\nЯ проверил файл.")
    assert not lint(off)[0], lint(off)[0]

    # Frontmatter, fenced code and blockquotes are not the author's prose.
    for ignored in ("---\ndescription: в целях проверки\n---\nЯ проверил файл.",
                    "```\nв целях проверки\n```\nЯ проверил файл.",
                    "> в целях проверки\nЯ проверил файл."):
        assert not lint(ignored)[0], ignored

    # Hedges are content, not style: never flagged.
    assert not lint("Возможно, сервер не ответил. Я не уверен.")[0]
    assert not lint("Я перенёс два файла в папку проекта.")[0]

    # A number satisfies the vague-quantity rule.
    assert "vague-quantity" in rules_in("В папке много заметок.")
    assert "vague-quantity" not in rules_in("В папке 69 заметок.")

    # One principle for severity: literal list — hard, morphology guess — advisory.
    assert SEVERITY["nominalization"] == SOFT
    assert SEVERITY["kantselyarit"] == HARD
    assert {f["severity"] for f in lint("Мне выдали производственное задание.")[0]} <= {SOFT}

    # Two sentence-length rules instead of one rule with two severities.
    assert "sentence-length" in rules_in("Я " + "очень " * 16 + "устал.")
    assert "long-sentence" in rules_in("Я " + "очень " * 25 + "устал.")

    # Chained nouns, and two names for one thing.
    assert "noun-chain" in rules_in("Повышение качества обслуживания клиентов компании.")
    assert "synonym-rotation" in rules_in("Я открыл файл и положил документ в папку.")

    # Disabling a rule silences it.
    assert "kantselyarit" not in rules_in("В целях проверки я открыл файл.",
                                          disabled={"kantselyarit"})

    with tempfile.TemporaryDirectory() as tmp:
        paths = [os.path.join(tmp, name) for name in ("a.md", "b.md")]
        for path in paths:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("Я проверил файл.\n")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            # Flags may appear between file paths.
            assert main([paths[0], "--json", paths[1]]) == 0
        payload = json.loads(out.getvalue())
        assert payload["words"] == 6, payload
        assert payload["baseline"] == 0, payload

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert main(["--rules"]) == 0
        assert "kantselyarit" in out.getvalue()

    # A bad command line or an unreadable file exits 2, not 1.
    for bad in (["--baseline", "-1"], ["--baseline", "x"], ["--disable", "нетакого"],
                ["--base", "3"], [os.path.join(tempfile.gettempdir(), "нет-такого.md")]):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                main(bad)
        except SystemExit as exit_code:
            assert exit_code.code == 2, (bad, exit_code.code)
        else:
            raise AssertionError(f"не отклонено: {bad}")

    print("selftest OK")
    return 0


if __name__ == "__main__":
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    sys.exit(main())
