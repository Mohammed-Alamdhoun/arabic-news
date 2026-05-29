#!/usr/bin/env python3
"""
fix_articles_deepseek.py
Use Google Gemini 2.5 Flash-Lite via OpenRouter to fix Arabic word-boundary
errors in title + body of articles_2026_05.jsonl.

Processes BATCH_SIZE articles in parallel, waits for the full batch to
finish, then writes them in original order before moving to the next batch.
Resume logic is correct: output line count maps exactly to input line count.

Output: data/raw/articles_2026_05_fixed.jsonl
"""

import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from pathlib import Path

import requests

# Signals all threads to stop (e.g. on 402 credits exhausted)
_stop_event = threading.Event()

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
INPUT_PATH  = PROJECT_DIR / "data" / "raw" / "articles_2026_05.jsonl"
OUTPUT_PATH = PROJECT_DIR / "data" / "raw" / "articles_2026_05_fixed.jsonl"

# ── API ────────────────────────────────────────────────────────────────────────
# Read the key from the environment so secrets never live in the repo.
# Set it before running, e.g.:  export OPENROUTER_API_KEY="sk-or-v1-..."
API_KEY    = os.environ.get("OPENROUTER_API_KEY", "")
API_URL    = "https://openrouter.ai/api/v1/chat/completions"
MODEL      = "google/gemini-2.5-flash-lite"
BATCH_SIZE = 5

PRICE_IN  = 0.10 / 1_000_000   # $0.10 per 1M input tokens
PRICE_OUT = 0.40 / 1_000_000   # $0.40 per 1M output tokens

RETRY_DELAYS = (5, 15, 40)

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
أنت خبير متخصص في المعالجة المسبقة للنصوص العربية (NLP Pre-processing).
مهمتك الوحيدة: تصحيح أخطاء حدود الكلمات الناتجة عن نموذج تقطيع آلي معيب .

القواعد الصارمة التي لا استثناء منها إطلاقاً:
١. يُحظر حذف أي كلمة أو جملة أو جزء من النص مهما كان السبب.
٢. يُحظر إعادة صياغة الجمل أو تغيير المرادفات أو الأسلوب أو إضافة أي كلمة جديدة.
٣. يُحظر تغيير علامات الترقيم أو الأرقام أو الرموز الواردة في النص الأصلي.
٤. النوع الأول من الأخطاء — التجزئة (Fragmentation): مسافة زائدة أُدخلت داخل كلمة واحدة تُقسّمها، يجب دمج الأجزاء لتُشكّل كلمة صحيحة.
٥. النوع الثاني من الأخطاء — الدمج (Merging): غياب مسافة بين كلمتين مستقلتين ملتصقتين بلا فاصل، يجب إدراج المسافة الصحيحة بينهما. هذا النوع شائع جداً في النصوص المستخرجة من HTML حيث تلتصق الكلمة الأخيرة من رابط بالكلمة الأولى التالية له.
٦. يجب تصحيح كل حالة دمج في النص بدون استثناء، حتى لو تكررت عشرات المرات.
٧. أخرج النص المصحح داخل وسوم XML المطلوبة فقط، بلا مقدمات ولا شروح.\
"""

USER_TEMPLATE = """\
فيما يلي أمثلة شاملة على نوعَي الأخطاء المطلوب تصحيحها:

  "الع المية"    ->  "العالمية"          "الاو ورب ية"  ->  "الأوروبية"
  "المت فاقم"    ->  "المتفاقم"          "مفاوض اتها"   ->  "مفاوضاتها"
  "وت داعي اته"  ->  "وتداعياته"         "بال فيديو"    ->  "بالفيديو"
  "مستويات ها"   ->  "مستوياتها"         "باحتجاج ات"   ->  "باحتجاجات"
  "الجيوسيا سية" ->  "الجيوسياسية"       "لبضائ عهم"    ->  "لبضائعهم"


  "الإسرائيليفيجنوب لبنان"    ->  "الإسرائيلي في جنوب لبنان"
  "استخدامالطائرات المسيرة"    ->  "استخدام الطائرات المسيرة"
  "مزودةبكابل رفيع يمتد"       ->  "مزودة بكابل رفيع يمتد"
  "صحيفةنيويورك تايمز"         ->  "صحيفة نيويورك تايمز"
  "حلفاءإيران الآخرين"         ->  "حلفاء إيران الآخرين"
  "وكالةأسوشيتد برس"           ->  "وكالة أسوشيتد برس"
  "قتلىمؤخرا"                  ->  "قتلى مؤخرا"
  "الجيشالإسرائيلي"            ->  "الجيش الإسرائيلي"
  "حزبالله"                    ->  "حزب الله"
  "منظمةالأمم المتحدة"         ->  "منظمة الأمم المتحدة"
  "رئيسالوزراء"                ->  "رئيس الوزراء"
  "وزيرالخارجية"               ->  "وزير الخارجية"

  "أ وروج واي"         ->  "أوروغواي"
  "تاكايت شي"          ->  "تاكايتشي"
  "فايننش ال"          ->  "فايننشال"
  "بطرس بورغ"          ->  "بطرسبورغ"

النصوص المراد تصحيحها:
<TITLE>TITLE_PLACEHOLDER</TITLE>
<BODY>BODY_PLACEHOLDER</BODY>

أخرج النتيجة بهذه الصيغة بالضبط ولا شيء آخر:
<TITLE>العنوان المصحح هنا</TITLE>
<BODY>المتن المصحح هنا</BODY>\
"""

_RE_TITLE = re.compile(r"<TITLE>(.*?)</TITLE>", re.DOTALL)
_RE_BODY  = re.compile(r"<BODY>(.*?)</BODY>",   re.DOTALL)


# ── API call (one record) ──────────────────────────────────────────────────────

def fix_record(record: dict) -> tuple[dict, int, int, int]:
    """Fix title + body in one API call.
    Returns (fixed_record, in_tokens, out_tokens, fail_count).
    """
    title = (record.get("title") or "").strip()
    body  = (record.get("body")  or "").strip()

    if not title and not body:
        return record, 0, 0, 0

    # Use placeholder replacement to safely handle { } chars in article text
    user_msg = (USER_TEMPLATE
                .replace("TITLE_PLACEHOLDER", title)
                .replace("BODY_PLACEHOLDER",  body))

    if not API_KEY:
        sys.exit("OPENROUTER_API_KEY is not set. Run: export OPENROUTER_API_KEY='sk-or-v1-...'")

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model":    MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens":  4096,
    }

    for attempt, delay in enumerate(RETRY_DELAYS, 1):
        if _stop_event.is_set():
            return record, 0, 0, 1

        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)

            if resp.status_code == 200:
                data    = resp.json()
                content = data["choices"][0]["message"]["content"] or ""
                usage   = data.get("usage", {})
                in_tok  = usage.get("prompt_tokens", 0)
                out_tok = usage.get("completion_tokens", 0)

                title_m = _RE_TITLE.search(content)
                body_m  = _RE_BODY.search(content)
                fail    = 0

                if title_m:
                    record["title"] = title_m.group(1).strip()
                elif title:
                    fail += 1

                if body_m:
                    record["body"] = body_m.group(1).strip()
                elif body:
                    fail += 1

                return record, in_tok, out_tok, fail

            elif resp.status_code == 402:
                print("\n  [402] Credits exhausted — stopping. Progress saved.")
                _stop_event.set()
                return record, 0, 0, 1

            elif resp.status_code == 429:
                print(f"\n  [rate-limit] waiting {delay}s ...")
                time.sleep(delay)

            else:
                print(f"\n  [HTTP {resp.status_code}] {resp.text[:150]} — retry {attempt}")
                time.sleep(delay)

        except requests.exceptions.Timeout:
            print(f"\n  [timeout] retry {attempt} after {delay}s ...")
            time.sleep(delay)
        except requests.RequestException as exc:
            print(f"\n  [error] {exc} — retry {attempt}")
            time.sleep(delay)

    return record, 0, 0, 1


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Resume ────────────────────────────────────────────────────────────────
    done = 0
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            done = sum(1 for line in f if line.strip())

    records = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total     = len(records)
    remaining = total - done

    print(f"Input  : {INPUT_PATH.name}  ({total:,} records)")
    print(f"Output : {OUTPUT_PATH.name}")
    print(f"Model  : {MODEL}  (batch={BATCH_SIZE} parallel)")
    print(f"Already done : {done:,}   Remaining : {remaining:,}")
    print(f"Est. cost    : ${remaining * 1768 * PRICE_IN + remaining * 668 * PRICE_OUT:.2f}")
    print()

    if remaining == 0:
        print("All records already processed.")
        return

    total_in_tok  = 0
    total_out_tok = 0
    total_failed  = 0
    n_done        = done
    todo          = records[done:]

    try:
        with (
            open(OUTPUT_PATH, "a", encoding="utf-8") as fout,
            ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool,
        ):
            for batch_start in range(0, len(todo), BATCH_SIZE):
                if _stop_event.is_set():
                    print("\n  Stopping — credits exhausted. Progress saved.")
                    break

                batch   = todo[batch_start : batch_start + BATCH_SIZE]
                futures = [pool.submit(fix_record, rec) for rec in batch]
                wait(futures, return_when=ALL_COMPLETED)

                for future in futures:
                    rec, in_tok, out_tok, fail = future.result()
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_in_tok  += in_tok
                    total_out_tok += out_tok
                    total_failed  += fail
                    n_done        += 1

                fout.flush()

                cost = total_in_tok * PRICE_IN + total_out_tok * PRICE_OUT
                pct  = 100.0 * n_done / total
                print(
                    f"  {n_done:>6}/{total}  ({pct:5.1f}%)"
                    f"  failed={total_failed}"
                    f"  tokens={total_in_tok + total_out_tok:,}"
                    f"  cost=${cost:.4f}",
                    end="\r", flush=True,
                )

    except KeyboardInterrupt:
        print("\n\nInterrupted — progress saved. Re-run to continue.")
        sys.exit(0)

    print()
    total_cost = total_in_tok * PRICE_IN + total_out_tok * PRICE_OUT
    print(f"\nDone.")
    print(f"  Total records      : {total:,}")
    print(f"  Failed (kept orig) : {total_failed}")
    print(f"  Input  tokens      : {total_in_tok:,}")
    print(f"  Output tokens      : {total_out_tok:,}")
    print(f"  Total cost         : ${total_cost:.4f}")
    print(f"  Output → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
