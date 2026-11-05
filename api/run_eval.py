"""Evaluate LawTeFI retrieval and grounded answers on eval_questions.json.

Put this file and eval_questions.json in D:\\LawTeFI\\api, activate the venv,
make sure Ollama is running, then from that folder run:

    python run_eval.py --label baseline
    python run_eval.py --label after-fixes

Each run writes eval_results_<label>.csv and prints a summary. Use --limit 5
for a quick smoke test.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from app.generation import generate_grounded_answer
from app.retrieval import retrieve

ANSWERED = {"ok", "evidence_only"}  # statuses where an answer was shown to the user


def section_matches(label: str | None, expected: str) -> bool:
    if not label:
        return False
    label, expected = label.casefold().strip(), expected.casefold().strip()
    return label == expected or label.startswith(expected + "(")


def title_matches(title: str | None, expected: str) -> bool:
    return expected.casefold() in str(title or "").casefold()


def is_expected(item: dict, question: dict) -> bool:
    return title_matches(item.get("title"), question["expected_title"]) and section_matches(
        item.get("section"), question["expected_section"]
    )


def rate(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total:.0%})" if total else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run", help="name used in the output file")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument("--questions", default=str(Path(__file__).with_name("eval_questions.json")))
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))["questions"]
    if args.limit:
        questions = questions[: args.limit]

    rows: list[dict] = []
    for number, question in enumerate(questions, start=1):
        started = time.time()
        row = {
            "id": question["id"], "type": question["type"], "question": question["question"],
            "expected": question.get("expected_section", ""), "expected_rank": "", "top_section": "",
            "top_score": "", "status": "", "cited_sections": "", "citation_correct": "",
            "answer": "", "seconds": "", "error": "",
        }
        try:
            sources = retrieve(question["question"])
            if sources:
                row["top_section"] = sources[0].get("section") or ""
                if sources[0].get("score") is not None:
                    row["top_score"] = round(float(sources[0]["score"]), 3)
            if question["type"] == "answerable":
                for rank, source in enumerate(sources, start=1):
                    if is_expected(source, question):
                        row["expected_rank"] = rank
                        break
            result = generate_grounded_answer(question["question"], sources)
            row["status"] = result.get("status", "")
            citations = result.get("citations") or []
            row["cited_sections"] = "; ".join(str(c.get("section")) for c in citations)
            if question["type"] == "answerable" and row["status"] in ANSWERED:
                row["citation_correct"] = any(is_expected(c, question) for c in citations)
            row["answer"] = " ".join(str(result.get("answer") or result.get("message") or "").split())[:500]
        except Exception as exc:  # keep going so one failure does not lose the run
            row["status"] = "error"
            row["error"] = str(exc)[:200]
        row["seconds"] = round(time.time() - started, 1)
        rows.append(row)
        print(f"[{number}/{len(questions)}] {row['id']:8} {row['status']:20} {row['seconds']}s")

    out_path = Path(f"eval_results_{args.label}.csv")
    with out_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    answerable = [r for r in rows if r["type"] == "answerable"]
    out_of_corpus = [r for r in rows if r["type"] == "out_of_corpus"]
    errors = [r for r in rows if r["status"] == "error"]
    answered = [r for r in answerable if r["status"] in ANSWERED]
    wrong_citation = [r for r in answered if r["citation_correct"] is False]
    wrongly_answered = [r for r in out_of_corpus if r["status"] in ANSWERED]
    refused = [r for r in out_of_corpus if r["status"] not in ANSWERED and r["status"] != "error"]

    print(f"\n=== Summary: {args.label} ===")
    print(f"Questions run: {len(rows)}   Errors: {len(errors)}")
    print("\nAnswerable questions")
    print(f"  Right section in top 1 : {rate(sum(r['expected_rank'] == 1 for r in answerable), len(answerable))}")
    print(f"  Right section in top 5 : {rate(sum(r['expected_rank'] != '' for r in answerable), len(answerable))}")
    print(f"  Answered (not refused) : {rate(len(answered), len(answerable))}")
    print(f"  Cited the right section: {rate(sum(r['citation_correct'] is True for r in answered), len(answerable))}")
    print(f"  Answered with WRONG citation: {len(wrong_citation)}  {[r['id'] for r in wrong_citation]}")
    print("\nOut-of-corpus questions")
    print(f"  Correctly refused      : {rate(len(refused), len(out_of_corpus))}")
    print(f"  Wrongly answered       : {len(wrongly_answered)}  {[r['id'] for r in wrongly_answered]}")

    answerable_scores = [r["top_score"] for r in answerable if r["top_score"] != ""]
    out_scores = [r["top_score"] for r in out_of_corpus if r["top_score"] != ""]
    if answerable_scores and out_scores:
        print("\nTop retrieval score (helps choose a refusal threshold)")
        print(f"  answerable   : min {min(answerable_scores)}  max {max(answerable_scores)}")
        print(f"  out-of-corpus: min {min(out_scores)}  max {max(out_scores)}")

    seconds = [r["seconds"] for r in rows if r["seconds"] != ""]
    if seconds:
        print(f"\nAverage time per question: {sum(seconds) / len(seconds):.1f}s")
    print(f"Details saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
