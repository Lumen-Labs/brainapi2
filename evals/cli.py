from __future__ import annotations

import argparse
from typing import Any

from rich.console import Console
from rich.table import Table

from evals.config import OPTIONAL_USE_CASES, REQUIRED_USE_CASES, Settings, USE_CASES
from evals.datasets.loader import load_use_case
from evals.evaluate import evaluate, judge_agreement
from evals.report import load_report, render_markdown, write_report

console = Console()


def _parse_list(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return list(default)
    return [part.strip() for part in raw.split(",") if part.strip()]


def cmd_list_datasets(_args: argparse.Namespace, settings: Settings) -> int:
    table = Table(title="Use-case datasets")
    for col in ("use_case", "docs", "smoke", "dev", "test", "gold_pairs"):
        table.add_column(col)
    for name in USE_CASES:
        bundle = load_use_case(name, settings.datasets_dir)
        q = bundle["questions"]
        table.add_row(
            name,
            str(len(bundle["corpus"])),
            str(len(q["smoke"])),
            str(len(q["dev"])),
            str(len(q["test"])),
            str(len(bundle["gold_pairs"])),
        )
    console.print(table)
    return 0


def cmd_selftest(_args: argparse.Namespace, settings: Settings) -> int:
    from evals.baseline.fusion import reciprocal_rank_fusion, weighted_fusion
    from evals.baseline.dates import in_force
    from evals.metrics import conflict_prf, exact_match, must_include

    assert exact_match("Net 30 days", "net 30 days")
    assert must_include("Inspection moved from 10 days to 5 days", ["10", "5"])
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    assert fused[0][0] in {"a", "b"}
    weighted = weighted_fusion([("a", 10.0), ("b", 1.0)], [("b", 5.0), ("a", 1.0)], dense_weight=0.9)
    assert weighted[0][0] == "b"
    assert in_force({"valid_from": "2020-01-01", "valid_to": "2020-12-31"}, "2020-06-01")
    assert not in_force({"valid_from": "2020-01-01", "valid_to": "2020-12-31"}, "2021-01-01")
    prf = conflict_prf([("x", "y")], [("y", "x")])
    assert prf["f1"] == 1.0
    agreement = judge_agreement(settings)
    assert agreement["n"] >= 8
    console.print("[green]selftest ok[/green]")
    return 0


def cmd_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    use_cases = _parse_list(args.use_case, list(REQUIRED_USE_CASES))
    if args.include_optional:
        for extra in OPTIONAL_USE_CASES:
            if extra not in use_cases:
                use_cases.append(extra)
    systems = _parse_list(args.systems, ["hybrid"])
    unknown = [name for name in use_cases if name not in USE_CASES]
    if unknown:
        raise SystemExit(f"Unknown use cases: {unknown}")
    report = evaluate(
        settings,
        use_cases=use_cases,
        split=args.split,
        systems=systems,
        run_id=args.run,
        answerer=args.answerer,
        tune=not args.no_tune,
        smoke=args.smoke or args.split == "smoke",
        use_judge=args.judge,
        skip_enrichment=True
        if (args.smoke or args.split == "smoke")
        else bool(getattr(args, "skip_enrichment", False)),
    )
    run_dir = settings.runs_dir / report["run_id"]
    md_path = write_report(report, run_dir)
    console.print(f"[green]JSON[/green] {run_dir / 'report.json'}")
    console.print(f"[green]Markdown[/green] {md_path}")
    if not report.get("brainapi_attempted") and "brainapi" in systems:
        console.print(
            "[yellow]BrainAPI arm skipped (no BRAINPAT_TOKEN / EVAL_API_KEY). "
            "Hybrid numbers only; scale remains local/small-scale.[/yellow]"
        )
    _print_summary(report)
    return 0


def cmd_smoke(args: argparse.Namespace, settings: Settings) -> int:
    args.split = "smoke"
    args.smoke = True
    args.use_case = args.use_case or ",".join(
        list(REQUIRED_USE_CASES) + list(OPTIONAL_USE_CASES)
    )
    args.systems = args.systems or "hybrid,brainapi"
    args.no_tune = True
    args.judge = False
    args.skip_enrichment = True
    args.include_optional = True
    args.run = args.run or "local-smoke"
    settings.scale = "local/small-scale"
    return cmd_evaluate(args, settings)


def cmd_report(args: argparse.Namespace, settings: Settings) -> int:
    report = load_report(settings, args.run)
    run_dir = settings.runs_dir / args.run
    path = write_report(report, run_dir)
    console.print(render_markdown(report))
    console.print(f"[dim]rewrote {path}[/dim]")
    return 0


def cmd_download_far(args: argparse.Namespace, _settings: Settings) -> int:
    from evals.datasets.versioned_rules.download_far import main as download_main
    import sys

    sys.argv = ["download_far.py", *(["--out", str(args.out)] if args.out else [])]
    return download_main()


def _print_summary(report: dict[str, Any]) -> None:
    table = Table(title=f"eval {report.get('run_id')} ({report.get('scale')})")
    for col in ("use_case", "system", "accuracy", "p50", "verdict"):
        table.add_column(col)
    for use in report.get("use_cases") or []:
        verdict = (use.get("comparison") or {}).get("verdict", "")
        for sys_name, payload in (use.get("systems") or {}).items():
            metrics = payload.get("metrics") or {}
            lat = metrics.get("latency") or {}
            acc = metrics.get("accuracy")
            acc_s = "n/a" if acc is None else f"{100*acc:.1f}%"
            n_degraded = int(metrics.get("n_degraded") or 0)
            if n_degraded:
                acc_s += f" degraded={n_degraded}"
            extra = metrics.get("conflict_pairs")
            if extra:
                acc_s += f" pairF1={extra.get('f1', 0):.2f}"
            p50 = lat.get("p50_ms")
            table.add_row(
                str(use.get("use_case")),
                sys_name,
                acc_s,
                "n/a" if p50 is None else f"{p50:.0f}ms",
                verdict,
            )
    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evals",
        description="BrainAPI vs tuned hybrid-search baseline, per use case.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-datasets", help="Show fixture sizes and splits")
    sub.add_parser("selftest", help="Pure-Python sanity checks (no BrainAPI)")

    ev = sub.add_parser("evaluate", help="Run one or more use cases")
    ev.add_argument("--use-case", help="Comma-separated use cases")
    ev.add_argument("--split", default="test", choices=("smoke", "dev", "test"))
    ev.add_argument("--systems", default="hybrid", help="hybrid,brainapi")
    ev.add_argument("--run", help="Run id (directory under evals/runs)")
    ev.add_argument("--answerer", default="extractive", choices=("extractive", "llm"))
    ev.add_argument("--no-tune", action="store_true")
    ev.add_argument("--judge", action="store_true")
    ev.add_argument(
        "--skip-enrichment",
        action="store_true",
        default=False,
        help="CI smoke only. Real BrainAPI comparisons run graph enrichment.",
    )
    ev.add_argument("--include-optional", action="store_true")
    ev.add_argument("--smoke", action="store_true")
    ev.set_defaults(func=cmd_evaluate)

    sm = sub.add_parser("smoke", help="Tiny local/small-scale run of all fixtures")
    sm.add_argument("--use-case")
    sm.add_argument("--systems", default="hybrid,brainapi")
    sm.add_argument("--run", default="local-smoke")
    sm.add_argument("--answerer", default="extractive", choices=("extractive", "llm"))
    sm.add_argument("--skip-enrichment", action="store_true", default=True)
    sm.add_argument("--judge", action="store_true")
    sm.add_argument("--no-tune", action="store_true")
    sm.add_argument("--include-optional", action="store_true")
    sm.add_argument("--split", default="smoke")
    sm.add_argument("--smoke", action="store_true", default=True)
    sm.set_defaults(func=cmd_smoke)

    rp = sub.add_parser("report", help="Rebuild markdown from report.json")
    rp.add_argument("--run", required=True)
    rp.set_defaults(func=cmd_report)

    dl = sub.add_parser("download-far", help="Fetch public-domain FAR HTML")
    dl.add_argument("--out")
    dl.set_defaults(func=cmd_download_far)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.load()
    func = getattr(args, "func", None)
    if func is None:
        mapping = {
            "list-datasets": cmd_list_datasets,
            "selftest": cmd_selftest,
        }
        func = mapping[args.command]
    return func(args, settings)
