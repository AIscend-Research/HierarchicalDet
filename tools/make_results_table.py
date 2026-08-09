"""
Assemble the reproduction's headline comparison table: HierarchicalDet against
the baselines, per detection tier, next to the numbers the paper reports.

Inputs are the JSON files the other tools already write:
  --hierarchical   tier_metrics.json from tools/evaluate_tiers.py
  --baseline       NAME=metrics.json, repeatable, from tools/coco_eval_standalone.py
  --paper          a JSON of the paper's reported numbers

The paper column is deliberately not baked into this script. Someone has to
read the numbers out of the paper and write them into a JSON file, e.g.

  {"HierarchicalDet": {"2": {"AP50": 00.0}}, "DiffusionDet": {"2": {"AP50": 00.0}}}

keyed by method, then tier index, then metric. Hardcoding remembered values
into reproduction tooling is how wrong reference numbers end up in a report.
"""
import argparse
import json
import os

from repro_common import TIERS  # noqa: E402

METRICS = ["AP", "AP50", "AP75"]


def load(path):
    with open(path) as f:
        return json.load(f)


def hierarchical_rows(path):
    payload = load(path)
    rows = {}
    for tier, result in payload["results_by_tier"].items():
        box = result.get("bbox", {})
        rows[str(tier)] = {metric: box.get(metric) for metric in METRICS}
        rows[str(tier)]["per_class_AP"] = {
            k[len("AP-"):]: v for k, v in box.items() if k.startswith("AP-")}
    return payload.get("label", "HierarchicalDet"), rows


def baseline_rows(path):
    payload = load(path)
    rows = {}
    for tier, result in payload["results_by_tier"].items():
        rows[str(tier)] = {metric: result.get(metric) for metric in METRICS}
        rows[str(tier)]["per_class_AP"] = result.get("per_class_AP", {})
    return rows


def fmt(value):
    return "{:.2f}".format(value) if isinstance(value, (int, float)) else "n/a"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hierarchical", required=True, help="tier_metrics.json")
    p.add_argument("--hierarchical-standalone", default=None,
                   help="optional: the same checkpoint scored by tools/coco_eval_standalone.py, "
                        "to show the repo's forked evaluator and a stock COCOeval side by side")
    p.add_argument("--baseline", action="append", default=[], metavar="NAME=PATH",
                   help="e.g. --baseline RetinaNet=out/retinanet/metrics.json (repeatable)")
    p.add_argument("--paper", default=None, help="JSON of the paper's reported numbers")
    p.add_argument("--output", required=True, help="Markdown file to write")
    args = p.parse_args()

    label, hierarchical = hierarchical_rows(args.hierarchical)
    methods = {label: hierarchical}
    if args.hierarchical_standalone:
        methods["{} (stock COCOeval)".format(label)] = baseline_rows(args.hierarchical_standalone)
    for entry in args.baseline:
        name, _, path = entry.partition("=")
        if not path:
            raise SystemExit("--baseline expects NAME=PATH, got {!r}".format(entry))
        methods[name] = baseline_rows(path)

    paper = load(args.paper) if args.paper else {}

    lines = ["# Reproduction results", "",
             "All numbers are box AP in percent on the same evaluation split.",
             "The **paper** column is what the original work reports; a blank means the",
             "paper does not report that combination.", ""]

    for tier in sorted({t for rows in methods.values() for t in rows}):
        lines += ["## Tier {} -- {}".format(tier, TIERS[int(tier)][1]), "",
                  "| Method | AP | AP50 | AP75 | paper AP50 |", "|---|---|---|---|---|"]
        for name, rows in methods.items():
            row = rows.get(tier, {})
            paper_value = paper.get(name.split(" (")[0], {}).get(tier, {}).get("AP50")
            lines.append("| {} | {} | {} | {} | {} |".format(
                name, fmt(row.get("AP")), fmt(row.get("AP50")), fmt(row.get("AP75")), fmt(paper_value)))
        lines.append("")

        class_names = sorted({
            c for rows in methods.values() for c in rows.get(tier, {}).get("per_class_AP", {})})
        if class_names:
            lines += ["### Per-class AP", "",
                      "| Method | " + " | ".join(class_names) + " |",
                      "|---" * (len(class_names) + 1) + "|"]
            for name, rows in methods.items():
                per_class = rows.get(tier, {}).get("per_class_AP", {})
                lines.append("| {} | {} |".format(
                    name, " | ".join(fmt(per_class.get(c)) for c in class_names)))
            lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote {}".format(args.output))


if __name__ == "__main__":
    main()
