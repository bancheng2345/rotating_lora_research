#!/usr/bin/env python
"""Builds a self-contained progress PPTX without external pptx dependencies."""

from __future__ import annotations

import json
import math
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs" / "runs" / "meta_llama_llama_3_1_8b"
REPORT_DIR = ROOT / "reports"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    """Loads JSONL records when the file exists."""

    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_metric_record(run_dir: Path, metric_name: str) -> dict:
    """Returns the latest eval record for a specific metric name."""

    records = [
        row
        for row in load_jsonl(run_dir / "eval_metrics.jsonl")
        if row.get("metric_name") == metric_name and row.get("metric_value") is not None
    ]
    if not records:
        return {}
    return sorted(records, key=lambda row: int(row.get("step") or -1))[-1]


def mean_std(values: list[float]) -> tuple[float, float | None]:
    if not values:
        return math.nan, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def summaries_for(pattern: str) -> list[dict]:
    rows = []
    for path in RUN_ROOT.glob(pattern):
        summary_path = path / "final_summary.json"
        if summary_path.exists():
            data = load_json(summary_path)
            data["_run_dir"] = str(path.relative_to(ROOT))
            data["_mtime"] = summary_path.stat().st_mtime
            rows.append(data)
    return rows


def latest_by_seed(rows: Iterable[dict], *, method: str, params: int | None = None) -> list[dict]:
    selected: dict[int, dict] = {}
    for row in rows:
        if row.get("method") != method:
            continue
        if params is not None and (row.get("parameter_counts") or {}).get("trainable_parameters") != params:
            continue
        seed = row.get("seed")
        if seed not in {0, 1, 2}:
            continue
        if seed not in selected or row["_mtime"] > selected[seed]["_mtime"]:
            selected[seed] = row
    return [selected[s] for s in sorted(selected)]


def exact_values(rows: list[dict]) -> list[float]:
    return [float(row["final_eval_metric"]) for row in rows if row.get("final_eval_metric") is not None]


def loss_values(rows: list[dict]) -> list[float]:
    values = []
    for row in rows:
        loss = (row.get("last_eval") or {}).get("eval_loss")
        if loss is not None:
            values.append(float(loss))
    return values


def field_values(rows: list[dict], field: str) -> list[float]:
    """Collects finite numeric values from top-level summary fields."""

    values = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def last_eval_values(rows: list[dict], field: str) -> list[float]:
    """Collects finite numeric values from the final eval record."""

    values = []
    for row in rows:
        value = (row.get("last_eval") or {}).get(field)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def format_pm(values: list[float]) -> str:
    mean, std = mean_std(values)
    if math.isnan(mean):
        return "-"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def format_mean(values: list[float], digits: int = 4) -> str:
    """Formats a mean value or a dash when missing."""

    if not values:
        return "-"
    return f"{statistics.mean(values):.{digits}f}"


def format_time_hours(values: list[float]) -> str:
    """Formats average wall-clock training time in hours."""

    if not values:
        return "-"
    return f"{statistics.mean(values) / 3600:.2f}h"


def commonsense_task_table(params: int, methods: list[tuple[str, str]]) -> list[list[str]]:
    rows = summaries_for("*commonsense_170k_train_165420*")
    table = [["方法", "seeds", "Exact/EM ↑", "Eval loss ↓", "Token acc ↑", "参数"]]
    for label, method in methods:
        selected = latest_by_seed(rows, method=method, params=params)
        seeds = ",".join(str(row.get("seed")) for row in selected) or "-"
        em = exact_values(selected)
        losses = loss_values(selected)
        accuracies = last_eval_values(selected, "eval_accuracy")
        table.append(
            [
                label,
                seeds,
                format_pm(em),
                format_mean(losses, 4),
                format_mean(accuracies, 4),
                f"{params:,}",
            ]
        )
    return table


def commonsense_mechanism_table(params: int, methods: list[tuple[str, str]]) -> list[list[str]]:
    rows = summaries_for("*commonsense_170k_train_165420*")
    table = [["方法", "omega_full ↓", "omega_resid ↓", "orth_A ↓", "rho_k"]]
    for label, method in methods:
        selected = latest_by_seed(rows, method=method, params=params)
        rho_values = field_values(selected, "final_rho_k_mean")
        table.append(
            [
                label,
                format_mean(field_values(selected, "final_omega_full_mean"), 4),
                format_mean(field_values(selected, "final_omega_residual_mean"), 4),
                format_mean(field_values(selected, "final_orth_error_A"), 4),
                format_mean(rho_values, 4) if rho_values else "not logged",
            ]
        )
    return table


def commonsense_efficiency_table(params: int, methods: list[tuple[str, str]]) -> list[list[str]]:
    rows = summaries_for("*commonsense_170k_train_165420*")
    table = [["方法", "Train time", "Peak GPU GB", "参数"]]
    for label, method in methods:
        selected = latest_by_seed(rows, method=method, params=params)
        table.append(
            [
                label,
                format_time_hours(field_values(selected, "total_train_time_sec")),
                format_mean(field_values(selected, "peak_gpu_memory_gb"), 2),
                f"{params:,}",
            ]
        )
    return table


def commonsense_mechanism_efficiency_table(params: int, methods: list[tuple[str, str]]) -> list[list[str]]:
    rows = summaries_for("*commonsense_170k_train_165420*")
    table = [["方法", "omega_full", "omega_resid", "orth_A", "rho_k", "time", "GPU GB"]]
    for label, method in methods:
        selected = latest_by_seed(rows, method=method, params=params)
        rho_values = field_values(selected, "final_rho_k_mean")
        table.append(
            [
                label,
                format_mean(field_values(selected, "final_omega_full_mean"), 4),
                format_mean(field_values(selected, "final_omega_residual_mean"), 4),
                format_mean(field_values(selected, "final_orth_error_A"), 4),
                format_mean(rho_values, 4) if rho_values else "not logged",
                format_time_hours(field_values(selected, "total_train_time_sec")),
                format_mean(field_values(selected, "peak_gpu_memory_gb"), 2),
            ]
        )
    return table


def metamath_table() -> list[list[str]]:
    table = [["方法", "Final-answer EM ↑", "EM count", "PPL 辅助↓", "Token acc 辅助↑"]]
    for run_dir in sorted(RUN_ROOT.glob("*metamathqa_train_100000*")):
        summary_path = run_dir / "final_summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        if summary.get("max_steps") not in {300, None}:
            continue
        method = str(summary.get("method"))
        if method not in {"lora", "lora_nsc", "oplora", "pc_lora"}:
            continue
        em_record = latest_metric_record(run_dir, "exact_match")
        ppl_record = latest_metric_record(run_dir, "perplexity")
        record = em_record or ppl_record
        table.append(
            [
                method,
                f"{float(record.get('eval_exact_match') or summary.get('final_eval_metric') or 0.0):.4f}"
                if record
                else "-",
                f"{int(record.get('eval_exact_match_count'))}/{int(record.get('eval_num_samples') or record.get('num_samples') or 0)}"
                if record and record.get("eval_exact_match_count") is not None
                else "-",
                f"{float(record.get('perplexity')):.6f}"
                if record and record.get("perplexity") is not None
                else "-",
                f"{float(record.get('eval_token_accuracy')):.4f}"
                if record and record.get("eval_token_accuracy") is not None
                else "-",
            ]
        )
    order = {"oplora": 0, "lora_nsc": 1, "pc_lora": 2, "lora": 3}
    header, body = table[0], sorted(table[1:], key=lambda row: order.get(row[0], 99))
    return [header] + body


def metamath_mechanism_efficiency_table() -> list[list[str]]:
    table = [["方法", "omega_full ↓", "omega_resid ↓", "orth_A ↓", "time", "GPU GB"]]
    for run_dir in sorted(RUN_ROOT.glob("*metamathqa_train_100000*")):
        summary_path = run_dir / "final_summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        method = str(summary.get("method"))
        if method not in {"lora", "lora_nsc", "oplora", "pc_lora"}:
            continue
        if summary.get("max_steps") not in {300, None} and not (run_dir / "math_generation_eval.json").exists():
            continue
        table.append(
            [
                method,
                f"{float(summary.get('final_omega_full_mean')):.4f}"
                if summary.get("final_omega_full_mean") is not None
                else "-",
                f"{float(summary.get('final_omega_residual_mean')):.4f}"
                if summary.get("final_omega_residual_mean") is not None
                else "-",
                f"{float(summary.get('final_orth_error_A')):.4f}"
                if summary.get("final_orth_error_A") is not None
                else "-",
                f"{float(summary.get('total_train_time_sec')) / 3600:.2f}h"
                if summary.get("total_train_time_sec") is not None
                else "-",
                f"{float(summary.get('peak_gpu_memory_gb')):.2f}"
                if summary.get("peak_gpu_memory_gb") is not None
                else "-",
            ]
        )
    order = {"oplora": 0, "lora_nsc": 1, "pc_lora": 2, "lora": 3}
    header, body = table[0], sorted(table[1:], key=lambda row: order.get(row[0], 99))
    return [header] + body


def table_to_md(table: list[list[str]]) -> str:
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    lines = []
    lines.append("| " + " | ".join(table[0][i].ljust(widths[i]) for i in range(len(widths))) + " |")
    lines.append("| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |")
    for row in table[1:]:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(widths))) + " |")
    return "\n".join(lines)


EMU_PER_INCH = 914400


def emu(value: float) -> int:
    return int(value * EMU_PER_INCH)


def xml_text(text: str) -> str:
    return escape(text).replace("\n", " ")


@dataclass
class TextBox:
    x: float
    y: float
    w: float
    h: float
    lines: list[str]
    size: int = 18
    bold: bool = False
    mono: bool = False


def paragraph_xml(text: str, *, size: int, bold: bool, mono: bool) -> str:
    bullet = text.startswith("- ")
    clean = text[2:] if bullet else text
    ppr = '<a:pPr marL="285750" indent="-171450"><a:buChar char="•"/></a:pPr>' if bullet else "<a:pPr/>"
    typeface = "Consolas" if mono else "Microsoft YaHei"
    bold_attr = ' b="1"' if bold else ""
    return (
        f"<a:p>{ppr}<a:r><a:rPr lang=\"zh-CN\" sz=\"{size * 100}\"{bold_attr}>"
        f"<a:latin typeface=\"{typeface}\"/><a:ea typeface=\"Microsoft YaHei\"/>"
        f"</a:rPr><a:t>{xml_text(clean)}</a:t></a:r></a:p>"
    )


def textbox_xml(idx: int, box: TextBox) -> str:
    paragraphs = "".join(
        paragraph_xml(line, size=box.size, bold=box.bold, mono=box.mono) for line in box.lines
    )
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{idx}" name="TextBox {idx}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{emu(box.x)}" y="{emu(box.y)}"/><a:ext cx="{emu(box.w)}" cy="{emu(box.h)}"/></a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln>
  </p:spPr>
  <p:txBody><a:bodyPr wrap="square" lIns="45720" tIns="45720" rIns="45720" bIns="45720"/><a:lstStyle/>{paragraphs}</p:txBody>
</p:sp>
"""


def slide_xml(title: str, body: list[str], table: list[list[str]] | None = None) -> str:
    boxes = [
        TextBox(0.35, 0.25, 12.6, 0.7, [title], size=28, bold=True),
        TextBox(0.65, 1.15, 12.0, 4.4 if table else 5.6, body, size=18),
    ]
    if table:
        table_lines = ["  ".join(row) for row in table]
        boxes.append(TextBox(0.55, 5.15, 12.4, 1.9, table_lines, size=10, mono=True))
    shapes = "\n".join(textbox_xml(i + 2, box) for i, box in enumerate(boxes))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    {shapes}
  </p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def content_types(num_slides: int) -> str:
    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    overrides += [
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, num_slides + 1)
    ]
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        + "".join(overrides)
        + "</Types>"
    )


def write_pptx(slides: list[tuple[str, list[str], list[list[str]] | None]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types(len(slides)))
        zf.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>""",
        )
        zf.writestr(
            "docProps/core.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<dc:title>PC-LoRA Progress 20260513</dc:title><dc:creator>Codex</dc:creator></cp:coreProperties>""",
        )
        zf.writestr(
            "docProps/app.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
<Application>Codex</Application></Properties>""",
        )
        slide_ids = "".join(
            f'<p:sldId id="{255 + i}" r:id="rId{i}"/>' for i in range(1, len(slides) + 1)
        )
        zf.writestr(
            "ppt/presentation.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{len(slides)+1}"/></p:sldMasterIdLst>
<p:sldIdLst>{slide_ids}</p:sldIdLst>
<p:sldSz cx="{emu(13.333)}" cy="{emu(7.5)}" type="screen16x9"/>
<p:notesSz cx="{emu(10)}" cy="{emu(7.5)}"/></p:presentation>""",
        )
        rels = [
            f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
            for i in range(1, len(slides) + 1)
        ]
        rels.append(
            f'<Relationship Id="rId{len(slides)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
        )
        zf.writestr(
            "ppt/_rels/presentation.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(rels)
            + "</Relationships>",
        )
        zf.writestr("ppt/theme/theme1.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office"><a:themeElements><a:clrScheme name="Office"><a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1><a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="1F1F1F"/></a:dk2><a:lt2><a:srgbClr val="F7F3EA"/></a:lt2><a:accent1><a:srgbClr val="1B5E5A"/></a:accent1><a:accent2><a:srgbClr val="C7782A"/></a:accent2><a:accent3><a:srgbClr val="5B6C8A"/></a:accent3><a:accent4><a:srgbClr val="7A5C42"/></a:accent4><a:accent5><a:srgbClr val="4E7D4A"/></a:accent5><a:accent6><a:srgbClr val="9B3A34"/></a:accent6><a:hlink><a:srgbClr val="0000FF"/></a:hlink><a:folHlink><a:srgbClr val="800080"/></a:folHlink></a:clrScheme><a:fontScheme name="Office"><a:majorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:majorFont><a:minorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:minorFont></a:fontScheme><a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>""")
        zf.writestr(
            "ppt/slideLayouts/slideLayout1.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>""",
        )
        zf.writestr(
            "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>""",
        )
        zf.writestr(
            "ppt/slideMasters/slideMaster1.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMap val="bg1" lastClr="FFFFFF"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst></p:sldMaster>""",
        )
        zf.writestr(
            "ppt/slideMasters/_rels/slideMaster1.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>""",
        )
        for i, (title, body, table) in enumerate(slides, start=1):
            zf.writestr(f"ppt/slides/slide{i}.xml", slide_xml(title, body, table))
            zf.writestr(
                f"ppt/slides/_rels/slide{i}.xml.rels",
                """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>""",
            )


def build_report() -> tuple[str, list[tuple[str, list[str], list[list[str]] | None]]]:
    matched_methods = [
        ("LoRA matched", "lora"),
        ("SVD-right / OPLoRA", "oplora"),
        ("PC-LoRA matched", "pc_lora"),
    ]
    rank_methods = [
        ("ActCov rank LoRA", "actcov_rank_lora"),
        ("ActCov rank + SVD-right", "actcov_rank_svd_right"),
        ("ActCov rank PC-LoRA", "actcov_rank_pc_lora"),
    ]
    full_methods = [
        ("LoRA full", "lora"),
        ("LoRA+NSC full", "lora_nsc"),
        ("OPLoRA full", "oplora"),
        ("PC-LoRA full", "pc_lora"),
    ]
    matched = commonsense_task_table(
        425_984,
        matched_methods,
    )
    rank = commonsense_task_table(
        417_792,
        rank_methods,
    )
    full = commonsense_task_table(
        3_407_872,
        full_methods,
    )
    matched_mech = commonsense_mechanism_table(425_984, matched_methods)
    rank_mech = commonsense_mechanism_table(417_792, rank_methods)
    full_mech_eff = commonsense_mechanism_efficiency_table(3_407_872, full_methods)
    math = metamath_table()
    math_mech_eff = metamath_mechanism_efficiency_table()

    slides = [
        (
            "PC-LoRA 进展更新（2026-05-13）",
            [
                "- 覆盖范围：从 2026-05-07 PPT 之后新增的实验与结论修正。",
                "- 主模型：Llama-3.1-8B，本地权重；数据：Commonsense170k、MetaMathQA。",
                "- 当前核心判断：SVD Preserve 仍最稳；Capture 改变了机制指标，但 PC-LoRA 尚未稳定胜出。",
                "- 本版增加任务指标、机制指标和效率指标；不再只看 loss。",
                "- 注意：matched 8-adapter 与 full 64-adapter 参数量不同，不能直接横向比较。",
            ],
            None,
        ),
        (
            "上版 PPT 之后新增了什么",
            [
                "- 补齐 ActCov rank PC-LoRA seed2，修正“当前最有希望”的早期判断。",
                "- 补齐 Commonsense full 64-adapter 的 LoRA+NSC / PC-LoRA 多 seed 结果。",
                "- 新增 MetaMathQA 100k/5k split，完成 8B 四方法训练。",
                "- 新增并修正 MetaMathQA generation evaluator，支持 query/response 字段。",
                "- 数学线改用 MetaMathQA；本版不纳入小样本 generation 预评估结果。",
            ],
            None,
        ),
        (
            "关键结论变化",
            [
                "- 旧结论：ActCov rank PC-LoRA seed0/1 = 0.6174 ± 0.0020，疑似最好。",
                "- 新结论：补 seed2 后变为 0.6132 ± 0.0074，低于 SVD-right/OPLoRA 的 0.6147 ± 0.0022。",
                "- Full 64-adapter 下，LoRA/LoRA+NSC 明显强于 OPLoRA/PC-LoRA；PC-LoRA seed 间波动大。",
                "- MetaMathQA 旧 PPL/token accuracy 口径已作废；后续只看 final-answer EM。",
            ],
            None,
        ),
        (
            "Commonsense：matched 8-adapter 主结果",
            [
                "- 公平比较 projection / allocation / capture 机制，参数量约 0.42M。",
                "- 表中同时看 Exact/EM、eval loss、token accuracy 和参数量。",
                "- SVD-right / OPLoRA 仍是最稳定 matched baseline：EM 方差最低。",
                "- ActCov allocation 类方法降低 loss，但 Exact/EM 均值和稳定性没有超过 SVD-right。",
                "- PC residual capture 当前没有形成稳定叠加收益。",
            ],
            matched + rank[1:],
        ),
        (
            "Commonsense：matched 机制指标",
            [
                "- omega_full / omega_resid 越低，表示 A 行空间捕获 activation signal 越多。",
                "- orth_A 越低，表示 A 行空间越接近正交。",
                "- matched 设置中 rho_k 未系统记录；后续需要补代表性 run。",
                "- 机制指标有变化，但没有稳定转化为更高 Exact/EM。",
            ],
            matched_mech + rank_mech[1:],
        ),
        (
            "Commonsense：full 64-adapter 结果",
            [
                "- Full 设置参数量约 3.41M，是 matched 的约 8 倍。",
                "- Full LoRA 和 LoRA+NSC 最强，说明容量增加比 projection 机制更直接有效。",
                "- Full OPLoRA 和 PC-LoRA 不稳定，尤其 PC-LoRA seed1 出现明显掉点。",
                "- 该结果提示：Preserve 投影在 full adapter 下可能限制任务适配空间。",
            ],
            full,
        ),
        (
            "Commonsense：full 机制与效率",
            [
                "- 这里补充机制和开销，不只看 task loss。",
                "- PC-LoRA/LoRA+NSC 的 omega 低，但 full setting 下任务 EM 没有同步领先。",
                "- PC-LoRA 显存和时间略高，主要来自 capture/projection 计算与机制日志。",
                "- rho_k 仍需补；否则 Preserve 干扰降低只能从设计上推断，缺少实测曲线。",
            ],
            full_mech_eff,
        ),
        (
            "方向替代实验：ActCov / Fisher / CUR",
            [
                "- ActCov hard projection：约 0.55~0.58，明显差于 SVD-right。",
                "- Fisher-right OPLoRA：seed0 exact 0.5524，负结果明确。",
                "- CUR-right OPLoRA：有效 eval 约 0.5994，低于 LoRA/SVD-right。",
                "- 解释：真实高激活方向适合解释 activation energy，但 hard projection 删除它会伤害任务信号。",
                "- 当前策略：保留 SVD 做 Preserve，ActCov 只作为 allocation/capture 的辅助信号。",
            ],
            None,
        ),
        (
            "新增方案：Data-Weight Joint Directions",
            [
                "- ActivationCov 只看数据分布：top eig(C_z) 找到输入中方差最大的方向。",
                "- SVD 只看权重几何：top eig(W0^T W0) 找到 W0 的高能输入映射方向。",
                "- 二者各有盲点：高方差方向未必被 W0 强烈使用；高 SVD 方向也未必在真实数据中出现。",
                "- 新方案构造 M_WA = C_z^{1/2} W0^T W0 C_z^{1/2}，同时考虑输入频率和输出能量。",
                "- 后续比较 ActivationCov、SVD、M_WA 三类方向的 InputCapture / OutputCapture。",
            ],
            [
                ["方向", "只看数据", "只看权重", "联合目标"],
                ["ActivationCov", "yes", "no", "high input variance"],
                ["SVD", "no", "yes", "high W0 mapping energy"],
                ["M_WA", "yes", "yes", "data-weight output energy"],
            ],
        ),
        (
            "M_WA 实验设计",
            [
                "- 对同一层/模块收集 activation z，估计 C_z = E[z z^T]。",
                "- 对同一 W0 计算 G_W = W0^T W0，形成 M_WA = C_z^{1/2} G_W C_z^{1/2}。",
                "- 取 M_WA top-k eigenvectors，经 C_z^{1/2} 映射回输入空间并正交化，得到 Q_WA。",
                "- InputCapture: ||Q Q^T z||^2 / ||z||^2，衡量解释真实输入能量。",
                "- OutputCapture: ||W0 Q Q^T z||^2 / ||W0 z||^2，衡量解释真实模型输出能量。",
                "- 若 Q_WA 的 OutputCapture 高于 ActCov/SVD，说明 joint direction 更贴近真实模型行为。",
            ],
            None,
        ),
        (
            "MetaMathQA：新增数学实验设置",
            [
                "- 原始数据：data/MetaMathQA/MetaMathQA-395K.json。",
                "- 本地 split：train 100,000，validation 5,000；字段 query/response。",
                "- 训练：Llama-3.1-8B，seed0，max_steps=300，rank=8，target q_proj/v_proj。",
                "- 主指标：final-answer Exact Match，即最终答案完全匹配样本数 / 总样本数。",
                "- PPL / token accuracy 仅作为辅助诊断，不再作为数学任务精度。",
            ],
            None,
        ),
        (
            "MetaMathQA：当前指标",
            [
                "- 旧的 PPL / token accuracy 表格已作废，不能作为数学能力结论。",
                "- 后续只用 final-answer EM 汇报方法优劣。",
                "- LoRA+NSC 和 PC-LoRA 的机制指标更低，但训练期任务指标没有同步领先。",
                "- 训练期 teacher-forced EM 可作快速筛查；正式结论使用 generation final-answer EM。",
            ],
            math,
        ),
        (
            "MetaMathQA：机制与效率指标",
            [
                "- LoRA+NSC / PC-LoRA 的 omega 更低，说明 capture loss 确实改变了 A 行空间。",
                "- OPLoRA 的 PPL 当前最好，但 omega 并不是最低。",
                "- 这说明 mechanism metric 与 task metric 不是简单单调关系。",
                "- 训练开销：capture/projection 方法显存约高 0.4GB，时间约高 5%~13%。",
            ],
            math_mech_eff,
        ),
        (
            "指标体系：现在汇报哪些指标",
            [
                "- 任务性能：final-answer EM；PPL、eval loss、token accuracy 只作辅助诊断。",
                "- 机制指标：omega_full、omega_residual、orth_error_A；rho_k 尚需补代表性 run。",
                "- 效率指标：total_train_time_sec、peak_gpu_memory_gb、trainable parameters。",
                "- 稳定性：seed mean/std，不再只看单次 run。",
                "- 当前缺口：forgetting_gap / retention_score 还没有正式跑跨域评估。",
            ],
            None,
        ),
        (
            "当前判断",
            [
                "- Preserve：SVD-right 是目前最可靠的保护方向；ActCov/Fisher/CUR 不适合直接替代 SVD 做 hard projection。",
                "- Capture：机制指标显示 capture loss 改变了 A 行空间，但任务收益还需要 full generation eval 验证。",
                "- PC-LoRA：理论组合合理，但当前实现/超参下没有稳定超过 OPLoRA 或 LoRA+NSC。",
                "- 新增 M_WA 方向用于回答：真实高信息方向是否应同时依赖数据分布和权重映射。",
                "- 论文叙事应从“PC-LoRA 已显著优于 baseline”调整为“机制有效但组合方式仍需校准”。",
            ],
            None,
        ),
        (
            "下一步实验优先级",
            [
                "- P0：跑 MetaMathQA 5000 generation EM，先获得正式数学能力指标。",
                "- P1：若 OPLoRA 仍领先，补 LoRA/OPLoRA/LoRA+NSC/PC-LoRA seed1/2。",
                "- P1：补代表性 rho_k：LoRA、OPLoRA、LoRA+NSC、PC-LoRA 各 seed0 selected layers。",
                "- P2：做 forgetting eval：commonsense adapter 评 MetaMathQA；MetaMathQA adapter 评 commonsense。",
                "- P2：调 lambda_cap / capture layers，而不是继续增加 hard projection 变体。",
            ],
            None,
        ),
    ]

    md_sections = [
        "# PC-LoRA / Preserve-and-Capture LoRA 进展更新\n\n日期：2026-05-13\n",
        "## 当前结论\n\n"
        "- SVD-right / OPLoRA 仍是最稳定的 Preserve baseline。\n"
        "- ActCov/Fisher/CUR 直接 hard projection 是负结果；ActCov 更适合 allocation/capture 辅助。\n"
        "- Capture 改变了机制指标，但 PC-LoRA 尚未稳定胜出。\n"
        "- 本版汇报任务性能、机制指标和效率指标，不只看 loss。\n"
        "- 下一步应优先完成 MetaMathQA 5000 generation EM，再决定是否补 seed。\n",
        "## Commonsense matched 8-adapter：任务指标\n\n" + table_to_md(matched + rank[1:]) + "\n",
        "## Commonsense matched 8-adapter：机制指标\n\n" + table_to_md(matched_mech + rank_mech[1:]) + "\n",
        "## Commonsense full 64-adapter：任务指标\n\n" + table_to_md(full) + "\n",
        "## Commonsense full 64-adapter：机制/效率指标\n\n" + table_to_md(full_mech_eff) + "\n",
        "## MetaMathQA seed0：任务指标\n\n" + table_to_md(math) + "\n",
        "## MetaMathQA seed0：机制/效率指标\n\n" + table_to_md(math_mech_eff) + "\n",
        "## 新增研究方法：Data-Weight Joint Directions\n\n"
        "ActivationCov 找到的是数据中方差最大的输入方向，可解释真实 activation energy，"
        "但这些方向不一定会被预训练权重 W0 强烈使用。SVD 找到的是 W0 的高能映射方向，"
        "但它只依赖权重矩阵本身，不考虑真实输入分布。为同时建模“数据中经常出现”和“权重会强烈映射”"
        "这两个因素，我们构造 data-weight joint high-information directions：\n\n"
        "`M_WA = C_z^{1/2} W0^T W0 C_z^{1/2}`\n\n"
        "其中 `C_z = E[z z^T]` 表示某层输入激活协方差，`W0^T W0` 表示该线性层输入方向被映射到输出的能量。"
        "取 `M_WA` 的 top-k eigenvectors 后映射回输入空间并正交化，得到联合方向 `Q_WA`。"
        "实验上比较三类方向：ActivationCov、SVD、M_WA。评价指标包括 "
        "`InputCapture = ||Q Q^T z||^2 / ||z||^2` 和 "
        "`OutputCapture = ||W0 Q Q^T z||^2 / ||W0 z||^2`。"
        "如果 `Q_WA` 在 OutputCapture 上优于 ActivationCov/SVD，说明真实模型行为中的高信息方向需要同时考虑数据分布和权重映射。\n",
        "## M_WA 实验方案\n\n"
        "1. 在同一模型、同一数据、同一 target layers/modules 上收集 activation `z`。\n"
        "2. 估计 `C_z`，并对每个原始权重 `W0` 计算 `G_W = W0^T W0`。\n"
        "3. 构造 `M_WA = C_z^{1/2} G_W C_z^{1/2}`，取 top-k 得到 joint directions。\n"
        "4. 对 k = 4, 8, 16, 32, 64 做 sweep，比较 ActivationCov / SVD / M_WA。\n"
        "5. 记录 InputCapture、OutputCapture、ResidualInputEnergy、ResidualOutputEnergy 和 principal angles。\n"
        "6. 若 M_WA 明显提升 OutputCapture，再考虑把 Preserve 投影从 SVD 替换为 joint-aware projection。\n",
        "## 结论变化\n\n"
        "- 上版 PPT 中 ActCov rank PC-LoRA 只有 seed0/1，0.6174 ± 0.0020；补 seed2 后为 0.6132 ± 0.0074。\n"
        "- Commonsense full setting 中 LoRA/LoRA+NSC 强于 OPLoRA/PC-LoRA，PC-LoRA 不稳定。\n"
        "- MetaMathQA 数学精度改为 final-answer EM；旧 PPL/token accuracy 结果不再纳入结论。\n",
    ]
    return "\n".join(md_sections), slides


def main() -> None:
    md, slides = build_report()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORT_DIR / "pc_lora_progress_20260513.md"
    pptx_path = REPORT_DIR / "pc_lora_progress_20260513.pptx"
    md_path.write_text(md, encoding="utf-8")
    write_pptx(slides, pptx_path)
    print(json.dumps({"markdown": str(md_path), "pptx": str(pptx_path), "slides": len(slides)}, indent=2))


if __name__ == "__main__":
    main()
