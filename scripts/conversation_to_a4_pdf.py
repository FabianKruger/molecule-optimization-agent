#!/usr/bin/env python3
"""Render molecule optimization conversations into a single-page A4 PDF."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from rdkit import Chem
from rdkit.Chem import Draw


@dataclass
class MoleculeRecord:
    iteration: int
    smiles: str
    ic50_nm: float | None
    qed: float | None
    novelty: str


def parse_records(conversation_json: Path, max_molecules: int) -> list[MoleculeRecord]:
    with conversation_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    messages = payload.get("conversation", [])
    records: dict[int, MoleculeRecord] = {}
    proposal_smiles: list[str] = []

    smiles_re = re.compile(r"SMILES:\s*([^\s]+)")
    ic50_re = re.compile(r"Predicted IC50:\s*([0-9]+(?:\.[0-9]+)?)\s*nM")
    qed_re = re.compile(r"QED:\s*([0-9]+(?:\.[0-9]+)?)")
    novelty_re = re.compile(r"Novelty:\s*(.+)")
    iter_re = re.compile(r"Iteration:\s*(\d+)\s*/\s*\d+")

    for msg in messages:
        if msg.get("role") == "AIMessage":
            content = msg.get("content", "")
            if '"smiles"' in content:
                try:
                    parsed = json.loads(content)
                    smiles = str(parsed.get("smiles", "")).strip()
                    if smiles:
                        proposal_smiles.append(smiles)
                except json.JSONDecodeError:
                    smiles_m = smiles_re.search(content)
                    if smiles_m:
                        proposal_smiles.append(smiles_m.group(1).strip())

        if msg.get("role") != "HumanMessage":
            continue

        content = msg.get("content", "")
        if "Evaluation of your last proposal" not in content:
            continue

        smiles_m = smiles_re.search(content)
        ic50_m = ic50_re.search(content)
        qed_m = qed_re.search(content)
        novelty_m = novelty_re.search(content)
        iter_m = iter_re.search(content)

        if not all([smiles_m, ic50_m, qed_m, novelty_m, iter_m]):
            continue

        iteration = int(iter_m.group(1))
        if iteration in records:
            continue

        records[iteration] = MoleculeRecord(
            iteration=iteration,
            smiles=smiles_m.group(1).strip(),
            ic50_nm=float(ic50_m.group(1)),
            qed=float(qed_m.group(1)),
            novelty=novelty_m.group(1).strip(),
        )

    completed: list[MoleculeRecord] = []
    for iteration in range(1, max_molecules + 1):
        if iteration in records:
            completed.append(records[iteration])
            continue

        if iteration - 1 < len(proposal_smiles):
            completed.append(
                MoleculeRecord(
                    iteration=iteration,
                    smiles=proposal_smiles[iteration - 1],
                    ic50_nm=None,
                    qed=None,
                    novelty="N/A",
                )
            )

    return completed


def render_a4_pdf(
    records: list[MoleculeRecord],
    output_pdf: Path,
    title: str,
    columns: int,
    rows: int,
) -> None:
    max_cells = columns * rows
    if max_cells < len(records):
        raise ValueError(
            f"Grid is too small for {len(records)} molecules. "
            f"Increase --rows/--columns (current capacity: {max_cells})."
        )

    fig = plt.figure(figsize=(8.27, 11.69), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    gs = fig.add_gridspec(
        nrows=rows,
        ncols=columns,
        left=0.018,
        right=0.982,
        top=0.948,
        bottom=0.02,
        wspace=0.04,
        hspace=0.07,
    )

    #fig.suptitle(title, fontsize=13, fontweight="bold", color="#202124")

    for idx in range(max_cells):
        r, c = divmod(idx, columns)
        ax = fig.add_subplot(gs[r, c])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Assign border color based on iteration range
        if idx >= len(records):
            border_color = "#d8d8d8"
        else:
            rec_iter = records[idx].iteration
            if rec_iter <= 10:
                border_color = "#a8c8e1"  # Light blue
            elif rec_iter <= 20:
                border_color = "#a8d9a8"  # Light green
            elif rec_iter <= 28:
                border_color = "#e8d4b8"  # Light peach
            else:
                border_color = "#d8c8d8"  # Light lavender

        card = FancyBboxPatch(
            (0.01, 0.01),
            0.98,
            0.98,
            boxstyle="round,pad=0.01,rounding_size=0.025",
            linewidth=1.2,
            edgecolor=border_color,
            facecolor="#ffffff",
            transform=ax.transAxes,
            zorder=0,
        )
        ax.add_patch(card)

        if idx >= len(records):
            continue

        record = records[idx]
        mol = Chem.MolFromSmiles(record.smiles)
        if mol is not None:
            img = Draw.MolToImage(mol, size=(360, 240), kekulize=True)
            img_ax = ax.inset_axes([0.055, 0.34, 0.89, 0.58])
            img_ax.imshow(img)
            img_ax.set_aspect("equal")
            img_ax.set_axis_off()
        else:
            ax.text(
                0.5,
                0.63,
                "Invalid SMILES",
                ha="center",
                va="center",
                fontsize=7,
                color="#a33a3a",
                fontweight="bold",
            )

        novelty_text = "N/A"
        if record.novelty != "N/A":
            novelty_text = "Novel" if "NOVEL" in record.novelty.upper() else "Exists"
        novelty_color = "#1f7a3a" if novelty_text == "Novel" else "#8f1f1f"
        if novelty_text == "N/A":
            novelty_color = "#6e6e6e"

        ax.text(0.06, 0.26, f"Iter {record.iteration}", fontsize=7.1, fontweight="bold", color="#202124")
        ic50_text = f"IC50: {record.ic50_nm:.2f} nM" if record.ic50_nm is not None else "IC50: N/A"
        qed_text = f"QED: {record.qed:.4f}" if record.qed is not None else "QED: N/A"
        ax.text(0.06, 0.19, ic50_text, fontsize=6.5, color="#2f3136")
        ax.text(0.06, 0.12, qed_text, fontsize=6.5, color="#2f3136")
        ax.text(0.06, 0.05, f"Novelty: {novelty_text}", fontsize=6.5, color=novelty_color)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a one-page A4 PDF showing molecules from a conversation JSON "
            "with iteration, IC50, QED, and novelty labels."
        )
    )
    parser.add_argument("input_json", type=Path, help="Path to conversation JSON file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("conversation_molecules_a4.pdf"),
        help="Output PDF path (default: conversation_molecules_a4.pdf)",
    )
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=50,
        help="Maximum number of molecules to render (default: 50)",
    )
    parser.add_argument("--columns", type=int, default=5, help="Grid columns (default: 5)")
    parser.add_argument("--rows", type=int, default=10, help="Grid rows (default: 10)")
    parser.add_argument(
        "--title",
        type=str,
        default="Molecule Optimization Trace (First 50 Iterations)",
        help="PDF title",
    )
    args = parser.parse_args()

    all_records = parse_records(args.input_json, args.max_molecules)
    if not all_records:
        raise RuntimeError("No evaluation records were found in the provided conversation JSON.")

    selected = all_records[: args.max_molecules]
    render_a4_pdf(
        records=selected,
        output_pdf=args.output,
        title=args.title,
        columns=args.columns,
        rows=args.rows,
    )

    print(f"Rendered {len(selected)} molecules to {args.output}")


if __name__ == "__main__":
    main()