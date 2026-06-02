"""
plantdrp.cli
============
Command-line interface for PlantDRP.

Commands
--------
    plantdrp predict    Run DR protein prediction on a FASTA file
    plantdrp ui         Launch the Streamlit web interface
    plantdrp info       Show version and model information

Usage examples
--------------
    # Basic prediction with DS2 (recommended)
    plantdrp predict --input proteins.fasta --model ds2

    # Save as TSV, use DS1, custom threshold
    plantdrp predict --input proteins.fasta --model ds1 \
                     --output results.tsv --format tsv \
                     --threshold 0.6

    # Launch Streamlit UI
    plantdrp ui

    # Launch on custom port
    plantdrp ui --port 8502
"""

import sys
import subprocess
from pathlib import Path
from typing import Optional

import typer
import threading
import time
import requests
import webbrowser
import uvicorn

app = typer.Typer(
    name="plantdrp",
    help="PlantDRP - Prediction of Plant Disease Resistance Proteins",
    add_completion=False,
)

def _version():
    try:
        from plantdrp import __version__
        return __version__
    except Exception:
        return "unknown"

def _print_banner():
    typer.echo("")
    typer.echo("  ╔═══════════════════════════════════════╗")
    typer.echo("  ║           PlantDRP  v" + _version() + "            ║")
    typer.echo("  ║  Plant Disease Resistance Protein     ║")
    typer.echo("  ║  Predictor · ICAR-IASRI, New Delhi   ║")
    typer.echo("  ╚═══════════════════════════════════════╝")
    typer.echo("")

# COMMAND 1 — predict


@app.command()
def predict(
    input: Path = typer.Option(
        ...,
        "--input", "-i",
        help="Path to input protein FASTA file.",
        exists=True,
        readable=True,
    ),
    model: str = typer.Option(
        "ds2",
        "--model", "-m",
        help="Model to use: **ds1** (experimentally validated) "
             "or **ds2** (extended database, recommended).",
    ),
    output: Path = typer.Option(
        "plantdrp_results.csv",
        "--output", "-o",
        help="Path to save results file.",
    ),
    format: str = typer.Option(
        "csv",
        "--format", "-f",
        help="Output format: csv, tsv, or json.",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold", "-t",
        min=0.0,
        max=1.0,
        help="Probability cutoff for DR label (0–1). Default: 0.5",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device for embedding: auto, cpu, cuda, or mps.",
    ),
    
    verbose: bool = typer.Option(
        True,
        "--verbose",
        help="Print per-sequence progress. Use --no-verbose to silence.",
    ),
    #),
):
    """
    **Predict** disease resistance proteins from a FASTA file.

    Runs the full pipeline:
    FASTA → ProtT5 embedding → SVM inference → results file
    """
    _print_banner()

    #  Validate model choice 
    model = model.lower()
    if model not in ("ds1", "ds2"):
        typer.echo(
            f"[ERROR] Invalid model '{model}'. "
            f"Choose 'ds1' or 'ds2'.",
            err=True
        )
        raise typer.Exit(code=1)

    #  Validate format choice ─
    format = format.lower()
    if format not in ("csv", "tsv", "json"):
        typer.echo(
            f"[ERROR] Invalid format '{format}'. "
            f"Choose 'csv', 'tsv', or 'json'.",
            err=True
        )
        raise typer.Exit(code=1)

    typer.echo(f"  Input     : {input}")
    typer.echo(f"  Model     : {model.upper()}")
    typer.echo(f"  Threshold : {threshold}")
    typer.echo(f"  Output    : {output}")
    typer.echo(f"  Format    : {format}")
    typer.echo(f"  Device    : {device}")
    typer.echo("")

    #  Run prediction ─
    try:
        from plantdrp import Predictor

        pred = Predictor(
            model=model,
            device=device,
            threshold=threshold,
        )
        df = pred.predict(str(input), verbose=verbose)

    except FileNotFoundError as e:
        typer.echo(f"[ERROR] {e}", err=True)
        raise typer.Exit(code=1)

    except Exception as e:
        typer.echo(f"[ERROR] Prediction failed: {e}", err=True)
        raise typer.Exit(code=1)

    #  Save results 
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        df.to_csv(output, index=False)
    elif format == "tsv":
        df.to_csv(output, sep="\t", index=False)
    elif format == "json":
        df.to_json(output, orient="records", indent=2)

    #  Print summary 
    dr_count  = (df["prediction"] == "DR").sum()
    ndr_count = (df["prediction"] == "Non-DR").sum()

    typer.echo("")
    typer.echo("   Summary ─")
    typer.echo(f"  Total sequences : {len(df)}")
    typer.echo(f"  DR proteins     : {dr_count}")
    typer.echo(f"  Non-DR proteins : {ndr_count}")
    typer.echo(f"  Results saved   : {output}")
    typer.echo("  ")
    typer.echo("")



# COMMAND 2 — ui

@app.command()
def ui():
    """
    Launch PlantDRP web interface.
    """

    def wait_and_open():
        url = "http://127.0.0.1:8000"

        while True:
            try:
                requests.get(url, timeout=1)
                webbrowser.open(url)
                break
            except Exception:
                time.sleep(1)

    threading.Thread(
        target=wait_and_open,
        daemon=True
    ).start()

    uvicorn.run(
        "plantdrp.server:app",
        host="127.0.0.1",
        port=8000
    )

# COMMAND 3 — info


@app.command()
def info():
    """Show PlantDRP version, model info, and system details."""
    import torch

    _print_banner()
    typer.echo("  Version      : " + _version())
    typer.echo("  Python       : " + sys.version.split()[0])
    typer.echo("  PyTorch      : " + torch.__version__)

    if torch.cuda.is_available():
        typer.echo("  GPU          : " + torch.cuda.get_device_name(0))
        mem = torch.cuda.get_device_properties(0).total_memory
        typer.echo(f"  GPU Memory   : {mem / 1e9:.1f} GB")
    else:
        typer.echo("  GPU          : Not available — using CPU")

    typer.echo("")
    typer.echo("  Models")
    typer.echo("  ─")
    for ds in ("ds1", "ds2"):
        pkl    = Path(__file__).parent / "models" / f"svm_{ds}.pkl"
        status = "✓ found"   if pkl.exists() else "✗ missing"
        size   = f"{pkl.stat().st_size / 1e6:.1f} MB" if pkl.exists() else "—"
        typer.echo(f"  svm_{ds}.pkl  : {status}  ({size})")
    typer.echo("")


# ENTRY POINT


if __name__ == "__main__":
    app()
