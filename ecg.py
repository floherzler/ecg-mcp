import json
from uuid import uuid4
from pathlib import Path
from collections.abc import Iterable
from typing import Any
import re
import numpy as np

import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt

from mcp.server.fastmcp import FastMCP

# ----------------------------------------
# MCP Server
# ----------------------------------------
mcp = FastMCP("ecg-mcp")

BASE = Path(__file__).resolve().parent

DATA_DIR = BASE / "ecg_data"
SIM_DATA_DIR = DATA_DIR / "simulations"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PLOT_DIR = DATA_DIR / "plots"

SIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------
# Helper: THIS fixes the JSON errors
# ----------------------------------------


def _to_json_serializable(value: Any) -> Any:
    """Convert NumPy and pandas types recursively to JSON-friendly Python types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_json_serializable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_json_serializable(val) for val in value]
    if isinstance(value, tuple):
        return tuple(_to_json_serializable(val) for val in value)
    return value


async def _resource_to_text(uri: str) -> str:
    """Read an MCP resource and normalize it to plain text."""
    content = await mcp.read_resource(uri)

    def _decode_bytes(data: bytes | bytearray) -> str:
        try:
            return data.decode()
        except UnicodeDecodeError:
            return f"<binary data {len(data)} bytes>"

    # Already plain
    if isinstance(content, str):
        return content
    if isinstance(content, (bytes, bytearray)):
        return _decode_bytes(content)

    # Iterable of ReadResourceContents
    if isinstance(content, Iterable):
        parts = []
        for chunk in content:
            # correct field is 'content', not 'data'
            if hasattr(chunk, "content"):
                c = chunk.content
                if isinstance(c, (bytes, bytearray)):
                    parts.append(_decode_bytes(c))
                else:
                    parts.append(str(c))
            else:
                # fallback
                parts.append(str(chunk))
        return "".join(parts)

    # Single ReadResourceContents
    if hasattr(content, "content"):
        c = content.content
        return c.decode() if isinstance(c, (bytes, bytearray)) else str(c)

    raise TypeError(f"Unsupported resource type: {type(content)}")


async def _read_json_resource(uri: str) -> Any:
    text = await _resource_to_text(uri)
    if not text.strip():
        raise ValueError(f"Resource {uri} is empty!")
    return json.loads(text)


def _resource_id_from_uri(uri: str) -> str:
    """Extract the unique identifier from an MCP URI."""
    candidate = uri.rstrip("/").split("/")[-1]
    return candidate or uuid4().hex


# ----------------------------------------
# RESOURCES
# ----------------------------------------


_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_resource_id(resource_id: str) -> None:
    """Allow only safe characters in resource ids to prevent path traversal."""
    if not isinstance(resource_id, str) or not _ID_PATTERN.match(resource_id):
        raise ValueError("Invalid resource id")


def _safe_join(base: Path, filename: str) -> Path:
    """
    Join and resolve a path, ensuring it stays under the base directory.

    Parameters
    ----------
    base : Path
        The base directory under which the file must reside.
    filename : str
        The filename or relative path to join with the base directory.

    Returns
    -------
    Path
        The resolved absolute path to the file.

    Raises
    ------
    ValueError
        If path traversal is detected (i.e., the resolved path is not under the base directory).
    """
    candidate = base / filename
    resolved_base = base.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_base)
    except ValueError:
        raise ValueError(
            f"Path traversal detected: resolved path '{resolved_candidate}' must be within the base directory '{resolved_base}'"
        )
    return resolved_candidate


@mcp.resource("ecg://simulations/{sim_id}", mime_type="application/json")
def read_simulation(sim_id: str) -> str:
    _validate_resource_id(sim_id)
    path = _safe_join(SIM_DATA_DIR, f"{sim_id}.json")
    if not path.exists():
        raise FileNotFoundError(f"Simulation resource '{sim_id}' not found")
    return path.read_text()


@mcp.resource("ecg://processed/{proc_id}", mime_type="application/json")
def read_processed(proc_id: str) -> str:
    _validate_resource_id(proc_id)
    path = _safe_join(PROCESSED_DATA_DIR, f"{proc_id}.json")
    if not path.exists():
        raise FileNotFoundError(f"Processed resource '{proc_id}' not found")
    return path.read_text()


@mcp.resource("ecg://plots/{plot_id}", mime_type="image/png")
def read_plot(plot_id: str) -> bytes:
    _validate_resource_id(plot_id)
    path = _safe_join(PLOT_DIR, f"{plot_id}.png")
    if not path.exists():
        raise FileNotFoundError(f"Plot resource '{plot_id}' not found")
    return path.read_bytes()


# ----------------------------------------
# TOOLS
# ----------------------------------------


@mcp.tool()
async def debug_resource(uri: str) -> dict:
    """
    Debug an MCP resource by returning its type, repr, and decoded text.

    Parameters
    ----------
    uri : str
        The URI of the MCP resource to debug.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'type': The type of the resource content as a string.
        - 'repr': The string representation of the resource content.
        - 'decoded': The decoded text content of the resource, if available.
    """
    content = await mcp.read_resource(uri)
    return {
        "type": str(type(content)),
        "repr": repr(content),
        "decoded": await _resource_to_text(uri),
    }


@mcp.tool()
async def ecg_simulate(
    duration: int = 10,
    sampling_rate: int = 1000,
    noise: float = 0.01,
    heart_rate: int = 70,
    heart_rate_std: int = 1,
    method: str = "ecgsyn",
):
    """Simulate a raw ECG signal and return a resource URI.

    Args:
        duration: Length of the signal in seconds.
        sampling_rate: Samples per second for the generated signal.
        noise: Amount of gaussian noise added to the signal.
        heart_rate: Target average heart rate.
        heart_rate_std: Standard deviation of the heart rate sampling.
        method: Simulation algorithm passed to `nk.ecg_simulate`.
    """

    ecg = nk.ecg_simulate(
        duration=duration,
        sampling_rate=sampling_rate,
        noise=noise,
        heart_rate=heart_rate,
        heart_rate_std=heart_rate_std,
        method=method,
    )

    sim_id = uuid4().hex
    path = SIM_DATA_DIR / f"{sim_id}.json"
    path.write_text(json.dumps(ecg.tolist()))

    return {
        "uri": f"ecg://simulations/{sim_id}",
        "metadata": {
            "duration": duration,
            "sampling_rate": sampling_rate,
            "n_samples": len(ecg),
        },
    }


@mcp.tool()
async def ecg_process(simulation_uri: str, sampling_rate: int) -> dict:
    """
    Load raw ECG simulation, run nk.ecg_process, and return processed URI.

    Parameters
    ----------
    simulation_uri : str
        URI of the raw ECG simulation resource. Should be in the format "ecg://simulations/{id}".
    sampling_rate : int
        Sampling rate of the ECG signal in Hz (e.g., 50, 1000).

    Returns
    -------
    dict
        Dictionary containing the processed ECG resource URI and metadata.
    """
    raw_ecg = await _read_json_resource(simulation_uri)

    signals, info = nk.ecg_process(raw_ecg, sampling_rate=sampling_rate)

    proc_id = _resource_id_from_uri(simulation_uri)
    serializable_signals = _to_json_serializable(signals.to_dict(orient="list"))
    serializable_info = _to_json_serializable(info)
    out = {
        "signals": serializable_signals,
        "info": serializable_info,
    }
    path = PROCESSED_DATA_DIR / f"{proc_id}.json"
    path.write_text(json.dumps(out))

    return {
        "uri": f"ecg://processed/{proc_id}",
        "metadata": {**serializable_info, "resource_id": proc_id},
    }


@mcp.tool()
async def ecg_plot(processed_uri: str) -> dict:
    """Plot a processed ECG and return a PNG resource URI.

    Args:
        processed_uri: URI of the processed simulation data to plot.
    """

    data = await _read_json_resource(processed_uri)

    signals = pd.DataFrame(data["signals"])
    info = data["info"]

    nk.ecg_plot(signals, info)
    fig = plt.gcf()
    fig.set_size_inches(12, 10)

    plot_id = _resource_id_from_uri(processed_uri)
    outpath = PLOT_DIR / f"{plot_id}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return {
        "uri": f"ecg://plots/{plot_id}",
        "metadata": {"plot_id": plot_id},
    }


@mcp.tool()
async def ecg_clean(signal: list[float], sampling_rate: int = 1000) -> list[float]:
    """Clean a raw ECG signal and return the filtered waveform.

    Args:
        signal: Raw ECG voltage samples.
        sampling_rate: Sampling rate for the signal (defaults to 1000 Hz).
    """
    cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)
    return cleaned.tolist()


# ----------------------------------------
# MAIN
# ----------------------------------------


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
