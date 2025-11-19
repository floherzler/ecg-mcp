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


mcp = FastMCP("ecg-mcp")

BASE: Path = Path(__file__).resolve().parent

DATA_DIR: Path = BASE / "ecg_data"
SIM_DATA_DIR: Path = DATA_DIR / "simulations"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
ANALYZED_DATA_DIR: Path = DATA_DIR / "analyzed"
PLOT_DIR: Path = DATA_DIR / "plots"

SIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
ANALYZED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _to_json_serializable(value: Any) -> Any:
    """
    Convert NumPy/pandas structures recursively into JSON-friendly Python types.

    Parameters
    ----------
    value : Any
        The value to normalize before JSON serialisation.

    Returns
    -------
    Any
        A pure Python structure matching the shape of the input.
    """
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
    """
    Read an MCP resource and convert it into a plain-text string.

    Parameters
    ----------
    uri : str
        The MCP URI to load with `mcp.read_resource`.

    Returns
    -------
    str
        Decoded text content, or a placeholder if the content is binary.
    """
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
    """
    Load an MCP resource and parse its JSON payload.

    Parameters
    ----------
    uri : str
        MCP URI containing JSON content.

    Returns
    -------
    Any
        Parsed JSON payload.

    Raises
    ------
    ValueError
        If the resource is empty.
    """
    text = await _resource_to_text(uri)
    if not text.strip():
        raise ValueError(f"Resource {uri} is empty!")
    return json.loads(text)


def _resource_id_from_uri(uri: str) -> str:
    """
    Extract the final identifier segment from an MCP URI.

    Parameters
    ----------
    uri : str
        MCP URI to parse.

    Returns
    -------
    str
        The resource identifier (generates a UUID if the URI ends with '/').
    """
    candidate = uri.rstrip("/").split("/")[-1]
    return candidate or uuid4().hex


# ----------------------------------------
# RESOURCES
# ----------------------------------------


_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_resource_id(resource_id: str) -> None:
    """
    Ensure resource identifiers use a safe character set.

    Parameters
    ----------
    resource_id : str
        Candidate identifier provided by the user.

    Raises
    ------
    ValueError
        If the identifier contains unsafe characters or is not a string.
    """
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
    """
    Return the raw simulation JSON text for a resource identifier.

    Parameters
    ----------
    sim_id : str
        Identifier for the simulated ECG resource.

    Returns
    -------
    str
        Raw JSON content of the simulated ECG.
    """
    _validate_resource_id(sim_id)
    path = _safe_join(SIM_DATA_DIR, f"{sim_id}.json")
    if not path.exists():
        raise FileNotFoundError(f"Simulation resource '{sim_id}' not found")
    return path.read_text()


@mcp.resource("ecg://processed/{proc_id}", mime_type="application/json")
def read_processed(proc_id: str) -> str:
    """
    Return the processed ECG JSON text for a resource identifier.

    Parameters
    ----------
    proc_id : str
        Identifier for the processed ECG resource.

    Returns
    -------
    str
        Raw JSON content of the processed ECG.
    """
    _validate_resource_id(proc_id)
    path = _safe_join(PROCESSED_DATA_DIR, f"{proc_id}.json")
    if not path.exists():
        raise FileNotFoundError(f"Processed resource '{proc_id}' not found")
    return path.read_text()


@mcp.resource("ecg://plots/{plot_id}", mime_type="image/png")
def read_plot(plot_id: str) -> bytes:
    """
    Return the PNG bytes for a plotted ECG resource.

    Parameters
    ----------
    plot_id : str
        Identifier for the plot resource.

    Returns
    -------
    bytes
        PNG data for the requested plot.
    """
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
) -> dict:
    """
    Simulate a raw ECG signal and return a resource URI.

    Parameters
    ----------
    duration : int, optional
        Length of the signal in seconds.
    sampling_rate : int, optional
        Samples per second for the generated signal.
    noise : float, optional
        Amount of gaussian noise added to the signal.
    heart_rate : int, optional
        Target average heart rate.
    heart_rate_std : int, optional
        Standard deviation of the heart rate sampling.
    method : str, optional
        Simulation algorithm passed to `nk.ecg_simulate`.

    Returns
    -------
    dict
        Dictionary containing the generated resource URI and metadata.
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
    """
    Plot a processed ECG and return a PNG resource URI.

    Parameters
    ----------
    processed_uri : str
        URI of the processed simulation data to plot.

    Returns
    -------
    dict
        Dictionary containing the plot URI and metadata.
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
    """
    Clean a raw ECG signal and return the filtered waveform.

    Parameters
    ----------
    signal : list[float]
        Raw ECG voltage samples.
    sampling_rate : int, optional
        Sampling rate for the signal (defaults to 1000 Hz).

    Returns
    -------
    list[float]
        Filtered ECG signal samples.
    """
    cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)
    return cleaned.tolist()


@mcp.tool()
async def ecg_analyze(processed_uri: str):
    """
    Analyze a processed ECG and return an analyzed resource URI.

    Parameters
    ----------
    processed_uri : str
        URI of the processed simulation data to analyze.

    Returns
    -------
    dict
        Dict
    """
    data: dict = await _read_json_resource(uri=processed_uri)
    signals: pd.DataFrame = pd.DataFrame(data=data["signals"])
    info: dict = data["info"]

    analyze_epochs: pd.DataFrame = nk.ecg_analyze(
        data=signals,
        sampling_rate=info["sampling_rate"],
    )

    # Convert the DataFrame return value into a JSON-serializable structure
    analyze_epochs = _to_json_serializable(analyze_epochs.to_dict(orient="list"))

    analyzed_id: str = _resource_id_from_uri(uri=processed_uri)
    path: Path = ANALYZED_DATA_DIR / f"{analyzed_id}.json"
    path.write_text(data=json.dumps(obj=analyze_epochs))
    return {"uri": f"ecg://analyzed/{analyzed_id}"}


# ----------------------------------------
# MAIN
# ----------------------------------------


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
