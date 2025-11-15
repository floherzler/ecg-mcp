import json
import neurokit2 as nk
from mcp.server.fastmcp import FastMCP
from uuid import uuid4

mcp = FastMCP("ecg-mcp")

"""
MCP Cheat Sheet

Server:
    - Declares tools via @mcp.tool() with JSON-schema args
    - Tool functions return JSON-serializable Python objects
      (dict, list, str, int, float, bool, None)
    - Can optionally expose read-only resources (e.g., files, APIs)

Client:
    - Connects to an MCP server and reads its manifest
    - Lists available tools and resources
    - Calls tools by sending JSON arguments, receives JSON results

Agent (optional):
    - Wraps an LLM with goals/policy/guardrails
    - Chooses which tools/resources to call
    - Interprets results and produces the next LLM message or action

"""


@mcp.tool()
async def ecg_simulate(
    duration: int = 10,
    sampling_rate: int = 1000,
    noise: float = 0.01,
    heart_rate: int = 70,
    heart_rate_std: int = 1,
    method: str = "ecgsyn",
) -> dict:
    """Simulate an ECG signal using NeuroKit2."""
    sim_ecg = nk.ecg_simulate(
        duration=duration,
        sampling_rate=sampling_rate,
        noise=noise,
        heart_rate=heart_rate,
        heart_rate_std=heart_rate_std,
        method=method,
    )
    uuid = str(uuid4())
    path = f"/tmp/ecg_simulation_{uuid}.json"
    with open(path, "w") as f:
        json.dump(sim_ecg.tolist(), f)

    # 3) return a lightweight handle in the tool result
    return {
        "resource_id": f"ecg://simulations/{uuid}",
        "info": {
            "duration": duration,
            "sampling_rate": sampling_rate,
            "n_samples": len(sim_ecg),
        },
    }


@mcp.tool()
async def ecg_clean(
    signal: list[float],
    sampling_rate: int = 1000,
) -> list[float]:
    """Clean an ECG signal using NeuroKit2."""
    cleaned_ecg = nk.ecg_clean(
        ecg_signal=signal,
        sampling_rate=sampling_rate,
    )
    return cleaned_ecg.tolist()


def main():
    # Initialize and run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
