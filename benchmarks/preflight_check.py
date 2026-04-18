#!/usr/bin/env python3
"""Preflight checks for M5 Pro benchmarking environment."""

import subprocess
import sys
from pathlib import Path

REQUIRED_MODELS = [
    "nomic-embed-text",
    "mxbai-embed-large",
    "embeddinggemma",
    "bge-m3"
]


def check_ollama_running() -> bool:
    """Check if Ollama daemon is running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_models_available() -> dict:
    """Check which required models are available."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        installed_models = result.stdout.lower()

        status = {}
        for model in REQUIRED_MODELS:
            status[model] = model.lower() in installed_models

        return status
    except Exception as e:
        return {m: False for m in REQUIRED_MODELS}


def check_python_version() -> tuple:
    """Check Python version is 3.10+."""
    version = sys.version_info
    return (version.major, version.minor)


def capture_hardware_specs(output_dir: Path):
    """Capture hardware specs for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hardware spec
    result = subprocess.run(
        ["system_profiler", "SPHardwareDataType"],
        capture_output=True,
        text=True
    )
    (output_dir / "hardware_spec.txt").write_text(result.stdout)

    # OS version
    result = subprocess.run(
        ["sw_vers"],
        capture_output=True,
        text=True
    )
    (output_dir / "os_version.txt").write_text(result.stdout)

    # Git commit
    repo_dir = Path(__file__).parent.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_dir
        )
        (output_dir / "git_commit.txt").write_text(result.stdout.strip())
    except Exception:
        (output_dir / "git_commit.txt").write_text("not a git repo")


def main():
    """Run all preflight checks."""
    print("=" * 60)
    print("M5 Pro Benchmarking Preflight Checks")
    print("=" * 60)

    errors = []

    # Check Ollama
    print("\n[1/4] Checking Ollama daemon...")
    if check_ollama_running():
        print("  ✅ Ollama is running")
    else:
        print("  ❌ Ollama is NOT running. Start with: ollama serve")
        errors.append("ollama_not_running")

    # Check models
    print("\n[2/4] Checking required models...")
    model_status = check_models_available()
    for model, installed in model_status.items():
        if installed:
            print(f"  ✅ {model}")
        else:
            print(f"  ❌ {model} - Run: ollama pull {model}")
            errors.append(f"missing_model:{model}")

    # Check Python
    print("\n[3/4] Checking Python version...")
    major, minor = check_python_version()
    if major >= 3 and minor >= 10:
        print(f"  ✅ Python {major}.{minor}")
    else:
        print(f"  ❌ Python {major}.{minor} - Need 3.10+")
        errors.append("python_version")

    # Capture specs
    print("\n[4/4] Capturing hardware specs...")
    output_dir = Path(__file__).parent.parent / "results"
    capture_hardware_specs(output_dir)
    print(f"  ✅ Specs saved to {output_dir}/")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"❌ FAILED: {len(errors)} issue(s) found")
        for e in errors:
            print(f"   - {e}")
        sys.exit(1)
    else:
        print("✅ All preflight checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()