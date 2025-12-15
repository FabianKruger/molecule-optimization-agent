"""
TDC Oracle wrapper using subprocess isolation.

This module calls TDC oracles via a subprocess running in a separate pixi environment
to avoid dependency conflicts (especially sklearn version issues).
"""

import json
import subprocess
import sys
from pathlib import Path

from .base import OracleResult


# Find the project root (where pyproject.toml lives)
def _find_project_root() -> Path:
    """Find the project root by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")


class TdcOracle:
    """
    Generic wrapper for TDC oracles using subprocess isolation.
    
    This oracle wraps any TDC oracle (e.g., DRD2, GSK3, JNK3, etc.)
    by calling a subprocess in the isolated 'tdc' pixi environment.
    """

    def __init__(self, oracle_name: str, target_smiles: str | None = None, **kwargs):
        """
        Initialize TDC oracle.
        
        Args:
            oracle_name: Name of the TDC oracle (e.g., 'drd2', 'gsk3b', 'jnk3')
            target_smiles: Optional target SMILES for meta-oracles
            **kwargs: Additional parameters (currently unused, for forward compatibility)
        """
        self.oracle_name = oracle_name
        self.target_smiles = target_smiles
        self._project_root = _find_project_root()
        self._script_path = self._project_root / "scripts" / "tdc_oracle_subprocess.py"
        
        if not self._script_path.exists():
            raise RuntimeError(f"TDC subprocess script not found at {self._script_path}")

    def __call__(self, smiles: str) -> OracleResult:
        """
        Evaluate a molecule using the TDC oracle via subprocess.
        
        Args:
            smiles: SMILES string of the molecule to evaluate
            
        Returns:
            OracleResult with score and empty explanation
        """
        # Build command
        cmd = [
            "pixi", "run", "-e", "tdc",
            "python", str(self._script_path),
            self.oracle_name,
            smiles,
        ]
        
        if self.target_smiles:
            cmd.extend(["--target_smiles", self.target_smiles])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self._project_root,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout for model loading
            )
            
            if result.returncode != 0:
                raise ValueError(
                    f"TDC subprocess failed with code {result.returncode}.\n"
                    f"stderr: {result.stderr}"
                )
            
            # Parse JSON output from the last line (TDC/RDKit may print noise before)
            stdout_lines = result.stdout.strip().split("\n")
            if not stdout_lines or not stdout_lines[-1]:
                raise ValueError(f"TDC subprocess produced no output.\nstderr: {result.stderr}")
            output = json.loads(stdout_lines[-1])
            
            if "error" in output:
                raise ValueError(f"TDC oracle error: {output['error']}")
            
            score = output["score"]
            
        except subprocess.TimeoutExpired:
            raise ValueError(f"TDC oracle timed out for molecule {smiles}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse TDC output: {e}\nOutput: {result.stdout}")
        
        return {
            "score": score,
            "explanation": "",  # No explainability for TDC oracles
        }

    def get_params(self) -> dict:
        """Return oracle configuration for sharing with objectives."""
        return {
            "oracle_name": self.oracle_name,
            "target_smiles": self.target_smiles,
        }
