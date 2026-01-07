"""
EMPIR Reconstruction Runner

This module provides functionality to run EMPIR reconstruction binaries
(pixel2photon and photon2event) on TPX3 and photon data files.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm
import json


class EMPIRRunner:
    """
    Handles running EMPIR reconstruction binaries on data files.
    """

    def __init__(self, empir_binaries_dir: str = "./export", verbosity: int = 1):
        """
        Initialize the EMPIR runner.

        Args:
            empir_binaries_dir: Path to directory containing EMPIR binaries
            verbosity: Output verbosity (0=quiet, 1=normal, 2=detailed)
        """
        self.empir_dir = Path(empir_binaries_dir)
        self.verbosity = verbosity

        # Check if directory exists
        if not self.empir_dir.exists():
            raise FileNotFoundError(
                f"EMPIR binaries directory not found: {self.empir_dir}\n"
                f"Set the path using --empir-binaries or EMPIR_PATH environment variable"
            )

        # Find required binaries
        self.binaries = {}
        binary_paths = {
            'pixel2photon': ['bin/empir_pixel2photon_tpx3spidr', 'empir_pixel2photon_tpx3spidr'],
            'photon2event': ['bin/empir_photon2event', 'empir_photon2event'],
            'export_photons': ['empir_export_photons'],
            'export_events': ['empir_export_events'],
            'export_pixels': ['empir_export_pixelActivations'],
        }

        for name, possible_paths in binary_paths.items():
            found = False
            for rel_path in possible_paths:
                full_path = self.empir_dir / rel_path
                if full_path.exists():
                    self.binaries[name] = full_path
                    found = True
                    break

            if not found and self.verbosity >= 1:
                print(f"Warning: {name} binary not found in {self.empir_dir}")

    def run_pixel2photon(
        self,
        tpx3_dir: Path,
        output_dir: Path,
        params: Dict[str, Any],
        n_threads: int = 4
    ) -> bool:
        """
        Run pixel-to-photon reconstruction on TPX3 files.

        Args:
            tpx3_dir: Directory containing .tpx3 files
            output_dir: Directory to save .empirphot files
            params: Dictionary with reconstruction parameters
            n_threads: Number of parallel processes

        Returns:
            True if successful, False otherwise
        """
        if 'pixel2photon' not in self.binaries:
            raise RuntimeError(
                "pixel2photon binary not found. "
                "Ensure empir_pixel2photon_tpx3spidr is in the EMPIR binaries directory."
            )

        tpx3_dir = Path(tpx3_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find TPX3 files
        tpx3_files = sorted(tpx3_dir.glob("*.tpx3"))
        if not tpx3_files:
            if self.verbosity >= 1:
                print(f"No .tpx3 files found in {tpx3_dir}")
            return False

        if self.verbosity >= 1:
            print(f"\nRunning pixel-to-photon reconstruction on {len(tpx3_files)} files...")

        # Create temporary params file
        params_file = output_dir / ".temp_params.json"
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)

        # Process files in parallel batches
        pids = []
        success = True

        try:
            iterator = tqdm(tpx3_files, desc="pixel2photon") if self.verbosity >= 1 else tpx3_files

            for tpx3_file in iterator:
                output_file = output_dir / f"{tpx3_file.stem}.empirphot"

                cmd = [
                    str(self.binaries['pixel2photon']),
                    "-i", str(tpx3_file),
                    "-o", str(output_file),
                    "--paramsFile", str(params_file)
                ]

                # Add TDC1 flag if specified
                if params.get("pixel2photon", {}).get("TDC1", False):
                    cmd.append("-T")

                if self.verbosity >= 2:
                    print(f"Running: {' '.join(cmd)}")
                    process = subprocess.Popen(cmd)
                else:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )

                pids.append(process)

                # Wait for batch to complete
                if len(pids) >= n_threads:
                    for p in pids:
                        if p.wait() != 0:
                            success = False
                            if self.verbosity >= 1:
                                print("Error occurred during pixel2photon processing")
                    pids = []

            # Wait for remaining processes
            for p in pids:
                if p.wait() != 0:
                    success = False
                    if self.verbosity >= 1:
                        print("Error occurred during pixel2photon processing")

        finally:
            # Clean up temp params file
            if params_file.exists():
                params_file.unlink()

        if self.verbosity >= 1:
            if success:
                print(f"✓ Successfully created {len(tpx3_files)} photon files")
            else:
                print("✗ Some errors occurred during pixel2photon processing")

        return success

    def run_photon2event(
        self,
        photon_dir: Path,
        output_dir: Path,
        params: Dict[str, Any],
        n_threads: int = 4
    ) -> bool:
        """
        Run photon-to-event reconstruction on photon files.

        Args:
            photon_dir: Directory containing .empirphot files
            output_dir: Directory to save .empirevent files
            params: Dictionary with reconstruction parameters
            n_threads: Number of parallel processes

        Returns:
            True if successful, False otherwise
        """
        if 'photon2event' not in self.binaries:
            raise RuntimeError(
                "photon2event binary not found. "
                "Ensure empir_photon2event is in the EMPIR binaries directory."
            )

        photon_dir = Path(photon_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find photon files
        photon_files = sorted(photon_dir.glob("*.empirphot"))
        if not photon_files:
            if self.verbosity >= 1:
                print(f"No .empirphot files found in {photon_dir}")
            return False

        if self.verbosity >= 1:
            print(f"\nRunning photon-to-event reconstruction on {len(photon_files)} files...")

        # Create temporary params file
        params_file = output_dir / ".temp_params.json"
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)

        # Process files in parallel batches
        pids = []
        success = True

        try:
            iterator = tqdm(photon_files, desc="photon2event") if self.verbosity >= 1 else photon_files

            for photon_file in iterator:
                output_file = output_dir / f"{photon_file.stem}.empirevent"

                cmd = [
                    str(self.binaries['photon2event']),
                    "-i", str(photon_file),
                    "-o", str(output_file),
                    "--paramsFile", str(params_file)
                ]

                if self.verbosity >= 2:
                    print(f"Running: {' '.join(cmd)}")
                    process = subprocess.Popen(cmd)
                else:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )

                pids.append(process)

                # Wait for batch to complete
                if len(pids) >= n_threads:
                    for p in pids:
                        if p.wait() != 0:
                            success = False
                            if self.verbosity >= 1:
                                print("Error occurred during photon2event processing")
                    pids = []

            # Wait for remaining processes
            for p in pids:
                if p.wait() != 0:
                    success = False
                    if self.verbosity >= 1:
                        print("Error occurred during photon2event processing")

        finally:
            # Clean up temp params file
            if params_file.exists():
                params_file.unlink()

        if self.verbosity >= 1:
            if success:
                print(f"✓ Successfully created {len(photon_files)} event files")
            else:
                print("✗ Some errors occurred during photon2event processing")

        return success

    def run_export_photons(self, photon_dir: Path, output_dir: Path) -> bool:
        """
        Export photon files to CSV format.

        Args:
            photon_dir: Directory containing .empirphot files
            output_dir: Directory to save CSV files

        Returns:
            True if successful, False otherwise
        """
        if 'export_photons' not in self.binaries:
            if self.verbosity >= 1:
                print("Warning: export_photons binary not found, skipping export")
            return False

        photon_dir = Path(photon_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        photon_files = sorted(photon_dir.glob("*.empirphot"))
        if not photon_files:
            return False

        if self.verbosity >= 1:
            print(f"\nExporting {len(photon_files)} photon files to CSV...")

        success = True
        iterator = tqdm(photon_files, desc="Exporting photons") if self.verbosity >= 1 else photon_files

        for photon_file in iterator:
            output_file = output_dir / f"exported_{photon_file.stem}.csv"

            cmd = [
                str(self.binaries['export_photons']),
                str(photon_file),
                str(output_file),
                "csv"
            ]

            try:
                if self.verbosity >= 2:
                    subprocess.run(cmd, check=True)
                else:
                    subprocess.run(
                        cmd,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
            except subprocess.CalledProcessError:
                success = False
                if self.verbosity >= 1:
                    print(f"Error exporting {photon_file.name}")

        return success

    def run_export_events(self, event_dir: Path, output_dir: Path) -> bool:
        """
        Export event files to CSV format.

        Args:
            event_dir: Directory containing .empirevent files
            output_dir: Directory to save CSV files

        Returns:
            True if successful, False otherwise
        """
        if 'export_events' not in self.binaries:
            if self.verbosity >= 1:
                print("Warning: export_events binary not found, skipping export")
            return False

        event_dir = Path(event_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        event_files = sorted(event_dir.glob("*.empirevent"))
        if not event_files:
            return False

        if self.verbosity >= 1:
            print(f"\nExporting {len(event_files)} event files to CSV...")

        success = True
        iterator = tqdm(event_files, desc="Exporting events") if self.verbosity >= 1 else event_files

        for event_file in iterator:
            output_file = output_dir / f"{event_file.stem}.csv"

            cmd = [
                str(self.binaries['export_events']),
                str(event_file),
                str(output_file),
                "csv"
            ]

            try:
                if self.verbosity >= 2:
                    subprocess.run(cmd, check=True)
                else:
                    subprocess.run(
                        cmd,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
            except subprocess.CalledProcessError:
                success = False
                if self.verbosity >= 1:
                    print(f"Error exporting {event_file.name}")

        return success


def get_default_params() -> Dict[str, Any]:
    """
    Get default EMPIR reconstruction parameters.

    Returns:
        Dictionary with default parameters for pixel2photon and photon2event
    """
    return {
        "pixel2photon": {
            "dSpace": 2.0,
            "dTime": 100e-9,
            "nPxMin": 8,
            "nPxMax": 100,
            "TDC1": False
        },
        "photon2event": {
            "dSpace_px": 50.0,
            "dTime_s": 50e-9,
            "durationMax_s": 500e-9
        }
    }
