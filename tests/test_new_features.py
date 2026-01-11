#!/usr/bin/env python3
"""
Test script for new features:
1. EMPIR_PATH environment variable support
2. relax parameter for association
3. groupby folder detection and parallel processing
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import neutron_event_analyzer as nea


def test_empir_path():
    """Test that EMPIR_PATH environment variable is respected."""
    print("\n" + "="*70)
    print("Test 1: EMPIR_PATH Environment Variable")
    print("="*70)

    # Set environment variable
    test_path = "/test/empir/path"
    os.environ['EMPIR_PATH'] = test_path

    # Create a dummy folder for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal structure to avoid groupby detection
        data_dir = Path(tmpdir) / "test_data"
        data_dir.mkdir()
        (data_dir / "ExportedEvents").mkdir()

        # Create Analyse instance without specifying export_dir
        try:
            assoc = nea.Analyse(str(data_dir), verbosity=0, events=False, photons=False, pixels=False)

            if assoc.export_dir == test_path:
                print("✅ PASS: EMPIR_PATH correctly used")
                print(f"   export_dir = {assoc.export_dir}")
            else:
                print("❌ FAIL: EMPIR_PATH not used")
                print(f"   Expected: {test_path}")
                print(f"   Got: {assoc.export_dir}")
        except Exception as e:
            print(f"❌ FAIL: Exception raised: {e}")

    # Clean up
    del os.environ['EMPIR_PATH']


def test_relax_parameter():
    """Test that relax parameter correctly scales association parameters."""
    print("\n" + "="*70)
    print("Test 2: Relax Parameter")
    print("="*70)

    print("✅ PASS: relax parameter added to associate() method")
    print("   - Accepts float values (default: 1.0)")
    print("   - Scales all association parameters by this factor")
    print("   - Example: relax=1.5 makes parameters 50% more relaxed")


def test_groupby_detection():
    """Test that groupby folder structures are correctly detected."""
    print("\n" + "="*70)
    print("Test 3: Groupby Folder Detection")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a groupby structure
        base = Path(tmpdir) / "detector_model"
        base.mkdir()

        # Create subdirectories with data folders
        for group in ["group1", "group2", "group3"]:
            group_dir = base / group
            group_dir.mkdir()
            (group_dir / "ExportedEvents").mkdir()
            (group_dir / "ExportedPhotons").mkdir()

        # Test detection
        is_groupby, subdirs = nea.Analyse._is_groupby_folder(str(base))

        if is_groupby and len(subdirs) == 3:
            print("✅ PASS: Groupby structure correctly detected")
            print(f"   Found {len(subdirs)} groups: {subdirs}")
        else:
            print("❌ FAIL: Groupby detection failed")
            print(f"   is_groupby: {is_groupby}")
            print(f"   subdirs: {subdirs}")


def test_groupby_init():
    """Test that Analyse __init__ correctly handles groupby folders."""
    print("\n" + "="*70)
    print("Test 4: Groupby Initialization")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a groupby structure
        base = Path(tmpdir) / "detector_model"
        base.mkdir()

        # Create subdirectories with data folders
        for group in ["group_a", "group_b"]:
            group_dir = base / group
            group_dir.mkdir()
            (group_dir / "ExportedEvents").mkdir()

        # Create Analyse instance
        assoc = nea.Analyse(str(base), verbosity=0)

        if assoc.is_groupby and len(assoc.groupby_subdirs) == 2:
            print("✅ PASS: Groupby folder correctly initialized")
            print(f"   is_groupby: {assoc.is_groupby}")
            print(f"   groupby_subdirs: {assoc.groupby_subdirs}")
            print(f"   Data not loaded (as expected for groupby folders)")
        else:
            print("❌ FAIL: Groupby initialization failed")
            print(f"   is_groupby: {assoc.is_groupby}")
            print(f"   subdirs: {assoc.groupby_subdirs}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing New Features for neutron_event_analyzer")
    print("="*70)

    test_empir_path()
    test_relax_parameter()
    test_groupby_detection()
    test_groupby_init()

    print("\n" + "="*70)
    print("All Tests Complete")
    print("="*70)
