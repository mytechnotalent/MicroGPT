#!/usr/bin/env python3

"""
Convert TUTORIAL.md to PDF with beautiful formatting.

This script converts the GPT-2 tutorial from Markdown to PDF using
pandoc with xelatex for proper Unicode support.

Requirements:
    brew install pandoc
    brew install --cask mactex  # or basictex for minimal install

Usage:
    python convert_tutorial_to_pdf.py

Author: Kevin Thomas (ket189@pitt.edu)
License: MIT
"""

import subprocess
import sys
from pathlib import Path


def _check_pandoc_installed() -> bool:
    """Check if pandoc is installed.

    Returns:
        True if pandoc is available.

    Example:
        >>> _check_pandoc_installed()
        True
    """
    try:
        subprocess.run(
            ["pandoc", "--version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _convert_with_pandoc(
    input_file: str,
    output_file: str,
) -> None:
    """Convert markdown to PDF using pandoc.

    Args:
        input_file: Path to input markdown file.
        output_file: Path to output PDF file.

    Returns:
        None

    Example:
        >>> _convert_with_pandoc("GPT2_Tutorial.md", "GPT2_Tutorial.pdf")
    """
    cmd = [
        "pandoc",
        input_file,
        "-o",
        output_file,
        "--pdf-engine=xelatex",
        "-V",
        "geometry:margin=1in",
        "-V",
        "fontsize=11pt",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Pandoc failed: {result.stderr}")


def convert_markdown_to_pdf(
    input_file: str,
    output_file: str,
) -> None:
    """Convert markdown file to PDF.

    Args:
        input_file: Path to input markdown file.
        output_file: Path to output PDF file.

    Returns:
        None

    Example:
        >>> convert_markdown_to_pdf("GPT2_Tutorial.md", "GPT2_Tutorial.pdf")
    """
    print(f"Reading: {input_file}")
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    content = input_path.read_text(encoding="utf-8")
    print(f"Content: {len(content):,} characters")
    print("Converting to PDF with pandoc...")
    _convert_with_pandoc(input_file, output_file)
    print(f"PDF Created Successfully: {output_file}")
    output_path = Path(output_file)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")


def main() -> None:
    """Main function to run the conversion.

    Returns:
        None

    Example:
        >>> main()  # Converts GPT2_Tutorial.md to GPT2_Tutorial.pdf
    """
    input_file = "GPT2_Tutorial.md"
    output_file = "GPT2_Tutorial.pdf"
    print("=" * 60)
    print("GPT-2 Tutorial -> PDF Converter")
    print("=" * 60)
    print()
    if not _check_pandoc_installed():
        print("Error: pandoc not installed!")
        print()
        print("Install with: brew install pandoc")
        sys.exit(1)
    try:
        convert_markdown_to_pdf(input_file, output_file)
        print()
        print("Conversion complete!")
        print(f"You can now open: {output_file}")
    except Exception as e:
        print()
        print(f"Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
