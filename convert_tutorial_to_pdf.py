#!/usr/bin/env python3
"""
Convert TUTORIAL.md to PDF with beautiful formatting.

This script converts the MicroGPT tutorial from Markdown to PDF while
preserving code blocks, syntax highlighting, tables, and formatting.

Requirements:
    pip install markdown weasyprint pygments

Usage:
    python convert_tutorial_to_pdf.py
"""

import markdown
from weasyprint import HTML, CSS
from pathlib import Path
import sys


def create_css_styles():
    """
    Create CSS styles for beautiful PDF formatting.

    Returns:
        str: CSS stylesheet for the PDF
    """
    return """
        @page {
            size: letter;
            margin: 1in 0.75in;
            
            @top-center {
                content: "MicroGPT Tutorial";
                font-size: 10pt;
                color: #666;
            }
            
            @bottom-center {
                content: "Page " counter(page) " of " counter(pages);
                font-size: 10pt;
                color: #666;
            }
        }
        
        body {
            font-family: 'Georgia', 'Times New Roman', serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
            max-width: 100%;
        }
        
        h1 {
            color: #2c3e50;
            font-size: 28pt;
            font-weight: bold;
            margin-top: 0.5in;
            margin-bottom: 0.3in;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10pt;
            page-break-after: avoid;
        }
        
        h2 {
            color: #34495e;
            font-size: 20pt;
            font-weight: bold;
            margin-top: 0.4in;
            margin-bottom: 0.2in;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8pt;
            page-break-after: avoid;
        }
        
        h3 {
            color: #34495e;
            font-size: 16pt;
            font-weight: bold;
            margin-top: 0.25in;
            margin-bottom: 0.15in;
            page-break-after: avoid;
        }
        
        h4 {
            color: #555;
            font-size: 13pt;
            font-weight: bold;
            margin-top: 0.2in;
            margin-bottom: 0.1in;
            page-break-after: avoid;
        }
        
        p {
            margin-bottom: 12pt;
            text-align: justify;
            orphans: 3;
            widows: 3;
        }
        
        code {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 9pt;
            background-color: #f4f4f4;
            padding: 2pt 4pt;
            border-radius: 3pt;
            color: #c7254e;
        }
        
        pre {
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-left: 4px solid #3498db;
            border-radius: 4pt;
            padding: 12pt;
            margin: 12pt 0;
            overflow-x: auto;
            page-break-inside: avoid;
        }
        
        pre code {
            background-color: transparent;
            padding: 0;
            color: #333;
            font-size: 9pt;
            line-height: 1.4;
        }
        
        ul, ol {
            margin-bottom: 12pt;
            padding-left: 30pt;
        }
        
        li {
            margin-bottom: 6pt;
        }
        
        blockquote {
            margin: 12pt 0;
            padding-left: 15pt;
            border-left: 4pt solid #3498db;
            color: #555;
            font-style: italic;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 12pt 0;
            page-break-inside: avoid;
        }
        
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
            padding: 8pt;
            text-align: left;
            border: 1px solid #2980b9;
        }
        
        td {
            padding: 8pt;
            border: 1px solid #ddd;
        }
        
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        
        a {
            color: #3498db;
            text-decoration: none;
        }
        
        hr {
            border: none;
            border-top: 2px solid #ddd;
            margin: 20pt 0;
        }
        
        .toc {
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            padding: 15pt;
            margin: 20pt 0;
            border-radius: 4pt;
        }
        
        .toc ul {
            list-style-type: none;
            padding-left: 15pt;
        }
        
        .toc li {
            margin-bottom: 6pt;
        }
        
        strong {
            color: #2c3e50;
            font-weight: bold;
        }
        
        em {
            font-style: italic;
            color: #555;
        }
        
        /* Prevent breaking inside important elements */
        .keep-together {
            page-break-inside: avoid;
        }
        
        /* Syntax highlighting for code */
        .codehilite .hll { background-color: #ffffcc }
        .codehilite .c { color: #408080; font-style: italic }
        .codehilite .k { color: #008000; font-weight: bold }
        .codehilite .o { color: #666666 }
        .codehilite .cm { color: #408080; font-style: italic }
        .codehilite .cp { color: #BC7A00 }
        .codehilite .c1 { color: #408080; font-style: italic }
        .codehilite .cs { color: #408080; font-style: italic }
        .codehilite .gd { color: #A00000 }
        .codehilite .ge { font-style: italic }
        .codehilite .gr { color: #FF0000 }
        .codehilite .gh { color: #000080; font-weight: bold }
        .codehilite .gi { color: #00A000 }
        .codehilite .go { color: #888888 }
        .codehilite .gp { color: #000080; font-weight: bold }
        .codehilite .gs { font-weight: bold }
        .codehilite .gu { color: #800080; font-weight: bold }
        .codehilite .gt { color: #0044DD }
        .codehilite .kc { color: #008000; font-weight: bold }
        .codehilite .kd { color: #008000; font-weight: bold }
        .codehilite .kn { color: #008000; font-weight: bold }
        .codehilite .kp { color: #008000 }
        .codehilite .kr { color: #008000; font-weight: bold }
        .codehilite .kt { color: #B00040 }
        .codehilite .m { color: #666666 }
        .codehilite .s { color: #BA2121 }
        .codehilite .na { color: #7D9029 }
        .codehilite .nb { color: #008000 }
        .codehilite .nc { color: #0000FF; font-weight: bold }
        .codehilite .no { color: #880000 }
        .codehilite .nd { color: #AA22FF }
        .codehilite .ni { color: #999999; font-weight: bold }
        .codehilite .ne { color: #D2413A; font-weight: bold }
        .codehilite .nf { color: #0000FF }
        .codehilite .nl { color: #A0A000 }
        .codehilite .nn { color: #0000FF; font-weight: bold }
        .codehilite .nt { color: #008000; font-weight: bold }
        .codehilite .nv { color: #19177C }
        .codehilite .ow { color: #AA22FF; font-weight: bold }
        .codehilite .w { color: #bbbbbb }
        .codehilite .mb { color: #666666 }
        .codehilite .mf { color: #666666 }
        .codehilite .mh { color: #666666 }
        .codehilite .mi { color: #666666 }
        .codehilite .mo { color: #666666 }
        .codehilite .sb { color: #BA2121 }
        .codehilite .sc { color: #BA2121 }
        .codehilite .sd { color: #BA2121; font-style: italic }
        .codehilite .s2 { color: #BA2121 }
        .codehilite .se { color: #BB6622; font-weight: bold }
        .codehilite .sh { color: #BA2121 }
        .codehilite .si { color: #BB6688; font-weight: bold }
        .codehilite .sx { color: #008000 }
        .codehilite .sr { color: #BB6688 }
        .codehilite .s1 { color: #BA2121 }
        .codehilite .ss { color: #19177C }
        .codehilite .bp { color: #008000 }
        .codehilite .vc { color: #19177C }
        .codehilite .vg { color: #19177C }
        .codehilite .vi { color: #19177C }
        .codehilite .il { color: #666666 }
    """


def convert_markdown_to_pdf(input_file: str, output_file: str):
    """
    Convert Markdown file to PDF with beautiful formatting.

    Args:
        input_file: Path to input Markdown file
        output_file: Path to output PDF file
    """
    print(f"📖 Reading {input_file}...")

    # Read markdown file
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ Error: {input_file} not found!")
        sys.exit(1)

    markdown_text = input_path.read_text(encoding="utf-8")

    print("🔄 Converting Markdown to HTML...")

    # Convert markdown to HTML with extensions
    md = markdown.Markdown(
        extensions=[
            "extra",  # Tables, fenced code blocks, etc.
            "codehilite",  # Syntax highlighting
            "toc",  # Table of contents
            "sane_lists",  # Better list handling
            "nl2br",  # Newline to <br>
        ]
    )

    html_content = md.convert(markdown_text)

    # Create complete HTML document
    html_doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>MicroGPT Tutorial</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    print("🎨 Applying styles...")

    # Create CSS
    css = CSS(string=create_css_styles())

    print(f"📄 Generating PDF: {output_file}...")

    # Convert HTML to PDF
    HTML(string=html_doc).write_pdf(output_file, stylesheets=[css])

    print(f"✅ PDF created successfully: {output_file}")

    # Get file size
    output_path = Path(output_file)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"📊 File size: {size_mb:.2f} MB")


def main():
    """Main function to run the conversion."""
    input_file = "TUTORIAL.md"
    output_file = "MicroGPT_Tutorial.pdf"

    print("=" * 60)
    print("MicroGPT Tutorial → PDF Converter")
    print("=" * 60)
    print()

    try:
        convert_markdown_to_pdf(input_file, output_file)
        print()
        print("🎉 Conversion complete!")
        print(f"📖 You can now open: {output_file}")

    except ImportError as e:
        print()
        print("❌ Missing dependencies!")
        print()
        print("Please install required packages:")
        print("  pip install markdown weasyprint pygments")
        print()
        print(f"Error details: {e}")
        sys.exit(1)

    except Exception as e:
        print()
        print(f"❌ Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
