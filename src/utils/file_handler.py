"""
File Handler — Reads uploaded files (PDF, TXT, CSV, MD, JSON)
and returns normalized text chunks.
"""

from __future__ import annotations
import io
import json
import csv
from typing import List, Dict


class FileHandler:

    def process_files(self, uploaded_files) -> List[Dict[str, str]]:
        """Process a list of Streamlit UploadedFile objects."""
        results = []
        for f in uploaded_files:
            ext = f.name.rsplit(".", 1)[-1].lower()
            try:
                text = self._extract(f, ext)
                if text.strip():
                    results.append({"filename": f.name, "content": text})
            except Exception as e:
                results.append({
                    "filename": f.name,
                    "content": f"[Error reading {f.name}: {str(e)}]"
                })
        return results

    def _extract(self, file, ext: str) -> str:
        if ext == "pdf":
            return self._read_pdf(file)
        elif ext in ("txt", "md"):
            return file.read().decode("utf-8", errors="ignore")
        elif ext == "csv":
            return self._read_csv(file)
        elif ext == "json":
            return self._read_json(file)
        else:
            return file.read().decode("utf-8", errors="ignore")

    def _read_pdf(self, file) -> str:
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                return "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except ImportError:
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
                return "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
            except ImportError:
                return "[PDF reading requires pdfplumber or PyPDF2. Install with: pip install pdfplumber]"

    def _read_csv(self, file) -> str:
        content = file.read().decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        if not rows:
            return content
        lines = []
        for row in rows[:200]:   # cap at 200 rows for embedding
            lines.append(", ".join(f"{k}: {v}" for k, v in row.items()))
        return "\n".join(lines)

    def _read_json(self, file) -> str:
        try:
            data = json.loads(file.read().decode("utf-8", errors="ignore"))
            return json.dumps(data, indent=2)[:10000]  # cap size
        except json.JSONDecodeError as e:
            return f"[JSON parse error: {e}]"
