from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from openpyxl import Workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from str_core.pipeline import convert_path


def test_html_heuristic_conversion():
    html = """
    <table>
      <tr><th>Company</th><th>Revenue</th><th>Profit</th></tr>
      <tr><td>A</td><td>100</td><td>20</td></tr>
      <tr><td>B</td><td>200</td><td>50</td></tr>
    </table>
    """
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.html"
        path.write_text(html, encoding="utf-8")
        triplet = convert_path(path, input_type="html", use_llm="never")
    assert triplet["shape"] == "3*3"
    assert triplet["features"] == ["Revenue", "Profit"]
    assert {"item": "B", "feature": "Profit", "value": "50"} in triplet["group"]


def test_excel_heuristic_conversion():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.append(["Company", "Revenue", "Profit"])
        ws.append(["A", "100", "20"])
        ws.append(["B", "200", "50"])
        wb.save(path)
        wb.close()
        triplet = convert_path(path, input_type="excel", use_llm="never")
    assert triplet["shape"] == "3*3"
    assert len(triplet["group"]) == 4


if __name__ == "__main__":
    test_html_heuristic_conversion()
    test_excel_heuristic_conversion()
    print("smoke tests passed")
