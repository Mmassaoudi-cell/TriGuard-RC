"""LaTeX needs the last row of an \\input'ed table body to be unterminated (the caller adds the final \\\\)."""
import glob
from pathlib import Path
for f in glob.glob(str(Path(__file__).parent.parent / "TriGuard-RC-Paper" / "table_*.tex")):
    s = Path(f).read_text(encoding="utf8").rstrip()
    while s.endswith("\\\\"):
        s = s[:-2].rstrip()
    Path(f).write_text(s + "\n", encoding="utf8")
