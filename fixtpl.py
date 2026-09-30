import re
from pathlib import Path
P = Path(__file__).parent.parent / "TriGuard-RC-Paper-R1"
s = (P / "sec_results.tex.tpl").read_text(encoding="utf8")
s = re.sub(r"(\\input\{table_[a-z_]+\.tex\})( \\\\)?\n", lambda m: m.group(1) + " \\\\\n", s)
(P / "sec_results.tex.tpl").write_text(s, encoding="utf8")
print(re.findall(r"\\input\{table_[a-z_]+\.tex\}[^\n]*", s))
