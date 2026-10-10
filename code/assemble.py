"""assemble.py [RESULTS_DIR] -- after run_cutest_batch.sh has finished: builds all tables (final_tables.py)
and writes every number quoted in Section 3.3 and in the response letter to tables/section33_numbers.txt
(final_text.py)."""
import sys, io, contextlib, runpy
from pathlib import Path
import final_tables

HERE = Path(__file__).parent
final_tables.main()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path(str(HERE / "final_text.py"), run_name="__main__")
(HERE / "tables" / "section33_numbers.txt").write_text(buf.getvalue())
print(buf.getvalue())
print("wrote tables/section33_numbers.txt")
