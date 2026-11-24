"""Run a script (an example) under the strict-forms recorder: ``python strict_run_script.py <script> <out dir>``."""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
script, out = sys.argv[1], sys.argv[2]
os.environ['STRICT_OUT'] = out
import strict_forms_plugin as sfp                       # noqa: E402

sfp.OUT_DIR = out
sfp.SCRIPT['name'] = os.path.relpath(script, sfp.REPO_ROOT)
sfp.install()
sys.argv = [script]
try:
    runpy.run_path(script, run_name='__main__')
finally:
    sfp.dump()
