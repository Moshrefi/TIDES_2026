"""Default notebook must be a safe preview from root or notebooks/, without GPU deps."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class NotebookContractTests(unittest.TestCase):
    def test_generated_notebook_matches_source(self):
        spec = importlib.util.spec_from_file_location('build_nb', ROOT/'scripts'/'build_eds_notebook.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual((ROOT/'notebooks'/'01_eds_paper_reproduction.ipynb').read_text(encoding='utf-8'), module.build())

    def test_default_run_all_never_launches_stages(self):
        notebook = json.loads((ROOT/'notebooks'/'01_eds_paper_reproduction.ipynb').read_text(encoding='utf-8'))
        display = types.ModuleType('IPython.display')
        display.display = lambda *args, **kwargs: None
        ipython = types.ModuleType('IPython')
        ipython.display = display
        env = {k:v for k,v in os.environ.items() if not k.startswith('TIDES_')}
        original = Path.cwd()
        for start in [ROOT, ROOT/'notebooks']:
            with self.subTest(start=start), patch.dict(os.environ, env, clear=True), \
                    patch.dict(sys.modules, {'IPython':ipython, 'IPython.display':display}), \
                    patch('subprocess.run', side_effect=AssertionError('Default notebook launched a subprocess')), \
                    patch('subprocess.Popen', side_effect=AssertionError('Default notebook launched a process')), \
                    patch('builtins.print'):
                try:
                    os.chdir(start)
                    ns = {'__name__':'__main__'}
                    for i, cell in enumerate(notebook['cells']):
                        if cell['cell_type'] == 'code':
                            code = ''.join(cell['source'])
                            ast.parse(code)
                            exec(compile(code, f'notebook-cell-{i}', 'exec'), ns)
                    self.assertEqual(ns['STAGES'], set())
                    self.assertEqual(ns['ROOT'], ROOT)
                finally:
                    os.chdir(original)


if __name__ == '__main__':
    unittest.main()
