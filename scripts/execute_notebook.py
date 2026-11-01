"""Execute the companion with the current interpreter, avoiding global kernels."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
import json
import tempfile
import argparse

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('notebook',nargs='?',default='notebooks/energy_price_pilot.ipynb')
args=parser.parse_args()
path=(root/args.notebook).resolve()
with tempfile.TemporaryDirectory() as temp:
    kernel=Path(temp)/'pilot'; kernel.mkdir()
    (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                         'display_name':'Energy pilot local','language':'python'}))
    manager=KernelSpecManager(kernel_dirs=[temp])
    notebook=nbformat.read(path,as_version=4)
    client=NotebookClient(notebook,timeout=1200,kernel_name='pilot',resources={'metadata':{'path':str(root)}},kernel_manager_class=__import__('jupyter_client').KernelManager)
    from jupyter_client import KernelManager
    client.km=KernelManager(kernel_name='pilot',kernel_spec_manager=manager)
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook,path)
print('Notebook executed top-to-bottom and saved:',path)
