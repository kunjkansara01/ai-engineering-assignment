"""Execute notebook Python cells in order without kernel sockets.

Uses an in-process IPython shell and saves captured outputs. This does not test
Google Colab or an external Jupyter kernel. Run from a fresh Python process.
"""
import argparse
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.io import capture_output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('notebook',nargs='?',default='Assignment_Solutions.ipynb')
    args=parser.parse_args()
    path=Path(args.notebook).resolve()
    nb=nbformat.read(path,as_version=4)
    shell=InteractiveShell.instance()
    count=0
    for cell in nb.cells:
        if cell.cell_type!='code':
            continue
        count+=1
        print('Executing code cell',count,flush=True)
        with capture_output() as captured:
            result=shell.run_cell(cell.source,store_history=False)
        outputs=[]
        for stream in ('stdout','stderr'):
            value=getattr(captured,stream)
            if value:
                outputs.append(nbformat.v4.new_output('stream',name=stream,text=value))
        for displayed in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data',data=displayed.data,metadata=displayed.metadata))
        cell.outputs=outputs
        cell.execution_count=count
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'Notebook cell {count} failed: {result.error_before_exec or result.error_in_exec}')
    nb.metadata['validation']={'method':'Fresh Python process; sequential IPython in-process cell execution',
                               'code_cells_passed':count,'external_jupyter_kernel_tested':False,
                               'reason':'Execution host blocks local TCP and IPC socket binding'}
    nbformat.validate(nb)
    nbformat.write(nb,path)
    print('ALL_NOTEBOOK_CELLS_PASSED',count)


if __name__=='__main__':
    main()
