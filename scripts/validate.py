"""Run the complete suite and retain its actual output and timestamp."""
from datetime import datetime,timezone
from pathlib import Path
import os
import subprocess
import sys

env=dict(os.environ,RUN_CUDA_TESTS='1')
run=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],
                   env=env,capture_output=True,text=True)
text=datetime.now(timezone.utc).isoformat()+'\n'+run.stdout+run.stderr
Path('results').mkdir(exist_ok=True)
Path('results/tests.txt').write_text(text,encoding='utf-8')
print(text)
sys.exit(run.returncode)
