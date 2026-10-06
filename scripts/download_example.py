"""Fetch the public SmolLM2-135M base checkpoint (~270 MB) for a real-model smoke run.

No remote Python code is executed. Check the model card/license before using
weights for another purpose. Download is explicit and never part of unit tests.
"""
from pathlib import Path
import urllib.request


def main():
    target=Path('checkpoints/SmolLM2-135M'); target.mkdir(parents=True,exist_ok=True)
    base='https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/'
    for name in ['config.json','tokenizer.json','tokenizer_config.json','model.safetensors']:
        path=target/name
        if path.exists(): continue
        temporary=path.with_suffix(path.suffix+'.partial')
        print('Downloading',name,flush=True)
        urllib.request.urlretrieve(base+name,temporary)
        temporary.replace(path)
    print('Saved',target)


if __name__=='__main__': main()
