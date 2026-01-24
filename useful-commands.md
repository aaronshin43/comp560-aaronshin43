# Useful commands

Add to the Python path then check to see what directories are actually searched for modules etc.
```
PYTHONPATH="../comp560-nanoGPT/":"." python -c "import sys; print('\n'.join(sys.path))"
```


Training:
```
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py  python -u ../../comp560-nanoGPT/train.py config/basic.py
```


Sampling:
```
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py  python -u ../../comp560-nanoGPT/sample.py config/basic.py --num_samples=1 --max_new_tokens=100 --seed=2
```

Copied from [Prof. MacCormick's repository](https://github.com/dson-comp560-sp26/comp560-jmac)