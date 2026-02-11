# Todo
- [ ] Research about how train/loss val/loss is evaluated (Check if inputs are considered as well)
- [ ] If input are considered, figure out way to only evaluate on outputs 
- [ ] Implement evaluation script 
- [ ] In `configurator.py` on line 32, `key, val = arg.split('=')` -> `key, val = arg.split('=', 1)` (allows `--start="1+1="`)