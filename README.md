```bash
cat ~/.condarc /opt/conda/.condarc /etc/conda/.condarc 2>/dev/null
conda config --show channels channel_alias default_channels
pip config list; cat /etc/pip.conf ~/.pip/pip.conf ~/.config/pip/pip.conf 2>/dev/null
env | grep -iE 'nexus|proxy|index'
grep -iE 'nexus|condarc|index-url|conda create' ~/.bash_history | tail -30

```
