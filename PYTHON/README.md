

# Python HDCluster

## How to Run

There are two options to run, traditional ```python-venv```, or ```uv```


### python-venv



### uv


1. Install uv using the instructions found: [uv Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)

2. Enter HDCluster/PYTHON and type ```uv sync```. You can test the import using:

```
uv run python
from hdcluster import hdcluster
```
This should return no errors.


### python-venv


Make sure you have Python3, any version, and pip3. Install ```pip install virtualenv``` to manage virtual Python environments.

Then create the environment ```virtualenv --python=3.13 .venv```.


```
source .venv/bin/activate
python
from hdcluster import hdcluster
```


This should return no errors.

