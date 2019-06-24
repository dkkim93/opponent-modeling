# opponent-modeling

## Dependency
Known dependencies are:
```
python (3.6.5)
pip (3.6)
virtualenv
```

## Setup
To avoid any conflict, please install Python virtual environment with [`virtualenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/):
```
pip3.6 install --upgrade virtualenv
```

## Run
After all the dependencies are installed, please run the code by running the following script.  
The script will start the virtual environment, install remaining Python dependencies in `requirements.txt`, and run the code.  
```
./_train.sh
```

## To-Dos
- [x] Make regression domain
- [x] Meta-learning method
- [x] No meta-learning baseline
- [ ] Experiments with multi regression domain
