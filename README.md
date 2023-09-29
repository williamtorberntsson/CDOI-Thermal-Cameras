# CDIO Characterizing Thermal Cameras


## Documentation guide
https://realpython.com/python-project-documentation-with-mkdocs/

## How to install

Run to start virtual enviroment
```bash

python3 -m venv .irvenv
source .irvenv/bin/activate     # For Linux
source .irvenv/Scripts/activate # For windows
```

Install needed packages
```bash
python3 -m pip install -r requirements.txt
```
verify with
```bash
pip list
```

Run main script with
```bash
python3 scripts/main.py
```

Run tests with
```bash
python -m doctest filename.py
```

## View documentation
The documentation can be viewed with
```bash
mkdocs serve
```
