## How to Run Locally

### 1. Installing Python
**Windows**
- Download Python from the [official Python website](https://www.python.org/downloads/)

**macOS**
```
$ brew install python
```

**Linux**
```
$ sudo apt update
$ sudo apt install python3
```

### 2. Install Dependencies

Run the command to download following packages:
```
$ pip install -r requirements.txt
```

### 3. Clone Repository
- Clone repo in terminal using HTTPS url
```
$ git clone https://github.com/soulpa/CS4100_Cooking_AI.git
```
- Navigate into the src directory
```
cd CS4100_Cooking_AI/src/
```

### 4. Run Website Remotely

```
python -m streamlit run app.py
```
- Website should automatically pop up!
