# CS 4100 Cooking AI Agent

### By: Kristen Cho and Ivina Wang
short description of some sorts lol

</br>

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

### 2. Install dependencies

Run the command to download following packages:
```
$ pip install streamlit torch
```

### 3. Clone Repository
- Clone repo in terminal using HTTPS url
```
$ git clone https://github.com/soulpa/CS4100_Cooking_AI.git
```
- Navigate into the main directory
```
cd CS4100_Cooking_AI/
```

### 4. Run Website Remotely

```
python -m streamlit run app.py
```
- Website should automatically pop up!

</br>

## Example Result
Input:`Ingredients: Chicken`, `Time: 30 minutes`

### Chicken Cordon Bleu
**Total Time:** 45 minutes
**Ingredients:**
- nonstick cooking spray
- 4 skinless, boneless chicken breast halves
- ¼ teaspoon salt
- ⅛ teaspoon ground black pepper
- 6 slices Swiss cheese
- 4 slices cooked ham
- ½ cup seasoned bread crumbs
[View Full Recipe Here](https://www.allrecipes.com/recipe/8495/chicken-cordon-bleu-i/)

</br>

## Project workflow
steps

</br>

## Weakenesses
- fuzzy wordmatching? can't spell words wrong or it won't match well
- not sure why the output bugs out at times