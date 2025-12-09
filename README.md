# CS 4100 Cooking AI Agent👨‍🍳

### By: Kristen Cho and Ivina Wang

Insallation instructions can be found in INSTALL.md!

## Abstract
Every day, people think to themselves one simple question: "What should I make to eat?" Some people might also be tired of making the same things repeatedly, too. We combated this problem by developing an AI agent to generate recipe recommendations given inputs of certain ingredients and a suggested cooking time. 

## Overview
Everyone needs to eat to live, but sometimes people might not know what to make. They also might get tired of making the same dishes they know how to cook well and would want to try something new. But what?

This problem is interesting because there are so many recipes across different cuisines, and there are a lot of dishes people who have not tried. By being able to introduce new foods through recipe recommendations, people can try new cuisines at home knowing what goes in the food and how it is made. This could also be helpful if someone lives in an area that does not have a lot of cultural diversity.

The approach we proposed to tackle this problem is to implement an AI cooking agent that gives recipe recommendations based on ingredients a user inputs and amount of time they want to spend cooking. More technical details to our apporach are explained below. Other approaches to this problem.. :D

The rationale behind our proposed approach was to learn from the starter code and be able to expand it to be able to implement features we are looking for. We tried generating recipes given ingredients on ChatGPT to see what it would output, and saw that it created its own recipes and didn't provide recipe links that could contain images and video instructions. Our approach differs from this because it would output the recipe name, the time it takes to make, the ingredients list, and the link to the original recipe.

## Approach
We followed along with agent workflow introduced in class. After cleaning our initial csv file and creating a corpus we...
1. Implemented TF-IDF search method to find recipes that match our input ingredients
2. Implemented prompting method to instruct the LLM to return the top 3 recipes that best match the ingredients and estimated cooking time
3. Integrated Qwen2.5-0.5B-Instruct model from HuggingFace
4. Built the agent workflow using a ReAct loop

Combining TF-IDF to find relevant context and creating a set prompting method in guide the Qwen LLM, builds into the agent workflow which uses ReAct loop to generate "Thoughts" and "Searches" until it gets it's final response.

Some limitations to our approach include our TF-IDF search method lacking semantic meaning but it doesn't affect us as much since we're mostly looking through a list of ingredients. However, if the ingredients are spelled wrong in the initial input, the model will have trouble matching up ingredients and may return with an error. Further exprimentation with fuzzy


## Results  
### Example Result

Input:`Ingredients: Chicken`, `Time: 30 minutes`

Output:

---
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

---

Lists 3 recipe recommendations, their corresponding cooking time (which includes prep, cooking, and additional time), and provides a link to detailed instructions.

## Discussion
**Weaknesses**
- Can potentially implement fuzzy wordmatching; if user inputs words that are spelled incorrectly, the TF-IDF won't match term well
- Time input can be improved to separate additional cooking time
- Sometimes it outputs one recipe, sometimes it outputs more than one 

## Conclusion

With this, we have achieved implementing an AI agent to recommend users recipes given ingredients they have or want to use. After searching for recipes, the agent outputs the dish name, time, ingredients needed and quantity, and link to the full original recipe. Now, people can try new foods with ingredients they have!


## References
**Bibliography**

[1]	Build a basic LLM chat app. Streamlit.io. Retrieved December 8, 2025 from https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps

[2]	Create an app. Streamlit.io. Retrieved December 8, 2025 from https://docs.streamlit.io/get-started/tutorials/create-an-app

[3]	Recipes Dataset. Kaggle.com. Retrieved December 8, 2025 from https://www.kaggle.com/datasets/thedevastator/better-recipes-for-a-better-life

[4] Class template. :D

