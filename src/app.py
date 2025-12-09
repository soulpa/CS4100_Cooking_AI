import streamlit as st
import agent_system as ags
import knowledge_base as kb
import language_model as lm
import pandas as pd
import ast

agent = ags.ReActAgent(lm.LLM, kb.TOOLS, ags.AgentConfig(max_steps=6, verbose=True))
recipes_data = pd.read_csv("data/clean_recipes.csv")

st.title("AI Cooking Agent👨‍🍳✨")

with st.chat_message("assistant"):
    st.write("Hello! If you provide me with ingredients and an estimated cooking time, I can provide you with yummy recipe recommendations. :)")

ingredients = st.text_input("Enter ingredients:")
time = st.number_input("Enter the estimated cooking time in minutes (optional):", min_value=0, step=10)

if st.button("Submit"):
    if time == 0:
        demo_q = f"What can I make using {ingredients}?"
    else:
        demo_q = f"What can I make using {ingredients}? The recipe should take around {time} minutes."
    result = agent.run(demo_q)
    # st.write(result["final_answer"])
    
    st.markdown("---")
    # if final answer is empty, sorry!
    if not result["final_answer"]:
        with st.chat_message("assistant"):
            st.write("I'm sorry, I can't find any recipes at this time. Please try again with different ingredients or cooking time.")
    else:
    # print recipies
        with st.chat_message("assistant"):
            st.write("Here are my recommendations for you!")

        for recipes in result["final_answer"].split(", "):
            recipe = recipes_data[recipes_data["Name"] == recipes].iloc[0]
            st.markdown(f"### {recipe['Name']}")
            st.markdown(f"**Total Time:** {recipe['Total Time']} minutes")
            st.markdown(f"**Ingredients:**")
            for ingredient in ast.literal_eval(recipe["Ingredients"]):
                st.markdown(f"- {ingredient['quantity']} {ingredient['unit']} {ingredient['name']}")
            st.markdown(f"[View Full Recipe Here]({recipe['url']})")
            st.markdown("---")
