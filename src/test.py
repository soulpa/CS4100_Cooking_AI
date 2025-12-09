# demo run
import agent_system as ags
import knowledge_base as kb
import language_model as lm
import prompting_techniques as pt

agent = ags.ReActAgent(lm.LLM, kb.TOOLS, ags.AgentConfig(max_steps=6, verbose=True))

while True:
    ingredients = input("\nEnter the ingredients you want to use: ")
    
    # exit bot
    if ingredients.lower() in {"done", "quit", "exit"}:
        print("\nClosing :D")
        break
    
    while True:
        time_limit = input("\nEnter the estimated cooking time in minutes (or press Enter to skip): ")
        if time_limit == "":
            break
        try:
            time_limit = int(time_limit)
        except ValueError:
            # must be a number
            print("must be a number")
            continue
        if time_limit < 0:
            print("cannot be negative")
            continue
        else:
            break

    # use an if statement to handle time input :D
    if not time_limit:
        demo_q = f"What can I make using {ingredients}?"
    else:
        demo_q = f"What can I make using {ingredients}? The recipe should take around {time_limit} minutes."
    # demo_q = "What can I make using chicken and lemon?"
    result = agent.run(demo_q)

    print("Question:", result["question"])
    print("\nFinal Answer:", result["final_answer"])
    print("\nTrajectory:")
    for i, s in enumerate(result["steps"], 1):
        print(f"\nStep {i}")
        print("Thought:", s["thought"])
        print(s["action"])
        print("Observation:", s["observation"][:500] + ("..." if len(s["observation"])>500 else ""))

def check_valid(time):
    try:
        time = int(time)
        if time > 0:
            return time
    except:
        time = input("\nEnter the estimated cooking time in minutes (or press Enter to skip): ")