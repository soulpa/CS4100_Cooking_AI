# demo run
import agent_system as ags
import knowledge_base as kb
import language_model as lm
import prompting_techniques as pt

agent = ags.ReActAgent(lm.LLM, kb.TOOLS, ags.AgentConfig(max_steps=6, verbose=True))

while True:
    user_q = input("\nEnter the ingredients you want to use: ")
    # user_q2 = input("\nEnter the estimated cooking time in minutes (or press Enter to skip): ")

    # exit bot
    if user_q.lower() in {"done", "quit", "exit"}:
        print("\nClosing :D")
        break

    # use an if statement to handle time input :D
    demo_q = f"What can I make using {user_q}"
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

