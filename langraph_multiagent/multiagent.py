import functools
research_node= functools.partial(agent_node, agent=research_agent, name="Researcher")


chart_node= functools.partial(agent_node, agent=chart_agent, name="Chart Generator")

workflow= StateGraph(AgentState)




workflow.add_node("Researcher", research_node)
workflow.add_node("Chart Generator", chart_node)
workflow.add_node("call_tool", tool_node)
workflow.add_conditional_edges(
    "Researcher",
    router,
    {"continue": "Chart Generator", "call_tool": "call_tool", "end": END},
)
workflow.add_conditional_edges(
    "Chart Generator",
    router,
    {"continue": "Researcher", "call_tool": "call_tool", "end": END},
)
workflow.add_conditional_edges(
    "call_tool",
# Each agent node updates the 'sender' field# the tool calling node does not, meaning
# this edge will route back to the original agent# who invoked the tool
		lambda x: x["sender"],
    {
        "Researcher": "Researcher",
        "Chart Generator": "Chart Generator",
    },
)
workflow.set_entry_point("Researcher")
graph= workflow.compile()


for s in graph.stream(
    {
        "messages": [
            HumanMessage(
                content="Fetch the Malaysia's GDP over the past 5 years,"
                " then draw a line graph of it."
                " Once you code it up, finish."
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 150},
):
    print(s)
    print("----")