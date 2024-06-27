from typing import List, TypedDict
from langgraph.graph import END, StateGraph
from pygraphml import GraphMLParser
from langchain_chroma import Chroma
import os
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import __config
from langchain_core.prompts import PromptTemplate
import pygraphml
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"] = "your-key"
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
collection = "graphml"
embeddings = OpenAIEmbeddings(
    model    = __config.modelEmbeddingsLarge,
)


# Define the state structure
class GraphState(TypedDict):
    error: str
    errorMsg: str
    messages: List
    generation: str
    iterations: int

# Define the nodes
def description_creator(state: GraphState):
    """
    Create detailed descriptions and break down tasks into sub-tasks.
    """
    print("---CREATING DESCRIPTIONS AND SUB-TASKS---")
    messages = state["messages"]
    # Assume we have a function `create_descriptions` to create detailed descriptions
    task_description = create_descriptions(messages[-1][1])
    print(task_description)
    messages += [("description_creator", task_description)]
    return {"generation": task_description, "messages": messages, "iterations": state["iterations"]}

def create_descriptions(task):
    """
    Break down a task and provide descriptions.
    """

    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Given a small input or a description of a process, elaborate and generate detailed steps to describe the task or process.
    Structure the text in a clear and readable format as if it were a flowchart, but provide only text. 
    Structure the text in a clear way also for DECISION cases or LOOPs.
    Ensure that each step is clearly numbered or labeled for easy understanding.
    When there are multiple options or IF statements make decision blocks that lead to different actions.     """),
        ("user", "{input}")
    ])   


    rag_chain = (        
        prompt
        | llm        
    )

    result = rag_chain.invoke({"input":task})    
    return f"Breakdown of the task: {result.content}"

def chart_generator(state: GraphState):
    """
    Generate graphML examples using a RAG tool.
    """
    print("---GENERATING GRAPHML CHART---")
    messages = state["messages"]
    # Assume we have a function `generate_graphml` that uses RAG tool
    graphml_chart = generate_graphml(messages[-1][1], state)
    print(graphml_chart)
    messages += [("chart_generator", graphml_chart)]
    return {"generation": graphml_chart, "messages": messages, "iterations": state["iterations"]}

def generate_graphml(description: str, state: GraphState):
    """
    Generate a graphML chart.
    """
    reflectionError = ""
    if state["error"] == "yes":
        reflectionError= description
    conversation_memory = ConversationBufferMemory()
    vectorstore = Chroma(collection, embeddings, persist_directory='db')

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    #prompt = hub.pull("rlm/rag-prompt")
    template = """Given a description of a process or a simple input, generate a detailed flowchart in GraphML format. Consider the context provided as useful examples for syntax.
    Start the Flowchart with  a START block and finish with an END block.
    Return only the GraphML text without any additional explanation or text. 
    Question: {question} 

    Context: {context} 

    Answer: """
    custom_rag_prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        #| conversation_memory
        | StrOutputParser()
    )

    result = rag_chain.invoke(description)
    return result.replace("\'","").replace("```","").replace("xml\n","").replace("\n","")


def validator(state: GraphState):
    """
    Validate the generated graphML chart.
    """
    print("---VALIDATING GRAPHML CHART---")
    messages = state["messages"]
    graphml_chart = state["generation"]

    try:
        validate_graphml(graphml_chart)
    except Exception as e:
        print("---GRAPHML VALIDATION FAILED---")
        error_message = [("validator", f"GraphML validation failed: {e}")]
        messages += error_message
        return {
            "generation": graphml_chart,
            "messages": messages,
            "iterations": state["iterations"],
            "error": "yes",
            "errorMsg":error_message
        }

    print("---GRAPHML VALIDATION PASSED---")
    return {
        "generation": graphml_chart,
        "messages": messages,
        "iterations": state["iterations"],
        "error": "no",
    }

def validate_graphml(graphml):
    """
    function to  graphML validation.
    """
    # Parse the GraphML file using pygraphml
    parser = GraphMLParser()
    graph = parser.parse_string(graphml.replace("'",""))
    
    errors = []
    # Check if the graph has nodes and edges
    if len(graph.nodes()) == 0:
        errors.append("No 'node' elements found in the graph.")
    if len(graph.edges()) == 0:
        errors.append("No 'edge' elements found in the graph.")

    # Additional checks to ensure yEd compatibility
    for node in graph.nodes():
        if not node.id:
            errors.append(f"A node is missing the 'id' attribute.")

    for edge in graph.edges():
        if not edge.node1 or not edge.node2:
            errors.append(f"An edge is missing 'source' or 'target' attributes.")

        if edge.node1 not in graph.nodes():
            errors.append(f"Edge source '{edge.node1.id}' does not exist in the list of nodes.")
        if edge.node2 not in graph.nodes():
            errors.append(f"Edge target '{edge.node2.id}' does not exist in the list of nodes.")
    if len(errors)>0:
        response= ""
        response("Validation Errors:")
        for error in errors:
            response+=(f"- {error}")
        raise ValueError("Invalid graphML - " + response)

def reflect(state: GraphState):
    """
    Reflect on errors and retry.
    """
    print("---REFLECTING ON ERRORS---")
    messages = state["messages"]
    iterations = state["iterations"]
    graphml_chart = state["generation"]

    # Add reflection and retry logic
    reflections = f"Reflected on errors: {messages[-1][1]}"
    messages += [("reflect", reflections)]
    return {"generation": graphml_chart, "messages": messages, "iterations": iterations}

def chart_corrector(state: GraphState):
    """
    Correct and fix graphML.
    """
    print("---Correct GRAPHML CHART---")
    
    messages = state["messages"]
    iterations = state["iterations"]
    graphml_chart = state["generation"]
    errorMsg = state["errorMsg"][0][1]
    # Assume we have a function `generate_graphml` that uses RAG tool
    graphml_chart = correct_chart(graphml_chart, errorMsg)
    messages += [("assistant", graphml_chart)]
    return {"generation": graphml_chart, "messages": messages, "iterations": state["iterations"]}

def correct_chart(chart, errorMsg:str ):
    """
    Generate a corrected chart.
    """

    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Given an error message  and a flowchart in GraphML format, fix and correct the flowchart. Generate only the GraphML syntax without explanations or descriptions."""),
        ("user", "{input}")
    ])   


    rag_chain = (        
        prompt
        | llm        
    )

    result = rag_chain.invoke({"input":"Given this error message: " + errorMsg + " fix the following chart in GraphML format: " + chart})    
    return result.content.replace("\'","").replace("```","").replace("xml\n","").replace("\n","")

# Define the decision logic
def decide_to_finish(state: GraphState):
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations >= max_iterations:
        print("---DECISION: FINISH---")
        print(state["generation"])
        # Parse the GraphML string
        parser = pygraphml.GraphMLParser()
        graph = parser.parse_string(state["generation"].replace("'", ""))

        # Validate parsed graph (optional)
        # You can add checks here to ensure the graph is parsed correctly,
        # such as checking for the presence of required elements like nodes and edges.

        # Write the parsed graph to an nformat GraphML file
        with open('output.graphml', 'w') as file:
            # Write text to the file
            file.write(state["generation"])

        print(f"GraphML string successfully parsed and saved to: graph001")

        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        return "reflect"

# Define the workflow graph
max_iterations = 3
workflow = StateGraph(GraphState)

# Add nodes to the workflow
workflow.add_node("description_creator", description_creator)
workflow.add_node("chart_generator", chart_generator)
workflow.add_node("validator", validator)
workflow.add_node("reflect", reflect)
workflow.add_node("chart_corrector", chart_corrector)

# Set entry point and define edges
workflow.set_entry_point("description_creator")
workflow.add_edge("description_creator", "chart_generator")
#workflow.set_entry_point("chart_generator")
workflow.add_edge("chart_generator", "validator")
workflow.add_conditional_edges(
    "validator",
    decide_to_finish,
    {
        "end": END,
        "reflect": "reflect",
    },
)
workflow.add_edge("reflect", "chart_corrector")
workflow.add_edge("chart_corrector", "validator")

# Compile and run the workflow
app = workflow.compile()
question = """make a  flowchart to describe the steps of purchasing a pair of shoes on a web shop.
Take into account different payments and deliveries.
make it  detailed but not more than 15 steps.
"""
state = {"messages": [("user", question)], "iterations": 0}
app.invoke(state)
