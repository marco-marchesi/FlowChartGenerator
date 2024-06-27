# FlowChartGenerator
**Overview**
This project is designed to generate flowcharts in GraphML format from a textual description of a process. The primary goal is to convert a given text input into a detailed flowchart that can be visualized and analyzed. The FlowCharts are compatible with free-to-use application yED.

**Features**
Text-to-Flowchart Conversion: Converts a text description of a process into a detailed flowchart in GraphML format.
Error Handling and Correction: Validates the generated GraphML and attempts to correct any errors.
Iterative Workflow: Utilizes a state graph to manage the workflow and allows for multiple iterations to refine the flowchart.
Integration with OpenAI: Uses OpenAI's GPT-3.5-turbo model to generate detailed descriptions and flowchart steps.
Requirements
Python 3.7+
langchain
pygraphml
openai
chroma
