import streamlit as st
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

sec_key = "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"
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

llm_gen = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=512,
    temperature=0.7,
    token=sec_key
)

# Optimal Material Suggestion for Component
component_material_template = '''
Suggest an optimal material composition for the component: {component}.
Component Composition:
'''
prompt_component_material = PromptTemplate(
    input_variables=['component'],
    template=component_material_template
)

# Component Performance Prediction
performance_pred_template = '''
Predict the performance metrics for the following component under {conditions}:
Component: {component}
Performance Metrics:
'''
prompt_performance = PromptTemplate(
    input_variables=['component', 'conditions'],
    template=performance_pred_template
)

# Environmental Resistance Prediction
env_resistance_template = '''
Predict the environmental resistance of the following component in {environment}:
Component: {component}
Resistance Level:
'''
prompt_env_resistance = PromptTemplate(
    input_variables=['component', 'environment'],
    template=env_resistance_template
)

# Predictive Maintenance Analysis
maintenance_analysis_template = '''
Provide predictive maintenance analysis for the component with the following usage data:
Component: {component}
Usage Data: {usage_data}
Maintenance Recommendation:
'''
prompt_maintenance = PromptTemplate(
    input_variables=['component', 'usage_data'],
    template=maintenance_analysis_template
)

st.title("BEL-AIxpert: Industrial AI for Bharat Electronics Limited")

st.sidebar.title("Select Analysis Task")
task = st.sidebar.selectbox(
    "Task",
    (
        "Optimal Material Composition",
        "Performance Prediction",
        "Environmental Resistance Prediction",
        "Predictive Maintenance Analysis"
    ),
)

# Optimal Material Composition Task
if task == "Optimal Material Composition":
    st.header("Optimal Material Composition")
    component = st.text_input("Enter component name (e.g., antenna, sensor casing):")
    if st.button("Suggest Material Composition"):
        if component:
            material_chain = LLMChain(llm=llm_gen, prompt=prompt_component_material)
            response = material_chain.run({"component": component})
            st.write("### Suggested Material Composition:")
            st.write(response)
        else:
            st.write("Please provide component name.")

# Performance Prediction Task
elif task == "Performance Prediction":
    st.header("Performance Prediction")
    component = st.text_input("Enter component name:")
    conditions = st.text_area("Describe operating conditions (e.g., temperature, pressure):")
    if st.button("Predict Performance"):
        if component and conditions:
            performance_chain = LLMChain(llm=llm_gen, prompt=prompt_performance)
            response = performance_chain.run({"component": component, "conditions": conditions})
            st.write("### Predicted Performance Metrics:")
            st.write(response)
        else:
            st.write("Please enter component and conditions.")

# Environmental Resistance Prediction Task
elif task == "Environmental Resistance Prediction":
    st.header("Environmental Resistance Prediction")
    component = st.text_input("Enter component name:")
    environment = st.selectbox("Select Environment:", ["Desert", "Arctic", "Marine"])
    if st.button("Predict Resistance Level"):
        if component:
            env_resistance_chain = LLMChain(llm=llm_gen, prompt=prompt_env_resistance)
            response = env_resistance_chain.run({"component": component, "environment": environment})
            st.write("### Environmental Resistance Level:")
            st.write(response)
        else:
            st.write("Please provide component and environment.")

# Predictive Maintenance Analysis Task
elif task == "Predictive Maintenance Analysis":
    st.header("Predictive Maintenance Analysis")
    component = st.text_input("Enter component name:")
    usage_data = st.text_area("Enter recent usage data (e.g., hours, stress levels):")
    if st.button("Provide Maintenance Analysis"):
        if component and usage_data:
            maintenance_chain = LLMChain(llm=llm_gen, prompt=prompt_maintenance)
            response = maintenance_chain.run({"component": component, "usage_data": usage_data})
            st.write("### Maintenance Recommendation:")
            st.write(response)
        else:
            st.write("Please provide component and usage data.")


st.sidebar.info("Developed by Rachit Ranjan")
