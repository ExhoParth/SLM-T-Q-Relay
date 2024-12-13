# usage.py

from slm_tree_cli import SLMNode, SLMTree, QueryRouter
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_query_routing():
    # Initialize the tokenizer and model for the root node
    root_model_name = 'distilgpt2'  # You can choose any model you want
    root_node = SLMNode(model_name=root_model_name)

    # Initialize a child model to be added to the root node
    child_model_name = 'gpt2'  # You can choose another model here
    child_node = SLMNode(model_name=child_model_name)

    # Add the child node to the root node
    root_node.add_child(child_node)

    # Create the SLM Tree with the root node
    slm_tree = SLMTree(root_node)

    # Initialize the query router with the SLM tree
    query_router = QueryRouter(slm_tree)

    # Test query routing
    query = "What is the capital of France?"
    response = query_router.handle_query(query)

    print(f"Query: {query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    test_query_routing()
