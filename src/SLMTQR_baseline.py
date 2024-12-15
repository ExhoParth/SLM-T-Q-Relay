
from transformers import AutoTokenizer, AutoModelForCausalLM
from difflib import get_close_matches
import networkx as nx
import matplotlib.pyplot as plt
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading the root model...")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B-Instruct").to(DEVICE)
print("Root model loaded successfully.")

class SLMNode:
    """
    Represents a node in the SLM Tree.
    Each node corresponds to a Small Language Model (SLM).
    """
    def __init__(self, name, description, shared_tokenizer, shared_model):
        self.name = name
        self.description = description
        self.tokenizer = shared_tokenizer
        self.model = shared_model
        self.children = {}  # Dictionary of child nodes

    def add_child(self, child_node):
        """
        Adds a child node to the current node.
        """
        self.children[child_node.name] = child_node
        print(f"Added child node '{child_node.name}' to parent '{self.name}'.")

    def display_tree(self, level=0):
        """
        Displays the tree structure from the current node downwards.
        """
        indent = "  " * level
        print(f"{indent}- {self.name}: {self.description}")
        for child in self.children.values():
            child.display_tree(level + 1)

    def should_route(self, output):
        """
        Decides whether to route the query based on the model's output.
        If the output contains a child node's name, route to that child.
        Otherwise, return False to generate a response.
        """
        for child_name in self.children:
            if child_name in output:
                print(f"Routing decision: Output '{output}' matches child '{child_name}'")
                return child_name  # Return the child name to route to it
        return None  # No routing needed, stay at the current node

    def route_query(self, query, path=""):
        """
        Routes a query to the appropriate child or generates a response at the current node.
        """
        # Generate system prompt describing the children
        if self.children:
            system_prompt = "Select the most relevant category:\n"
            for child_name, child_node in self.children.items():
                system_prompt += f"- {child_name}: {child_node.description}\n"
            system_prompt += f"Query: {query}\n"

            # Tokenize and generate a response
            input_ids = self.tokenizer.encode(system_prompt, return_tensors="pt").to(DEVICE)
            output_ids = self.model.generate(input_ids, max_new_tokens=100)
            output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Decide whether to route based on the output
            child_to_route = self.should_route(output)
            if child_to_route:
                # Add the current node to the path and route the query to the child
                new_path = f"{path} -> {self.name} (Routing to {child_to_route})"
                print(f"Routing path: {new_path}")
                return self.children[child_to_route].route_query(query, new_path)

            # If no relevant child is found, return the model's response
            print(f"Final path at node '{self.name}': {path} -> {self.name} (Response generated)")
            return output

        # If no children, generate a response from the current node
        input_ids = self.tokenizer.encode(query, return_tensors="pt").to(DEVICE)
        output_ids = self.model.generate(input_ids, max_new_tokens=100)
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Final path at node '{self.name}': {path} -> {self.name} (Response generated)")
        return output

class SLMTree:
    """
    Represents the SLM Tree with a root node.
    """
    def __init__(self, root_node):
        self.root = root_node

    def add_child_to_node(self, parent_name, child_name, child_description):
        """
        Adds a child node to a specified parent node by name.
        """
        parent_node = self._find_node_by_name(self.root, parent_name)
        if parent_node is None:
            print(f"Parent node '{parent_name}' not found!")
            return
        new_child = SLMNode(name=child_name, description=child_description,
                            shared_tokenizer=parent_node.tokenizer,
                            shared_model=parent_node.model)
        parent_node.add_child(new_child)

    def _find_node_by_name(self, node, name):
        """
        Recursively searches for a node by name.
        """
        if node.name == name:
            return node
        for child in node.children.values():
            found = self._find_node_by_name(child, name)
            if found:
                return found
        return None

    def display_tree(self):
        """
        Displays the entire tree starting from the root node.
        """
        if self.root:
            self.root.display_tree()

    def query(self, query):
        """
        Routes a query through the tree starting from the root.
        """
        print(f"Querying the SLM Tree with: {query}")
        return self.root.route_query(query)

root_node = SLMNode(
    name="Root",
    description="Handles general queries across all domains.",
    shared_tokenizer=tokenizer,
    shared_model=model
)

slm_tree = SLMTree(root_node)

slm_tree.add_child_to_node(
    parent_name="Root",
    child_name="Tech",
    child_description="Handles technical queries (e.g., programming, AI, engineering)."
)
slm_tree.add_child_to_node(
    parent_name="Root",
    child_name="Health",
    child_description="Handles health-related queries."
)

# Add second-level child nodes to 'Tech'
slm_tree.add_child_to_node(
    parent_name="Tech",
    child_name="AI",
    child_description="Specializes in Artificial Intelligence queries."
)
slm_tree.add_child_to_node(
    parent_name="Tech",
    child_name="Software",
    child_description="Deals with software development queries."
)

# Add second-level child nodes to 'Health'
slm_tree.add_child_to_node(
    parent_name="Health",
    child_name="MentalHealth",
    child_description="Focuses on mental health and well-being."
)
slm_tree.add_child_to_node(
    parent_name="Health",
    child_name="Fitness",
    child_description="Addresses fitness and physical health queries."
)

print("\nSLM Tree Structure:")
slm_tree.display_tree()

query = "What is the best way to fine-tune a model for sentiment analysis?"
response = slm_tree.query(query)

print("\nFinal Response:")
print(response)

