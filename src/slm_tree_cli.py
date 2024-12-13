import typer
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

app = typer.Typer()

class SLMNode:
    """
    Represents a node in the tree, which is a small language model.
    Each node either processes the query or routes it to another node in the tree.
    """
    def __init__(self, model_name: str, model_type="causal", tokenizer=None):
        """
        Initialize the SLM node with a specific model.

        :param model_name: The name of the model (e.g., 'distilbert-base-uncased')
        :param model_type: Type of model ('causal', 'encoder-decoder', etc.)
        :param tokenizer: Tokenizer associated with the model (optional)
        """
        self.model_name = model_name
        self.model_type = model_type
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.children = []  # child nodes that a query can be routed to

    def add_child(self, child_node):
        """
        Add a child node to this SLM node.

        :param child_node: Another SLMNode that will be a child of this node.
        """
        self.children.append(child_node)

    def forward(self, query: str):
        """
        Forward the query to the model, or route it to a child if needed.

        :param query: The input query to process.
        :return: The model's response or the result of forwarding to a child.
        """
        # Basic query processing (tokenization + model inference)
        inputs = self.tokenizer(query, return_tensors="pt")
        output = self.model.generate(**inputs)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return response

class SLMTree:
    """
    Represents the tree of Small Language Models (SLMs).
    Manages the root node and all the connections between nodes.
    """
    def __init__(self, root_node):
        """
        Initialize the SLM Tree with a root node.

        :param root_node: The root node of the SLM tree (SLMNode).
        """
        self.root_node = root_node

    def route_query(self, query: str):
        """
        Routes a query through the tree of SLM nodes.

        :param query: The query to route.
        :return: The final response after routing through the tree.
        """
        response = self._route_from_node(self.root_node, query)
        return response

    def _route_from_node(self, node, query: str):
        """
        Process the query from the current node and decide whether to forward to children.

        :param node: Current SLMNode to process the query.
        :param query: The query to be processed.
        :return: The response after processing or routing the query.
        """
        # Here you could add logic to decide whether to forward the query to children or not
        if node.children:  # If the node has children, decide where to route the query
            return self._route_from_node(node.children[0], query)  # Change this logic as needed
        else:
            # Process the query using the current node's model
            return node.forward(query)

class QueryRouter:
    """
    A utility class to handle and manage query routing for the SLM tree.
    """
    def __init__(self, slm_tree: SLMTree):
        """
        Initialize the QueryRouter with a given SLM tree.

        :param slm_tree: The SLMTree object that manages the SLM nodes.
        """
        self.slm_tree = slm_tree

    def handle_query(self, query: str) -> str:
        """
        Routes the query through the SLM tree and returns the response.

        :param query: The query to process.
        :return: The response from the SLM tree.
        """
        return self.slm_tree.route_query(query)

@app.command()
def create_node(model_name: str, model_type: str = "causal"):
    """
    Create a new SLM node with the given model name and type.

    :param model_name: Name of the language model (e.g., 'distilbert-base-uncased').
    :param model_type: Type of model ('causal', 'encoder-decoder', etc.)
    """
    node = SLMNode(model_name=model_name, model_type=model_type)
    typer.echo(f"Node created with model {model_name}")
    return node

@app.command()
def add_child_node(parent_model_name: str, child_model_name: str):
    """
    Add a child node to an existing SLM node.

    :param parent_model_name: Name of the parent node's model.
    :param child_model_name: Name of the child node's model to be added.
    """
    parent_node = SLMNode(parent_model_name)
    child_node = SLMNode(child_model_name)
    parent_node.add_child(child_node)
    typer.echo(f"Added {child_model_name} as a child of {parent_model_name}")
    return parent_node

@app.command()
def route_query(query: str, root_model_name: str):
    """
    Route the query through the tree of SLMs and get the response.

    :param query: The query to route.
    :param root_model_name: The root node's model name.
    """
    root_node = SLMNode(root_model_name)
    slm_tree = SLMTree(root_node)
    query_router = QueryRouter(slm_tree)
    response = query_router.handle_query(query)
    typer.echo(f"Response: {response}")

if __name__ == "__main__":
    app()
