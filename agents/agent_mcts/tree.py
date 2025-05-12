# from https://medium.com/pythoneers/getting-started-with-trees-in-python-a-beginners-guide-4e68818e7c05


class TreeNode:
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, child):
        self.children.append(child)


# Create a function for insertion
def insert_node(root, node):
    if root is None:
        root = node
    else:
        root.add_child(node)


# Create a function for deletion
def delete_node(root, target):
    if root is None:
        return None
    root.children = [child for child in root.children if child.data != target]
    for child in root.children:
        delete_node(child, target)


# Create a function to calculate the height of a tree
def tree_height(node):
    if node is None:
        return 0
    if not node.children:
        return 1
    return 1 + max(tree_height(child) for child in node.children)
