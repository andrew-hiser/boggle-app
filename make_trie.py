import pickle
import json

class Letter_Node:
    def __init__(self, letter: str):
        self.letter = letter
        self.children = {}
        self.is_terminal = False  # True if this node completes a word
    
    def add_child(self, node):
        self.children[node.letter] = node

    def get_child(self, letter):
        return self.children.get(letter)

def make_trie(words):
    root = Letter_Node('')  # Dummy root node

    for word in words:
        current_node = root
        for letter in word:
            child = current_node.get_child(letter)
            if child is None:
                child = Letter_Node(letter)
                current_node.add_child(child)
            current_node = child
        current_node.is_terminal = True  # Mark end of word

    return root
    
def load_words_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    
    words = [
        entry["word"].lower()
        for entry in entries
        if "word" in entry
        and entry["word"].isalpha()
        and len(entry["word"]) >= 3
    ]
    
    return words

def save_trie(trie, filename):
    with open(filename, 'wb') as f:
        pickle.dump(trie, f)

# words = load_words_from_txt('dictionary.txt')
words = load_words_from_json('EDMTDictionary.json')
trie = make_trie(words)
save_trie(trie, 'trie.pkl')