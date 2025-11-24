# loaders/txt_loader.py
def load_txt(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()
