import random
import json
import os
import networkx as nx

def generate_prosqa_data(num_samples=1000, output_dir="data/raw/prosqa"):
    """
    Genera datos sintéticos estilo ProsQA usando un DAG (Directed Acyclic Graph).

    Args:
        num_samples (int): Número de muestras a generar.
        output_dir (str): Directorio donde se guardarán los datos.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "train.json")

    entities = ["terpus", "brimpus", "lempus", "scrompus", "zumpus"]

    def generate_valid_dag():
        """
        Genera un DAG sencillo y conectado con un camino válido.
        """
        G = nx.DiGraph()
        random.shuffle(entities)
        
        # Conectar nodos secuencialmente para garantizar un camino válido
        for i in range(len(entities) - 1):
            G.add_edge(entities[i], entities[i + 1])

        # Agregar algunas aristas adicionales para variedad
        extra_edges = random.randint(1, 3)
        for _ in range(extra_edges):
            source, target = random.sample(entities, 2)
            if not nx.has_path(G, target, source):  # Evita ciclos
                G.add_edge(source, target)
        return G

    def extract_reasoning_and_answer(G):
        """
        Extrae pasos de razonamiento y genera una pregunta con respuesta.
        """
        # Seleccionar un camino válido directamente
        nodes = list(G.nodes)
        source, target = nodes[0], nodes[-1]
        path = list(nx.shortest_path(G, source=source, target=target))
        steps = [f"Every {path[i]} is a {path[i+1]}." for i in range(len(path) - 1)]
        question = f"Is Tom a {path[-1]} or a {random.choice(nodes[:-1])}?"
        answer = f"Tom is a {path[-1]}."
        return steps, question, answer

    # Generar muestras
    dataset = []
    for _ in range(num_samples):
        G = generate_valid_dag()
        steps, question, answer = extract_reasoning_and_answer(G)
        dataset.append({
            "question": question,
            "reasoning": steps,
            "answer": answer
        })

    # Guardar en archivo JSON
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"Datos generados y guardados en {output_file}")


