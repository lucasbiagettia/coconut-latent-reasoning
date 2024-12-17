import random
import json
import os

def generate_prontoqa_data(num_samples=1000, max_steps=5, output_dir="data/raw/prontoqa"):
    """
    Genera datos sintéticos estilo ProntoQA: preguntas lógicas con varios pasos de razonamiento.

    Args:
        num_samples (int): Número de muestras a generar.
        max_steps (int): Máximo número de pasos de razonamiento por problema.
        output_dir (str): Directorio donde se guardarán los datos.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Definición de palabras ficticias y relaciones
    entities = ["Brimpus", "Zumpus", "Gorpus", "Shumpus", "Terpus"]
    properties = ["luminous", "floral", "amenable", "voracious", "flimsy"]

    def generate_problem():
        """
        Genera un único problema lógico con razonamiento encadenado.
        """
        steps = []
        entity_chain = []

        # Crear una cadena de razonamiento aleatoria
        for i in range(max_steps):
            entity1 = random.choice(entities)
            entity2 = random.choice(entities)
            property1 = random.choice(properties)

            # Generar una relación lógica: "Entity1 are Property1."
            step = f"{entity1}s are {property1}."
            steps.append(step)
            entity_chain.append(entity1)

        # Generar la pregunta y respuesta
        final_entity = random.choice(entity_chain)
        target_property = random.choice(properties)
        question = f"True or false: {final_entity}s are {target_property}?"

        # Decidir si la respuesta es verdadera o falsa
        answer = "True" if any(target_property in step for step in steps) else "False"

        return {
            "question": question,
            "reasoning": steps,
            "answer": answer
        }

    # Generar todas las muestras
    dataset = [generate_problem() for _ in range(num_samples)]

    # Guardar en archivo JSON
    output_file = os.path.join(output_dir, "train.json")
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)
    
    print(f"Datos generados y guardados en {output_file}")


