from inference.inference import CoconutInference

if __name__ == "__main__":
    model_path = "checkpoints/coconut_stage_3.pth"  # Reemplaza con la ruta de tu checkpoint
    inference = CoconutInference(model_path)

    # Prueba en Modo Latente
    question = "What is 2 + 2?"
    response_latent = inference.infer(question, latent_steps=1, mode="latent")
    print("Modo Latente:", response_latent)

    # Prueba en Modo Lenguaje (CoT)
    response_language = inference.infer(question, mode="language")
    print("Modo Lenguaje:", response_language)
