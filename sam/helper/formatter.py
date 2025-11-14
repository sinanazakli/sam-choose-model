def format_pipeline_info(pipeline):
    """
    Gibt eine formatierte Übersicht der Pipeline zurück:
    - Scaler (falls vorhanden)
    - Modell mit Hyperparametern
    """
    scaler = pipeline.named_steps.get('scaler', None)
    model = pipeline.named_steps.get('model', None)

    scaler_info = f"Scaler: {scaler.__class__.__name__}" if scaler else "Scaler: Kein Scaler"
    model_info = f"Modell: {model.__class__.__name__}"

    # Hyperparameter des Modells extrahieren
    params = {k: v for k, v in model.get_params().items() if not k.startswith('_')}
    param_str = ", ".join([f"{k}={v}" for k, v in params.items() if k in ['kernel', 'C', 'max_iter', 'learning_rate', 'hidden_layer_sizes']])

    if param_str:
        model_info += f" ({param_str})"

    return f"{scaler_info}\n{model_info}"
