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

def format_model_list(model_list):
    """
    Gibt die Modellliste formatiert aus.
    Zeigt Name, Kategorie, CV-Strategie, Best-For und Anzahl der Hyperparameter.
    """
    print("\n📋 Verfügbare Modelle:\n" + "-"*50)
    for i, m in enumerate(model_list, start=1):
        name = m.get("name", "Unbekannt")
        category = m.get("category", "N/A")
        cv_strategy = m.get("cv_strategy", "N/A")
        best_for = m.get("best_for", "N/A")
        param_count = sum(len(v) if hasattr(v, '__len__') else 1 for v in m.get("param_grid", {}).values())

        print(f"{i}. {name}")
        print(f"   Kategorie     : {category}")
        print(f"   CV-Strategie  : {cv_strategy}")
        print(f"   Best geeignet : {best_for}")
        print(f"   Hyperparameter: {param_count} Kombinationen")
        print("-"*50)