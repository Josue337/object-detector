from model import load_model

_, model, _, _ = load_model()
total = sum(p.numel() for p in model.parameters())
print(f"{total / 1e6:.1f} millones de parámetros")