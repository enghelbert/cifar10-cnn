"""
Pregunta 4: Entrenamiento, métricas, curvas de aprendizaje y matriz de confusión.

Ejecutar desde la raíz del repositorio:
    python src/train.py                 # 15 épocas por defecto
    python src/train.py --epochs 20     # cambiar número de épocas

Genera en outputs/:
    loss_accuracy.png, confusion_matrix.png, classification_report.txt,
    metrics.json, best_model.pth
y escribe el análisis cuantitativo en la sección de resultados del README.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # guarda las figuras sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent))
from dataset import CLASSES, OUTPUT_DIR, ROOT_DIR, SEED, get_dataloaders, set_seed  # noqa: E402
from model import CNN_CIFAR10, count_parameters, validate_shapes  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento CNN en CIFAR-10")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Una pasada completa por el loader. Si hay optimizer, entrena; si no, evalúa."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)                 # (B, 10)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    for images, y in loader:
        out = model(images.to(device))
        preds.append(out.argmax(dim=1).cpu())
        labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def plot_curves(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1.plot(epochs, history["train_loss"], "o-", label="Entrenamiento")
    ax1.plot(epochs, history["val_loss"], "s-", label="Validación")
    ax1.set_title("Pérdida (CrossEntropyLoss)")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Entrenamiento")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s-", label="Validación")
    ax2.set_title("Exactitud (Accuracy)")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle(f"Curvas de aprendizaje - CNN CIFAR-10 (semilla {SEED})")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, path, test_acc):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(f"Matriz de confusión - test (accuracy {test_acc * 100:.2f}%, semilla {SEED})")
    thresh = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def diagnose(history, test_acc):
    """Interpreta cuantitativamente si hay overfitting o underfitting."""
    tl, vl = history["train_loss"], history["val_loss"]
    ta, va = history["train_acc"], history["val_acc"]
    n = len(tl)
    best = int(np.argmin(vl))
    gap_acc = (ta[-1] - va[-1]) * 100
    gap_loss = vl[-1] - tl[-1]
    rise = vl[-1] - vl[best]
    after = n - 1 - best

    if gap_acc > 10 or (after >= 3 and rise > 0.10):
        verdict = "overfitting"
        text = (f"El modelo muestra **síntomas de overfitting**: la exactitud de entrenamiento supera a la de "
                f"validación en {gap_acc:.2f} puntos porcentuales y la pérdida de validación alcanzó su mínimo "
                f"en la época {best + 1} ({vl[best]:.4f}) para luego subir a {vl[-1]:.4f} "
                f"(+{rise:.4f}) mientras la pérdida de entrenamiento siguió bajando. A partir de esa época el "
                f"modelo memoriza el conjunto de entrenamiento en lugar de generalizar. Se mitigaría con "
                f"early stopping en la época {best + 1}, más Dropout, weight decay o más aumento de datos.")
    elif ta[-1] < 0.65:
        verdict = "underfitting"
        text = (f"El modelo muestra **síntomas de underfitting**: la exactitud de entrenamiento final es solo "
                f"{ta[-1] * 100:.2f}% y la de validación {va[-1] * 100:.2f}%; ambas son bajas y cercanas "
                f"(brecha de {gap_acc:.2f} pp), lo que indica que la red aún no captura los patrones de los "
                f"datos. Se corregiría entrenando más épocas, aumentando la capacidad (más filtros o "
                f"bloques) o ajustando la tasa de aprendizaje.")
    else:
        verdict = "buen ajuste"
        text = (f"El modelo **no muestra overfitting ni underfitting significativos**: la brecha entre "
                f"entrenamiento y validación es de {gap_acc:.2f} pp en exactitud y {gap_loss:.4f} en pérdida, "
                f"y la pérdida de validación tuvo su mínimo en la época {best + 1} ({vl[best]:.4f}), "
                f"terminando en {vl[-1]:.4f}. Que la exactitud de validación sea similar o incluso mayor que "
                f"la de entrenamiento se explica porque el entrenamiento se mide sobre imágenes aumentadas "
                f"(RandomCrop/Flip) y con Dropout activo, lo que las hace más difíciles. Como ambas curvas "
                f"siguen mejorando, entrenar más épocas probablemente subiría la exactitud.")
    return {
        "veredicto": verdict, "texto": text, "mejor_epoca_val_loss": best + 1,
        "brecha_acc_pp": round(gap_acc, 2), "brecha_loss": round(gap_loss, 4),
        "subida_val_loss_desde_minimo": round(rise, 4), "test_acc": round(test_acc, 4),
    }


def update_readme(metrics, report_dict, args):
    """Escribe la tabla de resultados y el análisis entre los marcadores del README."""
    readme = ROOT_DIR / "README.md"
    start, end = "<!-- RESULTADOS_INICIO -->", "<!-- RESULTADOS_FIN -->"
    if not readme.exists():
        return
    content = readme.read_text(encoding="utf-8")
    if start not in content or end not in content:
        return

    h, d = metrics["history"], metrics["diagnostico"]
    rows = "\n".join(
        f"| {i + 1} | {h['train_loss'][i]:.4f} | {h['train_acc'][i] * 100:.2f}% | "
        f"{h['val_loss'][i]:.4f} | {h['val_acc'][i] * 100:.2f}% |"
        for i in range(len(h["train_loss"]))
    )
    per_class = sorted(
        ((c, report_dict[c]["recall"], report_dict[c]["precision"], report_dict[c]["f1-score"]) for c in CLASSES),
        key=lambda t: t[3],
    )
    class_rows = "\n".join(f"| {c} | {p * 100:.1f}% | {r * 100:.1f}% | {f * 100:.1f}% |"
                           for c, r, p, f in per_class)
    worst, best = per_class[0], per_class[-1]
    cm = np.array(metrics["confusion_matrix"])
    off = cm.copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)

    block = f"""{start}
> Resultados generados automáticamente por `src/train.py` el {metrics['fecha']} con semilla **{SEED}**,
> {args.epochs} épocas, batch {args.batch_size}, Adam (lr = {args.lr}), dispositivo `{metrics['device']}`.

**Exactitud en el conjunto de prueba (10 000 imágenes): {metrics['test_acc'] * 100:.2f}%**
(línea base aleatoria: 10%).

| Época | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
{rows}

Métricas por clase en prueba (ordenadas de peor a mejor F1):

| Clase | Precisión | Recall | F1 |
|---|---:|---:|---:|
{class_rows}

#### Análisis crítico

1. **Diagnóstico de ajuste.** {d['texto']}
2. **Generalización.** La exactitud final de validación fue {h['val_acc'][-1] * 100:.2f}% y la de prueba
   {metrics['test_acc'] * 100:.2f}%. La diferencia de {abs(h['val_acc'][-1] - metrics['test_acc']) * 100:.2f} pp
   indica que la validación estratificada fue un buen estimador del rendimiento sobre datos nunca vistos.
3. **Clases.** La clase con mejor F1 es **{best[0]}** ({best[3] * 100:.1f}%) y la peor es
   **{worst[0]}** ({worst[3] * 100:.1f}%). La confusión más frecuente es **{CLASSES[i]} → {CLASSES[j]}**
   ({cm[i, j]} imágenes), típico entre clases visualmente parecidas a 32×32 píxeles.
4. **Mejoras posibles.** Añadir BatchNorm, una segunda convolución por bloque, scheduler de tasa de
   aprendizaje y más épocas; si la validación empieza a empeorar, aplicar early stopping.

![Curvas de aprendizaje](outputs/loss_accuracy.png)

![Matriz de confusión](outputs/confusion_matrix.png)
{end}"""
    before = content.split(start)[0]
    after = content.split(end)[1]
    readme.write_text(before + block + after, encoding="utf-8")
    print(f"README.md actualizado con los resultados: {readme}")


def main():
    args = parse_args()
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device} | Semilla: {SEED}")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, val_size=args.val_size, num_workers=args.num_workers)

    model = CNN_CIFAR10().to(device)
    validate_shapes(CNN_CIFAR10(), batch_size=4)
    print(f"Parámetros entrenables: {count_parameters(model):,}")

    # Configuración del entrenamiento
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    EPOCHS = args.epochs

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_path = 0.0, OUTPUT_DIR / "best_model.pth"

    print(f"\n{'Época':>6} | {'Train loss':>10} | {'Train acc':>9} | {'Val loss':>8} | {'Val acc':>7} | {'Tiempo':>6}")
    print("-" * 66)
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        mark = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            mark = " *"
        print(f"{epoch + 1:>3}/{EPOCHS:<2} | {train_loss:>10.4f} | {train_acc * 100:>8.2f}% | "
              f"{val_loss:>8.4f} | {val_acc * 100:>6.2f}% | {time.time() - t0:>5.1f}s{mark}")

    # Evaluación final con el mejor modelo según validación
    model.load_state_dict(torch.load(best_path, map_location=device))
    y_true, y_pred = predict(model, test_loader, device)
    test_acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred)
    report_txt = classification_report(y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)

    print(f"\nExactitud en prueba: {test_acc * 100:.2f}%")
    print(report_txt)

    plot_curves(history, OUTPUT_DIR / "loss_accuracy.png")
    plot_confusion(cm, OUTPUT_DIR / "confusion_matrix.png", test_acc)
    (OUTPUT_DIR / "classification_report.txt").write_text(
        f"Semilla: {SEED}\nExactitud en prueba: {test_acc * 100:.2f}%\n\n{report_txt}", encoding="utf-8")

    diag = diagnose(history, test_acc)
    metrics = {
        "seed": SEED, "fecha": time.strftime("%Y-%m-%d %H:%M"), "device": str(device),
        "epochs": EPOCHS, "batch_size": args.batch_size, "lr": args.lr,
        "history": history, "test_acc": test_acc, "best_val_acc": best_val_acc,
        "confusion_matrix": cm.tolist(), "diagnostico": diag,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDIAGNÓSTICO:", diag["veredicto"].upper())
    print(diag["texto"].replace("**", ""))
    update_readme(metrics, report_dict, args)
    print(f"\nArchivos guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
