"""
Pregunta 1: Carga, visualización, transformaciones y partición estratificada de CIFAR-10.

Semilla individual: 68 (dos últimos dígitos del código de matrícula 019100568J).
Ejecutar desde la raíz del repositorio:  python src/dataset.py
"""
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------
SEED = 68  # código de matrícula 019100568J -> últimos dos dígitos = 68

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

CLASSES = ("avión", "automóvil", "pájaro", "gato", "ciervo",
           "perro", "rana", "caballo", "barco", "camión")

# Media y desviación estándar por canal RGB del conjunto de entrenamiento de CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int = SEED) -> None:
    """Fija todas las fuentes de aleatoriedad para que los resultados sean reproducibles."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Pipeline de transformaciones
# ---------------------------------------------------------------------------
def get_transforms():
    """
    Devuelve (transform_train, transform_eval).

    - Entrenamiento: Data Augmentation (RandomCrop + RandomHorizontalFlip),
      conversión a tensor y normalización de los 3 canales RGB.
    - Validación / prueba: solo conversión a tensor y normalización
      (no se aumenta para medir el rendimiento sobre imágenes reales).
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # recorte aleatorio con relleno de 4 px
        transforms.RandomHorizontalFlip(p=0.5),    # espejo horizontal con probabilidad 0.5
        transforms.ToTensor(),                     # PIL [0,255] HxWxC -> Tensor [0,1] CxHxW
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return transform_train, transform_eval


# ---------------------------------------------------------------------------
# Carga del dataset
# ---------------------------------------------------------------------------
def load_cifar10(transform_train=None, transform_eval=None, download: bool = True):
    """
    Descarga (si hace falta) y carga CIFAR-10.

    Se crean dos vistas del conjunto de entrenamiento con transformaciones
    distintas: una con aumento (para entrenar) y otra sin aumento (para validar).
    Ambas contienen las mismas 50 000 imágenes; la partición se hace por índices.
    """
    if transform_train is None or transform_eval is None:
        transform_train, transform_eval = get_transforms()

    train_aug = datasets.CIFAR10(DATA_DIR, train=True, download=download, transform=transform_train)
    train_eval = datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=transform_eval)
    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=download, transform=transform_eval)
    return train_aug, train_eval, test_set


# ---------------------------------------------------------------------------
# Partición estratificada
# ---------------------------------------------------------------------------
def stratified_split(targets, val_size: float = 0.1, seed: int = SEED):
    """
    Divide los índices del conjunto de entrenamiento en entrenamiento y validación
    manteniendo la misma proporción de cada clase (estratificación).
    La semilla determina qué imágenes caen en cada subconjunto.
    """
    indices = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        stratify=targets,
        random_state=seed,
        shuffle=True,
    )
    return train_idx, val_idx


def get_dataloaders(batch_size: int = 128, val_size: float = 0.1,
                    seed: int = SEED, num_workers: int = 0, verbose: bool = True):
    """Construye los DataLoaders de entrenamiento, validación y prueba."""
    set_seed(seed)
    train_aug, train_eval, test_set = load_cifar10()

    targets = np.array(train_aug.targets)
    train_idx, val_idx = stratified_split(targets, val_size, seed)

    train_subset = Subset(train_aug, train_idx)   # con Data Augmentation
    val_subset = Subset(train_eval, val_idx)      # sin Data Augmentation

    # Generador con semilla: el orden de mezcla (shuffle) de cada época es reproducible
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=g)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    if verbose:
        print_split_info(targets, train_idx, val_idx, test_set, train_loader)

    return train_loader, val_loader, test_loader


def print_split_info(targets, train_idx, val_idx, test_set, train_loader):
    """Imprime en consola las dimensiones resultantes de la partición."""
    x_batch, y_batch = next(iter(train_loader))
    print("=" * 60)
    print(f"PARTICIÓN DEL DATASET (semilla = {SEED})")
    print("=" * 60)
    print(f"Entrenamiento original : {len(targets):>6} imágenes")
    print(f"  -> Entrenamiento     : {len(train_idx):>6} imágenes")
    print(f"  -> Validación        : {len(val_idx):>6} imágenes")
    print(f"Prueba (test)          : {len(test_set):>6} imágenes")
    print(f"Shape de un batch X    : {tuple(x_batch.shape)}  (Batch, Canales, Alto, Ancho)")
    print(f"Shape de un batch y    : {tuple(y_batch.shape)}")
    print(f"Primeros 10 índices de validación: {np.sort(val_idx)[:10].tolist()}")
    print("-" * 60)
    print(f"{'Clase':<12}{'Entrenamiento':>15}{'Validación':>12}")
    train_counts = np.bincount(targets[train_idx], minlength=10)
    val_counts = np.bincount(targets[val_idx], minlength=10)
    for i, name in enumerate(CLASSES):
        print(f"{name:<12}{train_counts[i]:>15}{val_counts[i]:>12}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
def show_samples(n: int = 16, seed: int = SEED, save_path=None, show: bool = True):
    """
    Muestra una cuadrícula de n imágenes (8 a 16) de CIFAR-10 con su etiqueta real.
    Se usan las imágenes originales (sin normalizar) para que los colores sean correctos.
    """
    import matplotlib.pyplot as plt

    raw = datasets.CIFAR10(DATA_DIR, train=True, download=True)  # imágenes PIL originales
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(raw), size=n, replace=False)

    cols = 4 if n <= 16 else 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    for ax, i in zip(axes.flat, idx):
        img, label = raw[i]
        ax.imshow(img)
        ax.set_title(f"{CLASSES[label]} ({label})", fontsize=10)
        ax.axis("off")
    for ax in list(axes.flat)[len(idx):]:
        ax.axis("off")
    fig.suptitle(f"Muestras de CIFAR-10 con su etiqueta real (semilla {seed})", fontsize=12)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Cuadrícula guardada en: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    set_seed(SEED)
    get_dataloaders(batch_size=128)
    show_samples(n=16, save_path=OUTPUT_DIR / "samples_grid.png")
