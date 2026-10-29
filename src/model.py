"""
Pregunta 3: Arquitectura CNN en PyTorch y validación de shapes de tensores 4D.

Ejecutar desde la raíz del repositorio:  python src/model.py
"""
import torch
import torch.nn as nn

SEED = 68


class CNN_CIFAR10(nn.Module):
    """
    CNN con 3 bloques convolucionales (Conv2d -> ReLU -> MaxPool2d) y un clasificador denso.

    Flujo de tensores (B = tamaño de batch):
        entrada   (B,   3, 32, 32)
        bloque 1  (B,  32, 16, 16)
        bloque 2  (B,  64,  8,  8)
        bloque 3  (B, 128,  4,  4)
        flatten   (B, 2048)
        fc1       (B, 256)
        salida    (B, 10)   -> logits de las 10 clases
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3, img_size: int = 32):
        super().__init__()

        # Bloque 1: 3 -> 32 canales, 32x32 -> 16x16
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Bloque 2: 32 -> 64 canales, 16x16 -> 8x8
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Bloque 3: 64 -> 128 canales, 8x8 -> 4x4
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()  # (B, C, H, W) -> (B, C*H*W)

        # Cálculo dinámico del tamaño de entrada a la primera capa Linear:
        # se pasa un tensor vacío por los bloques convolucionales y se mide la salida.
        # Así se evita el error de "shape mismatch" si se cambia la arquitectura.
        self.flat_features = self._get_flatten_size(in_channels, img_size)

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def _get_flatten_size(self, in_channels: int, img_size: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            out = self.block3(self.block2(self.block1(dummy)))
        return out.numel()  # 128 * 4 * 4 = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x  # logits; CrossEntropyLoss aplica softmax internamente


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def validate_shapes(model: nn.Module, batch_size: int = 8, num_classes: int = 10) -> torch.Tensor:
    """
    Pasa un batch simulado por el modelo, imprime cada shape intermedio
    y verifica con aserciones que el flujo de tensores 4D es correcto.
    """
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32)
    print("=" * 60)
    print("VALIDACIÓN DEL FORWARD PASS (tensores 4D)")
    print("=" * 60)
    print(f"{'Entrada':<28} {tuple(x.shape)}")
    assert x.dim() == 4, "La entrada debe ser 4D: (Batch, Canales, Alto, Ancho)"

    with torch.no_grad():
        expected = {
            "block1": (batch_size, 32, 16, 16),
            "block2": (batch_size, 64, 8, 8),
            "block3": (batch_size, 128, 4, 4),
        }
        for name in ["block1", "block2", "block3"]:
            block = getattr(model, name)
            for layer in block:
                x = layer(x)
                print(f"{name + ' -> ' + layer.__class__.__name__:<28} {tuple(x.shape)}")
            assert tuple(x.shape) == expected[name], f"Shape incorrecto en {name}: {tuple(x.shape)}"

        x = model.flatten(x)
        print(f"{'Flatten':<28} {tuple(x.shape)}")
        assert x.shape == (batch_size, model.flat_features)

        for i, layer in enumerate(model.classifier):
            x = layer(x)
            print(f"{'classifier -> ' + layer.__class__.__name__:<28} {tuple(x.shape)}")

    assert x.shape == (batch_size, num_classes), f"Salida esperada {(batch_size, num_classes)}, se obtuvo {tuple(x.shape)}"

    # Comprobación cruzada con forward() completo
    with torch.no_grad():
        out = model(torch.randn(batch_size, 3, 32, 32))
    assert out.shape == (batch_size, num_classes)

    print("-" * 60)
    print(f"Tamaño de entrada a la capa Linear (calculado dinámicamente): {model.flat_features}")
    print(f"Parámetros entrenables: {count_parameters(model):,}")
    print("OK: todas las aserciones pasaron, no hay shape mismatch.")
    print("=" * 60)
    return out


if __name__ == "__main__":
    torch.manual_seed(SEED)
    model = CNN_CIFAR10()
    print(model)
    validate_shapes(model, batch_size=8)
