# Clasificación de imágenes CIFAR-10 con CNN en PyTorch

**Evaluación de la Segunda Unidad – Inteligencia Artificial**
Universidad Andina del Cusco – Departamento Académico de Ingeniería de Sistemas

| Dato | Valor |
|---|---|
| Estudiante | _(escribe tu nombre completo)_ |
| Código de matrícula | 019100568J |
| Semilla individual | **68** (`torch.manual_seed(68)`, `np.random.seed(68)`) |
| Dataset | CIFAR-10 (`torchvision.datasets.CIFAR10`), 60 000 imágenes RGB de 32×32 en 10 clases |

---

## Estructura del repositorio

```
.
├── src/
│   ├── dataset.py        # Pregunta 1: carga, transformaciones, partición y visualización
│   ├── model.py          # Pregunta 3: arquitectura CNN y validación de shapes
│   └── train.py          # Pregunta 4: entrenamiento, métricas y gráficos
├── notebooks/
│   └── experiment.ipynb  # Pruebas, EDA y visualización de muestras
├── outputs/
│   ├── loss_accuracy.png
│   ├── confusion_matrix.png
│   ├── samples_grid.png
│   ├── classification_report.txt
│   └── metrics.json
├── requirements.txt
└── README.md             # Documentación técnica (Preguntas 2 y 4)
```

## Cómo ejecutar

```bash
pip install -r requirements.txt
python src/dataset.py   # descarga CIFAR-10, imprime la partición y muestra 16 imágenes
python src/model.py     # imprime los shapes intermedios y valida el forward pass
python src/train.py     # entrena, guarda los gráficos en outputs/ y actualiza este README
```

El dataset se descarga en `data/`, carpeta excluida del repositorio por `.gitignore` (≈170 MB).

---

## Pregunta 1 – Carga, visualización, transformaciones y partición

Implementado en `src/dataset.py` y en el notebook.

- **Semilla:** `set_seed(68)` fija `random`, `numpy`, `torch` y CUDA. La semilla también controla la
  partición (`random_state=68`) y el orden de mezcla del `DataLoader` (`torch.Generator().manual_seed(68)`).
- **Visualización:** `show_samples(n=16)` muestra una cuadrícula 4×4 con la clase real de cada imagen.
- **Transformaciones:**

| Conjunto | Pipeline |
|---|---|
| Entrenamiento | `RandomCrop(32, padding=4)` → `RandomHorizontalFlip(0.5)` → `ToTensor()` → `Normalize(mean, std)` |
| Validación / prueba | `ToTensor()` → `Normalize(mean, std)` |

  Con `mean = (0.4914, 0.4822, 0.4465)` y `std = (0.2470, 0.2435, 0.2616)`, calculadas por canal RGB sobre el
  conjunto de entrenamiento. El aumento solo se aplica en entrenamiento para que la validación mida el
  rendimiento sobre imágenes reales.

- **Partición estratificada:** `train_test_split(..., stratify=targets, random_state=68)` divide las 50 000
  imágenes de entrenamiento en **45 000 de entrenamiento** y **5 000 de validación**, con exactamente
  4 500 y 500 imágenes por clase. El conjunto de prueba (10 000) se mantiene independiente.

---

## Pregunta 2 – Justificación teórica: CNN frente a MLP

### Por qué un MLP es ineficiente en CIFAR-10

1. **Explosión de parámetros.** Un MLP necesita aplanar cada imagen en un vector de
   32 × 32 × 3 = **3 072** valores. Una sola capa oculta de 512 neuronas requiere
   3 072 × 512 + 512 = **1 573 376** parámetros. Un MLP modesto 3072→512→256→10 suma **1 707 274**
   parámetros, casi 3 veces los de la CNN de este proyecto (620 362), y para imágenes más grandes el
   crecimiento es cuadrático e inviable (una imagen de 224×224×3 daría 77 millones de pesos en la primera capa).
2. **Pérdida de la estructura espacial.** Al aplanar, el MLP trata cada píxel como una variable
   independiente: no sabe que dos píxeles son vecinos ni que los tres canales RGB pertenecen al mismo punto.
   Esa relación espacial, que es justamente lo que forma bordes y texturas, debe reaprenderse desde cero.
3. **Sin invariancia a la traslación.** Si un gato aparece desplazado unos píxeles, activa neuronas de
   entrada totalmente distintas. El MLP tendría que aprender el mismo patrón en cada posición posible,
   lo que exige muchos más datos.
4. **Sobreajuste.** Con tantos parámetros y solo 45 000 imágenes, el MLP tiende a memorizar el
   entrenamiento; en la práctica un MLP en CIFAR-10 se queda alrededor de 50–55% de exactitud en prueba.

### Cómo lo resuelven las CNN

| Propiedad | Qué hace | Efecto |
|---|---|---|
| **Conexiones locales** | Cada neurona mira solo una ventana de 3×3 píxeles (campo receptivo local) | Aprovecha que los píxeles cercanos están correlacionados; detecta bordes, esquinas y texturas |
| **Weight sharing** (pesos compartidos) | El mismo filtro de 3×3×C se desliza por toda la imagen | La primera capa usa solo 3×3×3×32 + 32 = **896** parámetros, frente a 1,57 M del MLP |
| **Equivariancia + pooling → invariancia espacial** | Un patrón desplazado produce el mismo mapa de activación desplazado; `MaxPool2d` resume cada región 2×2 | El modelo reconoce el objeto aunque cambie ligeramente de posición |
| **Jerarquía de características** | Bloques sucesivos combinan rasgos simples en complejos (bordes → partes → objetos) | Representaciones más abstractas a medida que baja la resolución y aumentan los canales |

### Arquitectura diseñada

```
Entrada (B, 3, 32, 32)
   │
   ├─ Bloque 1: Conv2d(3→32, k=3, p=1) → ReLU → MaxPool2d(2)    → (B, 32, 16, 16)
   ├─ Bloque 2: Conv2d(32→64, k=3, p=1) → ReLU → MaxPool2d(2)   → (B, 64, 8, 8)
   ├─ Bloque 3: Conv2d(64→128, k=3, p=1) → ReLU → MaxPool2d(2)  → (B, 128, 4, 4)
   │
   ├─ Flatten                                                    → (B, 2048)
   ├─ Linear(2048→256) → ReLU → Dropout(0.5)                     → (B, 256)
   └─ Linear(256→10)                                             → (B, 10) logits
```

| Capa | Salida | Parámetros |
|---|---|---:|
| Conv2d 3→32 (3×3) | (B, 32, 32, 32) | 896 |
| MaxPool2d 2×2 | (B, 32, 16, 16) | 0 |
| Conv2d 32→64 (3×3) | (B, 64, 16, 16) | 18 496 |
| MaxPool2d 2×2 | (B, 64, 8, 8) | 0 |
| Conv2d 64→128 (3×3) | (B, 128, 8, 8) | 73 856 |
| MaxPool2d 2×2 | (B, 128, 4, 4) | 0 |
| Flatten | (B, 2048) | 0 |
| Linear 2048→256 | (B, 256) | 524 544 |
| Linear 256→10 | (B, 10) | 2 570 |
| **Total** | | **620 362** |

Decisiones de diseño:
- `padding=1` con kernel 3×3 conserva el alto y ancho en cada convolución; solo `MaxPool2d` los reduce a la mitad.
- Los canales se duplican (32→64→128) mientras la resolución baja, para compensar la pérdida de información espacial.
- `Dropout(0.5)` en la capa densa, que concentra el 85% de los parámetros, reduce el sobreajuste.
- La salida son logits: `CrossEntropyLoss` aplica `log_softmax` internamente.

---

## Pregunta 3 – Tensores 4D y validación del forward pass

Implementado en `src/model.py`:

- `forward()` encadena los tres bloques, `nn.Flatten()` y el clasificador.
- **Cálculo dinámico:** `_get_flatten_size()` pasa un tensor `torch.zeros(1, 3, 32, 32)` por los bloques
  convolucionales y mide `out.numel()` = 128 × 4 × 4 = **2048**. Así la primera capa `Linear` siempre recibe el
  tamaño correcto aunque se modifique la arquitectura, y se evita el *shape mismatch*.
- **Aserción:** `validate_shapes()` pasa un batch simulado `torch.randn(8, 3, 32, 32)`, imprime el shape tras
  cada capa y verifica con `assert` cada bloque y la salida `(8, 10)`.

Fórmula del tamaño de salida de una convolución: `H_out = (H_in + 2·padding − kernel) / stride + 1`.
Con H_in = 32, padding = 1, kernel = 3 y stride = 1: (32 + 2 − 3)/1 + 1 = 32. El MaxPool 2×2 con stride 2
da 32/2 = 16.

---

## Pregunta 4 – Entrenamiento y análisis de métricas

Configuración en `src/train.py`:

| Hiperparámetro | Valor |
|---|---|
| Función de pérdida | `nn.CrossEntropyLoss()` |
| Optimizador | `torch.optim.Adam`, lr = 0.001 |
| Épocas | 15 |
| Batch size | 128 |
| Selección del modelo | Pesos de la época con mayor exactitud de validación (`best_model.pth`) |

En cada época del bucle `for epoch in range(EPOCHS):` se registran loss y accuracy de entrenamiento y
validación. Al final se evalúa en el conjunto de prueba, se guardan `loss_accuracy.png` y
`confusion_matrix.png` y se escribe aquí el análisis con las cifras reales de la semilla 68.

### Resultados

<!-- RESULTADOS_INICIO -->
_Esta sección se completa automáticamente al ejecutar `python src/train.py`._
<!-- RESULTADOS_FIN -->
