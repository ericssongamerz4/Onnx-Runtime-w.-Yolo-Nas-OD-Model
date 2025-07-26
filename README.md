# Onnx Runtime w. Yolo-Nas OD Model
<div align="center">   
  <img src="MLFlujometros\Resources\giddings.jpg" alt="Giddings" width="300">
  <h1>📷 MLFlujometros</h1>
  <p><strong>Modelo de detección de dígitos en flujómetros utilizando YOLO-NAS-S</strong></p>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/ONNX-white?logo=onnx&logoColor=black&style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-FFDD00?logo=python&logoColor=black&style=for-the-badge" />
  <img src="https://img.shields.io/badge/YOLO NAS-3776AB?style=for-the-badge&logo=data:image/svg+xml;base64,..."/>  
  <img src="https://img.shields.io/badge/Made%20with-🖤-white?style=for-the-badge&logo=data:image/svg+xml;base64,..."/> 
</div>

---

## 📑 Tabla de Contenido

- [📌 Descripción](#-descripción)
- [📊 Métricas del Modelo](#-métricas-del-modelo)
- [🖼️ Ejemplos](#ejemplos-de-resultados)
- [🚀 Cómo Utilizar](#-cómo-utilizar)
  - [✅ Requisitos previos](#-requisitos-previos)
  - [🧪 Entrenar el modelo](#-entrenar-el-modelo)
  - [📦 Exportar el modelo](#-exportar-el-modelo)
  - [🔍 Probar-el-modelo](#-probar-el-modelo)
- [🧰 Tecnologías e Implementaciones Clave](#-tecnologías-e-implementaciones-clave)
- [🚧 Cosas por Implementar](#-cosas-por-implementar)
- [🧑‍💻 Colaboradores](#-colaboradores)
- [📄 Licencia](#-licencia)
- [📚 Referencias y Recursos Adicionales](#-referencias-y-recursos-adicionales)

---

## 📌 Descripción 

**MLFlujometros** es un módulo de visión por computadora diseñado para detectar y reconocer automáticamente los dígitos numéricos presentes en medidores de flujo de agua. Utiliza un modelo YOLO-NAS entrenado con un dataset personalizado de [Roboflow](https://roboflow.com) y permite exportación a ONNX para su integración con otros entornos, como .NET MAUI.

> ⚠️ **Nota:** Este proyecto se encuentra en desarrollo activo. Las funcionalidades están en evolución y podrían cambiar sin previo aviso.

---

## 📊 Métricas del Modelo 

| Métrica                      | Valor Final | Descripción                          |
|------------------------------|-------------|--------------------------------------|
| **📦 Total Loss**            | 1.659       | Pérdida global del modelo            |
| **🏷️ Cls Loss**             | 0.759       | Error en clasificación de objetos    |
| **📏 IOU Loss**             | 0.198       | Error en precisión de bounding boxes |
| **🔄 DFL Loss**             | 0.811       | Pérdida de distribución focal        |
| **🎯 Precision@0.5**        | 0.194       | Exactitud de detecciones (IOU=0.5)   |
| **🔍 Recall@0.5**           | 0.957       | Porcentaje de objetos detectados     |
| **💎 mAP@0.5**              | 0.911       | Precisión media (IOU=0.5)            |
| **⚡ F1-Score@0.5**         | 0.317       | Balance precisión-recall             |

>ℹ️**Nota:** Estas métricas se obtuvieron tras 101 épocas de entrenamiento.
---

## Ejemplos de resultados
Comparación entre imágenes originales de flujómetros y su procesamiento con el modelo YOLO-NAS S. Las imágenes procesadas muestran las detecciones de dígitos enmarcadas.

<details open>
  <summary>Imagenes</summary>
<table border="1" cellspacing="0" cellpadding="10" align="center" style="width:90%; text-align:center;">
  <tr>
    <th>Imagen Original</th>
    <th>Imagen Procesada (Detección)</th>
  </tr>
  <tr>
    <td>
      <img src="MLFlujometros\Resources\test-images\Test 1.jpg" alt="Original 1" style="width:100%;">
      <p><em>Flujómetro 1</em></p>
    </td>
    <td>
      <img src="MLFlujometros\Resources\test-images\Test 1 Detection.jpeg" alt="Procesada 1" style="width:100%;">
      <p><em>Detectado: ✅️369089</em></p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="MLFlujometros\Resources\test-images\Test 2.jpg" alt="Original 2" style="width:100%;">
      <p><em>Flujómetro 2</em></p>
    </td>
    <td>
      <img src="MLFlujometros\Resources\test-images\Test 2 Detection.jpeg" alt="Procesada 2" style="width:100%;">
      <p><em>Detectado: ✅️069657</em></p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="MLFlujometros\Resources\test-images\Test 3.jpeg" alt="Original 3" style="width:100%;">
      <p><em>Flujómetro 3</em></p>
    </td>
    <td>
      <img src="MLFlujometros\Resources\test-images\Test 3 Detection.jpeg" alt="Procesada 3" style="width:100%;">
      <p><em>Detectado: ❌ 01762</em></p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="MLFlujometros\Resources\test-images\Test 4.jpeg" alt="Original 4" style="width:100%;">
      <p><em>Flujómetro 4</em></p>
    </td>
    <td>
      <img src="MLFlujometros\Resources\test-images\Test 4 Detection.jpeg" alt="Procesada 4" style="width:100%;">
      <p><em>Detectado:✅️ 401762</em></p>
    </td>
  </tr>
  
  <tr>
    <td>
      <img src="MLFlujometros\Resources\test-images\Test 5.jpeg" alt="Original 5" style="width:100%;">
      <p><em>Flujómetro 5</em></p>
    </td>
    <td>
      <img src="MLFlujometros\Resources\test-images\Test 5 Detection.jpeg" alt="Procesada 5" style="width:100%;">
      <p><em>Detectado:✅️ 371157</em></p>
    </td>
  </tr>
</table>
</details>

---

## 🚀 Cómo Utilizar

Este proyecto se basa en el wrapper de YOLO-NAS de [naseemap47](https://github.com/naseemap47/YOLO-NAS/tree/master) para simplificar el proceso de entrenamiento del modelo y hacerlo desde la consola. Tambien tiene un script para exportar el modelo llamado **`export-model.py`**

---

### ✅ Requisitos previos
**1.** Crea un ambiente de Python con anaconda.
   ```bash
      conda create -n yolo-nas python=3.9 -y
      conda activate yolo-nas   
  ```
**2.** Tener instalado el paquete  `PyTorch v1.11.0`.
   ```bash
      pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
   ```
**3.** Tener instalado el paquete  `super-gradients`.
   ```bash
      pip install super-gradients==3.1.3
   ```   
**4.** Asegurarse de tener el archivo `data.yaml`.
   ```yaml
    # Rutas de las carpetas 
    Dir: 'dataset'
    images:
      test: images/test2017
      train: images/train2017
      val: images/val2017
    labels:
      test: annotations/instances_test2017.json
      train: annotations/instances_train2017.json
      val: annotations/instances_val2017.json
   ```
### 🧪 Entrenar el modelo

Puedes entrenar tu modelo YOLO-NAS con un solo comando en línea.

<details>
  <summary>Args</summary>
  
  `-i`, `--data`: ruta al archivo data.yaml <br>
  `-n`, `--name`: nombre del directorio de checkpoints <br>
  `-b`, `--batch`: tamaño del lote de entrenamiento <br>
  `-e`, `--epoch`: número de épocas de entrenamiento<br>
  `-s`, `--size`: tamaño de la imagen de entrada <br>
  `-j`, `--worker`: número de procesos de entrenamiento <br>
  `-m`, `--model`: tipo de modelo (Opciones: `yolo_nas_s`, `yolo_nas_m`, `yolo_nas_l`) <br>
  `-w`, `--weight`: ruta al peso preentrenado del modelo (`ckpt_best.pth`) (default: `coco` weight) <br>
 `--gpus`: Entrenar en múltiples GPUs <br>
  `--cpu`: Entrenar usando CPU <br>
  `--resume`: Reanudar el entrenamiento del modelo <br>
  
  **Otros parámetros de entrenamiento:**<br>
  `--warmup_mode`: Modo de calentamiento, por ejemplo: Linear Epoch Step <br>
  `--warmup_initial_lr`: Tasa de aprendizaje inicial durante el calentamiento <br>
  `--lr_warmup_epochs`: Número de épocas de calentamiento para la tasa de aprendizaje <br>
  `--initial_lr`: Tasa de aprendizaje inicial <br>
  `--lr_mode`: Modo de tasa de aprendizaje, por ejemplo: cosine <br>
  `--cosine_final_lr_ratio`: Proporción final de la tasa de aprendizaje en modo cosine <br>
  `--optimizer`: Optimizador, por ejemplo: Adam <br>
  `--weight_decay`: Decaimiento de pesos
  
</details>

**Ejemplo:**
```
python train.py --data C:\Users\Residente\Desktop\yolo-nas-v2\YOLO-NAS\data.yaml --epoch 100 --model yolo_nas_s --size 640 --weight C:\Users\Residente\Desktop\yolo-nas-v2\YOLO-NAS\runs\yolo_nas_s_test33\ckpt_latest.pth --resume
```
      
---

### 📦 Exportar el modelo
Para exportar un modelo se utiliza el script **`export-model.py`** 

**1.** Configura los siguientes parametros dentro del script.
`checkpoint_path`: Ruta al modelo que quieres exportar.
`num_classes`: Numero de clases que tiene el modelo.
`model-name`: Nombre del modelo exportado por ejemplo "yolo_nas_s_3.onnx"<br>

   ```python
    import torch
    from super_gradients.common.object_names import Models
    from super_gradients.training import models

    model = models.get(Models.YOLO_NAS_S, checkpoint_path="ckpt_best2.pth", num_classes=12)

    model.eval()
    model.prep_model_for_conversion(input_size=[1, 3, 640, 640])
    model.export("model-name", postprocessing=True, preprocessing=True)
  
   ```

**2.** Ejecuta el script.
      
---

### 🔎 Probar el modelo
Puedes realizar la inferencia de tu modelo **YOLO-NAS** con **un solo comando**

#### Soporta
- Imágenes
- Videos
- Cámara
- RTSP

<details>
  <summary>Argumentos</summary>
  
  `-n`, `--num`: Número de clases en las que se entrenó el modelo <br>
  `-m`, `--model`: Tipo de modelo (opciones: `yolo_nas_s`, `yolo_nas_m`, `yolo_nas_l`) <br>
  `-w`, `--weight`: ruta al peso del modelo entrenado, para modelo COCO: `coco` <br>
  `-s`, `--source`: ruta de video/id-de-cámara/RTSP <br>
  `-c`, `--conf`: confianza de predicción del modelo (0<conf<1) <br>
  `--save`: guardar video <br>
  `--hide`: ocultar ventana de video

</details>

**Ejemplo:**
```bash
python inference.py --num 12 --model yolo_nas_s --weight C:\Users\Residente\Desktop\yolo-nas-v2\YOLO-NAS\runs\yolo_nas_s_test33\ckpt_best.pth --source C:\Users\Residente\Desktop\ImagenesAppHuellaHidrica\1ef03701-1286-4e50-9183-948f11d21f26.jpg --conf 0.5
```
    
---

## 🧰 Tecnologías e Implementaciones Clave

- 🔍 **YOLO-NAS-S (Super Gradients):** Para detección de objetos.
- 🧠 **Modelo personalizado:** Entrenado con dataset de Roboflow para detectar dígitos del 0 al 9.
- 🧾 **Exportación a ONNX:** Facilita la interoperabilidad con otros entornos como .NET MAUI.
- 🐍 **Python 3.9:** Lenguaje base del entrenamiento y procesamiento.
- 🗃️ **Dataset anotado:** Roboflow project [water-meter-detection](https://universe.roboflow.com/test-qakhb/water-meter-detection-acmux-mmhgb/dataset/1)

---

## 🚧 Cosas por Implementar

- [ ] Volver a entrenar el modelo con datos reales tomados de flujometros utilizados por FRUITS-GIDDINGS, S.A. DE C.V. para aumentar la precisión del modelo.

---

## 🧑‍💻 Colaboradores

- [Erick Montaño](https://github.com/ericssongamerz4)

---

## 📄 Licencia

Este proyecto forma parte de un sistema empresarial interno. Su distribución está restringida a los colaboradores autorizados de FRUITS-GIDDINGS, S.A. DE C.V.

---

## 📚 Referencias y Recursos Adicionales

- [Ultralytics](https://github.com/ultralytics/ultralytics?tab=readme-ov-file)
- [Yolov8 FULL TUTORIAL | Detection | Classification | Segmentation | Pose | Computer vision](https://www.youtube.com/watch?v=Z-65nqxUdl4&t=9326s)
- [332 - All about image annotations​](https://youtu.be/NYeJvxe5nYw?si=ndt5-JarjZgyz8pG)
- [Netron](https://netron.app/) para ver de manera visual como funciona el modelo que estas utilizando
- [Roboflow](https://universe.roboflow.com/)