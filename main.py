# ========================
# 1. IMPORTS
# ========================
import os
import random
import cv2
import numpy as np
from tensorflow.keras import layers, Model, Input
from sklearn.model_selection import train_test_split

# ========================
# 2. CONFIG
# ========================
IMG_SIZE = 100
DATASET_PATH = "extract"
THRESHOLD = 0.7

# ========================
# 3. FUNÇÕES
# ========================
def load_image(path):
    img = cv2.imread(path, 0)

    if img is None:
        raise Exception(f"Erro ao carregar imagem: {path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img


def create_pairs(base_path):
    pairs = []
    labels = []

    persons = [p for p in os.listdir(base_path) if "_forg" not in p]

    for person in persons:
        real_path = os.path.join(base_path, person)
        forg_path = os.path.join(base_path, person + "_forg")

        if not os.path.exists(forg_path):
            continue

        real_imgs = os.listdir(real_path)
        forg_imgs = os.listdir(forg_path)

        # pares positivos (mesma pessoa)
        for i in range(len(real_imgs) - 1):
            img1 = load_image(os.path.join(real_path, real_imgs[i]))
            img2 = load_image(os.path.join(real_path, real_imgs[i+1]))

            pairs.append([img1, img2])
            labels.append(1)

        # pares negativos (real vs falso)
        for i in range(len(forg_imgs)):
            img1 = load_image(os.path.join(real_path, random.choice(real_imgs)))
            img2 = load_image(os.path.join(forg_path, forg_imgs[i]))

            pairs.append([img1, img2])
            labels.append(0)

    return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    input = Input(shape=input_shape)

    x = layers.Conv2D(32, (3,3), activation='relu')(input)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    return Model(input, x)


def compare_signatures(model, img1_path, img2_path):
    img1 = load_image(img1_path).reshape(1, IMG_SIZE, IMG_SIZE, 1)
    img2 = load_image(img2_path).reshape(1, IMG_SIZE, IMG_SIZE, 1)

    score = model.predict([img1, img2])[0][0]

    print("\n======================")
    print("Comparação de assinaturas")
    print("======================")
    print(f"Imagem 1: {img1_path}")
    print(f"Imagem 2: {img2_path}")
    print(f"Similaridade: {score:.4f}")

    if score > THRESHOLD:
        print("Resultado: MESMA PESSOA")
    else:
        print("Resultado: PESSOAS DIFERENTES")

# ========================
# 4. EXECUÇÃO
# ========================

print("🔄 Criando pares de imagens...")
pairs, labels = create_pairs(DATASET_PATH)

print(f"Total de pares: {len(pairs)}")

X1 = pairs[:,0].reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X2 = pairs[:,1].reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# ========================
# 5. SPLIT TREINO/TESTE
# ========================

X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
    X1, X2, labels, test_size=0.2, random_state=42
)

# ========================
# 6. MODELO SIAMÊS
# ========================

print("🧠 Criando modelo...")

base_network = create_base_network((IMG_SIZE, IMG_SIZE, 1))

input_a = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
input_b = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

feat_a = base_network(input_a)
feat_b = base_network(input_b)

distance = layers.Lambda(lambda x: abs(x[0] - x[1]))([feat_a, feat_b])
output = layers.Dense(1, activation='sigmoid')(distance)

model = Model([input_a, input_b], output)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ========================
# 7. TREINAMENTO
# ========================

print("🚀 Treinando modelo...")

model.fit(
    [X1_train, X2_train],
    y_train,
    validation_data=([X1_test, X2_test], y_test),
    batch_size=32,
    epochs=5
)

# ========================
# 8. AVALIAÇÃO
# ========================

print("\n📊 Avaliando modelo...")
loss, acc = model.evaluate([X1_test, X2_test], y_test)

print(f"Acurácia no teste: {acc:.4f}")

# ========================
# 9. TESTE PRÁTICO
# ========================

compare_signatures(
    model,
    "extract/001/1-001_01.jpg",
    "extract/001_forg/1-002_01.jpg"
)

print("\n✅ Finalizado!")