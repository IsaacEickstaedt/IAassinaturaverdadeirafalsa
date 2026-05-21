# ========================
# 1. IMPORTS
# ========================

import os
import random
import cv2
import numpy as np

from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# ========================
# 2. CONFIG
# ========================

IMG_SIZE = 100
DATASET_PATH = "extract"

# threshold mais tolerante
THRESHOLD = 0.4

# quantidade de pares positivos aleatórios
POSITIVE_RANDOM_PAIRS = 15

# quantidade de pares idênticos
IDENTICAL_PAIRS = 2

# seed
random.seed(42)
np.random.seed(42)

# ========================
# 3. LOAD IMAGE
# ========================

def load_image(path):

    img = cv2.imread(path, 0)

    if img is None:
        raise Exception(f"Erro ao carregar imagem: {path}")

    # resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # blur leve
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # threshold leve
    _, img = cv2.threshold(
        img,
        180,
        255,
        cv2.THRESH_BINARY
    )

    # normalização
    img = img.astype("float32") / 255.0

    return img


# ========================
# 4. CREATE PAIRS
# ========================

def create_pairs(base_path):

    pairs = []
    labels = []

    persons = [
        p for p in os.listdir(base_path)
        if "_forg" not in p
    ]

    for person in persons:

        real_path = os.path.join(base_path, person)
        forg_path = os.path.join(base_path, person + "_forg")

        if not os.path.exists(forg_path):
            continue

        real_imgs = os.listdir(real_path)
        forg_imgs = os.listdir(forg_path)

        positive_pairs = []
        negative_pairs = []

        # ========================
        # POSITIVOS
        # ========================

        # pares reais diferentes da mesma pessoa
        for _ in range(POSITIVE_RANDOM_PAIRS):

            img1_name = random.choice(real_imgs)
            img2_name = random.choice(real_imgs)

            # evita excesso de pares idênticos
            while img1_name == img2_name:
                img2_name = random.choice(real_imgs)

            img1 = load_image(
                os.path.join(real_path, img1_name)
            )

            img2 = load_image(
                os.path.join(real_path, img2_name)
            )

            positive_pairs.append([img1, img2])

        # poucos pares idênticos
        for _ in range(IDENTICAL_PAIRS):

            img_name = random.choice(real_imgs)

            img1 = load_image(
                os.path.join(real_path, img_name)
            )

            img2 = load_image(
                os.path.join(real_path, img_name)
            )

            positive_pairs.append([img1, img2])

        # ========================
        # NEGATIVOS
        # ========================

        for _ in range(len(positive_pairs)):

            real_name = random.choice(real_imgs)
            forg_name = random.choice(forg_imgs)

            img1 = load_image(
                os.path.join(real_path, real_name)
            )

            img2 = load_image(
                os.path.join(forg_path, forg_name)
            )

            negative_pairs.append([img1, img2])

        # ========================
        # BALANCEAMENTO
        # ========================

        min_len = min(
            len(positive_pairs),
            len(negative_pairs)
        )

        positive_pairs = positive_pairs[:min_len]
        negative_pairs = negative_pairs[:min_len]

        # adiciona positivos
        for pair in positive_pairs:

            pairs.append(pair)
            labels.append(1)

        # adiciona negativos
        for pair in negative_pairs:

            pairs.append(pair)
            labels.append(0)

    return np.array(pairs), np.array(labels)


# ========================
# 5. BASE NETWORK
# ========================

def create_base_network(input_shape):

    input_layer = Input(shape=input_shape)

    # bloco 1
    x = layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        padding='same'
    )(input_layer)

    x = layers.MaxPooling2D()(x)

    # bloco 2
    x = layers.Conv2D(
        64,
        (3, 3),
        activation='relu',
        padding='same'
    )(x)

    x = layers.MaxPooling2D()(x)

    # bloco 3
    x = layers.Conv2D(
        128,
        (3, 3),
        activation='relu',
        padding='same'
    )(x)

    x = layers.MaxPooling2D()(x)

    # flatten
    x = layers.Flatten()(x)

    # dense
    x = layers.Dense(
        256,
        activation='relu'
    )(x)

    # dropout
    x = layers.Dropout(0.3)(x)

    return Model(input_layer, x)


# ========================
# 6. COMPARE
# ========================

def compare_signatures(model, img1_path, img2_path):

    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    img1 = img1.reshape(
        1,
        IMG_SIZE,
        IMG_SIZE,
        1
    )

    img2 = img2.reshape(
        1,
        IMG_SIZE,
        IMG_SIZE,
        1
    )

    score = model.predict([img1, img2], verbose=0)[0][0]

    print("\n======================")
    print("Comparação de assinaturas")
    print("======================")

    print(f"Imagem 1: {img1_path}")
    print(f"Imagem 2: {img2_path}")

    print(f"Similaridade: {score:.4f}")

    if score >= THRESHOLD:
        print("Resultado: MESMA PESSOA")
    else:
        print("Resultado: PESSOAS DIFERENTES")


# ========================
# 7. EXECUÇÃO
# ========================

print("🔄 Criando pares...")

pairs, labels = create_pairs(DATASET_PATH)

print(f"Total pares: {len(pairs)}")

print("\nDistribuição:")
print("Positivos:", np.sum(labels == 1))
print("Negativos:", np.sum(labels == 0))

# ========================
# 8. INPUTS
# ========================

X1 = pairs[:, 0].reshape(
    -1,
    IMG_SIZE,
    IMG_SIZE,
    1
)

X2 = pairs[:, 1].reshape(
    -1,
    IMG_SIZE,
    IMG_SIZE,
    1
)

# ========================
# 9. SPLIT
# ========================

X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
    X1,
    X2,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ========================
# 10. MODELO SIAMÊS
# ========================

print("\n🧠 Criando rede siamesa...")

base_network = create_base_network(
    (IMG_SIZE, IMG_SIZE, 1)
)

input_a = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
input_b = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

feat_a = base_network(input_a)
feat_b = base_network(input_b)

# distância absoluta
distance = layers.Lambda(
    lambda tensors: abs(tensors[0] - tensors[1])
)([feat_a, feat_b])

# camada final
x = layers.Dense(
    128,
    activation='relu'
)(distance)

x = layers.Dropout(0.2)(x)

output = layers.Dense(
    1,
    activation='sigmoid'
)(x)

model = Model(
    [input_a, input_b],
    output
)

# ========================
# 11. COMPILE
# ========================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ========================
# 12. EARLY STOPPING
# ========================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ========================
# 13. TREINAMENTO
# ========================

print("\n🚀 Treinando modelo...")

history = model.fit(
    [X1_train, X2_train],
    y_train,

    validation_data=(
        [X1_test, X2_test],
        y_test
    ),

    batch_size=32,

    epochs=20,

    callbacks=[early_stop]
)

# ========================
# 14. AVALIAÇÃO
# ========================

print("\n📊 Avaliando modelo...")

loss, acc = model.evaluate(
    [X1_test, X2_test],
    y_test
)

print(f"\nAcurácia: {acc:.4f}")
print(f"Loss: {loss:.4f}")

# ========================
# 15. TESTES
# ========================

print("\n🧪 TESTE 1 - MESMA IMAGEM")

compare_signatures(
    model,
    "extract/001/1-001_01.jpg",
    "extract/001/1-001_01.jpg"
)

print("\n🧪 TESTE 2 - MESMA PESSOA")

compare_signatures(
    model,
    "extract/001/1-001_01.jpg",
    "extract/001/1-001_02.jpg"
)

print("\n🧪 TESTE 3 - FALSIFICAÇÃO")

# ajuste o nome do arquivo conforme existir na pasta
compare_signatures(
    model,
    "extract/001/1-001_01.jpg",
    "extract/001_forg/1-002_01.jpg"
)

print("\n✅ Finalizado!")