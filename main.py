# ========================
# 1. IMPORTS
# ========================

import os
import random
import cv2
import numpy as np

from tensorflow.keras import layers, Model, Input
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# ========================
# 2. CONFIGURAÇÕES
# ========================

IMG_SIZE = 100
DATASET_PATH = "extract"
MODEL_PATH = "modelo_assinatura.h5"

# limiar de decisão
THRESHOLD = 0.5

# épocas de treinamento
EPOCHS = 20

# ========================
# 3. FUNÇÕES AUXILIARES
# ========================

def is_image(file):

    return file.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.bmp')
    )


# ========================
# CARREGAR IMAGEM
# ========================

def load_image(path):

    img = cv2.imread(path, 0)

    if img is None:

        raise Exception(
            f"Erro ao carregar imagem: {path}"
        )

    # redimensiona
    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    # binariza assinatura
    img = cv2.threshold(
        img,
        127,
        255,
        cv2.THRESH_BINARY_INV
    )[1]

    # normaliza
    img = img / 255.0

    return img


# ========================
# CRIAR PARES
# ========================

def create_pairs(base_path):

    pairs = []
    labels = []

    persons = sorted([
        p for p in os.listdir(base_path)
        if "_forg" not in p
    ])

    for person in persons:

        real_path = os.path.join(
            base_path,
            person
        )

        forg_path = os.path.join(
            base_path,
            person + "_forg"
        )

        if not os.path.exists(forg_path):

            continue

        # imagens reais
        real_imgs = sorted([

            f for f in os.listdir(real_path)
            if is_image(f)

        ])

        # imagens falsas
        forg_imgs = sorted([

            f for f in os.listdir(forg_path)
            if is_image(f)

        ])

        # ========================
        # PARES POSITIVOS
        # ========================

        for i in range(len(real_imgs) - 1):

            img1 = load_image(
                os.path.join(
                    real_path,
                    real_imgs[i]
                )
            )

            img2 = load_image(
                os.path.join(
                    real_path,
                    real_imgs[i + 1]
                )
            )

            pairs.append([img1, img2])
            labels.append(1)

        # ========================
        # PARES NEGATIVOS
        # ========================

        for forg in forg_imgs:

            img1 = load_image(
                os.path.join(
                    real_path,
                    random.choice(real_imgs)
                )
            )

            img2 = load_image(
                os.path.join(
                    forg_path,
                    forg
                )
            )

            pairs.append([img1, img2])
            labels.append(0)

    return np.array(pairs), np.array(labels)


# ========================
# REDE BASE
# ========================

def create_base_network(input_shape):

    input_layer = Input(shape=input_shape)

    x = layers.Conv2D(
        32,
        (3, 3),
        activation='relu'
    )(input_layer)

    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    )(x)

    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)

    x = layers.Dense(
        128,
        activation='relu'
    )(x)

    x = layers.Dropout(0.3)(x)

    return Model(input_layer, x)


# ========================
# DISTÂNCIA ENTRE FEATURES
# ========================

def signature_distance(vectors):

    x, y = vectors

    return abs(x - y)


# ========================
# COMPARAR ASSINATURAS
# ========================

def compare_signatures(
    model,
    img1_path,
    img2_path
):

    img1 = load_image(img1_path).reshape(
        1,
        IMG_SIZE,
        IMG_SIZE,
        1
    )

    img2 = load_image(img2_path).reshape(
        1,
        IMG_SIZE,
        IMG_SIZE,
        1
    )

    score = model.predict(
        [img1, img2],
        verbose=0
    )[0][0]

    print("\n========================")
    print("COMPARAÇÃO DE ASSINATURAS")
    print("========================")

    print(f"Imagem 1: {img1_path}")
    print(f"Imagem 2: {img2_path}")

    print(f"\nSimilaridade: {score:.4f}")

    if score > THRESHOLD:

        print(
            "\n✅ RESULTADO: MESMA PESSOA"
        )

    else:

        print(
            "\n❌ RESULTADO: PESSOAS DIFERENTES"
        )


# ========================
# 4. VERIFICAR MODELO
# ========================

if os.path.exists(MODEL_PATH):

    print(
        "📂 Carregando modelo treinado..."
    )

    model = load_model(

        MODEL_PATH,

        compile=False,

        safe_mode=False
    )

    model.compile(

        loss='binary_crossentropy',

        optimizer='adam',

        metrics=['accuracy']
    )

else:

    # ========================
    # DATASET
    # ========================

    print(
        "🔄 Criando pares de imagens..."
    )

    pairs, labels = create_pairs(
        DATASET_PATH
    )

    print(
        f"Total de pares: {len(pairs)}"
    )

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
    # SPLIT
    # ========================

    (
        X1_train,
        X1_test,
        X2_train,
        X2_test,
        y_train,
        y_test

    ) = train_test_split(

        X1,
        X2,
        labels,

        test_size=0.2,

        random_state=42
    )

    # ========================
    # MODELO SIAMÊS
    # ========================

    print("🧠 Criando modelo...")

    base_network = create_base_network(
        (IMG_SIZE, IMG_SIZE, 1)
    )

    input_a = Input(
        shape=(IMG_SIZE, IMG_SIZE, 1)
    )

    input_b = Input(
        shape=(IMG_SIZE, IMG_SIZE, 1)
    )

    feat_a = base_network(input_a)
    feat_b = base_network(input_b)

    # distância entre embeddings
    distance = layers.Lambda(
        signature_distance
    )([feat_a, feat_b])

    output = layers.Dense(
        1,
        activation='sigmoid'
    )(distance)

    model = Model(
        [input_a, input_b],
        output
    )

    model.compile(

        loss='binary_crossentropy',

        optimizer='adam',

        metrics=['accuracy']
    )

    # ========================
    # TREINAMENTO
    # ========================

    print("🚀 Treinando modelo...")

    model.fit(

        [X1_train, X2_train],

        y_train,

        validation_data=(

            [X1_test, X2_test],
            y_test

        ),

        batch_size=32,

        epochs=EPOCHS
    )

    # ========================
    # AVALIAÇÃO
    # ========================

    print("\n📊 Avaliando modelo...")

    loss, acc = model.evaluate(

        [X1_test, X2_test],

        y_test
    )

    print(
        f"\n✅ Acurácia no teste: {acc:.4f}"
    )

    # ========================
    # SALVAR MODELO
    # ========================

    model.save(MODEL_PATH)

    print(
        "\n💾 Modelo salvo com sucesso!"
    )


# ========================
# LOOP INFINITO
# ========================

while True:

    print("\n========================")
    print("COMPARADOR DE ASSINATURAS")
    print("========================")

    img1 = input(
        "\nDigite caminho da assinatura VERDADEIRA: "
    )

    img2 = input(
        "Digite caminho da assinatura para TESTE: "
    )

    try:

        compare_signatures(
            model,
            img1,
            img2
        )

    except Exception as e:

        print(f"\n❌ ERRO: {e}")