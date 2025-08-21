import cv2
import joblib
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

TAMANHO = 64
PASTA_PESSOAS = "modelos-images/humano/"
PASTA_CACHORROS = "modelos-images/cachorro/"

images = []
rotulos = []

#PARTE 1 - Tratamento das imagens
print(f"📷 Carregando foto de PESSOAS...")
if os.path.exists(PASTA_PESSOAS):
    for foto in os.listdir(PASTA_PESSOAS):
        caminho = os.path.join(PASTA_PESSOAS, foto)
        if foto.startswith('.'):
            continue
        img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            img_pequena = cv2.resize(img, (TAMANHO, TAMANHO))
            vector = img_pequena.flatten() / 255.0

            images.append(vector)
            rotulos.append(0) #0 para pessoas e 1 para cachorros
    print(f"✅ Foram carregadas {len([r for r in rotulos if r ==0])} de humanos 👨🏻")
else:
    print("❌ pasta 'Pessoas' não encontrada!")


print(f"📷 Carregando foto de CACHORROS...")
if os.path.exists(PASTA_CACHORROS):
    for foto in os.listdir(PASTA_CACHORROS):
        caminho = os.path.join(PASTA_CACHORROS, foto)
        if foto.startswith('.'):
            continue
        img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            img_pequena = cv2.resize(img, (TAMANHO, TAMANHO))
            vector = img_pequena.flatten() / 255.0

            images.append(vector)
            rotulos.append(1) #0 para pessoas e 1 para cachorros
    print(f"✅ foram arregadas {len([r for r in rotulos if r ==1])} de cachorros 🐶")
else:
    print("❌ pasta 'Pessoas' não encontrada!")

#PARTE 2 - Dividir os dados para treino e teste do modelo
x = np.array(images)
y = np.array(rotulos)

print(" Dividindo os dados em treino (80%) e testes (20%)")
x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print(f"🏋️‍♀️ Foram coletados {len(x_treino)} imagens para treino")
print(f"🧪 Foram coletados {len(x_teste)} imagens para testes")

#PARTE 3 - Treinar o modelo
print("\n Criando o modelo de inteligência artificial...")
modelo = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=300,  # O 'adam' converge mais rápido, então 300 iterações são suficientes
    random_state=42,
    solver='adam', # Otimizador padrão e mais estável
    alpha=0.01,  # Aumentar a regularização para controlar os pesos
    verbose=False
)
"""
modelo = MLPClassifier(
    hidden_layer_sizes=(100),
    max_iter=500,
    random_state=42,
    solver='sgd', # Mudar para Stochastic Gradient Descent
    learning_rate='adaptive', # Taxa de aprendizado adaptativa
    learning_rate_init=0.001, # Taxa de aprendizado inicial
    momentum=0.9, # Ajuda a acelerar o SGD
    verbose=False # Mude para True se quiser ver o progresso do treino
)
"""

print("ARQUITETURA DO MODELO")
print(f"    🔴 Camada de entrada: {x.shape[1]} neurônios (um para cada pixel)")
print(f"    🟠 Camada oculta: 100 neurônios")
print(f"    🟢 Camada de saída: 1 ou 2 neurônios (0) pessoa, (1) cachorro")
print(f"    💻 Total de parâmetros: aprox1mdamente {(x.shape[1] * 100) + (100 * 2)} pesos")

print("Treinando o modelo... (pode demorar um pouco)")
modelo.fit(x_treino, y_treino)

#PARTE 4 - Verificar acurácia do modelo
print("Testando o modelo...")
acuracia = modelo.score(x_teste, y_teste)
print(f"📊 Acurácia: {acuracia:.2%}")

if acuracia > 0.8:
    print("☀️ EXCELENTE! O modelo está funcionando muito bem")
elif acuracia > 0.6:
    print("⛅️ BOM! O modelo está razovável")
else:
    print("⚠️ RUIM! O modelo pecisa de mais dados ou ajustes!")

# --- PARTE 5 - Salvar o modelo treinado ---
if acuracia > 0.70:
    print("\n💾 Salvando o modelo treinado em 'modelo_humano_cachorro.joblib'")
    joblib.dump(modelo, 'modelo_humano_cachorro.joblib')
    print("✅ Modelo salvo com sucesso!")