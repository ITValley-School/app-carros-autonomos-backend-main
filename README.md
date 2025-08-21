# Backend de Classificação Humano × Cachorro

Este repositório contém um conjunto de scripts em Python que demonstram como trabalhar com **visão computacional** (OpenCV) e **aprendizagem de máquina** para classificar imagens capturadas da webcam. O objetivo é treinar e testar um modelo capaz de identificar se a região central de uma imagem contém uma pessoa ou um cachorro.

## Visão geral do projeto

O repositório está dividido em pequenas etapas para facilitar o estudo e a evolução do projeto. Cada arquivo Python executa uma parte do pipeline:

| Arquivo                     | Descrição |
|----------------------------|-----------|
| `1-opencvCore.py`          | Demonstra o uso básico do OpenCV para capturar frames da webcam e exibi-los em tempo real. Encerra com a tecla `q`. |
| `2-tratarImagem.py`        | Carrega uma imagem da pasta `modelos-images/`, converte para tons de cinza, redimensiona e transforma em vetor. |
| `3-opencvDesenho.py`       | Mostra como desenhar um retângulo e adicionar texto sobre a imagem da webcam. |
| `4-treinarSalvarModelo.py` | Treina um MLP com imagens de `humano/` e `cachorro/`, salva o modelo em `modelo_humano_cachorro.joblib`. |
| `5-testarModelo.py`        | Usa a webcam para capturar imagens em tempo real, aplicando o modelo treinado para classificar entre humano e cachorro. |
| `modelos-images/`          | Pasta com imagens divididas em `humano/` e `cachorro/` usadas para treinar e testar o modelo. |
| `modelo_humano_cachorro.joblib` | Arquivo salvo com o modelo treinado. |
| `requirements.txt`         | Lista de dependências: `opencv-python`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`. |

## Pré-requisitos

- Python 3.8 ou superior
- Webcam funcional para testes em tempo real

## Instalação

```bash
git clone https://github.com/awazago/app-carros-autonomos-backend.git
cd app-carros-autonomos-backend

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```
*(Opcional)* Adicione mais imagens às pastas `modelos-images/humano` e `modelos-images/cachorro` para melhorar o treinamento.

## Como executar

### 1. Testar a webcam

```bash
python 1-opencvCore.py
```
Pressione q para encerrar.

### 2. Processar uma imagem
```
python 2-tratarImagem.py
```
Converte a imagem exemplo em vetor de entrada.

### 3. Desenhar e anotar frames da webcam
```
python 3-opencvDesenho.py
```
Mostra como adicionar elementos visuais ao vídeo.

### 4. Treinar e salvar o modelo
```
python 4-treinarSalvarModelo.py
```
O script executa as etapas:

- Carrega imagens de modelos-images/
- Divide os dados em treino/teste
- Treina um MLPClassifier
- Avalia o modelo e salva em modelo_humano_cachorro.joblib

### 5. Testar o modelo em tempo real
```
python 5-testarModelo.py
```
Captura vídeo da webcam, classifica o que há no centro da imagem e exibe o resultado com rótulo e cor.




# **🤖 Classificador em Tempo Real: Humanos vs. Cachorros**

Este projeto utiliza a biblioteca **OpenCV** para capturar vídeo da webcam e um modelo de Machine Learning (MLPClassifier) para classificar, em tempo real, se a imagem dentro de uma área de interesse é um **humano** ou um **cachorro**.

## **⚙️ Como Funciona**

O script é dividido em quatro etapas principais:

### **1\. Inicialização e Carregamento do Modelo**

O script começa importando as bibliotecas necessárias e carregando o modelo de classificação treinado.

```
import cv2  
import joblib  
import time

\# Carrega o modelo treinado do arquivo .joblib  
try:  
    modelo \=  joblib.load("modelo\_humano\_cachorro.joblib")  
    print("Modelo carregado com sucesso")  
except FileNotFoundError:  
    print("Não foi possível carregar o modelo.")  
    print("Certifique-se de que o arquivo 'modelo\_humano\_cachorro.joblib' esteja na mesma pasta.")  
    exit()
```
* **```joblib.load(...)```**: Esta função desserializa o arquivo ```.joblib```, recriando o objeto do modelo treinado na memória para que ele possa ser usado para fazer novas previsões.  
* ```try...except:``` Garante que o programa só continue se o modelo for carregado com sucesso.

### **2\. Acesso e Configuração da Webcam**

Após carregar o modelo, o script inicializa a webcam.
```
\# Tenta abrir a câmera padrão (índice 0\)  
camera \= cv2.VideoCapture(0)  
time.sleep(2) \# Pausa para a câmera estabilizar

if not camera.isOpened():  
    print("Não foi possível abrir a câmera")  
    exit()
```
* **```cv2.VideoCapture(0)```**: Cria um objeto para capturar vídeo da câmera padrão do sistema.  
* **```camera.isOpened()```**: Verifica se a conexão com a câmera foi bem-sucedida.

### **3\. Função de Previsão (```prever\_imagem```)**

Esta é a função central que processa uma imagem e retorna a previsão do modelo. Ela executa exatamente os mesmos passos de pré-processamento usados durante o treinamento.
```
def prever\_imagem(frame):  
    \# 1\. Converte para escala de cinza  
    gray \= cv2.cvtColor(frame, cv2.COLOR\_BGR2GRAY)  
    \# 2\. Redimensiona para 64x64 pixels  
    img\_pequena \= cv2.resize(gray, (64, 64))  
    \# 3\. Achata (Flatten) e Normaliza o vetor de pixels  
    vetor \= img\_pequena.flatten() / 255.0  
    \# 4\. Faz a previsão com o modelo  
    predicao \= modelo.predict(\[vetor\])\[0\]

    \# 5\. Retorna o rótulo e a cor para a exibição  
    if predicao \== 0:  
        return "HUMANO", (0, 255, 0\) \# Verde  
    else:  
        return "CACHORRO", (0, 165, 255\) \# Laranja
```
### **4\. Loop Principal de Execução**

Este é o coração da aplicação, onde a captura e a classificação acontecem em tempo real.
```
while True:  
    ret, frame \= camera.read()  
    \# ... (código do loop) ...
```
O loop faz o seguinte, continuamente:

1. **```camera.read()```**: Captura o quadro (frame) mais recente da câmera.  
2. **```cv2.flip()```**: Espelha o frame horizontalmente para que a imagem se pareça com um espelho.  
3. **Define a Área de Interesse (ROI)**: Um retângulo no centro da tela é definido como a área onde a detecção ocorrerá.  
4. **Otimização**: Para reduzir o uso da CPU, a previsão do modelo é executada apenas **a cada 5 frames**. Nos outros 4 frames, o resultado anterior é mantido.  
5. **Calcula a Confiança**: Usa m```modelo.predict\_proba()``` para obter a probabilidade (confiança) da previsão e a exibe na tela.  
6. **Exibe as Informações**: Desenha na tela o retângulo da ROI, o resultado da previsão ("HUMANO" ou "CACHORRO") e a confiança.  
7. **```cv2.imshow()```**: Mostra a janela com o frame final processado.  
8. **Condição de Saída**: O loop é interrompido quando o usuário pressiona a tecla **'q'**.

### Melhorias sugeridas

- Adicionar mais imagens de treino
- Testar diferentes hiperparâmetros no MLPClassifier
- Experimentar outros algoritmos de classificação
- Usar técnicas de pré-processamento (equalização, filtros)

Contribuições
Pull Requests e sugestões são bem-vindas!

Licença
Sem licença explícita. Consulte o autor para mais informações.

Autor
Desenvolvido por Adams Zago.

