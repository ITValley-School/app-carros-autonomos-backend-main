import cv2
import matplotlib.pyplot as plt

caminho_imagem = "modelos-images/rato.png"

imagemDeLinha = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
imagemResized = cv2.resize(imagemDeLinha, (64,64))

vetorizandoImagem = imagemResized.flatten()

plt.imshow(imagemResized, cmap='gray')
print("vetor de imagem rato:", vetorizandoImagem.shape)

plt.show()
