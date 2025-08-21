import cv2
import joblib

print(" TESTE DE MODELO: PESSOAS vs CACHORROS")

try:
    modelo =  joblib.load("modelo_humano_cachorro.joblib")
    print("Modelo carregado com sucesso")
except:
    print("Não foi possível carregar o modelo")
    print("Certifique-se que o arquivo com o modelo esteja no local informado")
    exit()

print("Tentando abrir a câmera...")
camera = cv2.VideoCapture(0)

print("Aguardando câmera inicializar...")
import time
time.sleep(2)

if not camera.isOpened():
    print("Não foi possível abrir a câmera")
    exit()

print("Câmera aberta com sucesso")
print("A resolução da câmera é: ", int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

print("Câmera inicializada!")


########################################################
def prever_imagem(frame):
    #converter para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #redimensionar imagem
    img_pequena = cv2.resize(gray, (64, 64))

    #Transformar em vetor
    vetor = img_pequena.flatten() / 255.0

    predicao = modelo.predict([vetor])[0]

    if predicao == 0:
        return "HUMANO", (0, 255, 0)
    else:
        return "CACHORRO", (0, 165, 255)
########################################################

frame_count = 0
resultado_atual = "Analisando..."
cor_atual = (255, 255, 255)
confianca_atual = 0

while True:
    ret, frame = camera.read()

    if not ret:
        print("Erro ao capturar frame")
        break

    frame = cv2.flip(frame, 1)

    altura, largura = frame.shape[:2]
    x1, y1 = largura//4, altura//4
    x2, y2 = 3*largura//4, 3*altura//4

    if frame_count % 5 == 0:
        area_teste = frame[y1:y2, x1:x2]

        resultado_atual, cor_atual = prever_imagem(area_teste)

        area_gray = cv2.cvtColor(area_teste, cv2.COLOR_BGR2GRAY)
        area_pequena = cv2.resize(area_gray, (64, 64))
        vetor = area_pequena.flatten() / 255.0
        confianca = modelo.predict_proba([vetor])[0]
        confianca_atual = max(confianca) * 100

    frame_count += 1

    cv2.rectangle(frame, (x1, y1), (x2, y2), cor_atual, 3)

    cv2.putText(frame, resultado_atual, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor_atual, 3)

    cv2.putText(frame, "DETECCAO EM TEMPO REAL OTIMIZADA ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, "Pressione Q para sair", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, "CPU análise a cada 5 frames", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.putText(frame, f"Confiança: {confianca_atual:.1f}%", (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_atual, 2)

    cv2.imshow("Teste: Humano vs Cachorro - TEMPO REAL", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Saindo...")
        break