import cv2
from deepface import DeepFace
import time
import os

# ===================== CONFIGURAÇÕES =====================
DATABASE_PATH = r"C:\Users\natan.alves\Desktop\foto"  # Caminho para sua base de dados
MODEL_NAME = "Facenet"  # Modelo de reconhecimento (pode usar "VGG-Face", "OpenFace", "Dlib", etc.)
DETECTOR_BACKEND = "opencv"  # Detector de rosto (pode usar "retinaface" para maior precisão, mas é mais lento)
INTERVALO_ANALISE = 3  # Analisa um frame a cada X segundos (evita sobrecarga)

# ===================== VERIFICAÇÃO INICIAL =====================
print(f"[INFO] Caminho do banco de dados: {DATABASE_PATH}")

# Verifica se a pasta existe
if not os.path.exists(DATABASE_PATH):
    print(f"[ERRO] A pasta '{DATABASE_PATH}' não foi encontrada.")
    print("[ERRO] Crie a pasta e organize as imagens em subpastas (veja o comentário acima).")
    exit()

# Verifica se a pasta do banco de dados não está vazia
if not os.listdir(DATABASE_PATH):
    print(f"[ERRO] A pasta '{DATABASE_PATH}' está vazia.")
    print("[ERRO] Adicione subpastas com nomes das pessoas e suas fotos.")
    exit()

# ===================== INICIALIZAÇÃO DA WEBCAM =====================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERRO] Não foi possível acessar a webcam.")
    exit()

print("[INFO] Webcam inicializada. Pressione 'q' para sair.")
ultimo_tempo = 0  # Controla o tempo da última análise

# ===================== LOOP PRINCIPAL =====================
while True:
    # 1. Captura o frame da webcam
    ret, frame = cap.read()
    if not ret:
        print("[ERRO] Falha ao capturar frame.")
        break

    tempo_atual = time.time()

    # 2. Processa o frame apenas após o intervalo definido
    if tempo_atual - ultimo_tempo > INTERVALO_ANALISE:
        # Converte o frame de BGR (OpenCV) para RGB (padrão DeepFace)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # 3. Tenta encontrar uma correspondência no banco de dados
            # 'enforce_detection=False' impede erro se não achar rosto
            resultados = DeepFace.find(
                img_path=frame_rgb,
                db_path=DATABASE_PATH,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,  # CRUCIAL: Não gera erro se nenhum rosto for detectado
                silent=True
            )

            # 4. Processa os resultados
            texto_tela = "Nenhum rosto detectado"
            cor = (0, 120, 255)  # Laranja

            # Verifica se a lista 'resultados' existe e não está vazia
            if resultados and isinstance(resultados, list) and len(resultados) > 0:
                df = resultados[0]  # Pega o DataFrame de resultados

                # Verifica se há alguma correspondência no DataFrame
                if len(df) > 0:
                    # Pega a correspondência com a menor distância (mais similar)
                    melhor_match = df.iloc[0]
                    distancia = melhor_match['distance']
                    caminho_identidade = melhor_match['identity']

                    # Extrai o nome da pessoa a partir do caminho da pasta
                    nome_pasta = os.path.basename(os.path.dirname(caminho_identidade))

                    # Define um limite (threshold) para considerar uma correspondência válida
                    limite_aceitacao = 0.5  # Ajuste conforme necessário (menor = mais rigoroso)

                    if distancia < limite_aceitacao:
                        texto_tela = f"Identificado: {nome_pasta}"
                        cor = (0, 255, 0)  # Verde
                    else:
                        texto_tela = f"Desconhecido (distancia: {distancia:.2f})"
                        cor = (0, 0, 255)  # Vermelho
                else:
                    # DataFrame existe mas está vazio = nenhuma correspondência encontrada
                    texto_tela = "Nenhuma correspondencia encontrada"
                    cor = (0, 0, 255)  # Vermelho

        except Exception as e:
            texto_tela = f"Erro na analise"
            cor = (0, 0, 255)  # Vermelho
            print(f"[EXCECAO] Ocorreu um erro: {e}")

        # 5. Atualiza o tempo da última análise
        ultimo_tempo = tempo_atual

    # 6. Desenha o texto de status no frame (sempre visível)
    cv2.putText(frame, texto_tela, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)

    # 7. Exibe o frame na janela
    cv2.imshow('Reconhecimento Facial - DeepFace + OpenCV', frame)

    # 8. Condição de saída: tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===================== FINALIZAÇÃO =====================
cap.release()
cv2.destroyAllWindows()
print("[INFO] Programa finalizado.")