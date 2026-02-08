import cv2
from deepface import DeepFace
import time
import os


class FaceRec():
    def __init__(self, camera_index, database_path, model, detect_backend, interval_analyse):
        # Carrega o classificador de rostos pré-treinado
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.camera_index = camera_index
        self.database_path = database_path
        self.model = model
        self.detect_backend = detect_backend
        self.interval_analyse = interval_analyse

        self.tempo_atual = 0
        self.ultimo_tempo = 0
        self.faces = None
        self.cap = None
        self.running = False
        self.last_recognition_results = []  # Armazena os últimos resultados de reconhecimento

    def FaceRecognition(self, frame):
        self.tempo_atual = time.time()
        
        # Detecta rostos no frame
        faces_detected = self.FaceFinder(frame)  # Não desenha boxes verdes aqui
        
        # Desenha boxes verdes para todos os rostos detectados
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(frame, 'Rosto', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Verifica se é hora de fazer uma nova análise
        if self.tempo_atual - self.ultimo_tempo > self.interval_analyse:
            self.last_recognition_results = []  # Limpa resultados antigos
            self.ultimo_tempo = self.tempo_atual
            
            # Se não há rostos detectados, não faz nada
            if len(faces_detected) == 0:
                return frame
                
            # Para cada rosto detectado, tenta reconhecer
            for (x, y, w, h) in faces_detected:
                # Extrai a região do rosto
                face_region = frame[y:y+h, x:x+w]
                
                # Converte apenas a região do rosto para RGB (DeepFace usa RGB)
                face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                
                try:
                    # Usa DeepFace para encontrar a pessoa mais similar na base de dados
                    dfs = DeepFace.find(
                        img_path=face_region_rgb,
                        db_path=self.database_path,
                        model_name=self.model,
                        detector_backend=self.detect_backend,
                        enforce_detection=False,
                        silent=True
                    )
                    
                    result = None
                    if len(dfs) > 0 and len(dfs[0]) > 0:
                        # Pega o resultado mais similar (primeiro da lista)
                        result = dfs[0].iloc[0]
                        
                        # Extrai o nome do arquivo (sem extensão)
                        identity = os.path.splitext(os.path.basename(result['identity']))[0]
                        distance = result['distance']
                        
                        # Se a distância for muito alta, considera desconhecido
                        if distance < 0.6:  # Ajuste este threshold conforme necessário
                            color = (255, 0, 0)  # Azul para conhecido
                        else:
                            identity = 'Desconhecido'
                            color = (0, 0, 255)  # Vermelho para desconhecido
                    else:
                        identity = 'Desconhecido'
                        color = (0, 0, 255)
                        
                    # Armazena o resultado para desenhar continuamente
                    self.last_recognition_results.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'identity': identity,
                        'color': color
                    })
                        
                except Exception as e:
                    print(f"Erro no reconhecimento: {e}")
                    # Armazena resultado de erro
                    self.last_recognition_results.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'identity': 'Erro',
                        'color': (0, 255, 255)  # Amarelo para erro
                    })
        
        # Desenha os resultados do reconhecimento (mantém desenhados até próxima análise)
        for result in self.last_recognition_results:
            x, y, w, h = result['x'], result['y'], result['w'], result['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), result['color'], 2)
            cv2.putText(frame, result['identity'], (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, result['color'], 2)
        
        return frame
    
    def FaceFinder(self, frame):
        # Converte para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta rostos
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Retorna apenas as coordenadas dos rostos
        return list(faces)
    
    def CameraView(self):
        # Inicia a camera
        self.cap = cv2.VideoCapture(self.camera_index)

        # Identifica se não ouve erro ao iniciar a camera
        if not self.cap.isOpened():
            print("ERRO: Não foi possível abrir a câmera")
            return
        
        self.running = True
        
        try:
            while self.running:
                # Obtem o frame e verifica se esta em bom estado
                ret, frame = self.cap.read()
                if not ret:
                    print("ERRO: Não foi possível ler o frame da câmera")
                    break
                
                # Aplica reconhecimento facial
                frame = self.FaceRecognition(frame)

                # Mostra o FPS
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                cv2.putText(frame, f'FPS: {int(fps)} | Intervalo: {self.interval_analyse}s', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Abre uma janela mostrando os frames
                cv2.imshow('Camera_View', frame)

                # Aguarda 1 milissegundo e verifica se a tecla 'q' foi pressionada ou janela fechada
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty('Camera_View', cv2.WND_PROP_VISIBLE) < 1:
                    self.running = False
                    break
                
        except KeyboardInterrupt:
            print("Interrompido pelo usuário")
        except Exception as e:
            print(f"Erro inesperado: {e}")
        finally:
            # Libera os recursos
            print("Liberando recursos...")
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            self.running = False