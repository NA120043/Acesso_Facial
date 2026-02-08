import cv2
import numpy as np
import os
import time
import pickle
import urllib.request
import sys

class FaceRec():
    def __init__(self, camera_index, database_path, detection_threshold=0.8, recognition_threshold=0.6, interval_analyse=2.0):
        """
        Sistema de Reconhecimento Facial usando YuNet + Embeddings do OpenCV
        """
        
        # 1. Inicializar apenas o detector YuNet (sem SFace)
        self.detection_model = "face_detection_yunet_2023mar.onnx"
        
        # Verificar se o modelo YuNet existe
        if not os.path.exists(self.detection_model):
            print(f"ERRO: Modelo {self.detection_model} não encontrado!")
            print("Baixando automaticamente de:")
            print("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
            
            # Tentar baixar o modelo automaticamente
            if self.download_yunet_model():
                print("✅ Download concluído com sucesso!")
            else:
                print("❌ Falha no download do modelo!")
                print("Usando Haar Cascade como fallback...")
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.use_yunet = False
        
        # Tentar inicializar o YuNet (se o download foi bem-sucedido ou o arquivo já existia)
        if not hasattr(self, 'use_yunet'):  # Só tenta se ainda não definimos o fallback
            try:
                self.face_detector = cv2.FaceDetectorYN.create(
                    self.detection_model,
                    "",
                    (320, 320),
                    score_threshold=detection_threshold,
                    nms_threshold=0.3,
                    top_k=5000
                )
                print("✅ YuNet inicializado com sucesso!")
                self.use_yunet = True
            except Exception as e:
                print(f"❌ Erro ao inicializar YuNet: {e}")
                print("Usando Haar Cascade como fallback...")
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.use_yunet = False
        
        # 2. Configurações do sistema (iguais ao seu código)
        self.camera_index = camera_index
        self.database_path = database_path
        self.recognition_threshold = recognition_threshold
        self.interval_analyse = interval_analyse
        
        # 3. Estado do sistema
        self.tempo_atual = 0
        self.ultimo_tempo = 0
        self.cap = None
        self.running = False
        self.last_recognition_results = []
        
        # 4. Banco de dados simplificado (armazenamos imagens dos rostos, não embeddings)
        self.known_faces = []      # Lista de imagens dos rostos
        self.known_names = []      # Lista de nomes correspondentes
        
        # Carregar banco de dados
        self.load_database()
        
        print(f"✅ Sistema iniciado com {len(self.known_names)} rostos conhecidos.")
    
    def download_yunet_model(self):
        """
        Função para baixar automaticamente o modelo YuNet
        Retorna True se o download foi bem-sucedido, False caso contrário
        """
        model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        
        try:
            # Função para mostrar o progresso do download
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100.0, downloaded * 100.0 / total_size) if total_size > 0 else 0
                sys.stdout.write(f"\rProgresso: {percent:.1f}% ({downloaded}/{total_size} bytes)")
                sys.stdout.flush()
            
            print("Iniciando download do modelo YuNet...")
            urllib.request.urlretrieve(model_url, self.detection_model, show_progress)
            print("\n✅ Download concluído!")
            return True
            
        except Exception as e:
            print(f"\n❌ Erro durante o download: {e}")
            return False

    def load_database(self):
        """Carrega imagens de referência do banco de dados"""
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)
            print(f"📁 Pasta '{self.database_path}' criada. Adicione fotos dos rostos conhecidos.")
            return
        
        print("📂 Carregando banco de dados...")
        
        for filename in os.listdir(self.database_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(self.database_path, filename)
                
                try:
                    # Carregar imagem
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"  ❌ Não foi possível ler: {filename}")
                        continue
                    
                    # Detectar rosto na imagem
                    if self.use_yunet:
                        height, width = img.shape[:2]
                        self.face_detector.setInputSize((width, height))
                        _, faces = self.face_detector.detect(img)
                        
                        if faces is None or len(faces) == 0:
                            print(f"  ⚠️  Nenhum rosto detectado em: {filename}")
                            continue
                        
                        # Extrair região do primeiro rosto
                        x, y, w, h = list(map(int, faces[0][:4]))
                    else:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                        
                        if len(faces) == 0:
                            print(f"  ⚠️  Nenhum rosto detectado em: {filename}")
                            continue
                        
                        x, y, w, h = faces[0]
                    
                    # Recortar e redimensionar rosto para tamanho padrão
                    face_region = img[y:y+h, x:x+w]
                    face_standard = cv2.resize(face_region, (100, 100))
                    
                    # Converter para escala de cinza e normalizar
                    face_gray = cv2.cvtColor(face_standard, cv2.COLOR_BGR2GRAY)
                    face_normalized = face_gray / 255.0
                    
                    # Armazenar no banco de dados
                    self.known_faces.append(face_normalized)
                    self.known_names.append(os.path.splitext(filename)[0])
                    
                    print(f"  ✅ Carregado: {os.path.splitext(filename)[0]}")
                    
                except Exception as e:
                    print(f"  ❌ Erro processando {filename}: {e}")

    def compare_faces_simple(self, face_img):
        """
        Compara um rosto detectado com os rostos conhecidos usando um método simples:
        Calcula a diferença absoluta média entre os pixels.
        """
        if len(self.known_faces) == 0:
            return "Banco vazio", 0.0
        
        # Preparar a imagem do rosto para comparação
        face_resized = cv2.resize(face_img, (100, 100))
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        face_normalized = face_gray / 255.0
        
        best_match_index = -1
        best_similarity = 0.0
        
        # Comparar com cada rosto conhecido
        for i, known_face in enumerate(self.known_faces):
            # Calcular similaridade (1 - diferença média)
            diff = np.abs(face_normalized - known_face)
            similarity = 1.0 - np.mean(diff)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_index = i
        
        # Verificar se a similaridade atinge o threshold
        if best_similarity > self.recognition_threshold:
            return self.known_names[best_match_index], best_similarity
        else:
            return "Desconhecido", best_similarity

    def FaceRecognition(self, frame):
        """Função principal de reconhecimento (igual à sua interface)"""
        self.tempo_atual = time.time()
        
        # 1. Detectar rostos
        faces_detected = self.FaceFinder(frame)
        
        # 2. Desenhar caixas verdes finas
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(frame, 'Rosto', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 3. Reconhecimento apenas no intervalo definido
        if self.tempo_atual - self.ultimo_tempo > self.interval_analyse:
            self.last_recognition_results = []
            self.ultimo_tempo = self.tempo_atual
            
            if len(faces_detected) == 0:
                return frame
            
            # 4. Para cada rosto detectado
            for (x, y, w, h) in faces_detected:
                try:
                    # Extrair região do rosto
                    face_region = frame[y:y+h, x:x+w]
                    
                    # Realizar reconhecimento simples
                    identity, similarity = self.compare_faces_simple(face_region)
                    
                    # Definir cor baseada no resultado
                    if identity == "Desconhecido":
                        color = (0, 0, 255)  # Vermelho
                        display_name = f"Desconhecido ({similarity:.2f})"
                    elif identity == "Banco vazio":
                        color = (255, 255, 0)  # Ciano
                        display_name = "Sem banco"
                    else:
                        color = (255, 0, 0)  # Azul
                        display_name = f"{identity} ({similarity:.2f})"
                    
                    # Armazenar resultado
                    self.last_recognition_results.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'identity': display_name,
                        'color': color
                    })
                    
                except Exception as e:
                    print(f"Erro no reconhecimento: {e}")
                    self.last_recognition_results.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'identity': 'Erro',
                        'color': (0, 255, 255)  # Amarelo
                    })
        
        # 5. Desenhar resultados
        for result in self.last_recognition_results:
            x, y, w, h = result['x'], result['y'], result['w'], result['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), result['color'], 2)
            cv2.putText(frame, result['identity'], (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, result['color'], 2)
        
        return frame

    def FaceFinder(self, frame):
        """Detecta rostos usando YuNet ou Haar Cascade"""
        if self.use_yunet:
            # Usar YuNet
            height, width = frame.shape[:2]
            self.face_detector.setInputSize((width, height))
            
            _, faces = self.face_detector.detect(frame)
            
            if faces is None:
                return []
            
            return [list(map(int, face[:4])) for face in faces]
        else:
            # Usar Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return list(faces)

    def CameraView(self):
        """Método principal (idêntico ao seu original)"""
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print("ERRO: Não foi possível abrir a câmera")
            return
        
        self.running = True
        print("Pressione 'q' para sair...")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("ERRO: Não foi possível ler o frame da câmera")
                    break
                
                frame = self.FaceRecognition(frame)

                fps = self.cap.get(cv2.CAP_PROP_FPS)
                cv2.putText(frame, f'FPS: {int(fps)} | Intervalo: {self.interval_analyse}s', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Adicionar informações sobre o detector
                detector_type = "YuNet" if self.use_yunet else "Haar Cascade"
                cv2.putText(frame, f'Detector: {detector_type} | Rostos: {len(self.known_names)}', (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Camera_View - Sistema Simples', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty('Camera_View - Sistema Simples', cv2.WND_PROP_VISIBLE) < 1:
                    self.running = False
                    break
                
        except KeyboardInterrupt:
            print("Interrompido pelo usuário")
        except Exception as e:
            print(f"Erro inesperado: {e}")
        finally:
            print("Liberando recursos...")
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            self.running = False