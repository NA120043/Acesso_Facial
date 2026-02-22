import cv2
import easyocr
import numpy as np
from collections import deque
import time
import os

class WebcamOCR:
    def __init__(self, idiomas, camera_index, frame_skip, confidence_threshold, 
                 database_path, save_interval):
        """
        Sistema de OCR em tempo real usando EasyOCR
        
        Args:
            idiomas: Lista de idiomas para o OCR (ex: ['pt', 'en'])
            camera_index: Índice da câmera (padrão: 0)
            frame_skip: Processa a cada N frames para melhor performance (padrão: 2)
            confidence_threshold: Limiar de confiança para resultados (padrão: 0.5)
            database_path: Pasta para salvar capturas (padrão: "ocr_database")
            save_interval: Intervalo para salvar resultados automaticamente (padrão: 5)
        """
        
        # 1. Configurações do sistema
        self.idiomas = idiomas
        self.camera_index = camera_index
        self.frame_skip = frame_skip
        self.confidence_threshold = confidence_threshold
        self.database_path = database_path
        self.save_interval = save_interval
        
        # 2. Estado do sistema
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.last_save_time = 0
        self.last_results = []  # Últimos resultados processados
        
        # 3. Buffer para histórico de resultados
        self.results_history = deque(maxlen=5)  # Guarda últimos 5 resultados
        
        # 4. Inicializar EasyOCR
        self.initialize_ocr()
        
        # 5. Criar pasta para banco de dados se não existir
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)
            print(f"Pasta '{self.database_path}' criada para salvar capturas.")
        
        print(f"Sistema OCR iniciado com idiomas: {idiomas}")
    
    def initialize_ocr(self):
        """Inicializa o leitor EasyOCR com fallback em caso de erro"""
        try:
            self.reader = easyocr.Reader(self.idiomas, gpu=False)
            print("EasyOCR inicializado com sucesso!")
            self.ocr_available = True
        except Exception as e:
            print(f"Erro ao inicializar EasyOCR: {e}")
            print("Sistema continuará sem OCR...")
            self.ocr_available = False
    
    def processar_frame(self, frame):
        """
        Executa OCR no frame
        Retorna lista de resultados filtrados
        """
        if not self.ocr_available:
            return []
        
        try:
            # Executar OCR
            resultados = self.reader.readtext(frame)
            
            # Filtrar por confiança
            resultados_filtrados = [r for r in resultados if r[2] > self.confidence_threshold]
            
            # Adicionar ao histórico
            if resultados_filtrados:
                self.results_history.append(resultados_filtrados)
                self.last_results = resultados_filtrados
            
            return resultados_filtrados
            
        except Exception as e:
            print(f"Erro no processamento OCR: {e}")
            return []
    
    def desenhar_resultados(self, frame, resultados):
        """
        Desenha bounding boxes e textos no frame
        """
        for (bbox, texto, confianca) in resultados:
            # Converte bbox para inteiros
            pts = np.array(bbox, dtype=np.int32)
            
            # Desenha bounding box verde
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            
            # Coordenadas para texto
            x = int(pts[0][0])
            y = int(pts[0][1]) - 10
            
            # Fundo preto para o texto
            text_size = cv2.getTextSize(f"{texto} ({confianca:.2f})", 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x-5, y-text_size[1]-5), 
                         (x+text_size[0]+5, y+5), (0, 0, 0), -1)
            
            # Texto em vermelho
            cv2.putText(frame, f"{texto} ({confianca:.2f})", 
                       (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 0, 255), 2)
        
        return frame
    
    def salvar_captura(self, frame, resultados):
        """
        Salva frame atual com timestamp e textos detectados
        """
        timestamp = int(time.time())
        
        # Criar nome do arquivo com os textos detectados
        if resultados:
            textos = "_".join([r[1][:10] for r in resultados[:3]])  # Máx 3 textos
            filename = f"{self.database_path}/ocr_{timestamp}_{textos}.jpg"
        else:
            filename = f"{self.database_path}/ocr_{timestamp}_sem_texto.jpg"
        
        # Salvar imagem
        cv2.imwrite(filename, frame)
        print(f"Captura salva: {filename}")
        
        # Salvar metadados em arquivo texto
        txt_filename = filename.replace('.jpg', '.txt')
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Textos detectados: {len(resultados)}\n\n")
            for i, (bbox, texto, confianca) in enumerate(resultados, 1):
                f.write(f"Texto {i}: {texto}\n")
                f.write(f"Confiança {i}: {confianca:.3f}\n")
                f.write(f"Posição {i}: {bbox}\n\n")
        
        print(f"Metadados salvos: {txt_filename}")
    
    def OCRProcessor(self, frame):
        """
        Função principal de processamento OCR (similar à FaceRecognition)
        """
        # 1. Processar OCR (apenas a cada N frames)
        self.frame_count += 1
        if self.frame_count % self.frame_skip == 0:
            resultados = self.processar_frame(frame)
        else:
            # Usar último resultado válido do histórico
            resultados = self.results_history[-1] if self.results_history else []
        
        # 2. Desenhar resultados
        frame_anotado = self.desenhar_resultados(frame.copy(), resultados)
        
        # 3. Salvar automaticamente em intervalos
        current_time = time.time()
        if current_time - self.last_save_time > self.save_interval:
            self.salvar_captura(frame, resultados)
            self.last_save_time = current_time
        
        return frame_anotado
    
    def CameraView(self):
        """
        Método principal para captura e exibição da câmera
        """
        # Abrir câmera
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print("ERRO: Não foi possível abrir a câmera")
            return
        
        # Configurar propriedades da câmera para melhor performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.running = True
        self.last_save_time = time.time()
        
        print("\n" + "="*50)
        print("Webcam OCR iniciado")
        print("="*50)
        print("Controles:")
        print("  'q' - Sair")
        print("  's' - Salvar frame atual")
        print("  'c' - Limpar histórico")
        print(f"  Intervalo auto-save: {self.save_interval}s")
        print("="*50)
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("ERRO: Não foi possível ler o frame da câmera")
                    break
                
                # Processar frame
                frame_anotado = self.OCRProcessor(frame)
                
                # Adicionar informações na tela
                height, width = frame.shape[:2]
                
                # Status do OCR
                status = "OCR: Ativo" if self.ocr_available else "OCR: Indisponível"
                color_status = (0, 255, 0) if self.ocr_available else (0, 0, 255)
                
                # Informações superiores
                cv2.putText(frame_anotado, status, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_status, 2)
                
                cv2.putText(frame_anotado, f"Idiomas: {self.idiomas}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Informações inferiores
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                textos_detectados = len(self.last_results)
                
                info_text = f"FPS: {fps:.1f} | Frame skip: {self.frame_skip} | Textos: {textos_detectados}"
                cv2.putText(frame_anotado, info_text, (10, height-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                
                # Mostrar frame
                cv2.imshow('Camera_View - Sistema OCR', frame_anotado)
                
                # Processar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or cv2.getWindowProperty('Camera_View - Sistema OCR', cv2.WND_PROP_VISIBLE) < 1:
                    self.running = False
                    break
                    
                elif key == ord('s'):
                    self.salvar_captura(frame, self.last_results)
                    
                elif key == ord('c'):
                    self.results_history.clear()
                    self.last_results = []
                    print("🧹 Histórico limpo!")
                
        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário")
        except Exception as e:
            print(f"Erro inesperado: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Libera recursos do sistema"""
        print("\nLiberando recursos...")
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.running = False
        print("Recursos liberados com sucesso!")

