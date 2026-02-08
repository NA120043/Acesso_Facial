import cv2
import numpy as np
import pytesseract
from typing import Optional, Tuple, List, Dict
import time

class PlateReader:
    def __init__(self, camera_id: int = 0, show_preview: bool = True):
        """
        Inicializa o leitor de placas.
        
        Args:
            camera_id: ID da câmera (0 para webcam padrão)
            show_preview: Se True, mostra preview da câmera
        """
        self.camera_id = camera_id
        self.show_preview = show_preview
        self.cap = None
        self.is_running = False
        
        # Configurações do OCR (Tesseract)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
        # Para Linux/Mac: geralmente já está no PATH
        
        # Parâmetros ajustáveis
        self.plate_detection_params = {
            'min_area': 2000,
            'max_area': 20000,
            'min_aspect_ratio': 2.0,
            'max_aspect_ratio': 5.0
        }
        
        # Histórico de placas reconhecidas
        self.plate_history = []
        
    def initialize_camera(self) -> bool:
        """
        Inicializa a conexão com a câmera.
        
        Returns:
            bool: True se a câmera foi inicializada com sucesso
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"Erro: Não foi possível acessar a câmera {self.camera_id}")
                return False
                
            # Configurações da câmera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"Câmera {self.camera_id} inicializada com sucesso")
            return True
            
        except Exception as e:
            print(f"Erro ao inicializar câmera: {e}")
            return False
    
    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Pré-processa a imagem para melhor detecção.
        
        Args:
            frame: Frame da câmera
            
        Returns:
            Imagem pré-processada
        """
        # Converte para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Aplica blur para reduzir ruído
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detecta bordas
        edges = cv2.Canny(blurred, 50, 150)
        
        return edges
    
    def detect_plate_regions(self, frame: np.ndarray) -> List[Tuple]:
        """
        Detecta regiões que podem conter placas.
        
        Args:
            frame: Frame da câmera
            
        Returns:
            Lista de retângulos (x, y, w, h) das regiões detectadas
        """
        # Pré-processamento
        processed = self.preprocess_image(frame)
        
        # Encontra contornos
        contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        potential_plates = []
        
        for contour in contours:
            # Aproxima o contorno para um polígono
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Calcula retângulo delimitador
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # Filtra por área e aspecto de placa
            if (self.plate_detection_params['min_area'] < area < 
                self.plate_detection_params['max_area'] and
                self.plate_detection_params['min_aspect_ratio'] < aspect_ratio < 
                self.plate_detection_params['max_aspect_ratio']):
                
                potential_plates.append((x, y, w, h))
        
        return potential_plates
    
    def recognize_characters(self, plate_region: np.ndarray) -> str:
        """
        Reconhece caracteres na região da placa usando OCR.
        
        Args:
            plate_region: Região da imagem contendo a placa
            
        Returns:
            Texto reconhecido
        """
        try:
            # Converte para escala de cinza
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            
            # Aplica threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Configurações do Tesseract para placas
            config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            
            # Aplica OCR
            text = pytesseract.image_to_string(thresh, config=config)
            
            # Limpa o texto
            text = ''.join(c for c in text if c.isalnum())
            
            return text
            
        except Exception as e:
            print(f"Erro no OCR: {e}")
            return ""
    
    def draw_detections(self, frame: np.ndarray, plates: List[Tuple], 
                       recognized_text: str = "") -> np.ndarray:
        """
        Desenha retângulos e texto na imagem.
        
        Args:
            frame: Frame original
            plates: Lista de regiões detectadas
            recognized_text: Texto reconhecido
            
        Returns:
            Frame com anotações
        """
        frame_copy = frame.copy()
        
        # Desenha retângulos para cada placa detectada
        for (x, y, w, h) in plates:
            # Retângulo verde para placa
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Texto da placa
            if recognized_text:
                cv2.putText(frame_copy, recognized_text, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Adiciona texto "PLACA DETECTADA"
            cv2.putText(frame_copy, "PLACA DETECTADA", (x, y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame_copy
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Processa um único frame.
        
        Args:
            frame: Frame da câmera
            
        Returns:
            Dicionário com resultados
        """
        # Detecta regiões de placa
        plates = self.detect_plate_regions(frame)
        
        recognized_text = ""
        
        # Se encontrou alguma placa, tenta reconhecer caracteres
        if plates:
            # Pega a primeira placa detectada
            x, y, w, h = plates[0]
            
            # Extrai região da placa
            plate_region = frame[y:y+h, x:x+w]
            
            # Reconhece caracteres
            recognized_text = self.recognize_characters(plate_region)
            
            # Adiciona ao histórico se não estiver vazio
            if recognized_text:
                self.plate_history.append({
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'plate': recognized_text,
                    'position': (x, y, w, h)
                })
        
        return {
            'plates_detected': len(plates),
            'recognized_text': recognized_text,
            'plate_regions': plates
        }
    
    def run(self):
        """
        Executa o loop principal de leitura da câmera.
        """
        if not self.initialize_camera():
            return
        
        self.is_running = True
        print("Sistema iniciado. Pressione 'q' para sair, 's' para salvar frame")
        
        while self.is_running:
            # Captura frame
            ret, frame = self.cap.read()
            
            if not ret:
                print("Erro ao capturar frame")
                break
            
            # Processa o frame
            results = self.process_frame(frame)
            
            # Desenha detecções
            annotated_frame = self.draw_detections(
                frame, 
                results['plate_regions'],
                results['recognized_text']
            )
            
            # Mostra informações na tela
            info_text = f"Placas detectadas: {results['plates_detected']}"
            cv2.putText(annotated_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Mostra preview
            if self.show_preview:
                cv2.imshow('Sistema de Leitura de Placas', annotated_frame)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Salva frame atual
                filename = f"captura_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Frame salvo como {filename}")
            elif key == ord('h'):
                # Mostra histórico
                self.show_history()
        
        # Libera recursos
        self.stop()
    
    def show_history(self):
        """Mostra histórico de placas detectadas."""
        print("\n" + "="*50)
        print("HISTÓRICO DE PLACAS DETECTADAS")
        print("="*50)
        
        if not self.plate_history:
            print("Nenhuma placa detectada ainda.")
        else:
            for i, entry in enumerate(self.plate_history[-10:], 1):
                print(f"{i}. [{entry['timestamp']}] - Placa: {entry['plate']}")
        print("="*50)
    
    def update_parameters(self, **kwargs):
        """
        Atualiza parâmetros do sistema.
        
        Args:
            **kwargs: Parâmetros para atualizar
        """
        for key, value in kwargs.items():
            if key in self.plate_detection_params:
                self.plate_detection_params[key] = value
                print(f"Parâmetro atualizado: {key} = {value}")
    
    def stop(self):
        """Para o sistema e libera recursos."""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("Sistema encerrado.")


# Funções auxiliares para uso direto
def create_plate_reader(camera_id: int = 0, show_preview: bool = True) -> PlateReader:
    """
    Cria uma instância do leitor de placas.
    
    Args:
        camera_id: ID da câmera
        show_preview: Mostrar preview
        
    Returns:
        Instância do PlateReader
    """
    return PlateReader(camera_id, show_preview)


def quick_detect(image_path: str) -> Dict:
    """
    Detecta placa em uma imagem estática.
    
    Args:
        image_path: Caminho da imagem
        
    Returns:
        Resultados da detecção
    """
    reader = PlateReader(show_preview=False)
    
    # Carrega imagem
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Imagem não encontrada"}
    
    # Processa imagem
    results = reader.process_frame(image)
    
    # Desenha resultados
    annotated = reader.draw_detections(
        image, 
        results['plate_regions'],
        results['recognized_text']
    )
    
    # Salva resultado
    output_path = "resultado_detecao.jpg"
    cv2.imwrite(output_path, annotated)
    
    results['output_image'] = output_path
    return results