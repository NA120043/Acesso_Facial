#import os
# from FaceRec import FaceRec
#from FaceRec2 import FaceRec
from PlateRead import WebcamOCR


# if __name__ == "__main__":
#     pasta_atual = os.getcwd()
#     print(f"Seu código está rodando na pasta: {pasta_atual}")
#     cam = FaceRec(
#         camera_index=0,  # 0 para webcam padrão
#         database_path=f"{pasta_atual}\\foto",
#         model="VGG-Face",
#         detect_backend="opencv",
#         interval_analyse = 2
#     )

#     cam.CameraView()


# if __name__ == "__main__":
#    pasta_atual = os.getcwd()
    
#     # configuração
#    recognizer = facerec(
#        camera_index=0,                    # webcam padrão
#        database_path=f"{pasta_atual}\\foto",        # pasta com fotos de referência
#        detection_threshold=0.8,           # threshold para detecção (0.0-1.0)
#        recognition_threshold=0.87,         # ajuste: valores mais baixos = mais sensível
#        interval_analyse=2.0               # intervalo entre reconhecimentos
#     )
    
#     # iniciar
#    recognizer.cameraview()



# Exemplo da leitura de placas
if __name__ == "__main__":
    # Configuração
    ocr_system = WebcamOCR(
        idiomas=['pt', 'en'],        # Português e Inglês
        camera_index=0,                # Câmera padrão
        frame_skip=2,                   # Processa a cada 2 frames
        confidence_threshold=0.5,       # Confiança mínima de 50%
        database_path="ocr_database",    # Pasta para salvar capturas
        save_interval=10                  # Salva automaticamente a cada 10 segundos
    )
    
    # Iniciar sistema
    ocr_system.CameraView()
