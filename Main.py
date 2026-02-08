import os
# from FaceRec import FaceRec
from FaceRec2 import FaceRec
import PlateRead


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


if __name__ == "__main__":
   pasta_atual = os.getcwd()
    
    # Configuração
   recognizer = FaceRec(
       camera_index=0,                    # Webcam padrão
       database_path=r"C:\Users\Loure\Downloads\foto",        # Pasta com fotos de referência
       detection_threshold=0.8,           # Threshold para detecção (0.0-1.0)
       recognition_threshold=0.87,         # AJUSTE: Valores mais baixos = mais sensível
       interval_analyse=2.0               # Intervalo entre reconhecimentos
    )
    
    # Iniciar
   recognizer.CameraView()



# Versão ultra-simples - apenas inicia e executa
# if __name__ == "__main__":
#     sistema = PlateRead.PlateReader()  # Usa configurações padrão
#     sistema.run()