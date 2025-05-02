
import cv2
from deepface import DeepFace
import numpy as np
import warnings

# Uyarıları gizle (DeepFace uyarılarını bastırır)
warnings.filterwarnings("ignore")

# Duygu - Emoji eşleştirmesi (İngilizce emotion isimlerine göre)
emotion_emojis = {
    'happy': '😄',
    'sad': '😢',
    'angry': '😠',
    'surprise': '😲',
    'fear': '😨',
    'neutral': '😐'
}

# Web kamerasını başlat
cap = cv2.VideoCapture(0)

print("Başlatılıyor... Kamera açılıyor. Çıkmak için 'q' tuşuna bas.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    try:
        # Duygu analizi (yüz tespiti dahil)
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Eğer birden fazla yüz varsa liste olarak gelir
        if isinstance(results, list):
            for result in results:
                region = result.get('region', {})
                if region:
                    x = region.get('x', 0)
                    y = region.get('y', 0)
                    w = region.get('w', 0)
                    h = region.get('h', 0)
                    emotion = result.get('dominant_emotion', 'neutral')
                    emoji = emotion_emojis.get(emotion, '')

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{emotion}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        else:
            region = results.get('region', {})
            if region:
                x = region.get('x', 0)
                y = region.get('y', 0)
                w = region.get('w', 0)
                h = region.get('h', 0)
                emotion = results.get('dominant_emotion', 'neutral')
                emoji = emotion_emojis.get(emotion, '')

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    except Exception as e:
        print("Hata:", e)
        # Devam et

    # Kullanıcıya çıkış komutu göster
    cv2.putText(frame, "Cikmak icin 'q' tusuna basin", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Görüntüyü göster
    cv2.imshow('Duygu Tanima', frame)

    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera kapatılır
cap.release()
cv2.destroyAllWindows()
