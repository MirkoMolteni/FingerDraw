import cv2
import mediapipe as mp
import numpy as np

# Inizializzazione di MediaPipe Hands e delle utilità di disegno
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inizializzazione della videocamera
cap = cv2.VideoCapture(0)

# Variabili per il disegno
drawing = False
prev_x, prev_y = None, None
color = (0, 255, 0)  # Colore iniziale verde
canvas = None  # Tela su cui disegnare

# Riquadri dei colori
color_buttons = {
    'red': {'rect': (20, 20, 60, 60), 'color': (0, 0, 255)},
    'blue': {'rect': (80, 20, 60, 60), 'color': (255, 0, 0)},
    'yellow': {'rect': (140, 20, 60, 60), 'color': (0, 255, 255)}
}

# Funzione per disegnare i riquadri dei colori
def draw_color_buttons(image):
    for btn in color_buttons.values():
        x, y, w, h = btn['rect']
        cv2.rectangle(image, (x, y), (x + w, y + h), btn['color'], -1)

# Funzione per controllare se il puntatore è su un riquadro
def check_color_selection(index_x, index_y):
    global color
    for btn_name, btn in color_buttons.items():
        x, y, w, h = btn['rect']
        if x <= index_x <= x + w and y <= index_y <= y + h:
            color = btn['color']

# Uso di Hands con rilevamento e tracciamento delle mani
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Errore nel leggere il video")
            break

        # Ingrandisci la finestra
        frame = cv2.resize(frame, (1280, 720))

        # Inverti l'immagine orizzontalmente per evitare l'effetto specchio
        frame = cv2.flip(frame, 1)

        # Crea la tela (se non esiste ancora)
        if canvas is None:
            h, w, _ = frame.shape
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Converti l'immagine in RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Rilevamento della mano
        results = hands.process(image)

        # Converti l'immagine di nuovo in BGR per OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Disegna i riquadri dei colori
        draw_color_buttons(image)

        # Se vengono rilevate mani, traccia i landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Landmark del pollice (4), indice (8), medio (12), anulare (16), e mignolo (20)
                thumb_tip = hand_landmarks.landmark[4]
                thumb_base = hand_landmarks.landmark[2]
                index_tip = hand_landmarks.landmark[8]
                index_base = hand_landmarks.landmark[6]
                middle_tip = hand_landmarks.landmark[12]
                middle_base = hand_landmarks.landmark[10]
                ring_tip = hand_landmarks.landmark[16]
                ring_base = hand_landmarks.landmark[14]
                pinky_tip = hand_landmarks.landmark[20]
                pinky_base = hand_landmarks.landmark[18]

                # Coordinate della punta dell'indice
                h, w, _ = image.shape
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

                # Condizioni per rilevare la "L" con pollice e indice
                if thumb_tip.x < thumb_base.x and index_tip.y < index_base.y:
                    drawing = True
                else:
                    drawing = False
                    prev_x, prev_y = None, None

                # Cambia colore se l'indice passa sopra un riquadro di colore
                check_color_selection(index_x, index_y)

                # Se tutte le 5 dita sono sollevate, cancella il disegno
                if (index_tip.y < index_base.y and
                        middle_tip.y < middle_base.y and
                        ring_tip.y < ring_base.y and
                        pinky_tip.y < pinky_base.y and
                        thumb_tip.x < thumb_base.x):
                    canvas = np.zeros((h, w, 3), dtype=np.uint8)

                # Se il disegno è attivo, traccia la linea dalla punta dell'indice
                if drawing:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), color, 5)

                    # Aggiorna le coordinate precedenti
                    prev_x, prev_y = index_x, index_y

        # Combina il frame corrente con il disegno
        image = cv2.addWeighted(image, 0.5, canvas, 0.5, 0)

        # Mostra l'immagine combinata
        cv2.imshow('Hand Tracking with Drawing', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
