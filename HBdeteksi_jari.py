import cv2
import mediapipe as mp
import math


def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Webcam
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Ujung jari
tip_ids = [4, 8, 12, 16, 20]


# Main Loop
while True:

    success, img = cap.read()

    if not success:
        print("Webcam tidak terdeteksi!")
        break

    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    total_fingers = 0
    gesture = ""

    if results.multi_hand_landmarks and results.multi_handedness:

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            hand_label = handedness.classification[0].label

            h, w, c = img.shape

            lm_list = []

        
            # Ambil Landmark
            for idx, lm in enumerate(hand_landmarks.landmark):

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                lm_list.append((cx, cy))

            fingers = []

            
            # Deteksi Ibu Jari
            if hand_label == "Right":

                if lm_list[4][0] < lm_list[3][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            else:

                if lm_list[4][0] > lm_list[3][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

           
            # Jari Lainnya
            for i in range(1, 5):

                if lm_list[tip_ids[i]][1] < lm_list[tip_ids[i] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers = fingers.count(1)

            # Deteksi Gesture OK 👌
            thumb_tip = lm_list[4]
            index_tip = lm_list[8]

            ok_distance = distance(
                thumb_tip,
                index_tip
            )

            if ok_distance < 30:
                gesture = "OK"
            else:
                gesture = ""

            # Gambar Landmark
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(
                    color=(0, 255, 255),
                    thickness=3,
                    circle_radius=4
                ),
                mp_draw.DrawingSpec(
                    color=(255, 255, 255),
                    thickness=2
                )
            )

    # Panel Angka
    cv2.rectangle(
        img,
        (20, 20),
        (180, 170),
        (255, 140, 0),
        -1
    )

    cv2.putText(
        img,
        "ANGKA",
        (35, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        img,
        str(total_fingers),
        (60, 135),
        cv2.FONT_HERSHEY_DUPLEX,
        2.5,
        (255, 255, 255),
        4
    )

    # Panel Gesture OK
    if gesture == "OK":

        cv2.rectangle(
            img,
            (220, 20),
            (380, 100),
            (0, 180, 0),
            -1
        )

        cv2.putText(
            img,
            "OK",
            (265, 75),
            cv2.FONT_HERSHEY_DUPLEX,
            1.8,
            (255, 255, 255),
            3
        )

   
    # Petunjuk Keluar
    cv2.putText(
        img,
        "ESC = Keluar",
        (20, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Deteksi Jari", img)

    key = cv2.waitKey(1)

    if key == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()