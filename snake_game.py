import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Snake config
initial_snake_length = 100
radius = 10

# Initialize game state
snake = []
snake_length = initial_snake_length
current_length = 0
score = 0
food_point = (300, 300)
game_over = False
restart_time = 0

# Distance function
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Reset game function
def reset_game():
    global snake, snake_length, current_length, score, food_point, game_over
    snake = []
    snake_length = initial_snake_length
    current_length = 0
    score = 0
    food_point = (300, 300)
    game_over = False

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Snake Game Controller", cv2.WINDOW_NORMAL)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not game_over and results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm = handLms.landmark[8]  # Index finger tip
            cx, cy = int(lm.x * w), int(lm.y * h)

            # Add the point to the snake body
            if len(snake) == 0:
                snake.append((cx, cy))
            else:
                prev = snake[-1]
                d = distance(prev, (cx, cy))
                if d > 5:
                    snake.append((cx, cy))
                    current_length += d

            # Trim the snake to keep it under max length
            while current_length > snake_length:
                d = distance(snake[0], snake[1])
                current_length -= d
                snake.pop(0)

            # Check collision with food
            if distance((cx, cy), food_point) < 20:
                food_point = (random.randint(50, w - 50), random.randint(50, h - 50))
                snake_length += 30
                score += 1

            # Check self collision
            if len(snake) > 20:  # Ignore small snake
                head = snake[-1]
                for pt in snake[:-20]:  # Ignore tail-end points
                    if distance(head, pt) < 20:
                        game_over = True
                        restart_time = time.time()

            # Draw snake body
            for i in range(1, len(snake)):
                cv2.line(frame, snake[i - 1], snake[i], (0, 255, 0), radius * 2)

            # Draw food
            cv2.circle(frame, food_point, radius, (0, 0, 255), -1)

            # Draw score
            cv2.putText(frame, f'Score: {score}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    elif game_over:
        # Show Game Over text
        cv2.putText(frame, "Game Over", (w // 2 - 150, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)
        if time.time() - restart_time > 2:
            reset_game()

    cv2.imshow("Snake Game Controller", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
