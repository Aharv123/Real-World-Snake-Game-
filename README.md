# 🐍 Snake Game Controller 🎮 (Hand Gesture-Based)

This project is a computer vision-based **Snake Game Controller** powered by **OpenCV** and **MediaPipe**.  
Use your **index fingertip** to control the movement of a snake on screen! As you move your finger, the snake follows and collects food points, growing longer and increasing your score.  

If the snake **collides with itself**, it's game over – and the game restarts automatically.

---

## 📽️ Demo

> 🔴 Live hand tracking
> 🟢 Green snake follows your finger
> 🍎 Red food dot spawns randomly
> 🎯 Collect food to grow the snake
> 💥 Collision with yourself = Game Over

---

## 🧠 Tech Stack

| Technology  | Purpose                            |
|-------------|------------------------------------|
| **OpenCV**  | Webcam access & rendering graphics |
| **MediaPipe** | Hand tracking & landmark detection |
| **NumPy**   | Coordinate & math operations       |
| **Python**  | Main programming language          |
| **Math**    | Distance calculations              |
| **Random**  | Food spawning                      |

---

## 🛠️ Installation

1. **Clone the repo:**

```bash
git clone https://github.com/your-username/snake-game-controller.git
cd snake-game-controller
