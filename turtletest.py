import cv2
import numpy as np
import pandas as pd
from statistics import mode
from turtle import Turtle, Screen
from handtracking import HandDetector
import random
import time
import pygame  


pygame.mixer.init()
pygame.mixer.music.load(r"sound\bgmusic.mp3") 
pygame.mixer.music.play(-1)  


t = Turtle()
sc = Screen()
sc.setup(width=800, height=600)
sc.bgcolor("CadetBlue1")

t.shape("turtle")
t.shapesize(stretch_wid=2, stretch_len=2, outline=2)
t.color("lightgreen")
t.penup()

home_x, home_y = 0, 0
t.goto(home_x, home_y)

sc.register_shape(r"Icon\bottlecup.gif")
sc.register_shape(r"Icon\broc.gif")
sc.bgpic(r'bgimg\bg.gif') 

num_bottles = 7
bottles = []
for _ in range(num_bottles):
    bottle = Turtle()
    bottle.shape(r"Icon\bottlecup.gif")
    bottle.penup()
    bottle.goto(random.randint(-300, 300), random.randint(-200, 200))
    bottles.append(bottle)

num_brocs = 15
brocs = []
for _ in range(num_brocs):
    broc = Turtle()
    broc.shape(r"Icon\broc.gif")
    broc.penup()
    broc.goto(random.randint(-300, 300), random.randint(-200, 200))
    brocs.append(broc)

screen_width = sc.window_width()
screen_height = sc.window_height()
boundary_padding = 20

score = 0
lives = 3
game_over = False

signs = ['W', 'H', 'C', 'D', 'Y']
train_data = pd.concat([pd.read_csv(f'gamejoint/{sign}_joints.csv') for sign in signs], ignore_index=True)

for col in train_data.columns:
    if col not in ['id', 'label']:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')

feature_cols = [col for col in train_data.columns if col not in ['id', 'label']]

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, complexity=0)

score_turtle = Turtle()
score_turtle.hideturtle()
score_turtle.penup()
score_turtle.goto(-screen_width // 2 + 20, screen_height // 2 - 40)
score_turtle.color("black")

lives_turtle = Turtle()
lives_turtle.hideturtle()
lives_turtle.penup()
lives_turtle.goto(screen_width // 2 - 120, screen_height // 2 - 40)
lives_turtle.color("black")

def update_display():
    score_turtle.clear()
    lives_turtle.clear()
    score_turtle.write(f"Score: {score}", font=("Arial", 16, "bold"))
    lives_turtle.write(f"Lives: {lives}", font=("Arial", 16, "bold"))

def bounds():
    x, y = t.position()
    return (
        -screen_width / 2 + boundary_padding <= x <= screen_width / 2 - boundary_padding
        and -screen_height / 2 + boundary_padding <= y <= screen_height / 2 - boundary_padding
    )

def check_collision(obj, distance_threshold=20):
    return t.distance(obj) < distance_threshold

def is_valid_position(x, y, particles, home_x, home_y, min_distance_from_home=150, min_distance_from_particles=70):
    if np.sqrt((x - home_x) ** 2 + (y - home_y) ** 2) < min_distance_from_home:
        return False
    for particle in particles:
        if np.sqrt((x - particle.xcor()) ** 2 + (y - particle.ycor()) ** 2) < min_distance_from_particles:
            return False
    return True

def spawn_particle(particle, particles, home_x, home_y, min_distance_from_home=150, min_distance_from_particles=70):
    while True:
        x = random.randint(-screen_width // 2 + boundary_padding, screen_width // 2 - boundary_padding)
        y = random.randint(-screen_height // 2 + boundary_padding, screen_height // 2 - boundary_padding)
        if is_valid_position(x, y, particles, home_x, home_y, min_distance_from_home, min_distance_from_particles):
            particle.goto(x, y)
            break

def knn_predict(test_point, train_data, k=5):
    distances = []
    for _, train_row in train_data.iterrows():
        dist = np.sqrt(np.sum((test_point - train_row[feature_cols]) ** 2))
        distances.append((dist, train_row['label']))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    neighbor_labels = [label for (_, label) in neighbors]
    predicted_label = mode(neighbor_labels)
    return predicted_label

def over():
    sc.clear()
    time.sleep(1)
    sc.clear()  # Clear the current screen
    sc.bgpic(r'bgimg\bgover.gif') 
    over_t = Turtle()
    over_t.hideturtle()
    over_t.penup()
    over_t.goto(0, 0)

predicted_label = None
update_display()
while not game_over:
    success, img = cap.read()
    if not success:
        break

    img, bboxes = detector.findHands(img)
    lmList = []

    if bboxes:
        lmList = detector.findPosition(img, draw=False)

        if lmList:
            test_point = []
            for joint in lmList:
                test_point.extend(joint[1:]) 

            if len(test_point) == len(feature_cols):
                predicted_label = knn_predict(np.array(test_point), train_data, k=5)
                print(f"Predicted Label: {predicted_label}")

                cv2.putText(img, f"Sign: {predicted_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if predicted_label == 'W':
        if bounds():  
            t.forward(10)
        else:
            t.goto(home_x, home_y)
    elif predicted_label == 'H':
        t.left(20)  
    elif predicted_label == 'C':
        t.right(20) 
    elif predicted_label == 'D':
        if bounds(): 
            t.backward(10) 
        else:
            t.goto(home_x, home_y)
    elif predicted_label == 'Y':
        pass  
    predicted_label = 'Y'

    for broc in brocs:
        if check_collision(broc, distance_threshold=20):
            score += 1
            print(f"Score: {score}")
            spawn_particle(broc, brocs + bottles, home_x, home_y)
            update_display()

    for bottle in bottles:
        if check_collision(bottle, distance_threshold=30):
            lives -= 1
            print(f"Lives Left: {lives}")
            if lives == 0:
                print("Game Over!")
                game_over = True
                over() 
            spawn_particle(bottle, brocs + bottles, home_x, home_y)
            update_display()

    cv2.imshow("Food Collection Game", img)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
sc.mainloop()