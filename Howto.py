from turtle import Turtle, Screen
import subprocess
import sys

import pygame  


pygame.mixer.init()
pygame.mixer.music.load(r"sound\bgmusic.mp3") 
pygame.mixer.music.play(-1)  


sc = Screen()
sc.setup(width=800, height=600)
sc.bgpic(r'bgimg\bg.gif')

selected_button = 0
buttons = []

def start_page():
    global buttons

    title = Turtle()
    title.hideturtle()
    title.penup()
    title.color("white")  
    title.goto(0, 110) 
    title.write("HOW TO PLAY", align="center", font=("ZF #2ndPixelus", 70, "bold"))

    buttons = [
        {"text": "Back To main menu", "action": start},
    ]

    for i, button in enumerate(buttons):
        t = Turtle()
        t.hideturtle()
        t.penup()
        t.color("white") 
        t.goto(0, -150)
        t.write(button["text"], align="center", font=("ZF #2ndPixelus", 35, "normal"))
        buttons[i]["turtle"] = t

    highlight_button(0)

    sc.listen()
    sc.onkey(up, "w")
    sc.onkey(up, "Up")
    sc.onkey(down, "s")
    sc.onkey(down, "Down")
    sc.onkey(select, "Return")

    sc.mainloop()

def highlight_button(index):
    global selected_button
    buttons[selected_button]["turtle"].clear()
    buttons[selected_button]["turtle"].color("white") 
    buttons[selected_button]["turtle"].write(buttons[selected_button]["text"], align="center", font=("ZF #2ndPixelus", 40, "normal"))

    selected_button = index
    buttons[selected_button]["turtle"].clear()
    buttons[selected_button]["turtle"].color("white") 
    buttons[selected_button]["turtle"].write(buttons[selected_button]["text"], align="center", font=("ZF #2ndPixelus", 45, "bold"))

def up():
    new_index = (selected_button - 1) % len(buttons)
    highlight_button(new_index)

def down():
    new_index = (selected_button + 1) % len(buttons)
    highlight_button(new_index)

def select():
    buttons[selected_button]["action"]()


def show_instructions():
    instructions = Turtle()
    instructions.hideturtle()
    instructions.penup()
    instructions.color("white") 
    instructions.goto(0, -80)
    instructions.write(
        "W : Move Forward\n"
        "C : Turn Right\n"
        "D : Move Backward\n"
        "H : Turn Left",
        align="center", font=("ZF #2ndPixelus", 35, "normal")
    )

def start_game():
    sc.bye()
    subprocess.run([sys.executable, "turtletest.py"])

def start():
    pygame.mixer.music.stop() 
    sc.bye()  
    subprocess.run([sys.executable, "start.py"])

show_instructions()
start_page()