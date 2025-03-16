from turtle import Turtle, Screen
import subprocess
import sys

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
    title.goto(0, 50) 
    title.write("แล้วเต่าทะเลล่ะ", align="center", font=("ZF #2ndPixelus", 80, "bold"))

    buttons = [
        {"text": "Collect Data", "action": collect_data},
        {"text": "How to Play", "action": show_instructions},
        {"text": "Start Game", "action": start_game}
    ]

    for i, button in enumerate(buttons):
        t = Turtle()
        t.hideturtle()
        t.penup()
        t.color("white") 
        t.goto(0, -i * 50)
        t.write(button["text"], align="center", font=("ZF #2ndPixelus", 40, "normal"))
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

def collect_data():
    print("Data collection started...")
    sc.bye()
    subprocess.run([sys.executable, "databutturtle.py"])

def show_instructions():
    print("Displaying instructions...")
    sc.bye()
    subprocess.run([sys.executable, "Howto.py"])
    
def start_game():
    print("Starting game...")
    sc.bye()
    subprocess.run([sys.executable, "turtletest.py"])

start_page()