import numpy as np
import turtle



def draw(t, p, length, color='black'):
    t.fillcolor(color)
    t.color(color)
    t.up()
    t.goto(p)
    t.down()
    t.begin_fill()
    t.goto(p + [length, 0])
    t.goto(p + [length, length])
    t.goto(p + [0, length])
    t.goto(p)
    t.end_fill()
    return 0

def cantor_dust(t, depth, p, length, color='black'):
    l = length/3
    if depth == 0:
        draw(t, p, l, color)
    else:
        cantor_dust(t, depth - 1, p, l, color)
        cantor_dust(t, depth - 1, p+[2*l/3, 0], l, color)
        cantor_dust(t, depth - 1, p+[2*l/3, 2*l/3], l, color)
        cantor_dust(t, depth - 1, p+[0, 2*l/3], l, color)
    return 0

def draw_cantor_dust(depth):
    t = turtle.Turtle()
    t.speed(speed=0)
    drawing = turtle.Screen()
    drawing.bgcolor('white')
    cantor_dust(t, depth, np.array([-500, -500]), 2000)
    t.hideturtle()
    drawing.exitonclick()
    return 0

def draw_cantor_dust_color(depth):
    t = turtle.Turtle()
    t.speed(speed=0)
    drawing = turtle.Screen()
    drawing.bgcolor('white')
    p = np.array([-500, -500])
    length = 3000
    cantor_dust(t, depth, p, length, 'black')
    t.hideturtle()
    drawing.exitonclick()
    return 0

"""
def draw_cantor_dust_color(depth):
    t = turtle.Turtle()
    t.speed(speed=0)
    drawing = turtle.Screen()
    drawing.bgcolor('white')
    p = np.array([-500, -500])
    length = 2000
    color = ['blue', 'red', 'green', 'yellow']
    cantor_dust(t, depth - 1, p, length/3, color[0])
    cantor_dust(t, depth - 1, p+[length/3, 0], length/3, color[1])
    cantor_dust(t, depth - 1, p+[length/3, length/3], length/3, color[2])
    cantor_dust(t, depth - 1, p+[0, length/3], length/3, color[3])
    t.hideturtle()
    drawing.exitonclick()
    return 0
"""

draw_cantor_dust_color(5)