import numpy as np
import turtle



def draw(t, points, color='black'):
    t.fillcolor(color)
    t.color(color)
    t.up()
    t.goto(points[0])
    t.down()
    t.begin_fill()
    t.goto(points[1])
    t.goto(points[2])
    t.goto(points[0])
    t.end_fill()
    return 0

def sierpinski_triangle(t, order, points, color='black'):
    if order == 0:
        draw(t, points, color)
    else:
        sierpinski_triangle(t, order - 1, [points[0], (points[0] + points[1])/2, (points[0] + points[2])/2], color)
        sierpinski_triangle(t, order - 1, [points[1], (points[0] + points[1])/2, (points[1] + points[2])/2], color)
        sierpinski_triangle(t, order - 1, [points[2], (points[2] + points[1])/2, (points[0] + points[2])/2], color)
    return 0

def draw_sierpinski_triangle(depth):
    t = turtle.Turtle()
    t.speed(speed=0)
    drawing = turtle.Screen()
    drawing.bgcolor('white')
    points = np.array([[-500, -400], [0, 500], [500, -400]])
    sierpinski_triangle(t, depth, points)
    t.up()
    t.goto([1000, 1000])
    drawing.exitonclick()
    return 0

def draw_sierpinski_triangle_color(depth):
    t = turtle.Turtle()
    t.speed(speed=0)
    drawing = turtle.Screen()
    drawing.bgcolor('white')
    points = np.array([[-500, -400], [0, 500], [500, -400]])
    colors = ['blue', 'red', 'yellow']
    sierpinski_triangle(t, depth - 1, [points[0], (points[0] + points[1])/2, (points[0] + points[2])/2], colors[0])
    sierpinski_triangle(t, depth - 1, [points[1], (points[0] + points[1])/2, (points[1] + points[2])/2], colors[1])
    sierpinski_triangle(t, depth - 1, [points[2], (points[2] + points[1])/2, (points[0] + points[2])/2], colors[2])
    t.hideturtle()
    drawing.exitonclick()
    return 0

draw_sierpinski_triangle(6)