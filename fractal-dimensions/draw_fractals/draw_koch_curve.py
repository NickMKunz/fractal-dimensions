import turtle



def koch_curve(t, depth, length, color='black'):
    if depth == 0:
        t.color(color)
        t.forward(length)
    else:
        for angle in [60, -120, 60, 0]:
            koch_curve(t, depth-1, length/3, color)
            t.left(angle)
    return 0

def draw_koch_curve(depth):
    t = turtle.Turtle()
    t.speed(speed=0)
    t.width(2)
    drawing = turtle.Screen()
    drawing.bgcolor('white')
    t.up()
    t.goto([-950, 0])
    t.down()
    koch_curve(t, depth, 1900)
    t.hideturtle()
    drawing.exitonclick()
    return 0


"""
def draw_koch_curve(depth):
    t = turtle.Turtle()
    t.speed(speed=0)
    t.width(3)
    drawing = turtle.Screen()
    drawing.bgcolor('white')
    t.up()
    t.goto([-950, 0])
    t.down()
    length = 1900
    colors = ['blue', 'red', 'green', 'yellow']
    angle = [60, -120, 60, 0]
    for i in range(4):
        koch_curve(t, depth-1, length/3, colors[i])
        t.left(angle[i])
    t.hideturtle()
    drawing.exitonclick()
    return 0
"""


def draw_koch_snowflake(depth):
    t = turtle.Turtle()
    t.speed(speed=0)
    t.width(2)
    drawing = turtle.Screen()
    drawing.bgcolor('white')
    t.up()
    t.goto([-450, 250])
    t.down()
    koch_curve(t, depth, 900)
    t.right(120)
    koch_curve(t, depth, 900)
    t.right(120)
    koch_curve(t, depth, 900)
    t.hideturtle()
    drawing.exitonclick()
    return 0

draw_koch_snowflake(4)