import turtle



def draw(t, x, y, height):
    t.fillcolor('black')
    t.up()
    t.goto([x, height])
    t.down()
    t.begin_fill()
    t.goto([x, height+10])
    t.goto([y, height+10])
    t.goto([y, height])
    t.goto([x, height])
    t.end_fill()
    return 0

def cantor_set(t, depth, x, y, height):
    if depth > 0:
        draw(t, x, y, height)
        cantor_set(t, depth-1, x, x + (y-x)/3, height-50)
        cantor_set(t, depth-1, y - (y-x)/3, y, height-50)
    return 0

def draw_cantor_set(depth):
    t = turtle.Turtle()
    t.speed(speed=0)
    drawing = turtle.Screen()
    drawing.bgcolor('white')
    height = 50*depth
    cantor_set(t, depth, -950, 950, height)
    t.up()
    t.goto([1500, 1500])
    drawing.exitonclick()
    return 0

draw_cantor_set(7)