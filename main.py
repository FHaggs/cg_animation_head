from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *

import time

from pygame import Color

from Objeto3D import Object3D
import Point

o: Object3D
tempo_antes = time.time()
soma_dt = 0
estado_animacao = "PLAY"  # Pode começar pausado ou em "PLAY"
tempo_animado = 0.0

def init():
    glClearColor(0.5, 0.5, 0.9, 1.0)
    glClearDepth(1.0)

    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    DefineLuz()
    PosicUser()


def DefineLuz():
    # Define cores para um objeto dourado
    luz_ambiente = [0.4, 0.4, 0.4]
    luz_difusa = [0.7, 0.7, 0.7]
    luz_especular = [0.9, 0.9, 0.9]
    posicao_luz = [2.0, 3.0, 0.0]  # PosiÃ§Ã£o da Luz
    especularidade = [1.0, 1.0, 1.0]

    # ****************  Fonte de Luz 0

    glEnable(GL_COLOR_MATERIAL)

    # Habilita o uso de iluminaÃ§Ã£o
    glEnable(GL_LIGHTING)

    # Ativa o uso da luz ambiente
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, luz_ambiente)
    # Define os parametros da luz nÃºmero Zero
    glLightfv(GL_LIGHT0, GL_AMBIENT, luz_ambiente)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, luz_difusa)
    glLightfv(GL_LIGHT0, GL_SPECULAR, luz_especular)
    glLightfv(GL_LIGHT0, GL_POSITION, posicao_luz)
    glEnable(GL_LIGHT0)

    # Ativa o "Color Tracking"
    glEnable(GL_COLOR_MATERIAL)

    # Define a reflectancia do material
    glMaterialfv(GL_FRONT, GL_SPECULAR, especularidade)

    # Define a concentraÃ§Ã£oo do brilho.
    # Quanto maior o valor do Segundo parametro, mais
    # concentrado serÃ¡ o brilho. (Valores vÃ¡lidos: de 0 a 128)
    glMateriali(GL_FRONT, GL_SHININESS, 51)


def PosicUser():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    # Configura a matriz da projeção perspectiva (FOV, proporção da tela, distância do mínimo antes do clipping, distância máxima antes do clipping
    # https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml
    gluPerspective(60, 16 / 9, 0.01, 50)  # Projecao perspectiva
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Especifica a matriz de transformação da visualização
    # As três primeiras variáveis especificam a posição do observador nos eixos x, y e z
    # As três próximas especificam o ponto de foco nos eixos x, y e z
    # As três últimas especificam o vetor up
    # https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/gluLookAt.xml
    gluLookAt(-2, 6, -8, 0, 0, 0, 0, 1.0, 0)


def DesenhaLadrilho():
    glColor3f(0.5, 0.5, 0.5)  # desenha QUAD preenchido
    glBegin(GL_QUADS)
    glNormal3f(0, 1, 0)
    glVertex3f(-0.5, 0.0, -0.5)
    glVertex3f(-0.5, 0.0, 0.5)
    glVertex3f(0.5, 0.0, 0.5)
    glVertex3f(0.5, 0.0, -0.5)
    glEnd()

    glColor3f(1, 1, 1)  # desenha a borda da QUAD
    glBegin(GL_LINE_STRIP)
    glNormal3f(0, 1, 0)
    glVertex3f(-0.5, 0.0, -0.5)
    glVertex3f(-0.5, 0.0, 0.5)
    glVertex3f(0.5, 0.0, 0.5)
    glVertex3f(0.5, 0.0, -0.5)
    glEnd()


def DesenhaPiso():
    glPushMatrix()
    glTranslated(-20, -1, -10)
    for x in range(-20, 20):
        glPushMatrix()
        for z in range(-20, 20):
            DesenhaLadrilho()
            glTranslated(0, 0, 1)
        glPopMatrix()
        glTranslated(1, 0, 0)
    glPopMatrix()


def DesenhaCubo():
    glPushMatrix()
    glColor3f(1, 0, 0)
    glTranslated(0, 0.5, 0)
    glutSolidCube(1)

    glColor3f(0.5, 0.5, 0)
    glTranslated(0, 0.5, 0)
    glRotatef(90, -1, 0, 0)
    glRotatef(45, 0, 0, 1)
    glutSolidCone(1, 1, 4, 4)
    glPopMatrix()


# Function called constantly (idle) to update the animation
def Animacao():
    global estado_animacao

    if estado_animacao == "PAUSE":
        return

    if estado_animacao == "REWIND":
        o.current_frame = max(0, o.current_frame - 20)
        print(f"REWIND <- currentFrame {o.current_frame}")
        estado_animacao = "PLAY"
        o.reproduz()

    if estado_animacao == "FOWARD":
        o.current_frame = min(len(o.baked_frames) - 1, o.current_frame + 20)
        print(f"FOWARD -> currentFrame {o.current_frame}")
        estado_animacao = "PLAY"
        o.reproduz()

    if estado_animacao == "PLAY":
        o.reproduz()

    glutPostRedisplay()


def desenha():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_MODELVIEW)

    DesenhaPiso()
    # DesenhaCubo()
    # o.Desenha()
    # o.DesenhaWireframe()
    o.draw_vertices()

    glutSwapBuffers()
    pass


def teclado(key, x, y):
    global estado_animacao, o

    if key == b'p':
        estado_animacao = "PLAY"

    elif key == b's':
        estado_animacao = "PAUSE"

    elif key == b'r':
        estado_animacao = "REWIND"

    elif key == b'f':
        estado_animacao = "FOWARD"
        
def main():
    import sys

    # 1. Inicializa o objeto e carrega o modelo
    global o, tempo_animado
    o = Object3D()
    o.load_file("Human_Head.obj")

    # 2. Fase de bake (sem renderização, só computação)
    tempo_animado = 0.0
    dt = 1.0 / 30.0


    print("Baking animation...")

    while not o.playback_mode :
        o.update(dt, tempo_animado)
        tempo_animado += dt

    print("Bake completo. Total de quadros:", len(o.baked_frames))

    # 3. Agora inicia a janela GLUT para exibição
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(1000, 600)
    glutInitWindowPosition(100, 100)
    glutCreateWindow(b"Computacao Grafica - 3D")

    init()

    glutDisplayFunc(desenha)
    glutKeyboardFunc(teclado)
    glutIdleFunc(Animacao)

    try:
        glutMainLoop()
    except SystemExit:
        pass

if __name__ == "__main__":
    main()
