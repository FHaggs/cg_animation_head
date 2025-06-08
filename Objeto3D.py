from copy import deepcopy
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
from pygame import Color
from Point import *
import time
import random
import math


class Object3D:
    def __init__(self):
        # ... (init code remains the same) ...
        self.vertices = []
        self.original_vertices = []
        self.faces = []
        self.position = Point(0, 0, 0)
        self.rotation = (0, 0, 0, 0)
        self.animation_state = "SWAY"       # estado atual da simulação
        self.animation_start_time = 0.0
        self.initial_time = time.time()
        self.velocities = []
        self.boids_view_radius = 2.0
        self.boids_max_speed = 3.0
        self.cohesion_weight = 1.0
        self.separation_weight = 1.5
        self.alignment_weight = 1.0
        self.separation_distance = 0.5

        self.baked_frames = []
        self.bake_complete = False
        self.playback_mode = False
        self.color_frames = []
        self.current_frame = 0  # <- Adicione isso aqui

    def load_file(self, file: str, vertex_sample_rate: int = 1):
        """
        Load a simplified version of the file by skipping some vertices
        and their associated faces. For example, vertex_sample_rate=2
        will keep every 2nd vertex.
        """

        vertex_map = {}  # Maps original index to new index
        vertex_index = 0
        new_index = 0

        with open(file, "r") as f:
            for line in f:
                values = line.split()
                if not values:
                    continue

                if values[0] == "v":
                    point = Point(
                        float(values[1]), float(values[2]), float(values[3])
                    )
                    self.vertices.append(point)
                    self.original_vertices.append(deepcopy(point))
                    self.velocities.append(Point(0, 0, 0))
                    vertex_map[vertex_index] = new_index
                    new_index += 1
                    vertex_index += 1

        with open(file, "r") as f:
            for line in f:
                values = line.split()
                if not values:
                    continue

                if values[0] == "f":
                    face = []
                    skip_face = False
                    for fVertex in values[1:]:
                        fInfo = fVertex.split("/")
                        original_idx = int(fInfo[0]) - 1
                        if original_idx in vertex_map:
                            face.append(vertex_map[original_idx])
                        else:
                            skip_face = True
                            break  # Don't add this face if any vertex was skipped

                    if not skip_face:
                        self.faces.append(face)

        print(
            f"Loaded {len(self.vertices)} vertices and {len(self.faces)} faces. Initial animation: {self.animation_state}"
        )

    def draw_vertices(self):
        glPushMatrix()
        glTranslatef(self.position.x, self.position.y, self.position.z)
        glRotatef(
            self.rotation[3], self.rotation[0], self.rotation[1], self.rotation[2]
        )

        colors = {
            "SWAY": (0.0, 0.0, 1.0),      # Blue
            "FALL": (1.0, 0.5, 0.0),      # Orange
            "TORNADO": (0.5, 0.5, 0.5),   # Grey
            "BOIDS": (0.0, 0.0, 0.0),     # Black
            "REASSEMBLE": (1.0, 1.0, 0.0),# Yellow
            "DONE": (1.0, 1.0, 1.0),      # White
        }
        glColor3f(*colors.get(self.color_frames[self.current_frame],(1, 1, 1)))
        glPointSize(8)

        for v in self.vertices:
            glPushMatrix()
            glTranslatef(v.x, v.y, v.z)
            glutSolidSphere(0.05, 10, 10)
            glPopMatrix()
        glPopMatrix()

    def update(self, dt, time_elapsed):
        # Executa simulação e salva os dados por partícula (posição + estado)
        self.simulate_animation(dt, time_elapsed)
        self.baked_frames.append(
            [(v.x, v.y, v.z,) for v in self.vertices]
        )
        self.color_frames.append(self.animation_state)

    def reproduz(self):
        # Só reproduz
        if self.current_frame < len(self.baked_frames):
            frame = self.baked_frames[self.current_frame + 1]
            self.vertices = [Point(x, y, z) for x, y, z in frame]
            self.current_frame += 1

    def simulate_animation(self, dt, animation_time):
        timeline = [
            ("SWAY", 0),
            ("FALL", 6),
            ("TORNADO", 12),
            ("BOIDS", 20),
            ("REASSEMBLE",30 ),
            ("DONE", 40),
        ]

        # Detecta em qual estado deve estar, baseado no tempo atual
        new_state = self.animation_state
        for i in range(len(timeline) - 1):
            state_name, start_time = timeline[i]
            _, next_time = timeline[i + 1]
            if start_time <= animation_time < next_time:
                new_state = state_name
                break
            else:
                new_state = "DONE"

        # Se o estado mudou, atualiza
        if new_state != self.animation_state:
            self.transition_to_state(new_state)

        # Executa animação conforme o estado atual
        if self.animation_state == "SWAY":
            elapsed = animation_time - 0
            self.animate_sway(elapsed)

        elif self.animation_state == "FALL":
            self.animate_fall(dt)

        elif self.animation_state == "TORNADO":
            elapsed = animation_time - 12
            self.animate_tornado(dt, elapsed)

        elif self.animation_state == "BOIDS":
            self.boids_update(dt)

        elif self.animation_state == "REASSEMBLE":
            self.animate_reassemble(dt)

        # Finaliza bake ao chegar no DONE
        if self.animation_state == "DONE":
            self.bake_complete = True
            self.playback_mode = True
            print("Bake completo com", len(self.baked_frames), "quadros")

    def transition_to_state(self, new_state):
        print(f"Transição de {self.animation_state} para {new_state}")
        self.animation_state = new_state

        if new_state == "FALL":
            self.velocities = [Point(0, 0, 0) for _ in self.velocities]

        elif new_state in ["TORNADO", "BOIDS"]:
            self.velocities = [
                Point(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
                for _ in self.velocities
            ]

        elif new_state == "SWAY":
            self.vertices = [Point(v.x, v.y, v.z) for v in self.original_vertices]

    def animate_sway(self, elapsed_time):
        max_angle, frequency = 12.0, 1.8
        angle = math.radians(max_angle * math.sin(elapsed_time * frequency))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for i, original_v in enumerate(self.original_vertices):
            self.vertices[i].y = original_v.y * cos_a - original_v.z * sin_a
            self.vertices[i].z = original_v.y * sin_a + original_v.z * cos_a
            self.vertices[i].x = original_v.x

    def animate_fall(self, dt):
        gravity = Point(0, -9.8, 0)
        ground_y, damping = -0.5, 0.7
        for i, v in enumerate(self.vertices):
            self.velocities[i] += gravity * dt
            self.vertices[i] = v + self.velocities[i] * dt
            v = self.vertices[i]
            if v.y < ground_y:
                v.y = ground_y
                self.velocities[i].y *= -damping
                self.velocities[i].x *= 0.9

    def animate_tornado(self, dt, time):
        center = Point(0, -2, 0)
        max_radius = 2.0
        up_speed = 7.0
        rot_speed = 10.0
        suck_strength = 6.0
        top_height = 3.5  # altura máxima no centro
        base_height = 1.0  # altura máxima nas bordas

        for i, v in enumerate(self.vertices):
            to_center = Point(center.x - v.x, 0, center.z - v.z)
            dist = to_center.magnitude() or 0.01

            # delay para sair baseado na distância
            start_delay = dist * 0.5
            if time < start_delay:
                continue

            # altura máxima baseada na distância (centro sobe mais)
            t = min(dist / max_radius, 1.0)
            max_height = (1 - t) * top_height + t * base_height

            # força radial
            if dist > max_radius:
                radial_force = to_center.normalized() * suck_strength
            else:
                radial_force = to_center.normalized() * (suck_strength * 0.3)

            # força tangencial (gira mais conforme sobe)
            height_factor = min(max((v.y + 2.0) / (top_height + 2.0), 0.0), 1.0)
            tangent = Point(-to_center.z, 0, to_center.x).normalized()
            tangent_force = tangent * (rot_speed * height_factor / dist)

            # subida até altura personalizada
            if v.y < max_height:
                height_boost = max(0.0, (max_height - v.y)) * 0.2
                up_factor = (1.5 - min(dist, 1.5)) / 1.5
                up_force = Point(0, (up_speed * (0.3 + up_factor)) + height_boost, 0)
            else:
                up_force = Point(0, 0, 0)

            total_force = radial_force + tangent_force + up_force

            self.velocities[i] += total_force * dt
            self.vertices[i] += self.velocities[i] * dt
            self.velocities[i] *= 0.965


    def boids_update(self, dt):
       return

    def animate_reassemble(self, dt):
        attraction_strength = 4.0
        for i, v in enumerate(self.vertices):
            target = self.original_vertices[i]
            to_target = target - v
            if to_target.magnitude() < 0.01:
                self.vertices[i] = target
                self.velocities[i] = Point()
            else:
                self.velocities[i] = to_target * attraction_strength
                self.vertices[i] = v + self.velocities[i] * dt
