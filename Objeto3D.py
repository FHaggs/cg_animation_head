from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
from Point import *
import time
import random
import math

class Object3D:
    def __init__(self):
        # Original geometry and rendering
        self.vertices = []
        self.original_vertices = []
        self.faces = []
        self.position = Point(0, 0, 0)
        self.rotation = (0, 0, 0, 0)

        # Animation state management
        self.animation_state = "SWAY"
        self.animation_start_time = time.time()
        self.initial_time = time.time()

        # Physics and animation properties for each vertex
        self.velocities = []

        # Boids parameters
        self.boids_view_radius = 2.0
        self.boids_max_speed = 3.0
        
    def load_file(self, file: str):
        with open(file, "r") as f:
            for line in f:
                values = line.split()
                if not values: continue
                if values[0] == "v":
                    point = Point(float(values[1]), float(values[2]), float(values[3]))
                    self.vertices.append(point)
                    self.original_vertices.append(point)
                    self.velocities.append(Point(0,0,0))
                elif values[0] == "f":
                    face = []
                    for fVertex in values[1:]:
                        fInfo = fVertex.split("/")
                        face.append(int(fInfo[0]) - 1)
                    self.faces.append(face)
        print(f"Loaded {len(self.vertices)} vertices. Initial animation: {self.animation_state}")

    def draw_vertices(self):
        glPushMatrix()
        glTranslatef(self.position.x, self.position.y, self.position.z)
        glRotatef(self.rotation[3], self.rotation[0], self.rotation[1], self.rotation[2])
        
        colors = {
            "SWAY": (0.0, 0.0, 1.0),       # Blue
            "FALL": (1.0, 0.5, 0.0),       # Orange
            "TORNADO": (0.5, 0.5, 0.5),    # Grey
            "BOIDS": (0.0, 1.0, 0.0),        # Green
            "REASSEMBLE": (1.0, 1.0, 0.0), # Yellow
            "DONE": (1.0, 1.0, 1.0)         # White
        }
        glColor3f(*colors.get(self.animation_state, (1,1,1)))

        glPointSize(8)
        for v in self.vertices:
            glPushMatrix()
            glTranslatef(v.x, v.y, v.z)
            glutSolidSphere(0.05, 10, 10)
            glPopMatrix()
        glPopMatrix()

    def update(self, dt):
        """Main update function, acts as a state machine."""
        current_time = time.time()
        elapsed_animation_time = current_time - self.animation_start_time

        if self.animation_state == "SWAY":
            self.animate_sway(elapsed_animation_time)
            if elapsed_animation_time > 6: self.transition_to_state("FALL")
        elif self.animation_state == "FALL":
            self.animate_fall(dt)
            if elapsed_animation_time > 6: self.transition_to_state("TORNADO")
        elif self.animation_state == "TORNADO":
            self.animate_tornado(dt)
            if elapsed_animation_time > 12: self.transition_to_state("BOIDS")
        elif self.animation_state == "BOIDS":
            
            self.animate_boids(dt)
            if elapsed_animation_time > 20: self.transition_to_state("REASSEMBLE")
        elif self.animation_state == "REASSEMBLE":
            self.animate_reassemble(dt)
            if elapsed_animation_time > 8: self.transition_to_state("DONE")

    def transition_to_state(self, new_state):
        print(f"Transitioning from {self.animation_state} to {new_state}")
        self.animation_state = new_state
        self.animation_start_time = time.time()
        
        if new_state == "FALL":
            for i in range(len(self.velocities)): self.velocities[i] = Point(0, 0, 0)
        elif new_state in ["TORNADO", "BOIDS"]:
            for i in range(len(self.velocities)): self.velocities[i] = Point(random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1))
        elif new_state == "SWAY":
             for i in range(len(self.vertices)): self.vertices[i] = self.original_vertices[i]

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
        ground_y, damping = -0.2, 0.7
        for i, v in enumerate(self.vertices):
            self.velocities[i] += gravity * dt
            # FIX: Explicitly assign the new Point object back to the list
            self.vertices[i] = v + self.velocities[i] * dt
            # Re-fetch 'v' as it's now a new object
            v = self.vertices[i]
            if v.y < ground_y:
                v.y = ground_y
                self.velocities[i].y *= -damping
                self.velocities[i].x *= 0.9

    def animate_tornado(self, dt):
        center = Point(0, -2, 0)
        up_speed, rot_speed, suck_str = 1.5, 15.0, 2.0
        for i, v in enumerate(self.vertices):
            to_center = Point(center.x - v.x, 0, center.z - v.z)
            dist = to_center.magnitude() or 0.1
            radial_force = to_center.normalize() * suck_str
            tangent_force = Point(-to_center.z, 0, to_center.x).normalize() * (rot_speed / dist)
            up_force = Point(0, up_speed / (1 + dist * 0.5), 0)
            self.velocities[i] += (radial_force + tangent_force + up_force) * dt
            # FIX: Explicitly assign the new Point object back to the list
            self.vertices[i] = v + self.velocities[i] * dt
            self.velocities[i] *= 0.98

    def animate_boids(self, dt):
        w_align, w_cohesion, w_separation = 0.05, 0.02, 0.3
        for i, v_i in enumerate(self.vertices):
            avg_vel, avg_pos, separation_force, neighbors = Point(), Point(), Point(), 0
            for j, v_j in enumerate(self.vertices):
                if i == j: continue
                dist = (v_i - v_j).magnitude()
                if dist < self.boids_view_radius:
                    avg_vel += self.velocities[j]
                    avg_pos += v_j
                    if dist < self.boids_view_radius / 2:
                        separation_force -= (v_j - v_i) / (dist * dist if dist != 0 else 0.01)
                    neighbors += 1
            if neighbors > 0:
                align_steer = ((avg_vel / neighbors) - self.velocities[i]) * w_align
                cohesion_steer = ((avg_pos / neighbors) - v_i) * w_cohesion
                self.velocities[i] += align_steer + cohesion_steer + (separation_force * w_separation)
            
            speed = self.velocities[i].magnitude()
            if speed > self.boids_max_speed: self.velocities[i] = self.velocities[i].normalize() * self.boids_max_speed
            
            # FIX: Explicitly assign the new Point object back to the list
            self.vertices[i] = v_i + self.velocities[i] * dt

            if self.vertices[i].magnitude() > 15: self.velocities[i] -= self.vertices[i].normalize() * 0.1

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
                # FIX: Explicitly assign the new Point object back to the list
                self.vertices[i] = v + self.velocities[i] * dt