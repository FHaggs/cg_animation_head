import math
import random
import time
from copy import deepcopy

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *

from Point import *


class Object3D:
    def __init__(self):
        # ... (init code remains the same) ...
        self.vertices = []
        self.original_vertices = []
        self.faces = []
        self.position = Point(0, 0, 0)
        self.rotation = (0, 0, 0, 0)
        self.animation_state = "SWAY"       # estado atual da simulação
        self.velocities = []

        # --- BOIDS PARAMETERS ---
        self.boids_view_radius = 5.0
        self.boids_max_speed = 2.5
        self.cohesion_weight = 3.0
        self.separation_weight = 3.2
        self.alignment_weight = 1.0
        self.separation_distance = 5.0


        # --- NEW GOAL PARAMETERS ---
        self.boids_goal   = Point(0, 3.3, -2.3)   # the point in space the boids will be attracted to
        self.flocks_count = 6          # number of flocks
        self.goal_weight  = 0.8             # how strongly each boid is pulled toward self.boids_goal

        # # --- NEW GOAL PARAMETERS ---
        # self.boids_goal = Point(0, 0, 0)     # The current position of the attractor
        # self.goal_weight = 0.8               # How strongly the boids are pulled to the goal


        self.baked_frames = []
        self.bake_complete = False
        self.color_frames = []
        self.current_frame = 0  


    def load_file(self, file: str, vertex_sample_rate: int = 1):
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

                if values[0] == "f":
                    face = []
                    for fVertex in values[1:]:
                        fInfo = fVertex.split("/")
                        original_idx = int(fInfo[0]) - 1
                        face.append(original_idx)
                    self.faces.append(face)

        print(
            f"Loaded {len(self.vertices)} vertices and {len(self.faces)} faces. Initial animation: {self.animation_state}"
        )

    def draw_vertices(self):
        # ... (draw_vertices code remains the same) ...
        glPushMatrix()
        glTranslatef(self.position.x, self.position.y, self.position.z)
        glRotatef(
            self.rotation[3], self.rotation[0], self.rotation[1], self.rotation[2]
        )

        colors = {
            "SWAY": (0.0, 0.0, 1.0),      # Blue
            "FALL": (1.0, 0.5, 0.0),      # Orange
            "TORNADO": (0.5, 0.5, 0.5),   # Grey
            "BOIDS": (0.3, 0.5, 1.0),     # Light Blue
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
            [(v.x, v.y, v.z,) for v in self.vertices] #[(x1, y1, z1), (x2, y2, z2), ..., (xn, yn, zn)]
        )
        self.color_frames.append(self.animation_state)

    def reproduz(self):
        # Só reproduz
        if self.current_frame < len(self.baked_frames):
            frame = self.baked_frames[self.current_frame]
            self.vertices = [Point(x, y, z) for x, y, z in frame]
            self.current_frame += 1


    def simulate_animation(self, dt, animation_time):
        timeline = [
            ("SWAY", 0),
            ("FALL", 6),
            ("TORNADO", 12),
            ("BOIDS", 20),
            ("BOIDS2", 30),
            ("REASSEMBLE", 40),
            ("DONE", 50),
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
        
        elif self.animation_state == "BOIDS2":
            self.separation_weight = 10.0
            self.cohesion_weight = 1.0
            self.boids_max_speed = 5.0
            self.boids_update(dt, explode=True)

        elif self.animation_state == "REASSEMBLE":
            self.animate_reassemble(dt)

        # Finaliza bake ao chegar no DONE
        if self.animation_state == "DONE":
            self.bake_complete = True
            print("Bake completo com", len(self.baked_frames), "quadros")

    def transition_to_state(self, new_state):
        print(f"Transição de {self.animation_state} para {new_state}")
        self.animation_state = new_state

        if new_state == "FALL":
            self.velocities = [Point(0, 0, 0) for _ in self.velocities]

        elif new_state == "BOIDS":
            self.velocities = [
                Point(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
                for _ in self.velocities
            ]

        elif new_state == "SWAY":
            self.vertices = [Point(v.x, v.y, v.z) for v in self.original_vertices]

    def animate_sway(self, elapsed_time):
        max_angle, frequency = 12.0, 2.0
        angle = math.radians(max_angle * math.sin(elapsed_time * frequency))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for i, original_v in enumerate(self.original_vertices): #rotation
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

    def animate_tornado(self, dt, time):
        center = Point(0, -2, 0)
        max_radius = 2.0
        up_speed = 7.0
        rot_speed = 10.0
        suck_strength = 6.0
        top_height = 2.2  
        base_height = 0.5  

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
                height_boost = max(0.0, (max_height - v.y)) * 0.2 #Aumenta o impulso vertical quanto mais longe estiver de max_height.
                up_factor = (1.5 - min(dist, 1.5)) / 1.5
                up_force = Point(0, (up_speed * (0.3 + up_factor)) + height_boost, 0)
            else:
                up_force = Point(0, 0, 0)

            total_force = radial_force + tangent_force + up_force

            self.velocities[i] += total_force * dt
            self.vertices[i] += self.velocities[i] * dt
            self.velocities[i] *= 0.965

    def apply_soft_boundary(self, pos, vel, strength=1.0, threshold=20.0):
        steer = Point(0.0, 0.0, 0.0)

        for axis in ['x', 'y', 'z']:
            val = getattr(pos, axis)
            v = getattr(vel, axis)
            min_val = -100  # Change as needed
            max_val = 100

            # Lower boundary
            if val - min_val < threshold:
                delta = (threshold - (val - min_val)) / threshold
                setattr(steer, axis, getattr(steer, axis) + delta * strength)

            # Upper boundary
            if max_val - val < threshold:
                delta = (threshold - (max_val - val)) / threshold
                setattr(steer, axis, getattr(steer, axis) - delta * strength)

        return steer



    def boids_update(self, dt, explode=False):
        """
        Very‐fast boids: four separate flocks, each using a 3D uniform‐grid (cell hashing)
        to limit neighbor searches to nearby cells.  
        
        Assumes:
        - self.vertices: list of Point (positions)
        - self.velocities: list of Point (velocities, same length as vertices)
        - self.boids_view_radius: float
        - self.boids_max_speed: float
        - self.cohesion_weight, self.separation_weight, self.alignment_weight: floats
        - self.separation_distance: float
        

        """

        N = len(self.vertices)
        if N == 0:
            return

        # 1) Determine flock‐assignment (4 flocks, by index range). 
        #    e.g. if N=1000, flock_size=250→
        #      flock 0 = [0..249], flock 1 = [250..499], flock 2 = [500..749], flock 3 = [750..999].
        n_flocks = self.flocks_count
        base = N // n_flocks
        flock_ranges = []
        for f in range(n_flocks):
            start = f * base
            # last flock takes any remainder
            end = (f + 1) * base if (f < n_flocks - 1) else N
            flock_ranges.append((start, end))

        # Cache some constants
        R = self.boids_view_radius
        R2 = R * R
        sep_dist = self.separation_distance
        sep_dist2 = sep_dist * sep_dist
        max_speed = self.boids_max_speed

        # Cell size = view radius (ensures any neighbor within R must be in same/adjacent cell)
        cell_size = R

        # For each flock, build a dict: cell_index (tuple) → list of boid indices
        # We will store four separate cell‐maps, one per flock.
        all_cell_maps = [ {} for _ in range(n_flocks) ]
        # Where cell index is (i, j, k) = ( floor(x/cell_size), floor(y/cell_size), floor(z/cell_size) ).

        # 2) Populate each flock's cell map
        for f_idx, (start, end) in enumerate(flock_ranges):
            cell_map = all_cell_maps[f_idx]
            for i in range(start, end):
                pos = self.vertices[i]
                # Compute the integer cell coordinates
                ci = int(math.floor(pos.x / cell_size))
                cj = int(math.floor(pos.y / cell_size))
                ck = int(math.floor(pos.z / cell_size))
                key = (ci, cj, ck)
                if key not in cell_map:
                    cell_map[key] = []
                cell_map[key].append(i)

        # 3) For each flock, for each boid in that flock, find neighbors in the 27 cells around it
        #    then accumulate cohesion, separation, alignment, update velocity and position.
        for f_idx, (start, end) in enumerate(flock_ranges):
            cell_map = all_cell_maps[f_idx]
            for i in range(start, end):
                pos_i = self.vertices[i]
                vel_i = self.velocities[i]

                # Determine this boid's cell
                ci = int(math.floor(pos_i.x / cell_size))
                cj = int(math.floor(pos_i.y / cell_size))
                ck = int(math.floor(pos_i.z / cell_size))

                # Accumulators
                #   Cohesion: steer toward average position,
                #   Separation: steer away from very‐close neighbors,
                #   Alignment: match average velocity.
                coh_acc = Point(0.0, 0.0, 0.0)
                sep_acc = Point(0.0, 0.0, 0.0)
                ali_acc = Point(0.0, 0.0, 0.0)
                n_neighbors = 0
                n_sep = 0

                # Loop over the 27 neighboring cells
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            cell_key = (ci + dx, cj + dy, ck + dz)
                            if cell_key not in cell_map:
                                continue
                            # Check each boid index in that cell
                            for j in cell_map[cell_key]:
                                if j == i:
                                    continue
                                # Quick bounding‐box check: squared distance
                                pj = self.vertices[j]
                                rx = pj.x - pos_i.x
                                ry = pj.y - pos_i.y
                                rz = pj.z - pos_i.z
                                dist2 = rx * rx + ry * ry + rz * rz
                                if dist2 > R2:
                                    continue  # outside view radius entirely

                                # It's a “neighbor” within view radius
                                n_neighbors += 1
                                # Cohesion: accumulate neighbor positions
                                coh_acc.x += pj.x
                                coh_acc.y += pj.y
                                coh_acc.z += pj.z
                                # Alignment: accumulate neighbor velocities
                                vj = self.velocities[j]
                                ali_acc.x += vj.x
                                ali_acc.y += vj.y
                                ali_acc.z += vj.z

                                # Separation: if too close (dist^2 < sep_dist2), steer away
                                if dist2 < sep_dist2:
                                    # vector pointing from neighbor → self
                                    if dist2 == 0:
                                        # avoid zero‐division (rare): random small vector
                                        rx2 = (random.random() - 0.5) * 0.01
                                        ry2 = (random.random() - 0.5) * 0.01
                                        rz2 = (random.random() - 0.5) * 0.01
                                        sep_acc.x += rx2
                                        sep_acc.y += ry2
                                        sep_acc.z += rz2
                                    else:
                                        inv_d = 1.0 / math.sqrt(dist2)
                                        sep_acc.x += (pos_i.x - pj.x) * inv_d
                                        sep_acc.y += (pos_i.y - pj.y) * inv_d
                                        sep_acc.z += (pos_i.z - pj.z) * inv_d
                                    n_sep += 1

                # If there are neighbors, finalize each component
                steer = Point(0.0, 0.0, 0.0)
                if n_neighbors > 0:
                    inv_n = 1.0 / n_neighbors
                    # Cohesion: average neighbor position minus self position
                    coh_acc.x = (coh_acc.x * inv_n - pos_i.x)
                    coh_acc.y = (coh_acc.y * inv_n - pos_i.y)
                    coh_acc.z = (coh_acc.z * inv_n - pos_i.z)
                    # Alignment: average neighbor velocity minus self velocity
                    ali_acc.x = (ali_acc.x * inv_n - vel_i.x)
                    ali_acc.y = (ali_acc.y * inv_n - vel_i.y)
                    ali_acc.z = (ali_acc.z * inv_n - vel_i.z)

                    # Weighted sum
                    steer.x += coh_acc.x * self.cohesion_weight
                    steer.y += coh_acc.y * self.cohesion_weight
                    steer.z += coh_acc.z * self.cohesion_weight

                    steer.x += ali_acc.x * self.alignment_weight
                    steer.y += ali_acc.y * self.alignment_weight
                    steer.z += ali_acc.z * self.alignment_weight

                # Separation term
                if n_sep > 0:
                    inv_sep = 1.0 / n_sep
                    steer.x += sep_acc.x * inv_sep * self.separation_weight
                    steer.y += sep_acc.y * inv_sep * self.separation_weight
                    steer.z += sep_acc.z * inv_sep * self.separation_weight


                # boundary_steer = self.apply_soft_boundary(pos_i, vel_i)
                # steer.x += boundary_steer.x
                # steer.y += boundary_steer.y
                # steer.z += boundary_steer.z
                if not explode:
                    # flock_goal_point = self.boids_goals[f_idx]
                    goal_dir = self.boids_goal - pos_i

                    if goal_dir.magnitude() > 0:
                        goal_dir = goal_dir.normalized()
                        steer.x += goal_dir.x * self.goal_weight
                        steer.y += goal_dir.y * self.goal_weight
                        steer.z += goal_dir.z * self.goal_weight

                # 4) Update velocity: v_new = v_old + steer * dt
                vel_i.x += steer.x * dt
                vel_i.y += steer.y * dt
                vel_i.z += steer.z * dt

                # 5) Limit speed to max_speed
                speed2 = vel_i.x * vel_i.x + vel_i.y * vel_i.y + vel_i.z * vel_i.z
                if speed2 > (max_speed * max_speed):
                    inv_sp = max_speed / math.sqrt(speed2)
                    vel_i.x *= inv_sp
                    vel_i.y *= inv_sp
                    vel_i.z *= inv_sp

                # 6) Finally, move position: p_new = p_old + v_new * dt
                pos_i.x += vel_i.x * dt
                pos_i.y += vel_i.y * dt
                pos_i.z += vel_i.z * dt

                # Write back
                self.velocities[i] = vel_i
                self.vertices[i] = pos_i

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
