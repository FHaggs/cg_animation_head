from copy import deepcopy
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
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
        self.animation_state = "SWAY"
        self.animation_start_time = time.time()
        self.initial_time = time.time()
        self.velocities = []
        self.boids_view_radius = 2.0
        self.boids_max_speed = 3.0
        self.cohesion_weight = 1.0
        self.separation_weight = 1.5
        self.alignment_weight = 1.0
        self.separation_distance = 0.5

    def load_file(self, file: str, vertex_sample_rate: int = 2):
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
                    if vertex_index % vertex_sample_rate == 0:
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
        # ... (draw_vertices code remains the same) ...
        glPushMatrix()
        glTranslatef(self.position.x, self.position.y, self.position.z)
        glRotatef(
            self.rotation[3], self.rotation[0], self.rotation[1], self.rotation[2]
        )

        colors = {
            "SWAY": (0.0, 0.0, 1.0),  # Blue
            "FALL": (1.0, 0.5, 0.0),  # Orange
            "TORNADO": (0.5, 0.5, 0.5),  # Grey
            "BOIDS": (0.0, 1.0, 0.0),  # Green
            "REASSEMBLE": (1.0, 1.0, 0.0),  # Yellow
            "DONE": (1.0, 1.0, 1.0),  # White
        }
        glColor3f(*colors.get(self.animation_state, (1, 1, 1)))

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
        # self.animation_state = "BOIDS" # Debug

        if self.animation_state == "SWAY":
            self.animate_sway(elapsed_animation_time)
            if elapsed_animation_time > 6:
                self.transition_to_state("FALL")
        elif self.animation_state == "FALL":
            self.animate_fall(dt)
            if elapsed_animation_time > 10:
                self.transition_to_state("TORNADO")
        elif self.animation_state == "TORNADO":
            self.animate_tornado(dt)
            if elapsed_animation_time > 12:
                self.transition_to_state("BOIDS")
        elif self.animation_state == "BOIDS":
            ## UPDATED TO CALL THE OPTIMIZED FUNCTION ##
            self.boids_update(dt)
            if elapsed_animation_time > 20:
                self.transition_to_state("REASSEMBLE")
        elif self.animation_state == "REASSEMBLE":
            self.animate_reassemble(dt)
            if elapsed_animation_time > 8:
                self.transition_to_state("DONE")

    # ... (transition_to_state, sway, fall, tornado, reassemble functions remain the same) ...
    def transition_to_state(self, new_state):
        print(f"Transitioning from {self.animation_state} to {new_state}")
        self.animation_state = new_state
        self.animation_start_time = time.time()

        if new_state == "FALL":
            for i in range(len(self.velocities)):
                self.velocities[i] = Point(0, 0, 0)
        elif new_state in ["TORNADO", "BOIDS"]:
            for i in range(len(self.velocities)):
                self.velocities[i] = Point(
                    random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)
                )
        elif new_state == "SWAY":
            for i in range(len(self.vertices)):
                self.vertices[i] = self.original_vertices[i]

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

    def animate_tornado(self, dt):
        center = Point(0, -2, 0)
        up_speed, rot_speed, suck_str = 1.5, 15.0, 2.0
        for i, v in enumerate(self.vertices):
            to_center = Point(center.x - v.x, 0, center.z - v.z)
            dist = to_center.magnitude() or 0.1
            radial_force = to_center.normalized() * suck_str
            tangent_force = Point(-to_center.z, 0, to_center.x).normalized() * (
                rot_speed / dist
            )
            up_force = Point(0, up_speed / (1 + dist * 0.5), 0)
            self.velocities[i] += (radial_force + tangent_force + up_force) * dt
            self.vertices[i] = v + self.velocities[i] * dt
            self.velocities[i] *= 0.98

    def boids_update(self, dt):
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
        
        Splits the vertices into exactly 4 flocks of equal (or nearly equal) size.
        """
        N = len(self.vertices)
        if N == 0:
            return

        # 1) Determine flock‐assignment (4 flocks, by index range). 
        #    e.g. if N=1000, flock_size=250→
        #      flock 0 = [0..249], flock 1 = [250..499], flock 2 = [500..749], flock 3 = [750..999].
        n_flocks = 4
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
