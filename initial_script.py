#!python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Config (tweak as needed)
# -----------------------------
N          = 50          # number of particles
RADIUS     = 0.003        # particle radius (all equal here)
DT         = 0.005       # time step
STEPS      = 1_000      # total simulated steps
SEED       = 2           # RNG seed (None for random)
SPEED_INIT = 0.5         # sets initial speed scale
BOX_MIN, BOX_MAX = 0.0, 1.0
ELASTIC_E  = 0.8         # coefficient of restitution (1.0 = elastic)

rng = np.random.default_rng(SEED)

# -----------------------------
# Helpers
# -----------------------------
def place_nonoverlapping(n, radius):
    """Random initial positions with no overlaps."""
    pos = np.empty((n, 2))
    for i in range(n):
        for _ in range(10000):
            candidate = rng.uniform(BOX_MIN + radius, BOX_MAX - radius, size=2)
            if i == 0:
                pos[i] = candidate
                break
            d2 = np.sum((pos[:i] - candidate) ** 2, axis=1)
            if np.all(d2 >= (2 * radius) ** 2):
                pos[i] = candidate
                break
        else:
            raise RuntimeError("Could not place particles without overlap.")
    return pos

def wall_collisions(pos, vel, radius):
    """Reflective walls with clamping."""
    for d in range(2):
        low_hit  = pos[:, d] < BOX_MIN + radius
        high_hit = pos[:, d] > BOX_MAX - radius
        vel[low_hit, d]  *= -1
        vel[high_hit, d] *= -1
        pos[:, d] = np.clip(pos[:, d], BOX_MIN + radius, BOX_MAX - radius)

def pairwise_collisions(pos, vel, mass, radius, e=1.0):
    """
    Resolve hard-sphere collisions (2D) with coefficient of restitution e.
    Handles different masses. Splits overlap proportionally to inverse mass.
    """
    n = pos.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            dist2 = rij @ rij
            min_dist = 2 * radius
            if dist2 < min_dist * min_dist:
                dist = np.sqrt(dist2) if dist2 > 0 else 1e-12
                n_hat = rij / dist

                # Separate overlapping positions (small correction)
                overlap = min_dist - dist
                if overlap > 0:
                    inv_m_sum = (1.0 / mass[i] + 1.0 / mass[j])
                    pos[i] += n_hat * (overlap * (1.0 / mass[i]) / inv_m_sum)
                    pos[j] -= n_hat * (overlap * (1.0 / mass[j]) / inv_m_sum)

                # Relative velocity along collision normal
                rel_v = vel[i] - vel[j]
                vn = rel_v @ n_hat
                if vn >= 0:
                    continue  # moving apart after position fix

                # Impulse for 1D elastic collision along n_hat
                j_imp = -(1 + e) * vn / (1.0 / mass[i] + 1.0 / mass[j])
                vel[i] += (j_imp / mass[i]) * n_hat
                vel[j] -= (j_imp / mass[j]) * n_hat

def kinetic_energy(vel, mass):
    return 0.5 * np.sum(mass * np.sum(vel**2, axis=1))

# -----------------------------
# Initialization
# -----------------------------
pos  = place_nonoverlapping(N, RADIUS)
# random directions, controlled speed
theta = rng.uniform(0, 2 * np.pi, size=N)
speed = SPEED_INIT * rng.uniform(0.7, 1.3, size=N)  # slight spread
vel   = np.stack([speed * np.cos(theta), speed * np.sin(theta)], axis=1)
mass  = rng.uniform(0.5, 3.0, size=N)               # different masses

# -----------------------------
# Animation setup
# -----------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.set_xlim(BOX_MIN, BOX_MAX)
ax.set_ylim(BOX_MIN, BOX_MAX)
ax.set_xticks([]); ax.set_yticks([])
scat = ax.scatter(pos[:, 0], pos[:, 1], s=(RADIUS * 1000) ** 2, alpha=0.8)

# running KE average
ke_hist = []
txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

def step():
    """Advance the system by one time step."""
    global pos, vel
    # free flight
    pos += vel * DT
    # walls
    wall_collisions(pos, vel, RADIUS)
    # pairwise collisions
    pairwise_collisions(pos, vel, mass, RADIUS, e=ELASTIC_E)

def init():
    scat.set_offsets(pos)
    ke = kinetic_energy(vel, mass)
    ke_hist.clear()
    ke_hist.append(ke)
    txt.set_text(f"KE: {ke:.3f}  ⟨KE⟩: {ke:.3f}")
    return scat, txt

def update(frame):
    # advance a few physics substeps per frame for smoother motion
    substeps = 2
    for _ in range(substeps):
        step()
    scat.set_offsets(pos)

    ke = kinetic_energy(vel, mass)
    ke_hist.append(ke)
    avg_ke = np.mean(ke_hist)
    txt.set_text(f"KE: {ke:.3f}  ⟨KE⟩: {avg_ke:.3f}")
    return scat, txt
def main():
    # parser block
    parser = argparse.ArgumentParser(description="A brief description of the script")
    parser.add_argument("--video", action="store_true", help="Make a video of the particles in the playground")
    args = parser.parse_args()

    ani = FuncAnimation(fig, update, frames=STEPS, init_func=init,
                        interval=20, blit=True)

    plt.show()
    if args.video:
        ani.save("particles.mp4", writer="ffmpeg", fps=30, dpi=200)

if __name__ == "__main__":
    main()
