#!python3
import argparse
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Config (tweak as needed)
# -----------------------------
N           = 50           # number of particles
RADIUS      = 0.003        # particle radius (all equal here)
DT          = 0.005        # time step
STEPS       = 1_000        # total simulated steps
SEED        = 2            # RNG seed (None for random)
SPEED_INIT  = 0.5          # sets initial speed scale
BOX_MIN     = 0.0
BOX_MAX     = 1.0
ELASTIC_E   = 0.8          # coefficient of restitution (1.0 = elastic)
SUBSTEPS    = 2            # physics substeps per frame

# -----------------------------
# Particle system
# -----------------------------
class ParticleSystem:
    def __init__(self, n, radius, dt, e, box_min=0.0, box_max=1.0,
                 seed=None, speed_init=0.5):
        self.n = n
        self.radius = radius
        self.dt = dt
        self.e = e
        self.box_min = box_min
        self.box_max = box_max

        self.rng = np.random.default_rng(seed)

        # state
        self.pos = self._place_nonoverlapping()
        theta = self.rng.uniform(0, 2*np.pi, size=n)
        speed = speed_init * self.rng.uniform(0.7, 1.3, size=n)
        self.vel = np.stack([speed*np.cos(theta), speed*np.sin(theta)], axis=1)
        self.mass = self.rng.uniform(0.5, 3.0, size=n)

        # diagnostics
        self.ke_hist = []

    # ----- helpers / physics -----
    def _place_nonoverlapping(self):
        pos = np.empty((self.n, 2))
        for i in range(self.n):
            for _ in range(10_000):
                cand = self.rng.uniform(self.box_min + self.radius,
                                        self.box_max - self.radius, size=2)
                if i == 0:
                    pos[i] = cand
                    break
                d2 = np.sum((pos[:i] - cand)**2, axis=1)
                if np.all(d2 >= (2*self.radius)**2):
                    pos[i] = cand
                    break
            else:
                raise RuntimeError("Could not place particles without overlap.")
        return pos

    def _wall_collisions(self):
        for d in range(2):
            low_hit  = self.pos[:, d] < self.box_min + self.radius
            high_hit = self.pos[:, d] > self.box_max - self.radius
            self.vel[low_hit, d]  *= -1
            self.vel[high_hit, d] *= -1
            self.pos[:, d] = np.clip(self.pos[:, d],
                                     self.box_min + self.radius,
                                     self.box_max - self.radius)

    def _pairwise_collisions(self):
        n = self.pos.shape[0]
        rmin = 2 * self.radius
        rmin2 = rmin * rmin
        for i in range(n):
            for j in range(i+1, n):
                rij = self.pos[i] - self.pos[j]
                dist2 = rij @ rij
                if dist2 < rmin2:
                    dist = np.sqrt(dist2) if dist2 > 0 else 1e-12
                    n_hat = rij / dist

                    # resolve overlap by inverse-mass split
                    overlap = rmin - dist
                    if overlap > 0:
                        inv_m_sum = (1.0 / self.mass[i] + 1.0 / self.mass[j])
                        self.pos[i] += n_hat * (overlap * (1.0 / self.mass[i]) / inv_m_sum)
                        self.pos[j] -= n_hat * (overlap * (1.0 / self.mass[j]) / inv_m_sum)

                    # normal impulse
                    rel_v = self.vel[i] - self.vel[j]
                    vn = rel_v @ n_hat
                    if vn >= 0:
                        continue
                    j_imp = -(1 + self.e) * vn / (1.0 / self.mass[i] + 1.0 / self.mass[j])
                    self.vel[i] += (j_imp / self.mass[i]) * n_hat
                    self.vel[j] -= (j_imp / self.mass[j]) * n_hat

    def step(self):
        # free flight
        self.pos += self.vel * self.dt
        # collisions
        self._wall_collisions()
        self._pairwise_collisions()

    def kinetic_energy(self):
        return 0.5 * np.sum(self.mass * np.sum(self.vel**2, axis=1))

    def reset_ke_history(self):
        self.ke_hist.clear()
        ke = self.kinetic_energy()
        self.ke_hist.append(ke)
        return ke

# -----------------------------
# Animation callbacks
# -----------------------------
def init_anim(scat, txt, particles: ParticleSystem):
    scat.set_offsets(particles.pos)
    ke = particles.reset_ke_history()
    txt.set_text(f"KE: {ke:.3f}  ⟨KE⟩: {ke:.3f}")
    return scat, txt

def update_anim(frame, scat, txt, particles: ParticleSystem):
    for _ in range(SUBSTEPS):
        particles.step()
    scat.set_offsets(particles.pos)

    ke = particles.kinetic_energy()
    particles.ke_hist.append(ke)
    avg_ke = np.mean(particles.ke_hist)
    txt.set_text(f"KE: {ke:.3f}  ⟨KE⟩: {avg_ke:.3f}")
    return scat, txt

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="2-D particle playground with hard-sphere collisions.")
    parser.add_argument("--video", action="store_true",
                        help="Save animation to particles.mp4 (requires ffmpeg).")
    args = parser.parse_args()

    # system
    particles = ParticleSystem(
        n=N, radius=RADIUS, dt=DT, e=ELASTIC_E,
        box_min=BOX_MIN, box_max=BOX_MAX,
        seed=SEED, speed_init=SPEED_INIT
    )

    # figure / artists
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_xlim(BOX_MIN, BOX_MAX)
    ax.set_ylim(BOX_MIN, BOX_MAX)
    ax.set_xticks([]); ax.set_yticks([])
    scat = ax.scatter(particles.pos[:, 0], particles.pos[:, 1],
                      s=(RADIUS * 1000) ** 2, alpha=0.8)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    # animation
    ani = FuncAnimation(
        fig,
        partial(update_anim, scat=scat, txt=txt, particles=particles),
        frames=STEPS,
        init_func=partial(init_anim, scat=scat, txt=txt, particles=particles),
        interval=20,
        blit=True
    )

    plt.show()

    if args.video:
        ani.save("particles.mp4", writer="ffmpeg", fps=30, dpi=200)

if __name__ == "__main__":
    main()
