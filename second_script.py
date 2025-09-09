#!python3
import argparse
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

# Import the ParticleSystem from the original script
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from initial_script import ParticleSystem

# -----------------------------
# Config (using same values as original)
# -----------------------------
N           = 100           # number of particles
RADIUS      = 0.003        # particle radius (all equal here)
DT          = 0.005        # time step
STEPS       = 1_000        # total simulated steps
SEED        = 2            # RNG seed (None for random)
SPEED_INIT  = 0.5          # sets initial speed scale
BOX_MIN     = 0.0
BOX_MAX     = 1.0
ELASTIC_E   = 0.6          # coefficient of restitution (1.0 = elastic)
SUBSTEPS    = 2            # physics substeps per frame

# Thermometer configuration
THERMO_WIDTH = 0.08
THERMO_HEIGHT = 0.6
THERMO_X = 1.05
THERMO_Y = 0.2
BULB_RADIUS = 0.06

class ThermometerVisualizer:
    def __init__(self, ax, initial_ke, x=THERMO_X, y=THERMO_Y, 
                 width=THERMO_WIDTH, height=THERMO_HEIGHT, bulb_radius=BULB_RADIUS):
        self.ax = ax
        self.initial_ke = initial_ke
        self.max_ke = initial_ke * 1.1  # Allow for slight increase
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.bulb_radius = bulb_radius
        
        # Create thermometer components
        self._create_thermometer()
        
    def _create_thermometer(self):
        # Thermometer tube (outline)
        self.tube_outline = Rectangle(
            (self.x - self.width/2, self.y), 
            self.width, self.height,
            linewidth=2, edgecolor='black', facecolor='white', alpha=0.8
        )
        self.ax.add_patch(self.tube_outline)
        
        # Thermometer bulb (outline)
        self.bulb_outline = plt.Circle(
            (self.x, self.y - self.bulb_radius/2), 
            self.bulb_radius,
            linewidth=2, edgecolor='black', facecolor='white', alpha=0.8
        )
        self.ax.add_patch(self.bulb_outline)
        
        # Mercury/fluid in bulb
        self.bulb_fluid = plt.Circle(
            (self.x, self.y - self.bulb_radius/2), 
            self.bulb_radius * 0.8,
            facecolor='red', alpha=0.8
        )
        self.ax.add_patch(self.bulb_fluid)
        
        # Mercury/fluid in tube (starts full)
        self.tube_fluid = Rectangle(
            (self.x - self.width/2 + 0.005, self.y), 
            self.width - 0.01, self.height,
            facecolor='red', alpha=0.8
        )
        self.ax.add_patch(self.tube_fluid)
        
        # Temperature scale labels
        self._add_scale_labels()
        
    def _add_scale_labels(self):
        # Add temperature scale on the right side
        scale_x = self.x + self.width/2 + 0.02
        
        # Hot label at top
        self.ax.text(scale_x, self.y + self.height, 'HOT', 
                    fontsize=10, fontweight='bold', color='red',
                    verticalalignment='top')
        
        # Cold label at bottom
        self.ax.text(scale_x, self.y, 'COLD', 
                    fontsize=10, fontweight='bold', color='blue',
                    verticalalignment='bottom')
        
        # Title
        self.ax.text(self.x, self.y + self.height + 0.08, 'Kinetic Energy', 
                    fontsize=12, fontweight='bold',
                    horizontalalignment='center')
        
    def update(self, current_ke):
        # Calculate the fill ratio based on kinetic energy
        # When KE decreases, thermometer level decreases
        if self.max_ke > 0:
            fill_ratio = max(0, min(1, current_ke / self.max_ke))
        else:
            fill_ratio = 0
            
        # Update the tube fluid height
        new_height = self.height * fill_ratio
        self.tube_fluid.set_height(new_height)
        
        # Change color based on energy level
        if fill_ratio > 0.7:
            color = 'red'
        elif fill_ratio > 0.4:
            color = 'orange'
        elif fill_ratio > 0.2:
            color = 'yellow'
        else:
            color = 'blue'
            
        self.tube_fluid.set_facecolor(color)
        self.bulb_fluid.set_facecolor(color)

# -----------------------------
# Animation callbacks
# -----------------------------
def init_anim(scat, txt, thermometer, particles: ParticleSystem):
    scat.set_offsets(particles.pos)
    ke = particles.reset_ke_history()
    thermometer.update(ke)
    txt.set_text(f"KE: {ke:.3f}  ⟨KE⟩: {ke:.3f}")
    return scat, txt

def update_anim(frame, scat, txt, thermometer, particles: ParticleSystem):
    for _ in range(SUBSTEPS):
        particles.step()
    scat.set_offsets(particles.pos)

    ke = particles.kinetic_energy()
    particles.ke_hist.append(ke)
    avg_ke = np.mean(particles.ke_hist)
    
    # Update thermometer
    thermometer.update(ke)
    
    txt.set_text(f"KE: {ke:.3f}  ⟨KE⟩: {avg_ke:.3f}")
    return scat, txt

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="2-D particle playground with thermometer showing kinetic energy.")
    parser.add_argument("--video", action="store_true",
                        help="Save animation to particles_thermo.mp4 (requires ffmpeg).")
    parser.add_argument("--add_red", action="store_true",
                        help="Adds a fast red particle to the playground.")
    args = parser.parse_args()

    # system
    particles = ParticleSystem(
        n=N, radius=RADIUS, dt=DT, e=ELASTIC_E,
        box_min=BOX_MIN, box_max=BOX_MAX,
        seed=SEED, speed_init=SPEED_INIT
    )

    if args.add_red:
        particles.add_red((0.8,0.01), (-1,4), mass=1.0)
        colors = np.tile([[0, 0, 1, 1]], (particles.n, 1))
        colors[-1] = [1, 0, 0, 1]
    else:
        colors = np.tile([[0, 0, 1, 1]], (particles.n, 1))

    # figure / artists with extra space for thermometer
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.set_xlim(BOX_MIN, BOX_MAX + 0.3)  # Extra space for thermometer
    ax.set_ylim(BOX_MIN, BOX_MAX)
    ax.set_xticks([]); ax.set_yticks([])
    
    # Particle scatter plot
    scat = ax.scatter(particles.pos[:, 0], particles.pos[:, 1],
                      s=(RADIUS * 1000) ** 2, alpha=0.8, c=colors)

    # Kinetic energy text
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")
    
    # Create thermometer
    initial_ke = particles.kinetic_energy()
    thermometer = ThermometerVisualizer(ax, initial_ke)

    # animation
    ani = FuncAnimation(
        fig,
        partial(update_anim, scat=scat, txt=txt, thermometer=thermometer, particles=particles),
        frames=STEPS,
        init_func=partial(init_anim, scat=scat, txt=txt, thermometer=thermometer, particles=particles),
        interval=20,
        blit=False  # Set to False to allow thermometer updates
    )

    plt.tight_layout()
    plt.show()

    if args.video:
        ani.save("particles_thermo.mp4", writer="ffmpeg", fps=30, dpi=200)

if __name__ == "__main__":
    main()
