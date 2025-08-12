import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from shapely.geometry import Polygon, Point # type: ignore

# LLPHits.py
# Takes in cavern geometry and data from Pythia LLP simulation csv to
# see for what phis and thetas the LLPs hit the cavern at a cross-section
# at z = 22.5 m. Plots results as histograms of phi and theta vs counts,
# and phi vs theta.

# --- Helper functions ---

def circular_arc(center, radius, start_angle, end_angle, num_points):
    """Generate points along a circular arc."""
    angles = np.linspace(start_angle, end_angle, num_points)
    return [
        (
            center[0] + radius * np.cos(angle),
            center[1] + radius * np.sin(angle)
        )
        for angle in angles
    ]

def arctan(center, amp, start, stop, points):
    """Generate points along an arctangent curve."""
    angles = np.linspace(start, stop, points)
    return [
        (
            center[1] + np.radians(angle),
            center[0] + amp * np.arctan(np.radians(angle) - center[1])
        )
        for angle in angles
    ]

# --- Define tunnel geometry (outer and inner boundary) ---

inside1 = [(-12, 12.1), (-13, 12.1), (-14, 12.1), (-15, 12.1), (-16, 12.1), (-17, 12.1), (-18, 12.1)]
outer1 = [(-12, 14.9), (-13, 14.9), (-14, 14.9), (-15, 14.9), (-16, 14.9), (-17, 14.9), (-18, 14.9)]

inside2 = circular_arc((-18, 6.5), 5.6, np.radians(90), np.radians(180), 15)
outer2 = circular_arc((-18, 6.5), 5.6 + 2.8, np.radians(90), np.radians(180), 15)

inside3 = [(-23.6, y) for y in np.arange(5.5, -4.5, -1.0)]
outer3 = [(-26.4, y) for y in np.arange(5.5, -4.5, -1.0)]

inside4 = circular_arc((-18, -3.5), 5.6, np.radians(180), np.radians(270), 15)
outer4 = circular_arc((-18, -3.5), 5.6 + 2.8, np.radians(180), np.radians(270), 15)

inside5 = [(-18, -9.1), (2, -9.1)]
outer5 = [(-18, -11.9), (2, -11.9)]

inside6 = arctan((-2.5, 2), 6.5, 90, 270, 20)
outer6 = arctan((-4.48, 0.02), 6.5, 90, 270, 20)

# Combine all parts to define detector polygon
outer = outer1 + outer2 + outer3 + outer4 + outer5  # + outer6 if desired
inside = inside1 + inside2 + inside3 + inside4 + inside5  # + inside6 if desired

c_detector = Polygon(outer + inside[::-1])  # Full polygon region

#/

df = pd.read_csv("LLP.csv")

# show first few rows
print(df.head())

print("\n")
# show column names to confirm formatting
#print("Columns:", df.columns.tolist())

# convert eta to theta
df['theta'] = 2 * np.arctan(np.exp(-df['eta']))

# convert to px, py, pz
df['px'] = df['momentum'] * np.sin(df['theta']) * np.cos(df['phi'])
df['py'] = df['momentum'] * np.sin(df['theta']) * np.sin(df['phi'])
df['pz'] = df['momentum'] * np.cos(df['theta'])

# show the first few to check
print(df[['event', 'id', 'px', 'py', 'pz']].head())

print("\n")

grouped = df.groupby('event')

# initialize empty array to store momentum vectors of each llp (as unit vectors)
llp_vectors = []

# Example: Loop over each event pair
for event_id, group in grouped:
    if len(group) != 2:
        print(f"Warning: Event {event_id} does not have 2 particles!")
        continue

    p1 = group.iloc[0]
    p2 = group.iloc[1]

    # print their px, py, pz, etc.
    #print(f"Event {event_id}:")
    #print(f"  Particle 1: px={p1['px']:.2f}, py={p1['py']:.2f}, pz={p1['pz']:.2f}")
    #print(f"  Particle 2: px={p2['px']:.2f}, py={p2['py']:.2f}, pz={p2['pz']:.2f}")

    # normalize vectors (can scale them later)
    v1 = np.array([p1['px'], p1['py'], p1['pz']])
    v2 = np.array([p2['px'], p2['py'], p2['pz']])
    
    # store both llps per event
    llp_vectors.append((v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)))

def simulate_particle(origin, direction, z_target=22.5):
    """
    Simulate a particle traveling from `origin` in `direction`,
    and check whether it hits the detector at z = z_target.
    
    Returns the (x, y) hit point if inside detector, else None.
    """
    if direction[2] == 0:
        return None  # Would never intersect the z = z_target plane

    t = (z_target - origin[2]) / direction[2]
    x = origin[0] + direction[0] * t
    y = origin[1] + direction[1] * t
    point = Point(x, y)

    if c_detector.contains(point):
        return (x, y)
    else:
        return None

# Initialize stats and containers
total_particles = 0
hits = 0
hit_coords = []
miss_coords = []
hit_theta = []
hit_phi = []
miss_theta = []
miss_phi = []

origin = np.array([0, 0, 0])
z_target = 22.5

for v1, v2 in llp_vectors:
    for v in (v1, v2):
        total_particles += 1
        hit = simulate_particle(origin, v)

        theta = np.arccos(v[2])  # polar angle
        phi = np.arctan2(v[1], v[0])  # azimuthal angle

        if hit is not None:
            hits += 1
            hit_coords.append(hit)
            hit_theta.append(np.degrees(theta))
            hit_phi.append(np.degrees(phi))
        else:
            # Compute projected point at z_target anyway for misses
            if v[2] != 0:
                t = (z_target - origin[2]) / v[2]
                if t > 0:
                    x_miss = origin[0] + v[0] * t
                    y_miss = origin[1] + v[1] * t
                    miss_coords.append((x_miss, y_miss))
                    miss_theta.append(np.degrees(theta))
                    miss_phi.append(np.degrees(phi))

print(f"\nTotal LLPs simulated: {total_particles}")
print(f"LLPs that hit detector: {hits}")
print(f"Detection efficiency: {hits / total_particles:.2%}")

# Convert to numpy arrays for plotting
hit_coords = np.array(hit_coords)
miss_coords = np.array(miss_coords) if miss_coords else np.empty((0, 2))

# --- Plot hits and misses with lines from origin ---
fig, ax = plt.subplots(figsize=(10, 8))

# Detector boundary (fill to show cavern nicely)
x_poly, y_poly = c_detector.exterior.xy
ax.fill(x_poly, y_poly, facecolor='lightgray', edgecolor='k', alpha=0.6, label='Cavern Boundary')

# Misses in faint gray
if len(miss_coords) > 0:
    ax.scatter(miss_coords[:, 0], miss_coords[:, 1], color='gray', s=10, alpha=0.3, label='Misses')

# Hits in blue
if len(hit_coords) > 0:
    ax.scatter(hit_coords[:, 0], hit_coords[:, 1], color='blue', s=15, alpha=0.9, label='Hits')
    # Draw lines from origin to hits
    for (xh, yh) in hit_coords:
        ax.plot([0, xh], [0, yh], color='blue', alpha=0.15, linewidth=0.8)

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title(f"LLP Impact Points on Cavern Plane (z = {z_target} m)\nHits = {hits}, Total = {total_particles}, Efficiency = {hits / total_particles:.2%}")
ax.axis('equal')
ax.legend()
plt.grid(alpha=0.3)

# Set sensible axis limits focusing on cavern and points
minx, miny, maxx, maxy = c_detector.bounds
margin = 5
ax.set_xlim(minx - margin, maxx + margin)
ax.set_ylim(miny - margin, maxy + margin)

plt.show()


# Plot theta histogram (hits only)
plt.figure()
plt.hist(hit_theta, bins=50, edgecolor='black', color='salmon')
plt.xlabel('Theta (degrees)')
plt.ylabel('Counts')
plt.title('Distribution of Allowed Theta Angles (Hits Only)')
plt.grid(True)
plt.show()

# Plot phi histogram (hits only)
plt.figure()
plt.hist(hit_phi, bins=50, edgecolor='black', color='dodgerblue')
plt.xlabel('Phi (degrees)')
plt.ylabel('Counts')
plt.title('Distribution of Allowed Phi Angles (Hits Only)')
plt.grid(True)
plt.show()

# 2D histogram (hits only)
plt.figure(figsize=(8, 6))
plt.hist2d(hit_phi, hit_theta, bins=[50, 50], cmap='Greys')
plt.colorbar(label='Counts')
plt.xlabel('Phi (degrees)')
plt.ylabel('Theta (degrees)')
plt.title('2D Histogram of Allowed Phi vs Theta Angles (Hits Only)')
plt.grid(True)
plt.show()
