#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      KIIT
#
# Created:     06-11-2023
# Copyright:   (c) KIIT 2023
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import imageio
import matplotlib.pyplot as plt
from random import randint, random
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

# Constants
N = 100  # Number of squares
h, w = 100, 100  # Height and width of the canvas
mutation_rate = 0.1  # Initial mutation rate

# Step 1: Generate a Canvas
canvas = np.ones((h, w, 3), dtype=np.float32)

# Step 2: Generate the Initial Population
population = []

for _ in range(N):
    square = {
        'x': randint(0, w - 1),
        'y': randint(0, h - 1),
        'color': np.random.rand(3),
        'opacity': random()
    }
    population.append(square)

# Define a function to render the canvas based on the population
def render_population(canvas, population):
    canvas_copy = canvas.copy()
    for square in population:
        x, y = square['x'], square['y']
        color = square['color']
        opacity = square['opacity']
        canvas_copy[y:y + 10, x:x + 10] = canvas_copy[y:y + 10, x:x + 10] * (1 - opacity) + color * opacity
    return canvas_copy

# Step 3: Crossover (not implemented in this code)

# Step 4: Mutation
def mutate(square):
    mutated_square = square.copy()
    if random() < mutation_rate:
        mutated_square['x'] = randint(0, w - 1)
    if random() < mutation_rate:
        mutated_square['y'] = randint(0, h - 1)
    if random() < mutation_rate:
        mutated_square['color'] = np.random.rand(3)
    if random() < mutation_rate:
        mutated_square['opacity'] = random()
    return mutated_square

# Step 5: Selection
# You can define your selection strategy here

# Load the target image from your directory
target_image_path = 'fruit.jpg'  # Specify the path to your image
target_im = imageio.imread(target_image_path)
target_im = np.asarray(target_im/255, dtype=float)

# Evolution loop
num_generations = 100
for generation in range(num_generations):
    # Crossover (not implemented in this code)

    # Mutation
    mutated_population = [mutate(square) for square in population]

    # Selection (you need to implement this)

    # Update the population
    population = mutated_population

    # Render the current population and display it
    canvas_with_squares = render_population(canvas, population)

pixelated_canvas = resize(canvas_with_squares, (h // 2, w // 2), anti_aliasing=False)
pixelated_canvas = resize(pixelated_canvas, (h, w), anti_aliasing=False)

# Display the final pixelated result
plt.imshow(pixelated_canvas)

plt.imshow(pixelated_canvas)
plt.title("Final Result")
plt.show()

# Save the final  image
output_image_path = 'output.jpg'
imageio.imwrite(output_image_path, (pixelated_canvas * 255).astype(np.uint8))
