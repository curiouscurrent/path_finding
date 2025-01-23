import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from ultralytics import YOLO
import streamlit as st
from pyswarm import pso  # Import PSO library

# Streamlit file uploader
st.title("Jalswarm: AI-powered Waste Collection, Navigation, and Disposal System")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Allow user to enter number of agents dynamically
num_agents = st.number_input("Enter number of agents", min_value=1, max_value=5, value=2)

if uploaded_file:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    
    # Convert the image to RGB for processing and display
    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)  # Convert to NumPy array for processing
    
    # Display the uploaded image
    st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

    # Load YOLO model
    model = YOLO('best_p6.pt')  # Specify the path to your model file

    # Perform inference to detect trash objects
    results = model.predict(source=image_np)

    # Initialize a list to store trash coordinates
    trash_coords = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            if class_id == 1:  # Assuming class 1 is trash
                x_center, y_center, _, _ = box.xywh[0]
                trash_coords.append((x_center, y_center))

    # Initialize agent parameters
    collected_trash = set()
    agent_positions = [trash_coords[0]] * num_agents  # Start all agents at the first trash object
    agent_paths = [[] for _ in range(num_agents)]  # Empty list for each agent's path

    # APF Parameters
    attraction_strength = 1.0
    repulsion_strength = 1.5
    repulsion_distance = 50  # Distance at which agents start to repel each other
    step_size = 5  # Step size for agents' movements

    # PSO Parameters
    def fitness_function(positions):
        """
        Fitness function for PSO, computes the total distance covered by all agents
        """
        global collected_trash, trash_coords
        total_distance = 0
        for idx, agent_position in enumerate(positions):
            closest_trash = min(range(len(trash_coords)), key=lambda x: np.linalg.norm(np.array(agent_position) - np.array(trash_coords[x])))
            total_distance += np.linalg.norm(np.array(agent_position) - np.array(trash_coords[closest_trash]))
        return total_distance  # Minimize this value

    lb = [0, 0] * num_agents  # Lower bounds for agent positions (example: start positions)
    ub = [image_np.shape[1], image_np.shape[0]] * num_agents  # Upper bounds for agent positions

    # Function to display the final frame with color-coded paths
    def display_final_frame(image_rgb, trash_coords, agent_paths, num_agents):
        fig, ax = plt.subplots()
        ax.imshow(image_rgb)
        ax.axis("off")

        # Plot all trash coordinates
        for coord in trash_coords:
            ax.plot(coord[0], coord[1], "bo", label="Trash")

        # Plot paths taken by agents
        colors = ["r", "g", "b", "c", "m"]  # Colors for agents
        for agent_idx in range(num_agents):
            if len(agent_paths[agent_idx]) > 0:  # Ensure the agent has a path
                path_coords = [trash_coords[node] for node in agent_paths[agent_idx]]
                for i in range(len(path_coords) - 1):
                    start_point = path_coords[i]
                    end_point = path_coords[i + 1]
                    ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], colors[agent_idx], lw=2, label=f"Agent {agent_idx + 1}")
                # Highlight the agent's final destination
                ax.plot(path_coords[-1][0], path_coords[-1][1], "o", color=colors[agent_idx], label=f"Final position of Agent {agent_idx + 1}")

        # Display the final image
        ax.legend()
        st.pyplot(fig)

    # Collect trash with dynamic recalculation
    def dynamic_path_recalculation(trash_coords, num_agents):
        frame_num = 0
        while len(collected_trash) < len(trash_coords):
            for agent_idx in range(num_agents):
                agent_position = agent_positions[agent_idx]

                # Check if only one piece of trash remains
                remaining_trash = [i for i in range(len(trash_coords)) if i not in collected_trash]
                if not remaining_trash:
                    break

                # Assign the agent to the only trash piece if there's exactly one
                if len(remaining_trash) == 1:
                    closest_trash = remaining_trash[0]
                    goal_position = trash_coords[closest_trash]

                    # Move directly towards the trash
                    agent_positions[agent_idx] = goal_position
                    collected_trash.add(closest_trash)
                    agent_paths[agent_idx].append(closest_trash)
                    break

                # Standard case for multiple trash pieces
                closest_trash = min(remaining_trash, key=lambda x: euclidean(agent_position, trash_coords[x]))
                goal_position = trash_coords[closest_trash]

                # Calculate the attractive force towards the goal
                direction_to_goal = np.array(goal_position) - np.array(agent_position)
                distance_to_goal = np.linalg.norm(direction_to_goal)

                if distance_to_goal != 0:
                    direction_to_goal /= distance_to_goal  # Normalize to get the direction vector

                # Repulsive forces from other agents
                repulsion_force = np.array([0.0, 0.0])
                for other_agent_idx in range(num_agents):
                    if other_agent_idx != agent_idx:
                        other_agent_position = agent_positions[other_agent_idx]
                        distance_between_agents = euclidean(agent_position, other_agent_position)
                        if distance_between_agents < repulsion_distance:
                            # Resolve collision using repulsion
                            repulsion_direction = np.array(agent_position) - np.array(other_agent_position)
                            repulsion_force += repulsion_direction / (distance_between_agents + 1e-6)  # Avoid division by zero

                # Apply attractive and repulsive forces
                force = attraction_strength * direction_to_goal + repulsion_strength * repulsion_force
                force_magnitude = np.linalg.norm(force)
                force /= force_magnitude if force_magnitude != 0 else 1  # Normalize force vector

                # Update agent position
                new_position = np.array(agent_position) + force * step_size
                agent_positions[agent_idx] = tuple(new_position)

                # Check if the agent reached its goal (within a threshold)
                if euclidean(agent_positions[agent_idx], goal_position) < 10:
                    collected_trash.add(closest_trash)
                    agent_paths[agent_idx].append(closest_trash)

            # Display the final frame once all trash is collected
            if len(collected_trash) == len(trash_coords):
                display_final_frame(image_rgb, trash_coords, agent_paths, num_agents)
                break

        return agent_paths

    # Call the dynamic path recalculation function for multiple trash objects
    agent_paths = dynamic_path_recalculation(trash_coords, num_agents)
