# Development Notes for Wildfire Simulation Project

## Tasks to Address Current Issues

1. **Spatial Model Inconsistency**
    - **Issue**: The current spatial model does not place new fire cells close to the existing fire, leading to an unrealistic spread pattern.
    - **Action**: Integrate the logic from the `get_perturbation` function in `spatial_model.py` into the spatial model to make the fire spread more realistically. This function, currently unused, models the desired spatial dynamics.

2. **Scattered Parameters**
    - **Issue**: Parameters are scattered across multiple files, making it difficult to tune or update the model.
    - **Action**: Identify all parameters and consolidate them into a unified location, such as a separate constants file or a configuration file. Document the units and reasonable realistic values for each parameter to facilitate tuning and updating.

## Tasks to Implement Desired Features

3. **Temporal and Spatial Model Integration**
    - **Goal**: Visualize the "most likely footprint" of the fire, incorporating both the temporal and spatial dynamics.
    - **Action**: Investigate how to combine the solvability of the temporal model with the spatial dynamics to produce a meaningful visualization.

4. **Advanced Firefighting Model**
    - **Goal**: Add more complexity to the firefighting aspect of the model.
    - **Action**: Implement a delay in firefighting response and introduce different types of firefighting strategies with unique spatial dynamics. Research realistic firefighting strategies and their effects on fire spread to incorporate into the model.

5. **Realistic Fire Spread**
    - **Goal**: Implement a more realistic fire spread model.
    - **Action**: Introduce topography and wind into the spatial model. This will require additional research on how these factors influence fire spread in real-world scenarios.

## Tasks for Model Validation

6. **Temporal Model Validation**
    - **Goal**: Validate the temporal model against real-world data.
    - **Action**: Collect relevant real-world data and compare the model's output against it. Adjust the model parameters as necessary to improve accuracy.

7. **Spatial Model Validation**
    - **Goal**: Validate the spatial model against real-world data.
    - **Action**: Collect relevant real-world data and compare the model's output against it. Adjust the model parameters as necessary to improve accuracy.

8. **Combined Model Validation**
    - **Goal**: Validate the combined model against real-world data.
    - **Action**: Collect relevant real-world data and compare the model's output against it. Adjust the model parameters as necessary to improve accuracy.

## Further Improvement

9. **Real-World Accuracy**
    - **Goal**: Increase the real-world accuracy of the model.
    - **Action**: Incorporate more real-world factors into the model, such as different types of vegetation, varying weather conditions, and human interventions. Source this from real-world data where possible using APIs or other data sources.

10. **Runtime Optimization**
    - **Goal**: Optimize the runtime of the model.
    - **Action**: Profile the code to identify bottlenecks and optimize these sections, potentially through parallelization or more efficient algorithms.

11. **Ease of Collaboration**
    - **Goal**: Improve the ease of collaboration on the project.
    - **Action**: Implement version control, write comprehensive documentation, and establish coding standards to facilitate collaboration among team members.